import json
import os
import subprocess
import tempfile
import matplotlib.pyplot as plt
import torch
from graph_dyno.models.model_factory import create_model
from graph_dyno.data.trajectory_generator import generate_data, generate_config_hash
from graph_dyno.data.dataloader import create_split_dataloaders_from_folder, CUDAPrefetcher
from graph_dyno.training.validate import validate_all_sitting_cells, plot_sitting_cell_validation
from graph_dyno.encoders.factory import get_companion_suffix
from graph_dyno.training.finetune import load_all_sdfs, finetune_decoder, check_finetune_cache
from tqdm.auto import tqdm
from pathlib import Path
import sys
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf
import mlflow
from graph_dyno.plotting.plot import plot_training, plot_generated_images, plot_embedding_colored_by_changing_params
from graph_dyno.precompute.precompute_embeddings import precompute_sdfs
from schedulefree import AdamWScheduleFree



def _flatten_dict(d, parent_key='', sep='.'):
    """Flatten nested dict for MLflow params logging."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _resolve_mlflow_uri(path):
    """Return an MLflow tracking URI from a path or full URI.

    Pass-through for full URIs (file://, sqlite:, http(s)://). Otherwise
    resolves to an absolute path before constructing the URI: `.db` suffix
    -> sqlite backend, anything else -> file:// directory store.
    """
    if isinstance(path, str) and path.startswith(
        ("file://", "sqlite:", "http://", "https://")
    ):
        return path
    p = Path(path).resolve()
    if p.suffix == ".db":
        p.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{p}"
    p.mkdir(parents=True, exist_ok=True)
    return f"file://{p}"


_RUN_INFO_FILE = Path(tempfile.gettempdir()) / "train_run_info.md"


@torch.no_grad()
def _compute_displacement_magnitude_std(dataloader, model, device):
    """Estimate per-group displacement-magnitude std over the training set.

    For each sample, computes ||Δ||₂ along the component axis for positions
    (states[..., :S]), velocities (states[..., S:2S]) and bottleneck L, then
    accumulates sums / sums-of-squares across all batches and returns a scalar
    std per group.

    Accumulates only over `valid & next_valid` when the batch carries those
    keys (all-True fallback, so no existing run changes). A missed or
    ID-swapped slot produces an enormous Δ, and these sigmas scale
    `input_noise_std` — so without the mask an ablated arm would train with
    *larger regularisation noise* than `none`, for reasons unrelated to
    segmentation.

    Returns (sigma_x, sigma_v, sigma_L) as 0-d tensors on `device`.
    """
    S = model.spatial_dim
    sums = torch.zeros(3, device=device)
    sqs = torch.zeros(3, device=device)
    count = 0
    was_training = model.training
    model.eval()
    for batch in dataloader:
        if batch is None:
            continue
        states = batch["states"].to(device, non_blocking=True)
        next_states = batch["next_states"].to(device, non_blocking=True)
        L = batch["L"].to(device, non_blocking=True)
        L_next = batch["next_L"].to(device, non_blocking=True)
        B, N, _ = states.shape

        if "valid" in batch:
            m = (batch["valid"].to(device, non_blocking=True)
                 & batch["next_valid"].to(device, non_blocking=True))
        else:
            m = torch.ones(B, N, dtype=torch.bool, device=device)
        w = m.to(states.dtype)

        dx_mag = (next_states[..., :S] - states[..., :S]).norm(dim=-1)  # (B, N)
        dv_mag = (next_states[..., S:2 * S] - states[..., S:2 * S]).norm(dim=-1)
        dL_mag = (L_next - L).norm(dim=-1)

        for i, mag in enumerate((dx_mag, dv_mag, dL_mag)):
            sums[i] += (mag * w).sum()
            sqs[i] += (mag * mag * w).sum()
        count += int(m.sum())

    if was_training:
        model.train()
    if count == 0:
        raise RuntimeError("No samples in dataloader for displacement-std computation")

    means = sums / count
    var = (sqs / count - means ** 2).clamp_min(0)
    std = var.sqrt()
    return std[0], std[1], std[2]


def _write_run_info(hydra_output_dir):
    """Write run metadata to a temp file."""
    script_path = os.path.abspath(sys.argv[0])
    full_command = " ".join(sys.argv)
    hydra_dir = str(hydra_output_dir)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=os.path.dirname(script_path) or ".",
    ).stdout.strip()
    rerun_command = f"python {script_path} --config-path={hydra_dir}/.hydra --config-name=config"
    clip_md = (
        f"script: `{script_path}`\n"
        f"hydra_output_dir: `{hydra_dir}`\n"
        f"git_commit: `{git_commit}`\n"
        f"command: `{full_command}`\n"
        f"rerun: `{rerun_command}`\n"
    )
    _RUN_INFO_FILE.write_text(clip_md)
    try:
        subprocess.run(["xclip", "-selection", "clipboard"], input=clip_md, text=True, check=True)
    except Exception:
        pass
    print(f"Run info written to {_RUN_INFO_FILE} (and copied to clipboard)")


def _load_companion_embeddings(cache_dir, cache_id, companion_suffix, traj_indices):
    """Load precomputed embeddings from companion files into a dict.

    Returns:
        dict mapping traj_idx -> {"embeddings": (T,N,D), "crop_sides": (T,N)}
    """
    embeddings = {}
    for traj_idx in traj_indices:
        filename = f"{companion_suffix}_{cache_id}_{traj_idx:04d}.pt"
        path = os.path.join(cache_dir, filename)
        embeddings[traj_idx] = torch.load(path, map_location='cpu')
    return embeddings


def _normalize_scale(embeddings, cache_dir, cache_id, traj_indices, group_key):
    """Rescale crop sides, positions and velocities so every group's median valid crop side is equal.

    Removes cell size as a group fingerprint (e.g. the 7.5x HeLa/PSC gap in
    real_dic_psc_late) while keeping each group's relative size variation:
    crop sides are multiplied by target / median_g, where target is the
    geometric mean of the group medians. `group_key` names a per-slot
    `sim_metadata` field such as "dataset_id" or "video_id". The scale input,
    its velocity and the scale target all read these crop sides. States and
    next_states (x, y, vx, vy) are multiplied by the same factor, so relative
    positions and velocities share the group's rescaled units.

    Returns (medians, target, factors), the dicts keyed by group code.
    """
    groups, valid = {}, {}
    for traj_idx in traj_indices:
        traj = torch.load(os.path.join(cache_dir, f"{cache_id}_{traj_idx:04d}.pt"),
                          map_location="cpu", mmap=True, weights_only=False)
        if group_key not in traj.get("sim_metadata", {}):
            raise KeyError(f"scale_norm={group_key!r} is not a sim_metadata field")
        groups[traj_idx] = traj["sim_metadata"][group_key].long()
        crop_sides = embeddings[traj_idx]["crop_sides"]
        valid[traj_idx] = traj.get("valid", torch.ones_like(crop_sides, dtype=torch.bool))

    medians = {}
    for g in torch.unique(torch.cat(list(groups.values()))).tolist():
        sides = torch.cat([embeddings[t]["crop_sides"][valid[t] & (groups[t] == g)].float()
                           for t in traj_indices])
        medians[g] = sides.median().item()
    target = torch.tensor(list(medians.values())).log().mean().exp().item()
    factors = {g: target / m for g, m in medians.items()}

    for t in traj_indices:
        f = torch.tensor([factors[g] for g in groups[t].tolist()])  # (N,)
        cs = embeddings[t]["crop_sides"]
        # Rounded back to the stored dtype: downstream readers expect integer sides.
        embeddings[t]["crop_sides"] = torch.round(cs.float() * f).to(cs.dtype)
        traj = torch.load(os.path.join(cache_dir, f"{cache_id}_{t:04d}.pt"),
                          map_location="cpu", mmap=True, weights_only=False)
        for key in ("states", "next_states"):
            embeddings[t][key] = traj[key] * f[None, :, None]
    return medians, target, factors


@torch.no_grad()
def _precompute_bottlenecks(embeddings, shape_decoder, device, batch_size=4096,
                            src_key="embeddings", dst_key="L", pop=True):
    """Replace raw embeddings with bottleneck vectors L in-place.

    Runs `shape_decoder.get_bottleneck` once over every cached frame so the
    training loop never has to re-encode. Mutates the input dict: each entry's
    "embeddings" field is popped and replaced by "L" of shape (T, N, bn).

    `pop=False` + `dst_key="L_clean"` runs the same pass without consuming the
    raw array — used to snapshot the clean bottleneck before a seg-ablation
    sidecar overwrites it, which is far cheaper than copying the (T, N, D)
    raw embeddings.

    Args:
        embeddings: dict mapping traj_idx -> {"embeddings": (T,N,D), "crop_sides": (T,N)}
        shape_decoder: frozen ShapeDecoder (must have get_bottleneck).
        device: Device to run the encoder on.
        batch_size: Flat-token batch size for the encoder forward.

    Returns:
        The same dict, now with "L" instead of "embeddings".
    """
    was_training = shape_decoder.training
    shape_decoder.eval()
    shape_decoder.to(device)
    for traj_idx in list(embeddings.keys()):
        entry = embeddings[traj_idx]
        emb = entry.pop(src_key) if pop else entry[src_key]
        T, N, D = emb.shape
        flat = emb.reshape(T * N, D)
        chunks = []
        for i in range(0, flat.shape[0], batch_size):
            chunk = flat[i:i + batch_size].to(device, non_blocking=True)
            chunks.append(shape_decoder.get_bottleneck(chunk).cpu())
        L = torch.cat(chunks, dim=0).view(T, N, -1)
        entry[dst_key] = L
    if was_training:
        shape_decoder.train()
    return embeddings


def _log_seg_stats(seg_stats):
    """Realized seg-ablation rates and balances -> MLflow params.

    Realized, not requested: the planner drops an episode when no free window
    exists for both roles, and on a sparsely-connected cache (measured: 1.8
    union-distinct partners per cell on more_osc_clusters against 22-25 on
    membrane_stiffness/adhesion) that shortfall is the difference between "we ablated 5%"
    and "we ablated what the contact graph allowed".
    """
    flat = {}
    for s in seg_stats:
        i = s["traj_idx"]
        for k in ("invalid_frac", "n_reencoded", "n_merge_clipped",
                  "edges_kept_frac", "centroid_sigma_px", "vel_snr",
                  "median_disp_px", "sigma_x", "sigma_v"):
            flat[f"segabl.t{i}.{k}"] = s[k]
        for kind, d in s["per_kind"].items():
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    flat[f"segabl.t{i}.{kind}.{k}"] = v
        for kind, n in s["n_infeasible_slots"].items():
            flat[f"segabl.t{i}.{kind}.n_infeasible_slots"] = n
    mlflow.log_params(flat)
    s0 = seg_stats[0]
    print("  realized per-kind (traj 0):")
    for kind, d in s0["per_kind"].items():
        if not d.get("n_episodes"):
            continue
        print(f"    {kind:9s} rate={d['realized_rate']:.4f} "
              f"(placed {d['n_episodes']} of {d['n_requested']} requested) "
              f"mean_len={d.get('mean_len', 0):.1f} "
              f"spread init/partner={d.get('slot_count_spread', 0)}/"
              f"{d.get('partner_count_spread', 0)} "
              f"slots touched {d.get('n_slots_touched', '-')}"
              f"/{d.get('n_feasible_slots', '-')}")
    print(f"    centroid sigma {s0['centroid_sigma_px']:.3f} px "
          f"(velocity SNR {s0['vel_snr']:.2f}), "
          f"{(1 - s0['edges_kept_frac']) * 100:.2f}% edges dropped")
    worst = s0.get("worst_label_correlation")
    if worst is not None:
        print(f"    worst per-slot/label correlation: r({worst[0]}, {worst[1]}) "
              f"= {worst[2]:+.3f}")
        mlflow.log_params({
            "segabl.worst_corr_count": worst[0],
            "segabl.worst_corr_factor": worst[1],
            "segabl.worst_corr_r": worst[2],
        })


def get_rollout_steps(epoch, schedule):
    """Determine rollout steps for a given epoch from the curriculum schedule.

    Args:
        schedule: List of [epoch_threshold, steps] pairs, sorted by epoch.

    Returns:
        Number of rollout steps to use.
    """
    steps = 1
    for entry in schedule:
        if epoch >= entry[0]:
            steps = entry[1]
        else:
            break
    return steps


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Write run info to temp file immediately (before slow data generation)
    hydra_cfg = HydraConfig.get()
    is_multirun = hydra_cfg.mode == RunMode.MULTIRUN
    if not is_multirun or hydra_cfg.job.num == 0:
        output_dir = Path(hydra_cfg.sweep.dir) if is_multirun else Path(hydra_cfg.runtime.output_dir)
        _write_run_info(output_dir)

    # Handle device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Shorthand for data config
    data = cfg.data
    si = data.params  # simulation-specific params

    # Generate trajectory data
    print("Generating trajectory data...")
    cache_dir = generate_data(cfg, device=device)
    print(f"\nData saved to: {cache_dir}")

    # Auto-resolve shape config from checkpoint (e.g. dino_model, dino_use_patches)
    from graph_dyno.models.model_factory import resolve_shape_config
    resolve_shape_config(cfg)

    has_shape = cfg.get("shape", None) is not None
    companion_suffix = get_companion_suffix(cfg) if has_shape else None

    # --- Precompute SDFs (shared by all shape types) ---
    from graph_dyno.precompute.precompute_embeddings import precompute_all
    sdf_workers = cfg.shape.get("sdf_workers", 32)
    target_size = cfg.shape.get("target_size", 224)
    precompute_sdfs(cache_dir=cache_dir, cfg=cfg,
                    sdf_workers=sdf_workers, target_size=target_size)

    cache_id = generate_config_hash(cfg.data)
    metadata = OmegaConf.load(os.path.join(cache_dir, "metadata.yaml"))
    n_trajectories = metadata.data.generator.n_trajectories
    traj_indices = list(range(n_trajectories))

    # --- Precompute & load embeddings (unified for DINO/Fourier/Conv) ---
    # Conv embeddings are valid across finetuning because the encoder is
    # frozen during finetune_decoder (only model.dino_decoder is trained).
    embeddings = None
    if has_shape:
        precompute_all(cache_dir=cache_dir, cfg=cfg)
        embeddings = _load_companion_embeddings(
            cache_dir, cache_id, companion_suffix, traj_indices)

    # --- Set up MLflow (needed before finetuning logs metrics) ---
    save_path = Path(hydra_cfg.runtime.output_dir)
    print(f"\nExperiment output directory: {save_path}")

    mlflow.set_tracking_uri(_resolve_mlflow_uri(
        cfg.get("mlflow_uri", os.environ.get("MLFLOW_URI", "./mlruns"))))
    mlflow.set_experiment(cfg.experiment_name)
    run_description = cfg.get("run_description", None)
    mlflow.start_run(description=run_description)

    flat_cfg = OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_params(_flatten_dict(flat_cfg))
    mlflow.set_tag("hydra_output_dir", str(save_path))
    # Optional user-defined tags from Hydra config: `+tags.foo=bar` -> tag foo=bar
    user_tags = cfg.get("tags", None)
    if user_tags is not None:
        for k, v in OmegaConf.to_container(user_tags, resolve=True).items():
            mlflow.set_tag(k, v)
    mlflow.log_artifact(save_path / ".hydra/config.yaml")

    # Create model and optionally finetune (before dataloader)
    model = create_model(cfg, device=device)

    if (hasattr(model, 'dino_decoder')
            and cfg.shape.type != "fourier"
            and cfg.training.get("finetune_decoder", {}).get("enabled", False)):
        # Conv finetune still needs an encoder (for the SDF→bottleneck step
        # in forward_fn) and SDF targets. Skip both when the finetune cache
        # will short-circuit. The encoder is frozen inside _setup_conv.
        encoder_for_ft = None
        all_sdf_for_ft = None
        if cfg.shape.type == "conv":
            ft_cached = check_finetune_cache(cfg, cache_dir) is not None
            if not ft_cached:
                from graph_dyno.encoders.factory import create_shape_encoder
                encoder_for_ft = create_shape_encoder(cfg, device=device)
                print("Loading precomputed SDFs for conv finetune...")
                all_sdf_for_ft, _, _, _ = load_all_sdfs(
                    cache_dir, cache_id, traj_indices)
        finetune_decoder(
            model, cfg, device, cache_dir=cache_dir,
            traj_indices=traj_indices, encoder=encoder_for_ft,
            all_sdf=all_sdf_for_ft, embeddings=embeddings)
        del encoder_for_ft, all_sdf_for_ft

    # --- Per-group scale normalization (`+dataloader.scale_norm=<sim_metadata key>`) ---
    # Equalizes each group's median crop side so cell size alone cannot separate
    # the groups. In memory and after finetuning, so the cache, the finetune key
    # and the finetuned decoder are untouched (finetune reads crop sides from disk).
    scale_norm = cfg.dataloader.get("scale_norm", None)
    if scale_norm is not None and embeddings is not None:
        medians, target, factors = _normalize_scale(
            embeddings, cache_dir, cache_id, traj_indices, str(scale_norm))
        print(f"\nscale_norm={scale_norm}: median crop side "
              + ", ".join(f"{g}: {m:.0f} -> x{factors[g]:.3f}" for g, m in medians.items())
              + f" (target {target:.1f})")
        mlflow.log_params({f"scale_norm.factor_{g}": round(f, 6) for g, f in factors.items()})

    # --- Segmentation / tracking ablation ---------------------------------
    # Applied AFTER finetuning and BEFORE the bottleneck pass, so that:
    #   * the shape decoder is finetuned on CLEAN crops identically for every
    #     level (`_setup_dino` pairs `all_emb` row-for-row with `all_sdf` from
    #     the clean `sdf_*.npy`, so applying first would train the decoder to
    #     *invert* the ablation), and
    #   * the corrupted embeddings flow through the same `get_bottleneck` path
    #     as the clean ones, with no special case.
    # `_compute_finetune_cache_key` hashes ckpt bytes + generate_config_hash(
    # cfg.data) + ft params and carries no ablation identifier — with cfg.data
    # unchanged by design, every arm resolves to the same
    # `finetuned_decoder_<key>.pt`, which is exactly what a controlled ablation
    # wants.
    seg_stats = None
    seg_spec = cfg.get("seg_ablation", None)
    if seg_spec is not None and embeddings is not None:
        from graph_dyno.seg_ablation.build import ensure_sidecars
        from graph_dyno.seg_ablation.apply import apply_all
        from graph_dyno.seg_ablation.plan import abl_id, normalize_spec

        seg_norm = normalize_spec(seg_spec)
        seg_abl = abl_id(seg_norm)
        # A bank-derived level needs no build step at all: every corrupted row
        # is already encoded, so materializing it is index selection plus the
        # CPU tail `_finalize_sidecar` shares with the direct builder — order a
        # minute per trajectory against a 13-30 min run, and the geometry half
        # is cached on disk for the sibling seeds. Everything downstream is
        # keyed off the spec rather than the files, so the two paths are
        # interchangeable from here on.
        use_bank = bool(seg_norm["bank"])
        print(f"\nSegmentation ablation {seg_abl} "
              f"({cfg.get('tags', {}).get('seg_ablation', '?')})"
              + (f" from bank {seg_norm['bank']}" if use_bank else ""))
        if not use_bank:
            ensure_sidecars(cache_dir, cfg, cache_id=cache_id,
                            companion_suffix=companion_suffix,
                            traj_indices=traj_indices, device=device)
        if cfg.get("log_clean_val_loss", True):
            # Bottleneck the CLEAN embeddings first (without consuming them),
            # then stash the clean crop_sides / next_states. Doing it in this
            # order means the raw (T, N, D) array is never duplicated — only
            # the much smaller bottleneck is.
            print("Stashing clean targets for the clean-target val loss...")
            _precompute_bottlenecks(embeddings, model.shape_decoder, device,
                                    dst_key="L_clean", pop=False)
            for traj_idx in traj_indices:
                e = embeddings[traj_idx]
                e["crop_sides_clean"] = e["crop_sides"].clone()
                disk = torch.load(
                    os.path.join(cache_dir, f"{cache_id}_{traj_idx:04d}.pt"),
                    map_location="cpu", mmap=True)
                e["next_states_clean"] = disk["next_states"].clone()
        if use_bank:
            from graph_dyno.seg_ablation.bank_apply import apply_bank
            seg_stats = apply_bank(embeddings, cache_dir, seg_norm,
                                   cache_id=cache_id,
                                   companion_suffix=companion_suffix,
                                   dt=cfg.data.dt)
        else:
            seg_stats = apply_all(embeddings, cache_dir, seg_norm,
                                  cache_id=cache_id,
                                  companion_suffix=companion_suffix,
                                  dt=cfg.data.dt)
        mlflow.set_tag("seg_ablation_abl_id", seg_abl)
        mlflow.log_params({"seg_ablation.abl_id": seg_abl})
        _log_seg_stats(seg_stats)
        if hasattr(model, "seg_ablation_meta"):
            model.seg_ablation_meta = {"abl_id": seg_abl, "stats": seg_stats}
        # Stamp the identity into the checkpoint too. MLflow stores get moved
        # and deleted; the .pth is what the analysis actually opens, and a
        # checkpoint that cannot say which ablation produced it is one mislaid
        # directory away from being unusable. NOT `_meta_`-prefixed, so
        # `_load_emb_meta`'s per-slot metadata scan ignores it.
        model.register_buffer(
            "_segabl_id", torch.tensor(int(seg_abl, 16), dtype=torch.int64),
            persistent=True)

    # Run shape_decoder.get_bottleneck once over every frame so the training
    # loop can consume L directly (no per-step encoder calls). Must follow
    # finetuning, which can mutate shape_decoder weights.
    if embeddings is not None and hasattr(model, 'shape_decoder'):
        print("Precomputing bottleneck vectors L for all trajectories...")
        embeddings = _precompute_bottlenecks(embeddings, model.shape_decoder, device)

    # --- Create dataloaders with embeddings ---
    # When rollout curriculum is enabled, dataset.rollout_steps is mutated
    # during training. With persistent_workers=True, worker processes keep
    # their own stale copy of the dataset and won't see the mutation →
    # batches lack the _seq keys that rollout_training_step expects.
    rollout_cfg_pre = cfg.training.get("rollout", {})
    if rollout_cfg_pre.get("enabled", False) and cfg.dataloader.get("persistent_workers", True):
        was_readonly = OmegaConf.is_readonly(cfg)
        OmegaConf.set_readonly(cfg, False)
        print("Rollout curriculum enabled — setting persistent_workers=False")
        cfg.dataloader.persistent_workers = False
        OmegaConf.set_readonly(cfg, was_readonly)

    print("\nCreating train/val/test dataloaders...")
    train_dataloader, val_dataloader, test_dataloader = create_split_dataloaders_from_folder(
        folder_path=cache_dir,
        cfg=cfg,
        embeddings=embeddings,
    )

    # Verify simulation metadata consistency across all trajectories
    train_dataloader.dataset.dataset.verify_metadata_consistency()

    # Store reference to underlying dataset for rollout curriculum mutation
    underlying_dataset = train_dataloader.dataset.dataset

    # Models with embeddings don't need images during training — disable for speed.
    # Images are loaded lazily for plotting (see load_images call before epoch loop).
    if underlying_dataset.has_embeddings:
        underlying_dataset.include_images = False

    # Rollout config
    rollout_cfg = cfg.training.get("rollout", {})
    rollout_enabled = rollout_cfg.get("enabled", False)
    rollout_schedule = list(rollout_cfg.get("schedule", [[0, 1]])) if rollout_enabled else []
    if rollout_enabled:
        max_steps = max(s[1] for s in rollout_schedule)
        print(f"\nMulti-step rollout enabled (max {max_steps} steps)")
        print(f"  Schedule: {rollout_schedule}")

    # Load simulation metadata from cache (computed during data generation)
    # (must be done before wrapping dataloaders in CUDAPrefetcher)
    sim_metadata = train_dataloader.dataset.dataset.get_simulation_metadata()

    # Dataset-level displacement-magnitude std for input noise (scalars per
    # group). Consumed by model._noise_inputs; computed here so the model
    # sees stable noise magnitudes regardless of batch composition.
    if getattr(model, "input_noise_std", 0.0) > 0:
        model.x_disp_std, model.v_disp_std, model.L_disp_std = \
            _compute_displacement_magnitude_std(train_dataloader, model, device)
        print(f"Dataset displacement-magnitude std: "
              f"x={model.x_disp_std.item():.4g}  "
              f"v={model.v_disp_std.item():.4g}  "
              f"L={model.L_disp_std.item():.4g}")
        if seg_stats is not None:
            # Pin the position/velocity sigmas to the CLEAN values cached in
            # the sidecars, so every arm of the ladder — `none` included —
            # trains with identical regularisation noise. The masked estimate
            # above already removes the miss/swap outliers; this removes the
            # residual drift too, and both numbers reach MLflow.
            masked = (model.x_disp_std.item(), model.v_disp_std.item())
            sx = sum(s["sigma_x"] for s in seg_stats) / len(seg_stats)
            sv = sum(s["sigma_v"] for s in seg_stats) / len(seg_stats)
            model.x_disp_std = torch.tensor(sx, device=device)
            model.v_disp_std = torch.tensor(sv, device=device)
            print(f"  pinned to clean sidecar values: x={sx:.4g} v={sv:.4g} "
                  f"(masked estimate was x={masked[0]:.4g} v={masked[1]:.4g})")
            mlflow.log_params({
                "segabl.disp_std_x_clean": sx,
                "segabl.disp_std_v_clean": sv,
                "segabl.disp_std_x_masked": masked[0],
                "segabl.disp_std_v_masked": masked[1],
                "segabl.disp_std_L_masked": model.L_disp_std.item(),
            })

    # When GPU cache is off, data lives on CPU — use CUDAPrefetcher to overlap
    # CPU→GPU transfers with compute.  When GPU cache is on, data is already on
    # GPU so no prefetcher is needed.
    use_gpu_cache = underlying_dataset.use_gpu_cache
    if not use_gpu_cache and device == "cuda":
        print("Using CUDAPrefetcher for async CPU→GPU transfers")
        train_dataloader = CUDAPrefetcher(train_dataloader, device)
        val_dataloader = CUDAPrefetcher(val_dataloader, device)

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "parameter_embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': cfg.training.optimizer.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ],
        lr=cfg.training.optimizer.lr
    )

    n_epochs = cfg.training.n_epochs
    grad_clip_norm = cfg.training.get("grad_clip_norm", None)
    best_val_loss = float('inf')
    prev_rollout = None  # track rollout changes to reset best_val_loss
    pending_best_reset_epoch = None  # defers best_val_loss reset until after LR ramp

    # LR schedule: at each rollout transition, LR drops to `lr_decay_on_transition × base`
    # then linearly ramps back to base over `lr_warmup_epochs` (0 = permanent drop, non-compounding).
    lr_decay = rollout_cfg.get("lr_decay_on_transition", 1.0) if rollout_enabled else 1.0
    lr_warmup_epochs = rollout_cfg.get("lr_warmup_epochs", 0) if rollout_enabled else 0
    transitions = [s[0] for i, s in enumerate(rollout_schedule)
                   if i > 0 and s[1] != rollout_schedule[i - 1][1]]

    def rollout_lr_factor(epoch):
        if lr_decay == 1.0:
            return 1.0
        active = max((t for t in transitions if t <= epoch), default=None)
        if active is None:
            return 1.0
        elapsed = epoch - active
        if lr_warmup_epochs == 0:
            return lr_decay
        if elapsed < lr_warmup_epochs:
            return lr_decay + (1.0 - lr_decay) * (elapsed / lr_warmup_epochs)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rollout_lr_factor)

    losses = torch.zeros((n_epochs, 2))  # column 0: train loss, column 1: val loss

    if sim_metadata is None:
        raise ValueError(
            "No simulation metadata found in cache. "
            "Please regenerate data with the latest code to include sim_metadata."
        )
    print(f"Loaded simulation metadata from cache:")
    for k, v in sim_metadata.items():
        if isinstance(v, torch.Tensor) and v.dim() == 0:
            print(f"  {k}: {v.item():.4f}")
        elif isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, min={v.min().item():.4f}, max={v.max().item():.4f}")
        else:
            print(f"  {k}: {v}")
    model.set_metadata(**sim_metadata)

    # Load images into cache for plotting (skipped during initial load when
    # embeddings are provided, to save memory during finetuning)
    if underlying_dataset.has_embeddings:
        underlying_dataset.load_images()
        if seg_stats is not None:
            print("\n" + "=" * 72)
            print("NOTE: training plots show CLEAN masks as 'ground truth'.")
            print("`load_images()` re-reads channeled_images straight from the")
            print("trajectory cache, which the sidecar never touches — so")
            print("plot.py renders undamaged cells beside model outputs driven")
            print("by CORRUPTED embeddings. Read the training_plots/ figures")
            print("accordingly.")
            print("=" * 72 + "\n")

    # Risk 10: on real data most slots are padding in most windows, so a wiring
    # bug that silently starved a whole video's embeddings of gradient would look
    # exactly like normal training. Over epoch 0 accumulate, per slot, the total
    # |grad| on `parameter_embedding`, plus two different "did this slot appear"
    # sets. Only real-data batches carry `valid`, so the simulation path never
    # enters this.
    #
    # `ever_masked` (the mask the loss actually applies) is the correctness
    # criterion: a slot the loss touched MUST get gradient, or the wiring is
    # broken. `ever_valid` is broader and is NOT a criterion — a short track
    # landing at a window edge is valid for a frame or two but can never form
    # the prev/cur/next triple a transition needs, so it legitimately never
    # trains. Those slots keep their random init and must be excluded from the
    # separation analysis; they are written to untrained_slots.json for it.
    slot_grad_abs = None
    slot_ever_valid = None
    slot_ever_masked = None

    snapshot_every = int(cfg.training.get("embedding_snapshot_every", 0) or 0)
    emb_snapshots = {}  # epoch -> parameter_embedding, saved to embedding_snapshots.pt
    # Diagnostic plots every `plot_every` epochs (0 = final epoch only). They
    # cost ~13% of a fast epoch and draw from the global RNG (fresh loader
    # iterators), so changing this changes the training trajectory like a seed.
    plot_every = int(cfg.training.get("plot_every", 5) or 0)

    for epoch in range(n_epochs):
        # Set rollout steps for this epoch via curriculum
        current_rollout = 1
        if rollout_enabled:
            current_rollout = get_rollout_steps(epoch, rollout_schedule)
            underlying_dataset.rollout_steps = current_rollout
            if current_rollout != prev_rollout:
                # Defer best_val_loss reset until after the LR warmup ramp
                # completes, so checkpoints during the dip (artificially high
                # val loss from reduced LR) don't get saved as "best."
                if prev_rollout is not None:
                    pending_best_reset_epoch = epoch + lr_warmup_epochs
                print(f"Epoch {epoch+1}: rollout steps → {current_rollout}; "
                      f"LR factor={rollout_lr_factor(epoch):.3g}"
                      + (f" (best_val_loss reset deferred to epoch {pending_best_reset_epoch+1})"
                         if pending_best_reset_epoch is not None and pending_best_reset_epoch > epoch
                         else ""))
            prev_rollout = current_rollout

        # Apply deferred best_val_loss reset once warmup is complete
        if pending_best_reset_epoch is not None and epoch >= pending_best_reset_epoch:
            best_val_loss = float('inf')
            print(f"Epoch {epoch+1}: warmup complete, reset best_val_loss")
            pending_best_reset_epoch = None

        # Training phase
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0
        n_clipped_steps = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", dynamic_ncols=True):
            if batch is None:
                continue  # skip empty batches from rollout boundary filtering
            optimizer.zero_grad()
            if current_rollout > 1:
                loss_dict = model.rollout_training_step(batch, current_rollout)
            else:
                loss_dict = model.training_step(batch)
            loss = sum(v * cfg.training.loss_terms[k] for k, v in loss_dict.items() if k in cfg.training.loss_terms)

            loss.backward()
            mask_fn = getattr(model, "loss_mask_for", None)
            if epoch == 0 and "valid" in batch and callable(mask_fn):
                # Read the grad before optimizer.step(); zero_grad() only runs at
                # the top of the next iteration, and clipping rescales but never
                # zeroes, so either side of the clip would do.
                emb_grad = getattr(model, "parameter_embedding", None)
                emb_grad = None if emb_grad is None else emb_grad.grad
                if emb_grad is not None:
                    g = emb_grad.abs().sum(dim=-1).detach()
                    v = batch["valid"].reshape(-1, batch["valid"].shape[-1]).any(dim=0)
                    m = mask_fn(batch)
                    m = m.reshape(-1, m.shape[-1]).any(dim=0)
                    slot_grad_abs = g if slot_grad_abs is None else slot_grad_abs + g
                    slot_ever_valid = v if slot_ever_valid is None else slot_ever_valid | v
                    slot_ever_masked = m if slot_ever_masked is None else slot_ever_masked | m
            if grad_clip_norm is not None:
                # An ablated arm whose loss is dominated by unpredictable
                # corrupted transitions would train permanently clipped while
                # `none` never clips — the arms would then differ in effective
                # step size, and any separation drop would be partly an
                # optimizer artifact. Log the rate so that is visible.
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       grad_clip_norm)
                n_clipped_steps += int(float(gnorm) > grad_clip_norm)
            optimizer.step()
            total_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = total_train_loss / max(n_train_batches, 1)

        if epoch == 0 and slot_ever_masked is not None:
            starved = slot_ever_masked & (slot_grad_abs == 0)
            n_trained = int(slot_ever_masked.sum())
            if bool(starved.any()):
                idx = torch.nonzero(starved).flatten()[:20].tolist()
                raise RuntimeError(
                    f"parameter_embedding got zero gradient on {int(starved.sum())} of "
                    f"{n_trained} slots the epoch-0 loss mask actually selected "
                    f"(first offenders: {idx}). The loss touched these slots and no "
                    f"gradient came back, so this is a wiring bug — check the slot "
                    f"offsets and the embedding gather before reading any result."
                )

            # Everything the gradient never reached keeps its random init. The
            # authoritative set is `grad == 0`, NOT `ever_valid & ~ever_masked`:
            # the split is temporal, so slots living only in validation windows
            # never train either, and pure-padding slots never train at all.
            # All of them are random noise in the embedding and all must be
            # dropped before clustering.
            untrained = slot_grad_abs == 0
            n_untrained = int(untrained.sum())
            print(f"  grad reaches all {n_trained} slots the loss selected in epoch 1")
            if n_untrained:
                slots = torch.nonzero(untrained).flatten().tolist()
                edge = int((slot_ever_valid & ~slot_ever_masked).sum())
                out = Path.cwd() / "untrained_slots.json"
                out.write_text(json.dumps(
                    {"n_slots": int(untrained.numel()),
                     "n_trained": n_trained,
                     "n_untrained": n_untrained,
                     "n_untrained_window_edge": edge,
                     "untrained_slots": slots}, indent=2))
                print(f"  WARNING: {n_untrained} of {int(untrained.numel())} slots "
                      f"never received gradient and keep their random init "
                      f"({edge} of them valid-but-never-a-transition, the rest "
                      f"padding or validation-only). Wrote {out} — EXCLUDE these "
                      f"from the embedding analysis. First few: {slots[:10]}")

        scheduler.step()

        # Reset rollout for validation (always single-step)
        if rollout_enabled:
            underlying_dataset.rollout_steps = 1

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_loss_accumulators = {}
        n_val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", dynamic_ncols=True):
                if batch is None:
                    continue  # skip empty batches from rollout boundary filtering
                loss_dict = model.training_step(batch)

                loss = sum(v * cfg.training.loss_terms[k] for k, v in loss_dict.items() if k in cfg.training.loss_terms)
                total_val_loss += loss.item()
                n_val_batches += 1

                for k, v in loss_dict.items():
                    val_loss_accumulators[k] = val_loss_accumulators.get(k, 0.0) + v.item()

                # Val losses against the CLEAN targets. `best_val_loss` below
                # is argmin over the *observed* val loss, which under a
                # seg-ablation is driven by unpredictable corrupted
                # transitions rather than embedding quality — which is why the
                # analysis reads graph_decoder_model.pth (the final
                # checkpoint). These are the numbers that say whether that
                # mattered.
                clean_fn = getattr(model, "clean_target_losses", None)
                if callable(clean_fn):
                    for k, v in clean_fn(batch).items():
                        val_loss_accumulators[k] = \
                            val_loss_accumulators.get(k, 0.0) + v.item()
        # `dataloader.val_ratio: 0.0` is a deliberate mode, not a mistake: on real
        # data the separation result is read off `parameter_embedding`, and every
        # timestep withheld is a timestep of embedding supervision lost (it also
        # strands whole slots — see the untrained-slot check above). With no
        # held-out split, report the training loss in the val slot so
        # checkpointing, plots and MLflow keep working unchanged. It is a FIT
        # metric, not a generalization metric; the print says so.
        have_val = n_val_batches > 0
        if have_val:
            avg_val_loss = total_val_loss / n_val_batches
            avg_val_losses = {k: v / n_val_batches
                              for k, v in val_loss_accumulators.items()}
        else:
            avg_val_loss = avg_train_loss
            avg_val_losses = {}

        # Run sitting cell validation (skip for DINO models)
        sitting_cell_rmse = None
        sitting_cell_df = None
        if not hasattr(model, 'dino_decoder'):
            sitting_cell_rmse, sitting_cell_df = validate_all_sitting_cells(
                model=model,
                cfg=cfg,
                device=device,
                n_steps=100,
            )

        losses[epoch, 0] = avg_train_loss
        losses[epoch, 1] = avg_val_loss

        if have_val:
            print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}")
            val_loss_parts = ", ".join(f"{k}: {v:.4f}" for k, v in avg_val_losses.items())
            print(f"  Val losses: {val_loss_parts}")
        else:
            print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_train_loss:.4f} "
                  f"(no held-out split; val_loss mirrors train_loss)")

        # Log metrics to MLflow
        mlflow_metrics = {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }
        if rollout_enabled:
            mlflow_metrics["rollout_steps"] = current_rollout
        if grad_clip_norm is not None:
            mlflow_metrics["grad_clip_frac"] = \
                n_clipped_steps / max(n_train_batches, 1)
        for k, v in avg_val_losses.items():
            mlflow_metrics[f"val_{k}"] = v
        if sitting_cell_rmse is not None:
            mlflow_metrics["sitting_cell_rmse"] = sitting_cell_rmse
        mlflow.log_metrics(mlflow_metrics, step=epoch)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path / "graph_decoder_model_best.pth")
            print(f"  → Best model saved (val_loss: {best_val_loss:.4f})")

        # Save the parameter_embedding + metadata snapshot every 20 epochs so
        # downstream figure scripts can render the embedding at intermediate
        # training stages (e.g. pre/post-grok). Cheap to store: just the L
        # matrix and the constant simulation metadata buffers.
        if (epoch + 1) % 20 == 0:
            full_sd = model.state_dict()
            emb_sd = {k: v for k, v in full_sd.items()
                      if k == "parameter_embedding" or k.startswith("_meta_")}
            emb_path = save_path / f"param_embedding_epoch_{epoch + 1}.pt"
            torch.save(emb_sd, emb_path)
            try:
                mlflow.log_artifact(emb_path)
            except Exception:
                pass  # non-fatal: filesystem-only sanity runs may not support artifacts

        # Opt-in trajectory of the per-cell embedding, for asking WHEN
        # separation peaks — the best and final checkpoints can disagree.
        if snapshot_every and (epoch + 1) % snapshot_every == 0:
            emb_snapshots[epoch + 1] = model.parameter_embedding.detach().cpu().clone()
            torch.save(emb_snapshots, save_path / "embedding_snapshots.pt")

        # show a sample image from the decoded images (every plot_every epochs, plus final)
        should_plot = (plot_every and (epoch + 1) % plot_every == 0) or (epoch == n_epochs - 1)
        with torch.no_grad():
            if not should_plot:
                if underlying_dataset.has_embeddings:
                    underlying_dataset.include_images = False
                continue
            # Temporarily enable images for plotting (fetch fresh batches)
            underlying_dataset.include_images = True
            # First non-empty batch: collate returns None when every sample in
            # a batch is filtered, which small batch sizes hit routinely.
            plot_batch = next(b for b in (val_dataloader if have_val else train_dataloader)
                              if b is not None)

            fig, _ = plot_training(
                batch=plot_batch,
                model=model,
                losses=losses,
                epoch=epoch,
                cfg=cfg
            )

            training_save_path = save_path / "training_plots"
            training_save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(training_save_path / f"training_epoch_{epoch+1:04d}.png")
            plt.close(fig)

            # Plot generated images from training data
            train_batch = next(b for b in train_dataloader if b is not None)
            fig_gen, _ = plot_generated_images(
                batch=train_batch,
                model=model,
                cfg=cfg,
                n_samples=4
            )
            fig_gen.savefig(training_save_path / f"generated_images_epoch_{epoch+1:04d}.png")
            plt.close(fig_gen)

            # Plot embedding colored by changing simulation parameters
            fig_emb = plot_embedding_colored_by_changing_params(model, use_subplots=True)
            if fig_emb is not None:
                fig_emb.savefig(training_save_path / f"embedding_params_epoch_{epoch+1:04d}.png")
                plt.close(fig_emb)

            if epoch == n_epochs - 1:
                # from graph_dyno.plotting.plot import plot_embedding
                # fig, ax = plot_embedding(model=model)
                # fig.savefig(save_path / "embedding_final.png")
                # mlflow.log_artifact(save_path / "embedding_final.png")
                # plt.close(fig)

                # Plot and log sitting cell validation
                if sitting_cell_df is not None:
                    plot_sitting_cell_validation(
                        sitting_cell_df,
                        save_path=save_path / "sitting_cell_validation.png"
                    )
                    mlflow.log_artifact(save_path / "sitting_cell_validation.png")

            # Restore include_images to config-based default
            if underlying_dataset.has_embeddings:
                underlying_dataset.include_images = False

    # Finally save the model
    torch.save(model.state_dict(), save_path / "graph_decoder_model.pth")
    print(f"\nTraining complete! Model saved to {save_path}")

    # Log final model and end MLflow run
    mlflow.log_artifact(save_path / "graph_decoder_model_best.pth")
    mlflow.end_run()

    return best_val_loss  # For Optuna hyperparameter optimization


if __name__ == "__main__":
    main()
