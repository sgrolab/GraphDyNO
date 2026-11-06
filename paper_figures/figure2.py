"""Figure 1b — autoregressive rollout vs ground truth.

Renders the original frame at t=START_TIMESTEP, the ground truth and
model-predicted frames at t=START_TIMESTEP+N, and a per-pixel disagreement
map between them.

The rollout runs on a held-out trajectory: one fresh simulation with the run's
data params and a new seed (HELDOUT_SEED_OFFSET), generated and cached next to
the training cache on first use. --training-data rolls out on a training
trajectory instead.
"""
import argparse
import os
import sys
from pathlib import Path


import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
plt.rcParams["font.family"] = "DejaVu Sans"
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from tqdm import tqdm

from omegaconf import OmegaConf

from graph_dyno.data.dataloader import TrajectoryDatasetFromFolder
from graph_dyno.encoders.factory import get_companion_suffix
from graph_dyno.models.model_factory import resolve_shape_config
from graph_dyno.plotting.plot import (
    _find_run_by_mlflow_name,
    generate_combined_image_colored,
    generate_combined_image_dino,
    load_run,
    rescale_dino_to_channeled,
)
from train import _load_companion_embeddings, _precompute_bottlenecks
from graph_dyno.data.trajectory_generator import generate_config_hash, generate_data
from graph_dyno.precompute.precompute_embeddings import precompute_all, precompute_sdfs

from _run_lookup import (  # noqa: E402  (importing from sibling after sys.path tweak)
    find_run_in_experiment,
    load_experiments_yaml,
    setup_mlflow_uri,
)


# ----- defaults (override via CLI flags) -----
N_STEPS = 128
N_CELLS = 5               # number of individual end-of-rollout cells to show
TRAJECTORY_INDEX = 0      # which training trajectory to use with --training-data
START_TIMESTEP = 100        # initial timestep (>=1 if model uses velocities)
HELDOUT_SEED_OFFSET = 1000  # simulation seed of the held-out trajectory (training: seed_offset + [0, n_trajectories))
CMAP = "viridis"
OUTPUT_PATH = "./plots/figure2_new.png"
DPI = 300

# error overlay colors
COLOR_PRED_ONLY = (0.85, 0.10, 0.10)
COLOR_GT_ONLY = (0.10, 0.30, 0.85)
COLOR_AGREEMENT = (0.85, 0.85, 0.85)


def resolve_cache_dir(cfg):
    """Match dataloader.create_split_dataloaders_from_folder's path resolution."""
    folder_path = cfg.dataloader.get("folder_path", "") or ""
    if folder_path:
        return folder_path
    return f"./data/cache_{generate_config_hash(cfg.data)}"


def load_trajectory(cfg, traj_idx):
    """Load a single trajectory cache file (CPU)."""
    cache_dir = resolve_cache_dir(cfg)
    cache_id = generate_config_hash(cfg.data)
    traj_path = os.path.join(cache_dir, f"{cache_id}_{traj_idx:04d}.pt")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    return torch.load(traj_path, map_location="cpu")


def heldout_cfg(cfg, seed_offset=HELDOUT_SEED_OFFSET):
    """Copy of the run's cfg pointing at one fresh trajectory: same data params, new seed.

    Per-cell parameters come from deterministic grids, so cell i keeps the
    parameters its learned lambda_i was trained on; only the placement and the
    stochastic forces change. The cache sits next to the training cache.
    """
    gen = cfg.data.generator
    train_seeds = range(gen.seed_offset, gen.seed_offset + gen.n_trajectories)
    if seed_offset in train_seeds:
        raise ValueError(f"held-out seed {seed_offset} is one of the training seeds "
                         f"{list(train_seeds)}")
    cache_dir = f"{resolve_cache_dir(cfg).rstrip('/')}_heldout_seed{seed_offset}/"
    out = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    out.data.generator.seed_offset = seed_offset
    out.data.generator.n_trajectories = 1
    out.data.generator.cache_dir = cache_dir
    out.dataloader.folder_path = cache_dir
    return out


def ensure_heldout_data(cfg, device):
    """Simulate, rasterize and encode the held-out trajectory unless it is cached (mirrors train.py)."""
    cache_dir = generate_data(cfg, device=device)
    precompute_sdfs(cache_dir=cache_dir, cfg=cfg,
                    sdf_workers=cfg.shape.get("sdf_workers", 32),
                    target_size=cfg.shape.get("target_size", 224))
    precompute_all(cache_dir=cache_dir, cfg=cfg)


def check_cell_params(model, sim_metadata):
    """Fail unless every per-cell parameter matches the trained model's, cell for cell."""
    mismatched = []
    for key, value in sim_metadata.items():
        ref = model.get_metadata(key)
        # Phases are drawn from the seed, so they legitimately differ.
        if ref is None or value.dim() != 1 or key.endswith("_phases"):
            continue
        if ref.shape != value.shape or not torch.allclose(ref.cpu().float(), value.float()):
            mismatched.append(key)
    if mismatched:
        raise ValueError(f"held-out trajectory's per-cell parameters differ from the "
                         f"training data: {mismatched}")


def make_init_batch(cfg, trajectory_index, start_timestep, device, shape_decoder):
    """Build a (B=1) init batch via TrajectoryDatasetFromFolder.

    Uses the dataset's own __getitem__ so every field (L_history, prev_L,
    etc.) is constructed exactly as during training, and history_window flows
    automatically from cfg.model.history_window.

    shape_decoder is required so raw companion embeddings can be converted to
    bottleneck L once before the dataset sees them.
    """
    cache_dir = resolve_cache_dir(cfg)
    cache_id = generate_config_hash(cfg.data)
    companion_suffix = get_companion_suffix(cfg)
    if companion_suffix is None:
        raise ValueError(
            "Conv shape encoder has no companion embeddings file; "
            "this script only supports DINO/Fourier configs."
        )
    # Mirror train.py: load companion embeddings for every trajectory the
    # dataset will iterate over (the GPU preload path indexes all of them).
    metadata = OmegaConf.load(os.path.join(cache_dir, "metadata.yaml"))
    traj_indices = list(range(metadata.data.generator.n_trajectories))
    embeddings = _load_companion_embeddings(
        cache_dir, cache_id, companion_suffix, traj_indices)
    embeddings = _precompute_bottlenecks(embeddings, shape_decoder, device)

    history_window = int(cfg.get("model", {}).get("history_window", 0))
    dataset = TrajectoryDatasetFromFolder(
        folder_path=cache_dir,
        cpu_cache_size=1,
        enable_gpu_cache=True,
        embeddings=embeddings,
        history_window=history_window,
    )
    # Embedding mode skips images during cache load (see _IMAGE_KEYS); load
    # them explicitly so __getitem__ can populate channeled_images.
    dataset.load_images()

    sample = dataset[trajectory_index * dataset.trajectory_length + start_timestep]
    if sample is None:
        raise ValueError(
            f"Sample at trajectory={trajectory_index}, t={start_timestep} unavailable "
            f"(need t >= max(1, history_window={history_window}))."
        )
    init_batch = {
        k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
        for k, v in sample.items()
    }
    # Precomputed GT bottleneck sequence for this trajectory — (T, N, bn).
    gt_L_seq = embeddings[trajectory_index]["L"]
    return init_batch, gt_L_seq


def build_interaction_graph(positions, r_cutoff, num_nodes, device):
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    dist = torch.norm(diff, dim=2)
    mask = (dist <= r_cutoff) & (dist > 0)
    indices = mask.nonzero(as_tuple=False).t()
    values = torch.ones(indices.shape[1], dtype=torch.bool, device=device)
    return torch.sparse_coo_tensor(
        indices, values, (num_nodes, num_nodes), dtype=torch.bool, device=device,
    )


def rollout(model, init_batch, n_steps, cfg, device):
    """Autoregressive N-step rollout.

    Returns a list of length n_steps of (positions_xy, channeled_imgs) on CPU,
    one entry per predicted step (i.e., entries correspond to t+1, ..., t+n_steps).
    """
    image_size = cfg.data.image_size
    r_cutoff = cfg.data.params.r_cutoff

    x = init_batch["states"][0]
    num_nodes = x.shape[0]

    L = init_batch["L"][0]
    current_scale = init_batch["crop_sides"][0].float()
    b = model.parameter_embedding.data

    prev_L = None
    prev_scale = None
    if model.use_velocities:
        prev_L = init_batch["prev_L"][0]
        prev_scale = init_batch["prev_crop_sides"][0].float()

    L_history = None
    if model.use_history:
        # (K, N, bn) → (N, K, bn) to match training layout.
        L_history = init_batch["L_history"][0].transpose(0, 1).contiguous()

    history = []
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(n_steps), desc="Rolling out", leave=False):
            W = build_interaction_graph(x[:, :2], r_cutoff, num_nodes, device)

            L_velocity = None
            scale_velocity = None
            if model.use_velocities:
                L_velocity = L - prev_L
                scale_velocity = current_scale - prev_scale

            new_x, L_out, next_scale = model.generate(
                x, W, b, L, current_scale, L_velocity, scale_velocity, L_history
            )

            if model.use_velocities:
                prev_L = L
                prev_scale = current_scale
            # Slide history: drop oldest, append the L we just consumed (input,
            # not output) — same convention as training_step.
            if model.use_history:
                L_history = torch.cat(
                    [L_history[:, 1:, :], L.unsqueeze(1)], dim=1
                )
            x = new_x
            L = L_out
            current_scale = next_scale

            decoded_sdf = model.decode_to_image(L)
            channeled = rescale_dino_to_channeled(decoded_sdf, current_scale, image_size)
            history.append((x.cpu(), channeled.cpu(), L.cpu()))

    return history


def compute_iou_curve(history, trajectory_data, start_timestep, domain_size, r_max):
    """Per-cell IoU quartiles between predicted and GT shape masks at each step.

    Returns (q0, q25, q50, q75) arrays of length n_steps summarizing the
    per-cell IoU distribution at each rollout step. q0 is the per-step min.
    """
    q0, q25, q50, q75 = [], [], [], []
    for k, (_pred_state, pred_channeled, _pred_L) in enumerate(
        tqdm(history, desc="Computing IoU", leave=False), start=1
    ):
        t = start_timestep + k
        gt_channeled = trajectory_data["channeled_images"][t]
        pred_masks = pred_channeled > 0.5
        gt_masks = gt_channeled > 0.5
        dims = (1, 2)
        intersection = (pred_masks & gt_masks).sum(dim=dims).float()
        union = (pred_masks | gt_masks).sum(dim=dims).float()
        per_cell = torch.where(union > 0, intersection / union,
                               torch.full_like(intersection, float("nan")))
        valid = per_cell[~torch.isnan(per_cell)]
        if valid.numel() == 0:
            for arr in (q0, q25, q50, q75):
                arr.append(float("nan"))
        else:
            qs = torch.quantile(valid, torch.tensor([0.25, 0.5, 0.75]))
            q0.append(valid.min().item())
            q25.append(qs[0].item()); q50.append(qs[1].item()); q75.append(qs[2].item())
    return np.array(q0), np.array(q25), np.array(q50), np.array(q75)


def _per_step_quartiles(per_cell):
    """Helper: (q0, q25, q50, q75) of per-cell error samples, NaN-safe."""
    valid = per_cell[~torch.isnan(per_cell)]
    if valid.numel() == 0:
        return (float("nan"),) * 4
    qs = torch.quantile(valid, torch.tensor([0.25, 0.5, 0.75]))
    return valid.min().item(), qs[0].item(), qs[1].item(), qs[2].item()


def compute_embedding_error_curve(history, gt_L_seq, start_timestep):
    """Per-cell L2 embedding error as % of the cross-cell spread of GT L.

    Normalizes by the typical distance from the per-step population mean of
    gt_L — i.e., how much L varies across cells at that step. 100% ≈ error
    comparable to swapping in a random other cell's L.
    """
    q0, q25, q50, q75 = [], [], [], []
    for k, (_x, _ch, pred_L) in enumerate(history, start=1):
        t = start_timestep + k
        gt_L = gt_L_seq[t].cpu().float()
        num = torch.norm(pred_L.float() - gt_L, dim=-1)
        gt_centered = gt_L - gt_L.mean(dim=0, keepdim=True)
        den = torch.norm(gt_centered, dim=-1).mean().clamp(min=1e-8)
        per_cell = num / den * 100
        a, b, c, d = _per_step_quartiles(per_cell)
        q0.append(a); q25.append(b); q50.append(c); q75.append(d)
    return np.array(q0), np.array(q25), np.array(q50), np.array(q75)


def compute_position_error_curve(history, trajectory_data, start_timestep):
    """Per-cell distance from the GT position as % of GT displacement.

    err_i = ||pred_pos[k] - gt_pos[k]|| / ||gt_pos[k] - init|| * 100
    """
    init_pos = trajectory_data["states"][start_timestep, :, :2]
    q0, q25, q50, q75 = [], [], [], []
    for k, (pred_state, _, _) in enumerate(history, start=1):
        t = start_timestep + k
        gt_pos = trajectory_data["states"][t, :, :2]
        dist = torch.norm(pred_state[:, :2] - gt_pos, dim=1)
        gt_d = torch.norm(gt_pos - init_pos, dim=1)
        per_cell = dist / gt_d.clamp(min=1e-8) * 100
        a, b, c, d = _per_step_quartiles(per_cell)
        q0.append(a); q25.append(b); q50.append(c); q75.append(d)
    return np.array(q0), np.array(q25), np.array(q50), np.array(q75)


def make_pastel_discrete(values):
    """Colorblind palette ListedColormap + BoundaryNorm with one bin per unique value.

    Returns (cmap, norm, unique_values). Bins are centered on each unique value
    so the colorbar shows N evenly-sized blocks, one per distinct group.
    """
    unique = np.unique(np.asarray(values))
    n = len(unique)
    if n > 10:
        raise ValueError(f"colorblind palette has 10 colors; got {n} unique values")
    palette = sns.color_palette("colorblind", n)
    cmap = ListedColormap(palette, name="colorblind")
    if n == 1:
        boundaries = np.array([unique[0] - 0.5, unique[0] + 0.5])
    else:
        mids = (unique[:-1] + unique[1:]) / 2
        boundaries = np.concatenate([
            [2 * unique[0] - mids[0]],
            mids,
            [2 * unique[-1] - mids[-1]],
        ])
    return cmap, BoundaryNorm(boundaries, n), unique


def render_combined(positions, channeled, domain_size, r_max, cmap, norm=None,
                    values=None):
    rgba, norm, _ = generate_combined_image_colored(
        positions=positions,
        channeled_image=channeled,
        domain_size=domain_size,
        r_max=r_max,
        cmap=cmap,
        norm=norm,
        values=values,
    )
    return rgba, norm


def render_error(pred_rgba, gt_rgba):
    pred_mask = pred_rgba[..., 3] > 0
    gt_mask = gt_rgba[..., 3] > 0
    out = np.ones((*pred_mask.shape, 3))
    out[pred_mask & ~gt_mask] = COLOR_PRED_ONLY
    out[gt_mask & ~pred_mask] = COLOR_GT_ONLY
    out[pred_mask & gt_mask] = COLOR_AGREEMENT
    return out


def render_cell_diff(pred_cell, gt_cell, threshold=0.5):
    pred_mask = pred_cell > threshold
    gt_mask = gt_cell > threshold
    out = np.ones((*pred_mask.shape, 3))
    out[pred_mask & ~gt_mask] = COLOR_PRED_ONLY
    out[gt_mask & ~pred_mask] = COLOR_GT_ONLY
    out[pred_mask & gt_mask] = COLOR_AGREEMENT
    return out


def _resolve_save_path(run_name, experiments_dir="./experiments"):
    """Mirror load_run's save_path resolution (mlflow-name or path-style)."""
    if "/" in str(run_name):
        return Path(experiments_dir) / run_name
    return _find_run_by_mlflow_name(run_name)


def _load_embedding_panel_data(run_or_path, device, epoch=None):
    """Read parameter_embedding + metadata from a run's checkpoint.

    Accepts either an MLflow run name, a Hydra path, or an already-resolved
    save_path. Reads the state_dict directly so this works even if the run's
    architecture is incompatible with the current code.

    If `epoch` is given, prefer the periodic snapshot saved by train.py at that
    epoch (`param_embedding_epoch_<N>.pt`). Otherwise fall back to the final
    full-model checkpoint (`graph_decoder_model.pth`).
    """
    if isinstance(run_or_path, Path):
        save_path = run_or_path
    else:
        save_path = _resolve_save_path(run_or_path)
    if epoch is not None:
        ckpt_path = save_path / f"param_embedding_epoch_{epoch}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No epoch-{epoch} embedding snapshot at {ckpt_path}. "
                "Make sure the run saved periodic embeddings (train.py "
                "saves every 20 epochs)."
            )
    else:
        ckpt_path = save_path / "graph_decoder_model.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Embedding-run checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    param_emb = state["parameter_embedding"].detach().cpu().numpy()
    metadata = {k[len("_meta_"):]: v for k, v in state.items() if k.startswith("_meta_")}
    return param_emb, metadata, ckpt_path


def _resolve_run(*, run_name=None, experiment=None, metric="best_val_loss"):
    """Either look up an explicit run-name (legacy) or pick the best run from an experiment."""
    if run_name:
        return load_run(run_name)
    if not experiment:
        raise ValueError("Must specify --run-name or --experiment.")
    return find_run_in_experiment(experiment, metric=metric, direction="min")


def make_figure(run_name, n_steps, trajectory_index, start_timestep,
                cmap, output_path, dpi, n_cells=N_CELLS,
                embedding_run_name=None,
                experiment=None, embedding_experiment=None,
                embedding_epoch=None, heldout_seed_offset=HELDOUT_SEED_OFFSET):
    run = _resolve_run(run_name=run_name, experiment=experiment)
    model = run["model"]
    cfg = run["cfg"]
    device = run["device"]
    save_path = run["save_path"]

    if embedding_run_name:
        embed_source = embedding_run_name
    elif embedding_experiment:
        embed_source = find_run_in_experiment(
            embedding_experiment, metric="best_val_loss", direction="min"
        )["save_path"]
    else:
        embed_source = save_path
    latest_param_embedding, embed_metadata, latest_ckpt = _load_embedding_panel_data(
        embed_source, device, epoch=embedding_epoch)
    print(f"Loaded parameter_embedding from: {latest_ckpt}")
    # train.py calls this before building dataloaders to populate cfg.shape.dino_model
    # (and dino_use_patches) from the decoder checkpoint; load_run doesn't, so we
    # do it here so get_companion_suffix returns the right "dino_<KEY>" suffix.
    resolve_shape_config(cfg)

    # Roll out on unseen data unless asked otherwise; the cfg swap only moves
    # the data source, the physics and model settings are the run's own.
    if heldout_seed_offset is not None:
        cfg = heldout_cfg(cfg, heldout_seed_offset)
        ensure_heldout_data(cfg, device)
        trajectory_index = 0
        print(f"Rolling out on held-out trajectory (seed {heldout_seed_offset}) "
              f"in {resolve_cache_dir(cfg)}")

    domain_size = model.domain_size
    r_max = model.get_metadata("r_max")
    if r_max is not None:
        r_max = r_max.item()

    trajectory_data = load_trajectory(cfg, trajectory_index)
    if heldout_seed_offset is not None:
        check_cell_params(model, trajectory_data["sim_metadata"])
    traj_len = trajectory_data["states"].shape[0]
    final_t = start_timestep + n_steps
    if final_t >= traj_len:
        raise ValueError(
            f"start_timestep + n_steps = {final_t} exceeds trajectory length {traj_len}"
        )

    init_batch, gt_L_seq = make_init_batch(
        cfg, trajectory_index, start_timestep, device, model.shape_decoder)

    history = rollout(model, init_batch, n_steps, cfg, device)
    pred_state, pred_channeled, _ = history[-1]
    pred_pos = pred_state[:, :2]

    init_pos = trajectory_data["states"][start_timestep, :, :2]
    init_channeled = trajectory_data["channeled_images"][start_timestep]
    gt_pos = trajectory_data["states"][final_t, :, :2]
    gt_channeled = trajectory_data["channeled_images"][final_t]

    growth_rates = model.get_metadata("growth_rates")
    if growth_rates is None:
        raise ValueError(
            "Model has no 'growth_rates' metadata; cannot color cells by γ."
        )
    growth_rates = growth_rates.detach().cpu().numpy()
    pastel_cmap, growth_norm, _ = make_pastel_discrete(growth_rates)

    init_rgba, _ = render_combined(
        init_pos, init_channeled, domain_size, r_max, pastel_cmap,
        norm=growth_norm, values=growth_rates)
    gt_rgba, _ = render_combined(
        gt_pos, gt_channeled, domain_size, r_max, pastel_cmap,
        norm=growth_norm, values=growth_rates)
    pred_rgba, _ = render_combined(
        pred_pos, pred_channeled, domain_size, r_max, pastel_cmap,
        norm=growth_norm, values=growth_rates)
    err_rgb = render_error(pred_rgba, gt_rgba)

    # Pixel scale used by generate_combined_image_colored — positions project
    # to pixels as (x/dx, y/dx), with imshow's x-axis = column = state y.
    image_size = cfg.data.image_size
    dx = 2 * r_max / (image_size - 1)
    gt_tracks = trajectory_data["states"][start_timestep:final_t + 1, :, :2].cpu().numpy()
    pred_tracks = np.concatenate(
        [gt_tracks[:1]] + [h[0][:, :2].cpu().numpy()[None] for h in history],
        axis=0,
    )

    iou_q0, iou_q25, iou_q50, iou_q75 = compute_iou_curve(
        history, trajectory_data, start_timestep, domain_size, r_max)
    pos_q0, pos_q25, pos_q50, pos_q75 = compute_position_error_curve(
        history, trajectory_data, start_timestep)

    top_panels = [
        (init_rgba,
         "Initial (t=0)",
         []),
        (gt_rgba,
         f"GT (t={n_steps})",
         []),
        (err_rgb,
         f"Predicted vs GT (t={n_steps})",
         [("false (+)", COLOR_PRED_ONLY), ("false (\u2212)", COLOR_GT_ONLY),
          ("agreement", COLOR_AGREEMENT)]),
    ]

    n_cells_total = pred_channeled.shape[0]
    n_cells = max(1, min(n_cells, n_cells_total))
    # Pick representative cells at quartiles of per-cell IoU at the final step
    pred_masks_final = pred_channeled > 0.5
    gt_masks_final = gt_channeled > 0.5
    dims = (1, 2)
    inter_final = (pred_masks_final & gt_masks_final).sum(dim=dims).float()
    union_final = (pred_masks_final | gt_masks_final).sum(dim=dims).float()
    per_cell_iou = torch.where(union_final > 0, inter_final / union_final,
                               torch.full_like(inter_final, float("nan")))
    valid_mask = ~torch.isnan(per_cell_iou)
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    sort_order = valid_idx[torch.argsort(per_cell_iou[valid_idx])].cpu().numpy()
    quantile_pos = np.linspace(0, len(sort_order) - 1, n_cells).astype(int)
    cell_idx = sort_order[quantile_pos]
    quantile_labels = np.linspace(0.0, 1.0, n_cells)

    n_top = len(top_panels)
    top_side = 1.70
    bot_side = n_top * top_side / n_cells
    mid_h = 0.78
    row_gap = 0.18

    left_m = 0.35
    right_m = 0.05
    top_m = 0.15
    bottom_m = 0.24

    fig_w = left_m + n_top * top_side + right_m
    fig_h = top_m + top_side + row_gap + mid_h + bottom_m

    ratios = [top_side, mid_h]
    hspace = row_gap * len(ratios) / sum(ratios)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=ratios,
        hspace=hspace,
        left=left_m / fig_w, right=1 - right_m / fig_w,
        top=1 - top_m / fig_h, bottom=bottom_m / fig_h,
    )
    top_gs = outer[0].subgridspec(1, n_top, wspace=0.05)
    mid_gs = outer[1].subgridspec(1, 3, wspace=0.55)

    top_axes = []
    for i, (img, title, legend_entries) in enumerate(top_panels):
        ax = fig.add_subplot(top_gs[0, i])
        top_axes.append(ax)
        ax.imshow(img)
        combined_size = img.shape[0]
        # Panel 0 (initial): no tracks. Panel 1 (GT): gt tracks. Panel 2
        # (predicted vs gt): predicted tracks so we see where the model went.
        tracks_source = (None, gt_tracks, pred_tracks)[i] if i < 3 else None
        track_label = (None, "GT track", "pred track")[i] if i < 3 else None
        if tracks_source is not None:
            # Break segments at periodic wrap-around so lines don't streak
            # across the domain.
            track_px_x = tracks_source[..., 1] / dx  # column = state y
            track_px_y = tracks_source[..., 0] / dx  # row    = state x
            jump_thresh = combined_size * 0.5
            for c in range(track_px_x.shape[1]):
                xs = track_px_x[:, c]
                ys = track_px_y[:, c]
                breaks = np.where(
                    (np.abs(np.diff(xs)) > jump_thresh)
                    | (np.abs(np.diff(ys)) > jump_thresh)
                )[0]
                start = 0
                for b in list(breaks) + [len(xs) - 1]:
                    seg = slice(start, b + 1)
                    if (b + 1) - start > 1:
                        ax.plot(xs[seg], ys[seg], color="black",
                                linewidth=0.5, alpha=0.7)
                    start = b + 1
        ax.set_xlim(-0.5, combined_size - 0.5)
        ax.set_ylim(combined_size - 0.5, -0.5)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        handles = [plt.matplotlib.patches.Patch(facecolor=c, edgecolor="black",
                                                linewidth=0.5, label=lbl)
                   for lbl, c in legend_entries]
        if tracks_source is not None:
            handles.append(plt.matplotlib.lines.Line2D(
                [0], [0], color="black", linewidth=0.5, label=track_label))
        if handles:
            ax.legend(handles=handles, loc="upper left",
                      fontsize=5, frameon=True, fancybox=False,
                      edgecolor="0.5", facecolor="white", framealpha=0.8,
                      handlelength=0.8, handleheight=0.6,
                      borderpad=0.3, labelspacing=0.2)

    steps = np.arange(1, n_steps + 1)
    ax_pos = fig.add_subplot(mid_gs[0, 0])
    ax_pos.fill_between(steps, pos_q25, pos_q75, color="C0", alpha=0.2, linewidth=0)
    ax_pos.plot(steps, pos_q50, linewidth=1.0, color="C0", label="position")
    ax_pos.set_xlabel("Time", labelpad=-1)
    ax_pos.set_ylabel("Position error (%)")
    ax_pos.set_xlim(1, n_steps)
    ax_pos.set_xticks([1, n_steps])
    ax_pos.set_ylim(bottom=0)
    ax_pos.grid(alpha=0.3)

    ax_shape = fig.add_subplot(mid_gs[0, 1])
    ax_shape.fill_between(steps, iou_q25, iou_q75, color="C2",
                          alpha=0.2, linewidth=0)
    ax_shape.plot(steps, iou_q50, linewidth=1.0, color="C2",
                  label="shape IoU")
    ax_shape.set_xlabel("Time", labelpad=-1)
    ax_shape.set_ylabel("Shape IoU")
    ax_shape.set_xlim(1, n_steps)
    ax_shape.set_xticks([1, n_steps])
    ax_shape.set_ylim(0, 1)
    ax_shape.grid(alpha=0.3)

    ax_embed = fig.add_subplot(mid_gs[0, 2])
    embed = latest_param_embedding

    changing_params = {
        k: v for k, v in embed_metadata.items()
        if isinstance(v, torch.Tensor) and v.numel() > 0
        and len(torch.unique(v)) > 1
    }
    color_key = None
    color_values = None
    if len(changing_params) == 1:
        color_key, t = next(iter(changing_params.items()))
        color_values = t.detach().cpu().numpy()
    elif len(changing_params) > 1:
        print("Changing metadata parameters:")
        for k in changing_params:
            print(f"  - {k}")
        sel = input("Select changing variable for embedding color: ").strip()
        if sel in changing_params:
            color_key = sel
            color_values = changing_params[sel].detach().cpu().numpy()

    if embed.shape[1] == 1:
        ax_embed.hist(embed.squeeze(), bins=30, color="C0", alpha=0.7)
        ax_embed.set_xlabel(r"$\lambda_0$", labelpad=2)
        ax_embed.set_ylabel("Count", labelpad=4)
    else:
        if embed.shape[1] > 2:
            from sklearn.decomposition import PCA
            embed_2d = PCA(n_components=2).fit_transform(embed)
            xlabel, ylabel = r"$PC_1$", r"$PC_2$"
        else:
            embed_2d = embed
            xlabel, ylabel = r"$\lambda_0$", r"$\lambda_1$"
        if color_values is not None:
            cvals = color_values[:embed_2d.shape[0]]
            embed_cmap, embed_norm, embed_unique = make_pastel_discrete(cvals)
            sns.scatterplot(x=embed_2d[:, 0], y=embed_2d[:, 1],
                            hue=cvals, palette=embed_cmap, hue_norm=embed_norm,
                            legend=False, ax=ax_embed, s=5)
            sm = plt.cm.ScalarMappable(cmap=embed_cmap, norm=embed_norm)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax_embed, ticks=embed_unique)
            cb.ax.set_yticklabels([f"{v:.1f}" for v in embed_unique])
            cb.set_label(r"$\gamma$", rotation=0, labelpad=6,
                         fontsize=8, va="center")
        else:
            sns.scatterplot(x=embed_2d[:, 0], y=embed_2d[:, 1],
                            color="C0", legend=False, ax=ax_embed, s=5)
        ax_embed.set_xlabel(xlabel, labelpad=-1)
        ax_embed.set_ylabel(ylabel, labelpad=4, rotation=0)
        ax_embed.set_yticks([0, 4])

    for ax in top_axes:
        p = ax.get_position()
        ax.set_position([p.x0 - 0.02, p.y0, p.width, p.height])

    shift = 0.04
    pos = ax_embed.get_position()
    ax_embed.set_position([pos.x0 - shift, pos.y0, pos.width, pos.height])
    for child_ax in fig.get_axes():
        if child_ax is not ax_embed and child_ax not in top_axes + [ax_pos, ax_shape]:
            cp = child_ax.get_position()
            child_ax.set_position([cp.x0 - shift, cp.y0, cp.width, cp.height])

    # Place panel labels in figure coords so a/d, b/e, c/f share an x-anchor.
    panel_top = list(top_axes)
    panel_mid = [ax_pos, ax_shape, ax_embed]
    for col in range(3):
        top_bbox = panel_top[col].get_position()
        mid_bbox = panel_mid[col].get_position()
        x = min(top_bbox.x0, mid_bbox.x0) - 0.010
        fig.text(x, top_bbox.y1 + 0.005, "abc"[col],
                 fontsize=9, fontweight="bold", va="bottom", ha="left")
        fig.text(x, mid_bbox.y1 + 0.04, "def"[col],
                 fontsize=9, fontweight="bold", va="bottom", ha="left")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {pdf_path}")

    pred_cells = pred_channeled.cpu().numpy()
    gt_cells = gt_channeled.cpu().numpy()
    iou_np = per_cell_iou.cpu().numpy()
    fig_cells, axes_cells = plt.subplots(
        1, n_cells, figsize=(bot_side * n_cells, bot_side), dpi=dpi,
        squeeze=False,
    )
    for i, ci in enumerate(cell_idx):
        ax = axes_cells[0, i]
        ax.imshow(render_cell_diff(pred_cells[int(ci)], gt_cells[int(ci)]))
        q = quantile_labels[i]
        ax.set_title(f"Q{int(round(q * 100))}  IoU={iou_np[int(ci)]:.2f}",
                     fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig_cells.tight_layout()
    cells_path = output_path.with_name(output_path.stem + "_cells" + output_path.suffix)
    fig_cells.savefig(cells_path, bbox_inches="tight")
    print(f"Saved {cells_path}")
    cells_pdf = cells_path.with_suffix(".pdf")
    fig_cells.savefig(cells_pdf, bbox_inches="tight")
    print(f"Saved {cells_pdf}")
    return fig


def _parse_args():
    cfg = load_experiments_yaml().get("figure2", {})
    p = argparse.ArgumentParser(description="Figure 1b — rollout visualization")
    p.add_argument("--experiment", default=cfg.get("experiment"),
                   help="MLflow experiment name; the best run by val_loss is "
                        "selected. Override with --run-name for an explicit run.")
    p.add_argument("--embedding-experiment", default=cfg.get("embedding_experiment"),
                   help="Experiment name for the parameter-embedding panel "
                        "(defaults to --experiment).")
    p.add_argument("--run-name", default=None,
                   help="Explicit MLflow run name; overrides --experiment.")
    p.add_argument("--embedding-run-name", default=None,
                   help="Explicit MLflow run name for the embedding panel.")
    p.add_argument("--epoch", type=int, default=None,
                   help="Load the embedding panel from the periodic "
                        "param_embedding_epoch_<N>.pt snapshot saved by "
                        "train.py every 20 epochs. Default: use the final "
                        "graph_decoder_model.pth.")
    p.add_argument("--n-steps", type=int, default=N_STEPS)
    p.add_argument("--n-cells", type=int, default=N_CELLS)
    p.add_argument("--heldout-seed-offset", type=int, default=HELDOUT_SEED_OFFSET,
                   help="Simulation seed of the held-out trajectory the rollout runs "
                        "on (the run's data params; generated and cached on first use).")
    p.add_argument("--training-data", action="store_true",
                   help="Roll out on training trajectory --trajectory-index instead "
                        "of the held-out trajectory.")
    p.add_argument("--trajectory-index", type=int, default=TRAJECTORY_INDEX,
                   help="Training trajectory to use with --training-data.")
    p.add_argument("--start-timestep", type=int, default=START_TIMESTEP)
    p.add_argument("--cmap", default=CMAP)
    p.add_argument("--output", default=OUTPUT_PATH)
    p.add_argument("--dpi", type=int, default=DPI)
    return p.parse_args()


if __name__ == "__main__":
    setup_mlflow_uri()
    args = _parse_args()
    make_figure(
        run_name=args.run_name,
        experiment=args.experiment,
        n_steps=args.n_steps,
        trajectory_index=args.trajectory_index,
        start_timestep=args.start_timestep,
        cmap=args.cmap,
        output_path=args.output,
        dpi=args.dpi,
        n_cells=args.n_cells,
        embedding_run_name=args.embedding_run_name,
        embedding_experiment=args.embedding_experiment,
        embedding_epoch=args.epoch,
        heldout_seed_offset=None if args.training_data else args.heldout_seed_offset,
    )
