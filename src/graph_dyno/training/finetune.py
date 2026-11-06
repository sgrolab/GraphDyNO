"""Fine-tune shape decoders before GNN training.

The core training loop is shared across all shape types. Each type provides:
  - A dataset (input, target pairs)
  - A forward function (input → prediction + loss dict)
  - Modules to optimize and optionally re-freeze

To add a new shape type, implement a _setup_<type>() function that returns
(dataset, modules, forward_fn, N) and add a branch in finetune_decoder().
"""

import hashlib
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm.auto import tqdm
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import mlflow

from graph_dyno.data.dataloader import CUDAPrefetcher
from graph_dyno.precompute.precompute_embeddings import precompute_sdfs, _sdf_filename


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _compute_finetune_cache_key(cfg, legacy=False):
    """Hash checkpoint content + data config + finetune params + split config into an 8-char key.

    legacy=True reproduces the pre-6af18be key, which did not hash
    finetune_decoder.val_ratio (see _finetune_cache_keys).
    """
    h = hashlib.sha256()
    ckpt_path = Path(cfg.shape.checkpoint_path)
    h.update(ckpt_path.read_bytes())
    from graph_dyno.data.trajectory_generator import generate_config_hash
    h.update(generate_config_hash(cfg.data).encode())
    ft = cfg.training.finetune_decoder
    ft_fields = {
        "n_epochs": ft.n_epochs, "lr": ft.lr, "batch_size": ft.batch_size,
        "loss_terms": OmegaConf.to_container(ft.loss_terms, resolve=True),
        "n_trajectories": ft.get("n_trajectories", None),
    }
    if not legacy:
        # Part of the key: it overrides dataloader.val_ratio for the finetune
        # split below, so two runs can share every other hashed field and still
        # need different weights.
        ft_fields["val_ratio"] = ft.get("val_ratio", None)
    h.update(OmegaConf.to_yaml(OmegaConf.create(ft_fields)).encode())
    dl = cfg.dataloader
    split_params = OmegaConf.to_yaml(OmegaConf.create({
        "seed": dl.seed, "train_ratio": dl.train_ratio, "val_ratio": dl.val_ratio,
        "val_split_mode": dl.get("val_split_mode", "trajectory"),
    }))
    h.update(split_params.encode())
    return h.hexdigest()[:8]


def _finetune_cache_keys(cfg):
    """Keys to look up, current first. New weights are only ever saved under keys[0].

    Adding val_ratio to the hash (6af18be) re-keyed every cache, including the
    ones it changes nothing for, and orphaned their finetuned weights. The
    legacy key still names equivalent weights when the finetune split is the
    one the old code ran: finetune_decoder.val_ratio unset, a non-empty
    dataloader val split (old code crashed on 0), and train + val == 1 (old
    code took train_ratio as given, new code uses 1 - val_ratio).
    """
    keys = [_compute_finetune_cache_key(cfg)]
    dl = cfg.dataloader
    if (cfg.training.finetune_decoder.get("val_ratio", None) is None
            and dl.val_ratio > 0
            and abs(dl.train_ratio + dl.val_ratio - 1.0) < 1e-9):
        keys.append(_compute_finetune_cache_key(cfg, legacy=True))
    return keys


def _find_finetune_cache(cfg, cache_dir):
    """Path of the first existing cached-weights file for cfg, else None."""
    for key in _finetune_cache_keys(cfg):
        path = _get_finetune_cache_path(cache_dir, key)
        if path.exists():
            return path
    return None


def _get_finetune_cache_path(cache_dir, cache_key):
    """Return the path to the cached finetuned decoder weights."""
    return Path(cache_dir) / f"finetuned_decoder_{cache_key}.pt"


def check_finetune_cache(cfg, cache_dir):
    """Check if cached finetuned weights exist.

    Returns:
        Path to cache file if it exists and should be used, else None.
    """
    ft_cfg = cfg.training.get("finetune_decoder", {})
    if not ft_cfg.get("enabled", False):
        return None
    if not ft_cfg.get("cache_weights", False):
        return None
    if ft_cfg.get("force_refinetune", False):
        return None
    return _find_finetune_cache(cfg, cache_dir)



def load_all_sdfs(cache_dir, cache_id, traj_indices):
    """Load all precomputed SDFs and crop_sides from disk, flattened across (timestep, entity).

    Returns:
        all_sdf: (total, S, S) float16 tensor
        all_crop_sides: (total,) int32 tensor
        T: number of timesteps per trajectory
        N: number of entities per timestep
    """
    from graph_dyno.precompute.precompute_embeddings import _crop_sides_filename

    first_sdf_path = os.path.join(cache_dir, _sdf_filename(cache_id, traj_indices[0]))
    sdf_shape = np.load(first_sdf_path, mmap_mode='r').shape
    T, N, S = sdf_shape[0], sdf_shape[1], sdf_shape[2]
    total = len(traj_indices) * T * N

    all_sdf = torch.empty(total, S, S, dtype=torch.float16)
    all_crop_sides = torch.empty(total, dtype=torch.int32)

    offset = 0
    for traj_idx in traj_indices:
        n = T * N
        sdf_path = os.path.join(cache_dir, _sdf_filename(cache_id, traj_idx))
        with open(sdf_path, 'rb') as f:
            version = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format._read_array_header(f, version) \
                if hasattr(np.lib.format, '_read_array_header') \
                else (np.lib.format.read_array_header_1_0(f) if version == (1, 0)
                      else np.lib.format.read_array_header_2_0(f))
            assert dtype == np.float16 and not fortran
            dst = all_sdf[offset:offset + n].numpy().reshape(shape)
            f.readinto(dst.data)

        cs_path = os.path.join(cache_dir, _crop_sides_filename(cache_id, traj_idx))
        cs_np = np.load(cs_path)
        all_crop_sides[offset:offset + n] = torch.from_numpy(cs_np.reshape(n))
        del cs_np

        offset += n

    return all_sdf, all_crop_sides, T, N


def _compute_iou(pred, target):
    """Mean IoU for SDF predictions, over samples that HAVE a target.

    Real-data caches carry a fixed slot axis, so most entries in any frame are
    padding with an all-background target (83% of slots on ctc_mix_b_late).
    IoU is undefined there — union is 0 — and counting those as 0 rescales the
    mean by the valid fraction instead of measuring reconstruction quality: the
    ctc_mix_b_late_ft finetune reported 0.144 while its decoder was in fact
    reconstructing real cells at 0.866 (0.866 x 0.166 valid = 0.144). Simulation
    caches have every slot occupied, so this changes nothing there.
    """
    pred_bin = (pred > 0).float()
    gt_bin = (target > 0).float()
    intersection = (pred_bin * gt_bin).sum(dim=(1, 2))
    union = pred_bin.sum(dim=(1, 2)) + gt_bin.sum(dim=(1, 2)) - intersection
    has_target = gt_bin.sum(dim=(1, 2)) > 0
    if not bool(has_target.any()):
        return torch.zeros((), device=pred.device)
    return (intersection[has_target] / (union[has_target] + 1e-6)).mean()


def _save_finetune_cache(cache_dir, cache_key, save_dict):
    """Atomically save finetuned weights to cache."""
    cache_path = _get_finetune_cache_path(cache_dir, cache_key)
    fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".pt.tmp")
    os.close(fd)
    try:
        torch.save(save_dict, tmp_path)
        Path(tmp_path).rename(cache_path)
        print(f"Saved finetuned weights to {cache_path}")
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise



# ---------------------------------------------------------------------------
# Generic finetuning loop
# ---------------------------------------------------------------------------

def _run_finetune_loop(*, train_loader, val_loader, forward_fn, modules,
                       optimizer, loss_terms, n_epochs, device, label="FT"):
    """Run the shared train/val finetuning loop.

    Args:
        train_loader: DataLoader yielding (input, target) batches.
        val_loader: DataLoader yielding (input, target) batches.
        forward_fn: Callable(input_batch, target_batch) -> (prediction, loss_dict).
            loss_dict maps loss name -> scalar tensor.
        modules: list of nn.Module to toggle train/eval.
        optimizer: Optimizer over all trainable parameters.
        loss_terms: dict mapping loss name -> weight.
        n_epochs: Number of finetuning epochs.
        device: Torch device string.
        label: Label for progress bar.

    Returns:
        (final_val_loss, final_val_iou)
    """
    def _weighted_loss(loss_dict):
        return sum(v * loss_terms[k] for k, v in loss_dict.items() if k in loss_terms)

    # --- Pre-finetuning baseline ---
    for m in modules:
        m.eval()
    pre_val_loss = 0.0
    pre_val_iou = 0.0
    with torch.no_grad():
        n_val = len(val_loader)
        if n_val == 0:
            raise ValueError(
                "Finetune validation loader is empty — nothing to measure the "
                "decoder against. This happens when the split leaves no held-out "
                "frames (e.g. dataloader.val_ratio: 0.0); set "
                "training.finetune_decoder.val_ratio to a small non-zero value."
            )
        for inp, tgt in CUDAPrefetcher(val_loader, device):
            pred, loss_dict = forward_fn(inp, tgt)
            pre_val_loss += _weighted_loss(loss_dict).item()
            pre_val_iou += _compute_iou(pred, tgt).item()
    pre_val_loss /= n_val
    pre_val_iou /= n_val

    print(f"Pre-finetune baseline: Val Loss={pre_val_loss:.6f}  Val IoU={pre_val_iou:.4f}")
    mlflow.log_metrics({"ft_val_loss": pre_val_loss, "ft_val_iou": pre_val_iou}, step=-1)

    val_loss = pre_val_loss
    val_iou = pre_val_iou

    for epoch in range(n_epochs):
        for m in modules:
            m.train()
        train_loss = 0.0

        for inp, tgt in tqdm(CUDAPrefetcher(train_loader, device),
                              desc=f"{label} Epoch {epoch+1}/{n_epochs} [Train]",
                              total=len(train_loader), dynamic_ncols=True):
            _, loss_dict = forward_fn(inp, tgt)
            loss = _weighted_loss(loss_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        for m in modules:
            m.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            n_val = len(val_loader)
            for inp, tgt in CUDAPrefetcher(val_loader, device):
                pred, loss_dict = forward_fn(inp, tgt)
                val_loss += _weighted_loss(loss_dict).item()
                val_iou += _compute_iou(pred, tgt).item()

        val_loss /= n_val
        val_iou /= n_val

        print(f"{label} Epoch {epoch+1}: Train Loss={train_loss:.4f}  "
              f"Val Loss={val_loss:.4f}  Val IoU={val_iou:.4f}")

        mlflow.log_metrics({
            "ft_train_loss": train_loss,
            "ft_val_loss": val_loss,
            "ft_val_iou": val_iou,
        }, step=epoch)

    return val_loss, val_iou


# ---------------------------------------------------------------------------
# Per-shape-type setup
# ---------------------------------------------------------------------------

def _setup_dino(model, embeddings_dict, cache_dir, cache_id, traj_indices):
    """Build dataset and forward_fn for DINO finetuning (embeddings → SDF).

    Loads SDFs from disk (only needed here for DINO path).

    Args:
        embeddings_dict: dict mapping traj_idx -> {"embeddings": (T,N,D), "crop_sides": (T,N)}

    Returns:
        (TensorDataset, modules, forward_fn, N)
    """
    first = embeddings_dict[traj_indices[0]]["embeddings"]
    T, N, D = first.shape
    total = len(traj_indices) * T * N

    all_emb = torch.empty(total, D)
    offset = 0
    for traj_idx in traj_indices:
        emb = embeddings_dict[traj_idx]["embeddings"]
        n = T * N
        all_emb[offset:offset + n] = emb.reshape(n, D).cpu()
        offset += n

    all_sdf, _, _, _ = load_all_sdfs(cache_dir, cache_id, traj_indices)

    decoder = model.dino_decoder

    def forward_fn(emb, targets):
        loss_dict = decoder.training_step(emb, targets)
        pred = decoder(emb)
        return pred, loss_dict

    return TensorDataset(all_emb, all_sdf), [decoder], forward_fn, T, N


def _setup_conv(model, encoder, all_sdf, cache_dir, cache_id, traj_indices):
    """Build dataset and forward_fn for Conv finetuning (SDF → SDF, end-to-end).

    Uses the pre-loaded all_sdf tensor passed from main (no reload).

    Returns:
        (TensorDataset, modules, forward_fn, T, N)
    """
    # Infer T, N from SDF file shape on disk
    first_sdf_path = os.path.join(cache_dir, _sdf_filename(cache_id, traj_indices[0]))
    sdf_shape = np.load(first_sdf_path, mmap_mode='r').shape
    T, N = sdf_shape[0], sdf_shape[1]

    decoder = model.dino_decoder

    encoder.eval()
    encoder.requires_grad_(False)

    def forward_fn(sdf_in, sdf_target):
        with torch.no_grad():
            bottleneck = encoder(sdf_in.float())
        pred = decoder.decode_from_bottleneck(bottleneck)
        loss_dict = {
            'image_mse': F.mse_loss(pred, sdf_target.float()),
            'embedding_l2': bottleneck.pow(2).mean(),
        }
        return pred, loss_dict

    return TensorDataset(all_sdf, all_sdf), [decoder], forward_fn, T, N


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def finetune_decoder(model, cfg, device, cache_dir, traj_indices,
                     encoder=None, all_sdf=None, embeddings=None):
    """Fine-tune shape decoder before GNN training.

    Called before the main dataloader is created. Computes its own train/val
    split from cfg and builds temporary dataloaders for finetuning.

    Args:
        model: GraphModelDINO with shape_decoder.
        cfg: Full Hydra config.
        device: Torch device.
        cache_dir: Directory with precomputed SDFs.
        traj_indices: List of trajectory indices.
        encoder: ConvEncoder instance (required for conv, ignored otherwise).
        all_sdf: Pre-loaded (total, S, S) SDF tensor. Required for conv to
            avoid loading SDFs twice. Ignored for DINO (loads internally).
        embeddings: Pre-loaded embeddings dict (required for DINO, ignored for conv).
    """
    ft_cfg = cfg.training.finetune_decoder
    shape_type = cfg.shape.type

    # --- Cache: check for cached finetuned weights ---
    cache_weights = ft_cfg.get("cache_weights", False)
    force_refinetune = ft_cfg.get("force_refinetune", False)

    # A seg-ablation ladder needs the shape prior pinned: every level must see
    # the *same* decoder, finetuned on the *clean* crops, or a measured drop in
    # separation is partly a different decoder. `_compute_finetune_cache_key`
    # hashes ckpt bytes + generate_config_hash(cfg.data) + ft params and carries
    # no ablation identifier, so with cfg.data unchanged by design every arm
    # resolves to the same `finetuned_decoder_<key>.pt` — which is exactly what
    # a controlled ablation wants, and force_refinetune is what would break it.
    if force_refinetune and cfg.get("seg_ablation", None) is not None:
        from graph_dyno.seg_ablation.plan import spec_is_active
        if spec_is_active(cfg.seg_ablation):
            raise ValueError(
                "training.finetune_decoder.force_refinetune=true is refused "
                "under seg_ablation. The finetune cache key carries no "
                "ablation identifier, so re-finetuning would overwrite the "
                "shared decoder every arm of the ladder depends on. Rebuild "
                "the decoder in a separate run without +seg_ablation=, then "
                "run the ladder."
            )

    if cache_weights and not force_refinetune:
        cache_path = _find_finetune_cache(cfg, cache_dir)
        if cache_path is not None:
            cached = torch.load(cache_path, map_location=device, weights_only=True)
            model.dino_decoder.load_state_dict(cached["decoder_state_dict"])
            if ft_cfg.refreeze:
                model.dino_decoder.eval()
                model.dino_decoder.requires_grad_(False)
            print(f"\n{'='*60}")
            print(f"Loaded cached finetuned weights from {cache_path}")
            print(f"  Val Loss={cached['val_loss']:.4f}  Val IoU={cached['val_iou']:.4f}")
            print(f"{'='*60}\n")
            return

    print(f"\n{'='*60}")
    print(f"Fine-tuning {shape_type} decoder for {ft_cfg.n_epochs} epochs (lr={ft_cfg.lr})")
    print(f"{'='*60}")

    from graph_dyno.data.trajectory_generator import generate_config_hash
    cache_id = generate_config_hash(cfg.data)

    sdf_workers = ft_cfg.get("sdf_workers", cfg.shape.get("sdf_workers", 32))
    precompute_sdfs(cache_dir=cache_dir, cfg=cfg, sdf_workers=sdf_workers)

    # Under a temporal split only the first ft.n_trajectories are ever indexed
    # (see the active_trajs slice below), so loading the rest costs ~200 GB of
    # SDF for nothing. The trajectory-split branch indexes by real trajectory id
    # and does need them all — leave that case alone.
    _ft_n = ft_cfg.get("n_trajectories", None)
    _temporal = cfg.dataloader.get("val_split_mode", "trajectory") == "temporal"
    setup_traj_indices = (
        traj_indices[:_ft_n]
        if _ft_n and (_temporal or len(traj_indices) == 1)
        else traj_indices
    )

    # Shape-type-specific setup
    print("Loading precomputed data...")
    if shape_type == "dino":
        if embeddings is None:
            raise ValueError("DINO finetuning requires pre-loaded embeddings")
        full_dataset, modules, forward_fn, T, N = _setup_dino(
            model, embeddings, cache_dir, cache_id, setup_traj_indices)
    elif shape_type == "conv":
        if encoder is None:
            raise ValueError("Conv finetuning requires an encoder instance")
        if all_sdf is None:
            raise ValueError("Conv finetuning requires pre-loaded all_sdf tensor")
        full_dataset, modules, forward_fn, T, N = _setup_conv(
            model, encoder, all_sdf, cache_dir, cache_id, traj_indices)
    else:
        raise ValueError(f"Finetuning not supported for shape type: {shape_type}")

    # Compute train/val split (same logic as dataloader.split_dataset)
    n_traj = len(traj_indices)
    dl_cfg = cfg.dataloader
    train_ratio = dl_cfg.train_ratio
    val_ratio = dl_cfg.val_ratio
    seed = dl_cfg.seed
    val_split_mode = dl_cfg.get("val_split_mode", "trajectory")

    # `dataloader.val_ratio: 0.0` is set on the real-data configs so that no
    # timestep is withheld from the per-slot `parameter_embedding` — a held-out
    # tail strands whole slots (see the untrained-slot check in train.py). That
    # reasoning does NOT carry over here: decoder finetuning is a per-crop
    # reconstruction task with no per-slot parameters, so its validation split
    # costs nothing that motivated val_ratio=0. Taken literally it would leave
    # the val loader empty and divide by zero in _run_finetune_loop. Fall back
    # to a small monitoring split, overridable as finetune_decoder.val_ratio.
    val_ratio = float(ft_cfg.get("val_ratio", None) or val_ratio or 0.1)
    train_ratio = 1.0 - val_ratio

    if val_split_mode == "temporal" or n_traj == 1:
        # Temporal split: first timesteps train, last timesteps val, per trajectory
        n_val_t = int(T * val_ratio)
        n_train_t = T - n_val_t

        # Optionally limit to fewer trajectories
        ft_n_trajectories = ft_cfg.get("n_trajectories", None)
        active_trajs = traj_indices[:ft_n_trajectories] if ft_n_trajectories else traj_indices

        train_flat = []
        val_flat = []
        for i, _ in enumerate(active_trajs):
            traj_base = i * T * N
            for t in range(n_train_t):
                entity_base = traj_base + t * N
                train_flat.extend(range(entity_base, entity_base + N))
            for t in range(n_train_t, T):
                entity_base = traj_base + t * N
                val_flat.extend(range(entity_base, entity_base + N))
    else:
        # Trajectory split: split by whole trajectories
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_traj, generator=g).tolist()
        n_train = int(n_traj * train_ratio)
        n_val = int(n_traj * val_ratio)
        train_trajs = perm[:n_train]
        val_trajs = perm[n_train:n_train + n_val]

        ft_n_trajectories = ft_cfg.get("n_trajectories", None)
        if ft_n_trajectories is not None:
            train_trajs = train_trajs[:ft_n_trajectories]
            print(f"Limiting finetune training to {len(train_trajs)} trajectory(ies)")

        def _traj_to_flat(traj_list):
            flat = []
            for t in traj_list:
                base = t * T * N
                flat.extend(range(base, base + T * N))
            return flat

        train_flat = _traj_to_flat(train_trajs)
        val_flat = _traj_to_flat(val_trajs)

    print(f"  Train: {len(train_flat)} samples, Val: {len(val_flat)} samples")

    ft_batch_size = ft_cfg.batch_size
    ft_train_loader = DataLoader(
        Subset(full_dataset, train_flat),
        batch_size=ft_batch_size, shuffle=True, pin_memory=False,
    )
    ft_val_loader = DataLoader(
        Subset(full_dataset, val_flat),
        batch_size=ft_batch_size, shuffle=False, pin_memory=False,
    )

    # Unfreeze all modules
    for m in modules:
        m.requires_grad_(True)
        m.train()

    all_params = [p for m in modules for p in m.parameters()]
    optimizer = torch.optim.Adam(all_params, lr=ft_cfg.lr)
    loss_terms = OmegaConf.to_container(ft_cfg.loss_terms, resolve=True)
    print(f"  Loss terms: {loss_terms}")

    val_loss, val_iou = _run_finetune_loop(
        train_loader=ft_train_loader,
        val_loader=ft_val_loader,
        forward_fn=forward_fn,
        modules=modules,
        optimizer=optimizer,
        loss_terms=loss_terms,
        n_epochs=ft_cfg.n_epochs,
        device=device,
    )

    del full_dataset, ft_train_loader, ft_val_loader

    # Re-freeze if configured
    if ft_cfg.refreeze:
        for m in modules:
            m.eval()
            m.requires_grad_(False)
        print(f"{shape_type} decoder re-frozen after fine-tuning.")
    else:
        print(f"{shape_type} decoder left unfrozen (will be included in main optimizer).")

    # Cache weights
    if cache_weights:
        cache_key = _compute_finetune_cache_key(cfg)
        save_dict = {
            "decoder_state_dict": model.dino_decoder.state_dict(),
            "val_loss": val_loss,
            "val_iou": val_iou,
        }
        _save_finetune_cache(cache_dir, cache_key, save_dict)

    print(f"{'='*60}\n")
