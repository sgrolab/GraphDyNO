"""
Precompute shape embeddings for simulation trajectory data using any encoder.

For each trajectory file {cache_id}_{traj_idx:04d}.pt, produces a companion file
{companion_suffix}_{cache_id}_{traj_idx:04d}.pt containing:
    embeddings:  (T, N, D) float32  - embedding vector per object per timestep
    crop_sides:  (T, N) int32       - square crop side length in original image pixels

Supports chunked processing of timesteps within a trajectory for parallelism.

Usage:
    # Process all trajectories
    python precompute_embeddings.py cache_dir=./data/cache_abc model.decoder_type=fourier

    # Process specific trajectories
    python precompute_embeddings.py cache_dir=./data/cache_abc traj_idx='[0,1,5]'

    # Chunk a trajectory's timesteps across 4 jobs
    python precompute_embeddings.py cache_dir=./data/cache_abc traj_idx=5 n_chunks=4 chunk_idx=0
    python precompute_embeddings.py cache_dir=./data/cache_abc traj_idx=5 n_chunks=4 chunk_idx=1

    # Merge chunks back into a single companion file
    python precompute_embeddings.py cache_dir=./data/cache_abc traj_idx=5 merge=true
"""

import glob
import math
import os

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import hydra

from graph_dyno.encoders.factory import create_shape_encoder, get_companion_suffix


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _companion_filename(companion_suffix: str, cache_id: str, traj_idx: int,
                        chunk_idx: int | None = None) -> str:
    base = f"{companion_suffix}_{cache_id}_{traj_idx:04d}"
    if chunk_idx is not None:
        return f"{base}_chunk{chunk_idx}.pt"
    return f"{base}.pt"


def _chunk_glob_pattern(companion_suffix: str, cache_id: str, traj_idx: int) -> str:
    return f"{companion_suffix}_{cache_id}_{traj_idx:04d}_chunk*.pt"


# ---------------------------------------------------------------------------
# Core precomputation
# ---------------------------------------------------------------------------

def precompute_all(
    cache_dir: str,
    cfg: DictConfig = None,
    encoder=None,
    device: torch.device | str | None = None,
    batch_size: int = 64,
    trajectory_indices: list[int] | None = None,
    n_chunks: int = 1,
    chunk_idx: int = 0,
) -> None:
    """Precompute and cache embeddings for trajectories in a cache directory.

    Provide either cfg (to create the encoder) or a pre-built encoder instance.

    When n_chunks > 1, each trajectory's timesteps are split into chunks.
    Each chunk is saved with chunk_idx in the filename. Use merge_chunks()
    to combine them afterward.

    Args:
        cache_dir: Directory containing metadata.yaml and trajectory .pt files
        cfg: Hydra config (used to create encoder if encoder is None)
        encoder: Pre-built ShapeEncoder instance (overrides cfg)
        device: Device for inference (None for auto-detect)
        batch_size: Batch size for encoding
        trajectory_indices: Specific trajectory indices to process. If None,
            processes all trajectories.
        n_chunks: Number of timestep chunks per trajectory (for parallelism).
        chunk_idx: Which chunk to process (0-indexed).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Derive companion suffix without loading encoder weights
    if encoder is not None:
        companion_suffix = encoder.companion_suffix
    else:
        from graph_dyno.encoders.factory import get_companion_suffix
        companion_suffix = get_companion_suffix(cfg)

    # Load metadata
    metadata_path = os.path.join(cache_dir, "metadata.yaml")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No metadata.yaml found in '{cache_dir}'")

    metadata = OmegaConf.load(metadata_path)
    n_trajectories = metadata.data.generator.n_trajectories

    from graph_dyno.data.trajectory_generator import generate_config_hash
    cache_id = generate_config_hash(metadata.data)

    if trajectory_indices is None:
        trajectory_indices = list(range(n_trajectories))

    use_chunks = n_chunks > 1

    # Check if all requested files already exist (before loading encoder weights)
    all_exist = True
    for traj_idx in trajectory_indices:
        filename = _companion_filename(companion_suffix, cache_id, traj_idx,
                                       chunk_idx if use_chunks else None)
        if not os.path.exists(os.path.join(cache_dir, filename)):
            all_exist = False
            break

    if all_exist:
        print(f"Embeddings already cached for all {len(trajectory_indices)} trajectories "
              f"(suffix={companion_suffix}). Skipping.")
        return

    # Only now create the encoder (loads weights for DINO)
    if encoder is None:
        encoder = create_shape_encoder(cfg, device=str(device))

    print(f"Precomputing embeddings for {len(trajectory_indices)} trajectories "
          f"(suffix={companion_suffix}"
          + (f", chunk {chunk_idx}/{n_chunks})" if use_chunks else ")") + "...")

    for traj_idx in tqdm(trajectory_indices, desc="Trajectories"):
        filename = _companion_filename(companion_suffix, cache_id, traj_idx,
                                       chunk_idx if use_chunks else None)
        companion_path = os.path.join(cache_dir, filename)

        if os.path.exists(companion_path):
            continue

        # Load trajectory
        traj_filename = f"{cache_id}_{traj_idx:04d}.pt"
        traj_path = os.path.join(cache_dir, traj_filename)
        trajectory_data = torch.load(traj_path, map_location="cpu", mmap=True)
        channeled_images = trajectory_data["channeled_images"]  # (T, N, H, W)
        T, N, H, W = channeled_images.shape

        # Determine timestep range for this chunk
        if use_chunks:
            ts_per_chunk = math.ceil(T / n_chunks)
            t_start = chunk_idx * ts_per_chunk
            t_end = min(t_start + ts_per_chunk, T)
            channeled_images = channeled_images[t_start:t_end]
            chunk_T = t_end - t_start
        else:
            chunk_T = T

        # Encode
        flat_masks = channeled_images.reshape(chunk_T * N, H, W)
        embeddings, crop_sides = encoder.encode_masks(
            flat_masks, batch_size=batch_size,
        )

        torch.save({
            "embeddings": embeddings.view(chunk_T, N, -1),
            "crop_sides": crop_sides.view(chunk_T, N),
        }, companion_path)

        del trajectory_data, channeled_images, embeddings, crop_sides

    print(f"Done.")


# ---------------------------------------------------------------------------
# SDF precomputation
# ---------------------------------------------------------------------------

def _sdf_filename(cache_id: str, traj_idx: int) -> str:
    return f"sdf_{cache_id}_{traj_idx:04d}.npy"


def _crop_sides_filename(cache_id: str, traj_idx: int) -> str:
    return f"crop_sides_{cache_id}_{traj_idx:04d}.npy"


def _assert_sdf_target_size(cache_dir: str, cache_id: str, traj_idx: int,
                            target_size: int) -> None:
    """Refuse to reuse an SDF file that was built at a different resolution."""
    path = os.path.join(cache_dir, _sdf_filename(cache_id, traj_idx))
    if not os.path.exists(path):
        return
    cached = int(np.load(path, mmap_mode="r").shape[-1])
    if cached != target_size:
        raise ValueError(
            f"{path} holds {cached}x{cached} SDFs but shape.target_size={target_size}. "
            f"The SDF filename carries only the data hash, so a cache dir cannot hold "
            f"both. Use a separate cache_dir, or delete the sdf_*/crop_sides_* files."
        )


def precompute_sdfs(
    cache_dir: str,
    cfg: DictConfig = None,
    sdf_workers: int = 32,
    target_size: int = 224,
) -> None:
    """Precompute and cache SDF targets for all trajectories.

    For each trajectory, converts channeled_images (binary masks) to signed
    distance fields and saves them as a separate file. SDFs are stored as
    float16 at (T, N, target_size, target_size).

    Args:
        cache_dir: Directory containing metadata.yaml and trajectory .pt files
        cfg: Hydra config (used to derive cache_id)
        sdf_workers: CPU workers for SDF conversion
        target_size: Spatial size for SDF output (224 for DINO)
    """
    from graph_dyno.precompute.precompute_dino_embeddings import _compute_sdfs_multiprocess

    metadata_path = os.path.join(cache_dir, "metadata.yaml")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No metadata.yaml found in '{cache_dir}'")

    metadata = OmegaConf.load(metadata_path)
    n_trajectories = metadata.data.generator.n_trajectories

    from graph_dyno.data.trajectory_generator import generate_config_hash
    cache_id = generate_config_hash(metadata.data)

    # Check if all SDF + crop_sides files already exist
    all_exist = all(
        os.path.exists(os.path.join(cache_dir, _sdf_filename(cache_id, i)))
        and os.path.exists(os.path.join(cache_dir, _crop_sides_filename(cache_id, i)))
        for i in range(n_trajectories)
    )
    if all_exist:
        # The SDF filename carries only the data hash, not target_size — so two
        # runs at different target_size in one cache dir would silently reuse
        # each other's crops. Refuse rather than serve the wrong resolution.
        _assert_sdf_target_size(cache_dir, cache_id, 0, target_size)
        print(f"SDFs already cached for all {n_trajectories} trajectories. Skipping.")
        return

    print(f"Precomputing SDFs for {n_trajectories} trajectories "
          f"(workers={sdf_workers}, target_size={target_size})...")

    for traj_idx in range(n_trajectories):
        sdf_path = os.path.join(cache_dir, _sdf_filename(cache_id, traj_idx))
        crop_sides_path = os.path.join(cache_dir, _crop_sides_filename(cache_id, traj_idx))
        if os.path.exists(sdf_path) and os.path.exists(crop_sides_path):
            _assert_sdf_target_size(cache_dir, cache_id, traj_idx, target_size)
            continue

        traj_filename = f"{cache_id}_{traj_idx:04d}.pt"
        traj_path = os.path.join(cache_dir, traj_filename)
        trajectory_data = torch.load(traj_path, map_location="cpu", mmap=True)
        channeled_images = trajectory_data["channeled_images"]  # (T, N, H, W)
        T, N, H, W = channeled_images.shape

        flat_masks = channeled_images.reshape(T * N, H, W).cpu()
        del trajectory_data, channeled_images

        sdf_masks, crop_sides = _compute_sdfs_multiprocess(
            flat_masks, target_size=target_size, num_workers=sdf_workers,
        )
        del flat_masks

        # Save as (T, N, target_size, target_size) float16
        np.save(sdf_path, sdf_masks.numpy().reshape(T, N, target_size, target_size))
        del sdf_masks

        # Save crop_sides as (T, N) int32
        np.save(crop_sides_path, crop_sides.numpy().reshape(T, N))
        del crop_sides

    print("Done.")


# ---------------------------------------------------------------------------
# Merge chunks
# ---------------------------------------------------------------------------

def merge_chunks(
    cache_dir: str,
    cfg: DictConfig = None,
    companion_suffix: str | None = None,
    trajectory_indices: list[int] | None = None,
) -> None:
    """Merge chunked companion files into single companion files per trajectory.

    Finds all chunk files matching the pattern and concatenates them along
    the timestep dimension.

    Args:
        cache_dir: Directory containing chunk files
        cfg: Hydra config (used to derive companion_suffix and cache_id)
        companion_suffix: Override companion suffix (otherwise derived from cfg)
        trajectory_indices: Which trajectories to merge. If None, discovers
            all trajectories that have chunk files.
    """
    if companion_suffix is None:
        companion_suffix = get_companion_suffix(cfg)

    metadata_path = os.path.join(cache_dir, "metadata.yaml")
    metadata = OmegaConf.load(metadata_path)

    from graph_dyno.data.trajectory_generator import generate_config_hash
    cache_id = generate_config_hash(metadata.data)

    if trajectory_indices is None:
        # Discover trajectories with chunk files
        n_trajectories = metadata.data.generator.n_trajectories
        trajectory_indices = []
        for traj_idx in range(n_trajectories):
            pattern = os.path.join(cache_dir,
                                   _chunk_glob_pattern(companion_suffix, cache_id, traj_idx))
            if glob.glob(pattern):
                trajectory_indices.append(traj_idx)

    if not trajectory_indices:
        print("No chunk files found to merge.")
        return

    print(f"Merging chunks for {len(trajectory_indices)} trajectories...")

    for traj_idx in tqdm(trajectory_indices, desc="Merging"):
        final_path = os.path.join(
            cache_dir, _companion_filename(companion_suffix, cache_id, traj_idx))

        if os.path.exists(final_path):
            continue

        # Find and sort chunk files
        pattern = os.path.join(cache_dir,
                               _chunk_glob_pattern(companion_suffix, cache_id, traj_idx))
        chunk_files = sorted(glob.glob(pattern))

        if not chunk_files:
            print(f"  WARNING: no chunks found for trajectory {traj_idx}, skipping")
            continue

        # Load and concatenate
        all_embeddings = []
        all_crop_sides = []
        for chunk_path in chunk_files:
            data = torch.load(chunk_path, map_location="cpu")
            all_embeddings.append(data["embeddings"])
            all_crop_sides.append(data["crop_sides"])

        torch.save({
            "embeddings": torch.cat(all_embeddings, dim=0),
            "crop_sides": torch.cat(all_crop_sides, dim=0),
        }, final_path)

        # Clean up chunk files
        for chunk_path in chunk_files:
            os.remove(chunk_path)

    print("Done.")


# ---------------------------------------------------------------------------
# Hydra CLI
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cache_dir = cfg.get("cache_dir", None)
    n_chunks = cfg.get("n_chunks", 1)
    chunk_idx = cfg.get("chunk_idx", 0)
    traj_idx = cfg.get("traj_idx", None)
    batch_size = cfg.get("batch_size", 64)
    device = cfg.get("device", None)
    do_merge = cfg.get("merge", False)

    if cache_dir is None:
        cache_dir = cfg.data.generator.get("cache_dir", None)
    if cache_dir is None:
        from graph_dyno.data.trajectory_generator import generate_config_hash
        cache_id = generate_config_hash(cfg.data)
        cache_dir = f"./data/cache_{cache_id}"

    # Parse traj_idx into a list
    if traj_idx is not None:
        if isinstance(traj_idx, int):
            trajectory_indices = [traj_idx]
        else:
            trajectory_indices = list(traj_idx)
    else:
        trajectory_indices = None

    if do_merge:
        merge_chunks(
            cache_dir=cache_dir,
            cfg=cfg,
            trajectory_indices=trajectory_indices,
        )
    else:
        precompute_all(
            cache_dir=cache_dir,
            cfg=cfg,
            device=device,
            batch_size=batch_size,
            trajectory_indices=trajectory_indices,
            n_chunks=n_chunks,
            chunk_idx=chunk_idx,
        )


if __name__ == "__main__":
    import sys
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra.run.dir=.")
    main()
