"""
Precompute DINO embeddings for simulation trajectory data and save as companion files.

For each trajectory file {cache_id}_{traj_idx:04d}.pt, produces a companion file
{cache_id}_{traj_idx:04d}_dino_{model_key}.pt containing:
    cls_tokens:   (T, N, D) float32  - CLS token per object per timestep
    patch_means:  (T, N, D) float32  - mean-pooled patch tokens
    crop_sides:   (T, N) int32       - square crop side length in original image pixels

Uses the same cv2 + scipy SDF pipeline as cell_mask_extractor._ann_to_sdf
to guarantee identical preprocessing to what the DINODecoder was trained on.

The CLI below is DINOv3-only: it hardcodes DINOv3FeatureExtractor and leaves
_SimMaskDataset on the default 3-channel InputSpec. Nothing on the training path
reaches it — train.py goes precompute_all -> create_shape_encoder — so the
Cell-DINO backend is plumbed through encoders/dino.py instead of threading a
spec parameter through this legacy entry point.

Usage:
    python precompute_dino_embeddings.py --cache_dir ./data/cache_abc123 --dino_model SP
    python precompute_dino_embeddings.py --cache_dir ./data/cache_abc123 --dino_model B --batch_size 128
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import psutil
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from dino.feature_extractor import DINOv3FeatureExtractor, InputSpec, ModelType
from dino.dataloader import CUDAPrefetcher


# ---------------------------------------------------------------------------
# SDF conversion — identical to cell_mask_extractor._ann_to_sdf
# ---------------------------------------------------------------------------

def _mask_to_sdf(mask_np: np.ndarray, target_size: int = 224) -> tuple[np.ndarray, int] | None:
    """Crop a binary mask to the smallest enclosing square and compute the SDF.

    Finds the tight bounding box, expands it to a square (object centered),
    then follows the cell_mask_extractor SDF pipeline:
        crop → cv2.resize(INTER_LINEAR) → threshold 0.5 → distance_transform_edt → normalize

    Using a square crop preserves aspect ratio when resizing to (target_size, target_size).

    Args:
        mask_np: (H, W) binary mask (uint8 or float with values 0/1)
        target_size: Output spatial size (default 224 for DINO)

    Returns:
        (sdf, crop_side) where sdf is (target_size, target_size) float16 in [-1, 1]
        and crop_side is the square side length in original pixels,
        or None if the mask is empty.
    """
    H, W = mask_np.shape

    # Find bounding box of nonzero region
    ys, xs = np.nonzero(mask_np)
    if len(ys) == 0:
        return None

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    bbox_h, bbox_w = y1 - y0, x1 - x0

    # Expand to square — center the object, clamp to image boundaries
    side = max(bbox_h, bbox_w)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, min(cy - side // 2, H - side))
    x0 = max(0, min(cx - side // 2, W - side))

    # Extract square crop (zero-pad if side > image dimension)
    if side <= H and side <= W:
        crop = mask_np[y0:y0 + side, x0:x0 + side].astype(np.float32)
    else:
        crop = np.zeros((side, side), dtype=np.float32)
        # Copy the region that overlaps with the image
        copy_h = min(side, H - y0)
        copy_w = min(side, W - x0)
        crop[:copy_h, :copy_w] = mask_np[y0:y0 + copy_h, x0:x0 + copy_w]

    # Resize (bilinear, matching cv2.INTER_LINEAR)
    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    # Threshold and compute SDF
    mask_bin = resized > 0.5
    dist_in = distance_transform_edt(mask_bin)
    dist_out = distance_transform_edt(~mask_bin)
    sdf = dist_in - dist_out

    return np.clip(sdf / (target_size / 2), -1, 1).astype(np.float16), side


def _mask_to_sdf_worker(args):
    """Worker wrapper for ProcessPoolExecutor (top-level for pickling)."""
    mask_np, target_size = args
    result = _mask_to_sdf(mask_np, target_size)
    if result is None:
        # Empty mask fallback: all-outside SDF
        sdf = np.full((target_size, target_size), -1.0, dtype=np.float16)
        return sdf, 0
    return result


# ---------------------------------------------------------------------------
# Dataset for feeding SDF masks to DINO
# ---------------------------------------------------------------------------

class _SimMaskDataset(Dataset):
    """Wraps precomputed SDF masks into backbone-ready channeled tensors.

    Matches the interface of _RGBMaskDataset from dino/precompute_embeddings.py.
    """

    def __init__(self, sdf_masks: torch.Tensor, input_spec: InputSpec | None = None):
        """
        Args:
            sdf_masks: (N, H, W) float SDF tensor in [-1, 1]
            input_spec: How to build channels. Defaults to the 3-channel
                replicate layout every existing `dino_*` cache was built with.
        """
        self.masks = sdf_masks
        self.input_spec = input_spec if input_spec is not None else InputSpec(3, "replicate")

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        return (self.input_spec(self.masks[idx]),)


# ---------------------------------------------------------------------------
# Core precomputation
# ---------------------------------------------------------------------------

def _compute_sdfs_multiprocess(
    flat_masks: torch.Tensor,
    target_size: int = 224,
    num_workers: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert binary masks to SDFs using multiprocessing (cv2 + scipy).

    Args:
        flat_masks: (M, H, W) binary masks
        target_size: Output spatial size
        num_workers: Number of worker processes

    Returns:
        sdf_masks: (M, target_size, target_size) float16 tensor
        crop_sides: (M,) int32 tensor — square crop side length per mask
    """
    M = flat_masks.shape[0]
    masks_np = flat_masks.numpy()

    # Build work items
    work_items = [(masks_np[i], target_size) for i in range(M)]

    # Process in parallel
    sdfs = np.empty((M, target_size, target_size), dtype=np.float16)
    crop_sides = np.empty(M, dtype=np.int32)

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for i, (sdf, crop_side) in enumerate(
            tqdm(pool.map(_mask_to_sdf_worker, work_items, chunksize=64),
                 total=M, desc="  SDF conversion", leave=False)
        ):
            sdfs[i] = sdf
            crop_sides[i] = crop_side

    return torch.from_numpy(sdfs), torch.from_numpy(crop_sides)


@torch.no_grad()
def precompute_trajectory(
    channeled_images: torch.Tensor,
    feature_extractor: DINOv3FeatureExtractor,
    batch_size: int,
    device: torch.device,
    num_workers: int = 4,
    sdf_workers: int = 32,
    target_size: int = 224,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DINO embeddings for all objects across all timesteps in one trajectory.

    Args:
        channeled_images: (T, N, H, W) binary masks from simulation
        feature_extractor: DINOv3FeatureExtractor on device
        batch_size: Batch size for DINO inference
        device: torch device for inference
        num_workers: DataLoader workers for DINO inference
        sdf_workers: CPU workers for SDF conversion
        target_size: Spatial size for SDF images (224 for DINO)

    Returns:
        cls_tokens:  (T, N, D) float32 on CPU
        patch_means: (T, N, D) float32 on CPU
        crop_sides:  (T, N) int32 on CPU — square crop side length in original pixels
    """
    T, N, H, W = channeled_images.shape

    # Flatten to (T*N, H, W) and convert to SDF via cv2/scipy multiprocessing
    flat_masks = channeled_images.reshape(T * N, H, W).cpu()
    sdf_masks, crop_sides_flat = _compute_sdfs_multiprocess(
        flat_masks, target_size=target_size, num_workers=sdf_workers,
    )

    # Create dataset and dataloader for DINO inference
    ds = _SimMaskDataset(sdf_masks)
    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        pin_memory=True, shuffle=False,
    )
    prefetcher = CUDAPrefetcher(loader, device)

    cls_list, pm_list = [], []
    for (rgb,) in tqdm(prefetcher, desc="  DINO inference", total=len(loader), leave=False):
        cls_token, patch_tokens = feature_extractor(rgb)
        cls_list.append(cls_token.cpu())
        pm_list.append(patch_tokens.mean(dim=1).cpu())

    cls_all = torch.cat(cls_list, dim=0)    # (T*N, D)
    pm_all = torch.cat(pm_list, dim=0)      # (T*N, D)

    D = cls_all.shape[-1]
    return (
        cls_all.view(T, N, D),
        pm_all.view(T, N, D),
        crop_sides_flat.view(T, N),
    )


@torch.no_grad()
def _precompute_trajectory_chunked(
    channeled_images: torch.Tensor,
    feature_extractor: DINOv3FeatureExtractor,
    batch_size: int,
    device: torch.device,
    num_workers: int = 4,
    sdf_workers: int = 32,
    target_size: int = 224,
    timestep_chunk_size: int = 400,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Memory-efficient variant of precompute_trajectory that processes timesteps in chunks.

    Used automatically when the full trajectory would exceed available RAM.
    Same interface and return values as precompute_trajectory.
    """
    T, N, H, W = channeled_images.shape

    cls_chunks, pm_chunks, cs_chunks = [], [], []

    for t_start in tqdm(range(0, T, timestep_chunk_size), desc="  Timestep chunks", leave=False):
        t_end = min(t_start + timestep_chunk_size, T)
        chunk = channeled_images[t_start:t_end]  # (chunk_T, N, H, W)
        chunk_T = chunk.shape[0]

        # Flatten chunk to (chunk_T*N, H, W) and convert to SDF
        flat_masks = chunk.reshape(chunk_T * N, H, W).cpu()
        sdf_masks, crop_sides_flat = _compute_sdfs_multiprocess(
            flat_masks, target_size=target_size, num_workers=sdf_workers,
        )
        del flat_masks

        # DINO inference on this chunk
        ds = _SimMaskDataset(sdf_masks)
        loader = DataLoader(
            ds, batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, shuffle=False,
        )
        prefetcher = CUDAPrefetcher(loader, device)

        cls_list, pm_list = [], []
        for (rgb,) in tqdm(prefetcher, desc="  DINO inference", total=len(loader), leave=False):
            cls_token, patch_tokens = feature_extractor(rgb)
            cls_list.append(cls_token.cpu())
            pm_list.append(patch_tokens.mean(dim=1).cpu())

        del sdf_masks, ds, loader, prefetcher

        cls_chunk = torch.cat(cls_list, dim=0)  # (chunk_T*N, D)
        pm_chunk = torch.cat(pm_list, dim=0)
        D = cls_chunk.shape[-1]

        cls_chunks.append(cls_chunk.view(chunk_T, N, D))
        pm_chunks.append(pm_chunk.view(chunk_T, N, D))
        cs_chunks.append(crop_sides_flat.view(chunk_T, N))

    return (
        torch.cat(cls_chunks, dim=0),   # (T, N, D)
        torch.cat(pm_chunks, dim=0),    # (T, N, D)
        torch.cat(cs_chunks, dim=0),    # (T, N)
    )


# ---------------------------------------------------------------------------
# Cache path helpers
# ---------------------------------------------------------------------------

def _get_dino_cache_path(cache_dir: str, cache_id: str, traj_idx: int, model_key: str) -> str:
    """Get the file path for a companion DINO embedding file.

    Returns:
        Path like {cache_dir}/{cache_id}_{traj_idx:04d}_dino_{model_key}.pt
    """
    filename = f"{cache_id}_{traj_idx:04d}_dino_{model_key}.pt"
    return os.path.join(cache_dir, filename)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def precompute_trajectory_dino_embeddings(
    cache_dir: str,
    dino_model_key: str,
    batch_size: int = 64,
    device: torch.device | str | None = None,
    num_workers: int = 4,
    sdf_workers: int = 32,
) -> None:
    """Precompute and cache DINO embeddings for all trajectories in a cache directory.

    Called once before training. Discovers trajectory files via metadata.yaml,
    checks if all companion DINO files already exist (skip if complete),
    otherwise loads DINOv3, processes each trajectory, and saves companion files.

    Each companion file contains:
        {
            "cls_tokens":  (T, N, D) float32,
            "patch_means": (T, N, D) float32,
            "crop_sides":  (T, N) int32  — square crop side length in original image pixels
        }

    Args:
        cache_dir: Directory containing metadata.yaml and trajectory .pt files
        dino_model_key: DINO model key (e.g. "SP", "B", "S", "L", "H", "F")
        batch_size: Batch size for DINO inference
        device: Device for DINO inference (None for auto-detect)
        num_workers: DataLoader workers for DINO inference
        sdf_workers: CPU workers for SDF conversion (cv2/scipy)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load metadata
    metadata_path = os.path.join(cache_dir, "metadata.yaml")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No metadata.yaml found in '{cache_dir}'")

    metadata = OmegaConf.load(metadata_path)
    n_trajectories = metadata.data.generator.n_trajectories

    # Compute cache_id from data config
    from graph_dyno.data.trajectory_generator import generate_config_hash
    cache_id = generate_config_hash(metadata.data)

    # Check if ALL companion files already exist
    all_exist = True
    for traj_idx in range(n_trajectories):
        dino_path = _get_dino_cache_path(cache_dir, cache_id, traj_idx, dino_model_key)
        if not os.path.exists(dino_path):
            all_exist = False
            break

    if all_exist:
        print(f"DINO embeddings already cached for all {n_trajectories} trajectories "
              f"(model={dino_model_key}). Skipping.")
        return

    # Load DINO feature extractor
    print(f"Loading DINOv3 feature extractor (model={dino_model_key})...")
    model_type = ModelType[dino_model_key] if dino_model_key in ModelType.__members__ else dino_model_key
    feature_extractor = DINOv3FeatureExtractor(model_type).to(device)

    print(f"Precomputing DINO embeddings for {n_trajectories} trajectories...")

    for traj_idx in tqdm(range(n_trajectories), desc="Trajectories"):
        dino_path = _get_dino_cache_path(cache_dir, cache_id, traj_idx, dino_model_key)

        # Skip if this trajectory's companion file already exists
        if os.path.exists(dino_path):
            continue

        # Load trajectory (mmap to avoid pulling full tensor into RAM)
        traj_filename = f"{cache_id}_{traj_idx:04d}.pt"
        traj_path = os.path.join(cache_dir, traj_filename)
        trajectory_data = torch.load(traj_path, map_location="cpu", mmap=True)
        channeled_images = trajectory_data["channeled_images"]  # (T, N, H, W)

        # Estimate peak memory: masks (T*N*H*W) + SDF output (T*N*224*224*2)
        T, N, H, W = channeled_images.shape
        est_bytes = T * N * H * W * channeled_images.element_size() \
                  + T * N * 224 * 224 * 2  # SDF float16
        avail_bytes = psutil.virtual_memory().available

        if est_bytes > avail_bytes * 0.5:
            print(f"  Large trajectory ({est_bytes / 1e9:.1f} GB est.), "
                  f"using chunked processing ({avail_bytes / 1e9:.1f} GB available)")
            cls_tokens, patch_means, crop_sides = _precompute_trajectory_chunked(
                channeled_images, feature_extractor, batch_size, device,
                num_workers=num_workers, sdf_workers=sdf_workers,
            )
        else:
            cls_tokens, patch_means, crop_sides = precompute_trajectory(
                channeled_images, feature_extractor, batch_size, device,
                num_workers=num_workers, sdf_workers=sdf_workers,
            )

        # Save companion file
        torch.save({
            "cls_tokens": cls_tokens,
            "patch_means": patch_means,
            "crop_sides": crop_sides,
        }, dino_path)

        # Free memory
        del trajectory_data, channeled_images, cls_tokens, patch_means, crop_sides

    print(f"Done. DINO embeddings cached for {n_trajectories} trajectories.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute DINO embeddings for simulation trajectory data"
    )
    parser.add_argument(
        "--cache_dir", type=str, required=True,
        help="Directory containing metadata.yaml and trajectory .pt files",
    )
    parser.add_argument(
        "--dino_model", type=str, default="SP",
        help="DINO model key (S, SP, B, L, H, F). Default: SP",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for DINO inference. Default: 64",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for inference (e.g. 'cuda', 'cpu'). Default: auto-detect",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader workers for DINO inference. Default: 4",
    )
    parser.add_argument(
        "--sdf_workers", type=int, default=8,
        help="CPU workers for SDF conversion. Default: 8",
    )

    args = parser.parse_args()

    precompute_trajectory_dino_embeddings(
        cache_dir=args.cache_dir,
        dino_model_key=args.dino_model,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        sdf_workers=args.sdf_workers,
    )
