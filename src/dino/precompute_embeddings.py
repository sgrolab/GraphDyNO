"""
Precompute DINO embeddings for all cell mask splits and save to disk.

Produces per split:
    {split}_cls.pt          (N, D) float32  - CLS token
    {split}_patch_mean.pt   (N, D) float32  - mean-pooled patch tokens

Usage:
    python -m dino.precompute_embeddings
    python -m dino.precompute_embeddings dino.model.dino_model=S
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from dino.feature_extractor import InputSpec, create_feature_extractor
from dino.dataloader import load_all_masks, CUDAPrefetcher


class _RGBMaskDataset(TensorDataset):
    """Wraps SDF masks -> (B, C, H, W) float [0,1] for the prefetcher."""

    def __init__(self, masks, input_spec: InputSpec | None = None):
        self.masks = masks  # (N, H, W) float16 SDF in [-1, 1]
        self.input_spec = input_spec if input_spec is not None else InputSpec(3, "replicate")

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        return (self.input_spec(self.masks[idx]),)


import os
@torch.no_grad()
def precompute(masks, feature_extractor, batch_size, device, num_workers=4):
    """Run frozen DINO on all masks and return CLS + mean-pooled patches on CPU.

    Args:
        masks: (N, H, W) float16 tensor on CPU
    Returns:
        cls_tokens:  (N, D) float32 on CPU
        patch_means: (N, D) float32 on CPU
    """
    from huggingface_hub import login
    if "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])
    ds = _RGBMaskDataset(masks, feature_extractor.input_spec)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True)
    prefetcher = CUDAPrefetcher(loader, device)

    cls_list, pm_list = [], []
    for (rgb,) in tqdm(prefetcher, desc="  Extracting", total=len(loader)):
        cls_token, patch_tokens = feature_extractor(rgb)
        cls_list.append(cls_token.cpu())
        pm_list.append(patch_tokens.mean(dim=1).cpu())

    return torch.cat(cls_list), torch.cat(pm_list)


def load_or_compute_split(split, masks, feature_extractor, embedding_dir,
                          batch_size, device, num_workers=4):
    """Load cached embeddings for a split, or compute and save them.

    Args:
        split: split name ("train", "val", "test")
        masks: (N, H, W) float16 SDF tensor on CPU
        feature_extractor: DINOv3FeatureExtractor on device
        embedding_dir: Path to directory for cached embeddings
        batch_size: batch size for DINO inference
        device: torch device
        num_workers: dataloader workers

    Returns:
        cls_tokens:  (N, D) float32 on CPU
        patch_means: (N, D) float32 on CPU
    """
    cls_path = embedding_dir / f"{split}_cls.pt"
    pm_path = embedding_dir / f"{split}_patch_mean.pt"

    if cls_path.exists() and pm_path.exists():
        print(f"  Loading cached embeddings for {split}...")
        cls_tokens = torch.load(cls_path, weights_only=True)
        patch_means = torch.load(pm_path, weights_only=True)
        print(f"  Loaded {cls_tokens.shape} from cache")
        return cls_tokens, patch_means

    print(f"  Computing embeddings for {split}...")
    cls_tokens, patch_means = precompute(
        masks, feature_extractor, batch_size, device, num_workers)

    embedding_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cls_tokens, cls_path)
    torch.save(patch_means, pm_path)
    print(f"  Saved {cls_tokens.shape} to {cls_path}")
    print(f"  Saved {patch_means.shape} to {pm_path}")

    return cls_tokens, patch_means


@hydra.main(version_base=None, config_path="../../configs", config_name="dino_config")
def main(cfg: DictConfig) -> None:
    model_cfg = cfg.dino.model
    data_cfg = cfg.dino.data
    training_cfg = cfg.dino.training

    data_dir = Path(data_cfg.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_key = model_cfg.dino_model
    backend = model_cfg.get("dino_backend", "hf")
    print(f"Loading feature extractor (backend={backend}, model={model_key})...")
    feature_extractor = create_feature_extractor(
        model_key,
        backend=backend,
        repo_dir=model_cfg.get("dino_repo_dir", None),
        weights_path=model_cfg.get("dino_weights_path", None),
        weights_url=model_cfg.get("dino_weights_url", None),
        trust_weights=model_cfg.get("dino_trust_weights", False),
    ).to(device)

    # Safe to key the cache on model_key alone: preprocessing is frozen per key.
    embedding_dir = data_dir / f"features_{model_key}"

    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        print(f"\n{split}:")
        masks = load_all_masks(split_dir)
        load_or_compute_split(
            split, masks, feature_extractor, embedding_dir,
            batch_size=training_cfg.batch_size,
            device=device,
            num_workers=training_cfg.num_workers,
        )


if __name__ == "__main__":
    main()
