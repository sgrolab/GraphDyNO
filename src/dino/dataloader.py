"""
Prefetching DataLoader utilities for DINO cell mask training.

CachedEmbeddingDataset: serves precomputed (cls_token, mask) from CPU tensors.
CUDAPrefetcher: overlaps host->device transfers with compute via a CUDA stream.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedEmbeddingDataset(Dataset):
    """Dataset backed by precomputed embeddings and SDF masks stored on CPU.

    Args:
        cls_tokens:  (N, D) float32 tensor on CPU
        masks:       (N, H, W) float16 SDF tensor on CPU
        patch_means: (N, D) float32 tensor on CPU, optional
    """

    def __init__(self, cls_tokens, masks, patch_means=None):
        self.cls_tokens = cls_tokens
        # Keep float16 and cast per item: an up-front .float() on the LIVECell
        # train split materializes a 120 GB fp32 copy of the 60 GB fp16 tensor.
        # The per-item cast runs in the DataLoader worker processes instead.
        self.masks = masks
        self.patch_means = patch_means

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        emb = self.cls_tokens[idx]
        if self.patch_means is not None:
            emb = torch.cat([emb, self.patch_means[idx]])
        return emb, self.masks[idx].float()


from graph_dyno.data.dataloader import CUDAPrefetcher  # noqa: F401


class SDFDataset(Dataset):
    """Returns (sdf, sdf) pairs for autoencoder training on SDF images.

    The input and target are identical — the autoencoder learns to reconstruct
    the SDF through a bottleneck.
    """

    def __init__(self, sdfs):
        """
        Args:
            sdfs: (N, H, W) float16 SDF tensor on CPU in [-1, 1]
        """
        self.sdfs = sdfs.float()  # promote float16 -> float32

    def __len__(self):
        return len(self.sdfs)

    def __getitem__(self, idx):
        return self.sdfs[idx], self.sdfs[idx]


def load_all_masks(split_dir):
    """Load all shard .pt files from a split directory into a single CPU tensor.

    Returns:
        masks: (N, H, W) float16 SDF tensor on CPU
    """
    split_dir = Path(split_dir)
    with open(split_dir / "meta.json") as f:
        meta = json.load(f)
    shards = []
    for i in range(meta["num_shards"]):
        shards.append(torch.load(split_dir / f"shard_{i:04d}.pt", weights_only=True))
    masks = torch.cat(shards, dim=0)[: meta["num_masks"]]
    print(f"  Loaded {len(masks)} masks from {split_dir}")
    return masks
