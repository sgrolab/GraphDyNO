"""Parallelized DINO embedding computation over the LIVECell mask shards.

A Hydra app over ``configs/dino_config.yaml`` — the same config
``dino/train_dino.py`` and ``dino/precompute_embeddings.py`` read — so
``dino/model=cell`` supplies backend, model key, repo dir and weights path with
no duplication, and the whole thing can be fanned out with ``--multirun``.

Work is addressed by a single flat ``+part`` index over
``(split, shard, sub_chunk)``, enumerated from each split's ``meta.json``.

Usage:
    # print the part table without touching a GPU
    python -m dino.compute_embeddings_worker dino/model=cell +sub_chunks=4 +list_parts=true

    # compute one part
    python -m dino.compute_embeddings_worker dino/model=cell +sub_chunks=4 +part=0

    # compute every part, one after another
    python -m dino.compute_embeddings_worker --multirun \
        dino/model=cell +sub_chunks=4 '+part=range(0,16)'

    # merge the parts back into {split}_cls.pt / {split}_patch_mean.pt
    python -m dino.compute_embeddings_worker dino/model=cell +merge=true +split=train
"""

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from dino.feature_extractor import create_feature_extractor
from dino.precompute_embeddings import precompute

# Only train/val are ever consumed: train_dino.py loads exactly those two, and
# the existing features_B/ directory contains only train_*/val_* files.
DEFAULT_SPLITS = ["train", "val"]


def _part_suffix(shard_idx, sub_chunks, sub_chunk_idx):
    """Build the file suffix for a compute output part.

    Keep `sub_chunks` uniform across a split: cmd_merge concatenates in
    sorted() filename order, and mixing "shard0000.pt" with
    "shard0000_sub00.pt" silently misorders the result.
    """
    if sub_chunks <= 1:
        return f"shard{shard_idx:04d}"
    return f"shard{shard_idx:04d}_sub{sub_chunk_idx:02d}"


def _enumerate_parts(data_dir, splits, sub_chunks):
    """Flat (split, shard_idx, sub_chunks, sub_chunk_idx) list, meta.json-driven.

    NB: enumerate from meta["num_shards"], never a glob. train/ contains a stale
    shard_0003.pt that meta.json does not cover (num_shards=3, and shards 0-2
    sum to exactly num_masks). A glob-based enumeration would silently corrupt
    the merge.
    """
    parts = []
    for split in splits:
        meta_path = Path(data_dir) / split / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        for shard in range(meta["num_shards"]):
            for k in range(sub_chunks):
                parts.append((split, shard, sub_chunks, k))
    return parts


def _build_feature_extractor(model_cfg, device):
    model_key = model_cfg.dino_model
    backend = model_cfg.get("dino_backend", "hf")
    print(f"Loading feature extractor (backend={backend}, model={model_key})...")
    return create_feature_extractor(
        model_key,
        backend=backend,
        repo_dir=model_cfg.get("dino_repo_dir", None),
        weights_path=model_cfg.get("dino_weights_path", None),
        weights_url=model_cfg.get("dino_weights_url", None),
        trust_weights=model_cfg.get("dino_trust_weights", False),
    ).to(device)


def cmd_compute(cfg, data_dir, embedding_dir, split, shard_idx, sub_chunks, sub_chunk_idx):
    """Process a single shard (or sub-chunk of a shard) on one GPU."""
    embedding_dir.mkdir(parents=True, exist_ok=True)

    suffix = _part_suffix(shard_idx, sub_chunks, sub_chunk_idx)
    cls_path = embedding_dir / f"{split}_cls_{suffix}.pt"
    pm_path = embedding_dir / f"{split}_patch_mean_{suffix}.pt"

    # Idempotent: skip if outputs already exist
    if cls_path.exists() and pm_path.exists():
        print(f"Part {split}/{suffix} already computed, skipping.")
        return

    shard_file = data_dir / split / f"shard_{shard_idx:04d}.pt"
    if not shard_file.exists():
        raise FileNotFoundError(f"Shard file not found: {shard_file}")
    print(f"Loading shard {shard_idx} from {shard_file} ...")
    # mmap so a sub-chunk reads only its own slice off NFS, not the whole shard.
    masks = torch.load(shard_file, weights_only=True, mmap=True)
    print(f"  Shard shape: {tuple(masks.shape)}")

    if sub_chunks > 1:
        n = len(masks)
        chunk_size = (n + sub_chunks - 1) // sub_chunks  # ceil division
        start = sub_chunk_idx * chunk_size
        end = min(start + chunk_size, n)
        masks = masks[start:end]
        print(f"  Sub-chunk {sub_chunk_idx}/{sub_chunks}: "
              f"masks[{start}:{end}] ({len(masks)} masks)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    feature_extractor = _build_feature_extractor(cfg.dino.model, device)

    training_cfg = cfg.dino.training
    cls_tokens, patch_means = precompute(
        masks, feature_extractor,
        training_cfg.batch_size, device, training_cfg.num_workers,
    )

    torch.save(cls_tokens, cls_path)
    torch.save(patch_means, pm_path)
    print(f"Saved {split}/{suffix}: "
          f"cls {tuple(cls_tokens.shape)}, patch_mean {tuple(patch_means.shape)}")


def cmd_merge(data_dir, embedding_dir, split):
    """Concatenate partial shard/sub-chunk results into final embedding files."""
    with open(data_dir / split / "meta.json") as f:
        meta = json.load(f)
    num_masks = meta["num_masks"]

    # Naming: {split}_cls_shard0000.pt or {split}_cls_shard0000_sub00.pt
    cls_parts = sorted(embedding_dir.glob(f"{split}_cls_shard*.pt"))
    pm_parts = sorted(embedding_dir.glob(f"{split}_patch_mean_shard*.pt"))

    if not cls_parts:
        raise FileNotFoundError(f"No shard files found in {embedding_dir} for {split}")
    if len(cls_parts) != len(pm_parts):
        raise ValueError(
            f"Mismatched part counts: {len(cls_parts)} cls vs {len(pm_parts)} patch_mean")

    print(f"Merging {len(cls_parts)} parts for {split} (num_masks={num_masks})")

    cls_shards = [torch.load(p, weights_only=True) for p in cls_parts]
    pm_shards = [torch.load(p, weights_only=True) for p in pm_parts]

    cls_all = torch.cat(cls_shards, dim=0)[:num_masks]
    pm_all = torch.cat(pm_shards, dim=0)[:num_masks]
    print(f"  Merged cls: {tuple(cls_all.shape)}, patch_mean: {tuple(pm_all.shape)}")

    final_cls = embedding_dir / f"{split}_cls.pt"
    final_pm = embedding_dir / f"{split}_patch_mean.pt"
    torch.save(cls_all, final_cls)
    torch.save(pm_all, final_pm)
    print(f"  Saved {final_cls}")
    print(f"  Saved {final_pm}")

    for p in cls_parts + pm_parts:
        p.unlink()
    print(f"  Cleaned up {len(cls_parts) + len(pm_parts)} part files.")


@hydra.main(version_base=None, config_path="../../configs", config_name="dino_config")
def main(cfg: DictConfig) -> None:
    data_dir = Path(cfg.dino.data.data_dir)
    model_key = cfg.dino.model.dino_model
    # Safe to key the cache on model_key alone: preprocessing is frozen per key.
    embedding_dir = data_dir / f"features_{model_key}"

    splits = cfg.get("splits", None)
    splits = list(splits) if splits is not None else list(DEFAULT_SPLITS)
    sub_chunks = int(cfg.get("sub_chunks", 1))

    if cfg.get("merge", False):
        split = cfg.get("split", None)
        merge_splits = [split] if split is not None else splits
        for s in merge_splits:
            cmd_merge(data_dir, embedding_dir, s)
        return

    parts = _enumerate_parts(data_dir, splits, sub_chunks)

    if cfg.get("list_parts", False):
        print(f"{len(parts)} parts (splits={splits}, sub_chunks={sub_chunks}):")
        for i, (split, shard, sc, k) in enumerate(parts):
            print(f"  part={i:3d}  split={split:5s}  shard={shard}  sub_chunk={k}/{sc}")
        return

    part = cfg.get("part", None)
    if part is None:
        raise ValueError(
            "Specify a part to compute: +part=<int> (see +list_parts=true), "
            "or +merge=true to merge.")
    part = int(part)
    if not 0 <= part < len(parts):
        raise IndexError(f"+part={part} out of range: {len(parts)} parts enumerated")

    split, shard_idx, sc, sub_chunk_idx = parts[part]
    print(f"part={part} -> split={split} shard={shard_idx} "
          f"sub_chunk={sub_chunk_idx}/{sc}")
    cmd_compute(cfg, data_dir, embedding_dir, split, shard_idx, sc, sub_chunk_idx)


if __name__ == "__main__":
    main()
