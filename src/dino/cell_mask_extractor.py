"""
Cell Mask Extractor
===================
Downloads LIVECell dataset and extracts individual cell crops from the
source images, saved as sharded PyTorch tensors for efficient DataLoader usage.

For each annotated cell the tight bounding box is found via the segmentation
mask, the corresponding region is cropped from the original image, and the SDF
(signed distance function) is computed and resized to (target_size x target_size).

Each shard is a .pt file containing a float16 tensor of shape (N, H, W).
Use CellMaskDataset to load them with a PyTorch DataLoader.

Usage:
    python -m dino.cell_mask_extractor
    python -m dino.cell_mask_extractor data.data_dir=./cell_masks data.raw_dir=./livecell_raw
"""

import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as mask_util
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

# ---------------------------------------------------------------------------
# LIVECell URLs
# ---------------------------------------------------------------------------
BASE_URL = "http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021/"

FILES = {
    "images": "images.zip",
    "train": "annotations/LIVECell/livecell_coco_train.json",
    "val": "annotations/LIVECell/livecell_coco_val.json",
    "test": "annotations/LIVECell/livecell_coco_test.json",
}


def _reporthook(pbar):
    """Return a urllib reporthook that updates a tqdm bar."""
    def hook(block_num, block_size, total_size):
        if pbar.total is None and total_size > 0:
            pbar.total = total_size
            pbar.refresh()
        pbar.update(block_size)
    return hook


def download(url, dest):
    if dest.exists():
        print(f"  [exists] {dest.name}")
        return
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024,
              desc=f"  {dest.name}") as pbar:
        urllib.request.urlretrieve(url, str(dest), reporthook=_reporthook(pbar))


def download_livecell(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for key, fname in FILES.items():
        dest = raw_dir / Path(fname).name
        download(BASE_URL + fname, dest)
        if fname.endswith(".zip"):
            print(f"  Extracting {fname} ...")
            with zipfile.ZipFile(dest, "r") as zf:
                zf.extractall(raw_dir)


def _flush_shard(shard, fill, shard_idx, output_dir):
    """Save a pre-allocated numpy shard (truncated to fill) as a .pt file."""
    tensor = torch.from_numpy(shard[:fill].copy())  # (fill, H, W) float16
    path = output_dir / f"shard_{shard_idx:04d}.pt"
    torch.save(tensor, path)


def _ann_to_sdf(rle, bbox, img_hw, min_area, target_size):
    """Decode one annotation RLE, crop to smallest enclosing square, and compute SDF.

    Returns None if skipped. Object is centered in the square crop to preserve
    aspect ratio when resizing to (target_size, target_size).
    """
    H, W = img_hw
    bx, by, bw, bh = [int(v) for v in bbox]
    x0, y0 = max(bx, 0), max(by, 0)
    x1, y1 = min(bx + bw, W), min(by + bh, H)
    if x1 <= x0 or y1 <= y0:
        return None

    mask = mask_util.decode(rle)  # full H x W uint8

    # Check area before expanding to square
    rect_crop = mask[y0:y1, x0:x1].astype(np.float32)
    if rect_crop.sum() < min_area:
        return None

    # Expand to square — center the object, clamp to image boundaries
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, min(cy - side // 2, H - side))
    x0 = max(0, min(cx - side // 2, W - side))

    if side <= H and side <= W:
        crop = mask[y0:y0 + side, x0:x0 + side].astype(np.float32)
    else:
        crop = np.zeros((side, side), dtype=np.float32)
        copy_h = min(side, H - y0)
        copy_w = min(side, W - x0)
        crop[:copy_h, :copy_w] = mask[y0:y0 + copy_h, x0:x0 + copy_w]

    resized = cv2.resize(crop, (target_size, target_size),
                         interpolation=cv2.INTER_LINEAR)
    mask_bin = resized > 0.5
    dist_in = distance_transform_edt(mask_bin)
    dist_out = distance_transform_edt(~mask_bin)
    sdf = dist_in - dist_out
    return np.clip(sdf / (target_size / 2), -1, 1).astype(np.float16)


def extract_masks(images_dir, coco_path, output_dir,
                  min_area=200, target_size=224, shard_size=200_000,
                  num_workers=8):
    coco = COCO(str(coco_path))
    img_ids = coco.getImgIds()
    print(f"  {len(img_ids)} images in {coco_path.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    shard = np.zeros((shard_size, target_size, target_size), dtype=np.float16)
    fill = 0
    shard_idx = 0
    count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for img_id in tqdm(img_ids, desc=f"  {coco_path.name}", unit="img"):
            info = coco.loadImgs(img_id)[0]
            img_hw = (info["height"], info["width"])

            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            jobs = []
            for ann in anns:
                if ann.get("area", 0) < min_area:
                    continue
                rle = coco.annToRLE(ann)
                jobs.append(pool.submit(
                    _ann_to_sdf, rle, ann["bbox"], img_hw,
                    min_area, target_size))

            for fut in jobs:
                result = fut.result()
                if result is None:
                    continue
                shard[fill] = result
                fill += 1
                count += 1

                if fill >= shard_size:
                    _flush_shard(shard, fill, shard_idx, output_dir)
                    shard_idx += 1
                    fill = 0

    if fill > 0:
        _flush_shard(shard, fill, shard_idx, output_dir)
        shard_idx += 1

    meta = {"num_masks": count, "num_shards": shard_idx,
            "shard_size": shard_size, "target_size": target_size}
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f)

    print(f"  -> {count} masks in {shard_idx} shards")
    return count


class CellMaskDataset(Dataset):
    """PyTorch Dataset that reads sharded .pt SDF tensors.

    Usage:
        ds = CellMaskDataset("./cell_masks/train")
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
        for batch in loader:   # batch shape: (B, 1, H, W) float32 in [-1, 1]
            ...
    """

    def __init__(self, root, transform=None):
        self.root = Path(root)
        with open(self.root / "meta.json") as f:
            meta = json.load(f)
        self.num_masks = meta["num_masks"]
        self.shard_size = meta["shard_size"]
        self.num_shards = meta["num_shards"]
        self.transform = transform
        self._cache_idx = None
        self._cache_tensor = None

    def __len__(self):
        return self.num_masks

    def _load_shard(self, shard_idx):
        if self._cache_idx != shard_idx:
            path = self.root / f"shard_{shard_idx:04d}.pt"
            self._cache_tensor = torch.load(path, weights_only=True)
            self._cache_idx = shard_idx
        return self._cache_tensor

    def __getitem__(self, idx):
        shard_idx = idx // self.shard_size
        local_idx = idx % self.shard_size
        tensor = self._load_shard(shard_idx)
        mask = tensor[local_idx].unsqueeze(0).float()  # (1, H, W) float32
        if self.transform is not None:
            mask = self.transform(mask)
        return mask


@hydra.main(version_base=None, config_path="../../configs", config_name="dino_config")
def main(cfg: DictConfig) -> None:
    data = cfg.dino.data

    raw_dir = Path(data.raw_dir)
    output_dir = Path(data.data_dir)
    skip_download = data.get("skip_download", False)

    if not skip_download:
        print("Downloading LIVECell...")
        download_livecell(raw_dir)

    images_dir = raw_dir  # search recursively

    total = 0
    for split in ["train", "val", "test"]:
        ann_path = raw_dir / Path(FILES[split]).name
        if not ann_path.exists():
            continue
        print(f"\nProcessing {split}...")
        n = extract_masks(images_dir, ann_path, output_dir / split,
                          min_area=data.min_area, target_size=data.target_size,
                          shard_size=data.shard_size,
                          num_workers=data.num_extract_workers)
        total += n

    print(f"\nTotal: {total} masks in {output_dir}")


if __name__ == "__main__":
    main()
