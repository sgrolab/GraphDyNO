"""Object-centred rasterization of CTC instance masks.

Matches the polyhoop contract (`polyhoop_simulation._generate_image` /
`_rasterize_polygon`): a fixed grid `linspace(-r_max, +r_max, image_size)` per
cell, centred on that cell's centre of mass, nearest-neighbour resampled,
binary. Shape lives in the mask; position lives only in `states`.

polyhoop maps `verts[:, 0] -> row`, so position axis 0 is the image row axis
here too — `states[..., 0] = row`, `states[..., 1] = col`.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from graph_dyno.real_data.ctc import load_seg


def rasterize_frame(lbl: np.ndarray, labels: np.ndarray, centroids: np.ndarray,
                    r_max: float, image_size: int) -> np.ndarray:
    """Render each (label, centroid) onto its own `[-r_max, r_max]` grid.

    Args:
        lbl: (H, W) instance-label image.
        labels: (n,) label values to render.
        centroids: (n, 2) float (row, col) centres of mass, in pixels.
        r_max: half-width of the grid, in pixels.
        image_size: grid side S.

    Returns:
        (n, S, S) uint8 binary masks. Out-of-frame grid samples are 0.
    """
    lbl = np.asarray(lbl)
    H, W = lbl.shape
    n = len(labels)
    out = np.zeros((n, image_size, image_size), dtype=np.uint8)
    if n == 0:
        return out

    grid = np.linspace(-r_max, r_max, image_size)
    for i in range(n):
        cy, cx = float(centroids[i, 0]), float(centroids[i, 1])
        rows = np.rint(cy + grid).astype(np.int64)
        cols = np.rint(cx + grid).astype(np.int64)
        in_r = (rows >= 0) & (rows < H)
        in_c = (cols >= 0) & (cols < W)
        patch = lbl[np.ix_(np.clip(rows, 0, H - 1), np.clip(cols, 0, W - 1))]
        out[i] = ((patch == labels[i]) & in_r[:, None] & in_c[None, :]).astype(np.uint8)
    return out


def _raster_frame_worker(args):
    """Top-level picklable worker; opens the TIFF itself (paths, not arrays)."""
    seg_path, labels, centroids, r_max, image_size = args
    if seg_path is None or len(labels) == 0:
        return np.zeros((len(labels), image_size, image_size), dtype=np.uint8)
    lbl = np.asarray(load_seg(seg_path))
    return rasterize_frame(lbl, labels, centroids, r_max, image_size)


def rasterize_window(seg_paths, frame_labels, frame_centroids,
                     r_max: float, image_size: int,
                     workers: int = 32) -> list[np.ndarray]:
    """Rasterize one window's frames, one TIFF read per frame.

    Args:
        seg_paths: length-T list of mask paths (or None).
        frame_labels: length-T list of (n_t,) label arrays.
        frame_centroids: length-T list of (n_t, 2) centroid arrays.

    Returns:
        Length-T list of (n_t, S, S) uint8 arrays.
    """
    items = [
        (p, np.asarray(lab), np.asarray(cen, dtype=np.float64), r_max, image_size)
        for p, lab, cen in zip(seg_paths, frame_labels, frame_centroids)
    ]
    n = len(items)
    if workers is None or workers <= 1 or n <= 1:
        return [_raster_frame_worker(it) for it in items]

    workers = min(int(workers), max(1, os.cpu_count() or 1), n)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_raster_frame_worker, items, chunksize=4))
