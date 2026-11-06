"""Per-frame geometry + contact graph for CTC instance masks.

**Criterion.** Dilate each cell's mask by `contact_px` and draw an edge to every
*other* mask it overlaps. Equivalently: two cells are neighbours iff the gap
between their pixel sets is <= `contact_px` (a negative gap = overlap). There is
**no occlusion** — a third cell lying between two others does not block their
edge. This is a pure proximity test.

Two implementations of that one criterion are provided and are exactly equal:

* :func:`frame_adjacency_edt` — per-label Euclidean distance transform inside a
  padded bounding box. Cost is O(sum of bbox areas), *independent of the
  radius*, because the EDT does the distance work instead of a structuring
  element. This is the production path for large radii.
* :func:`frame_adjacency_shift` — fully vectorized shifted array comparisons,
  one per offset in the disk of radius r. O(r^2 * H * W), so it wins only for
  small radii, but it needs no loop over labels. Doubles as the test oracle.

`skimage.segmentation.expand_labels` is deliberately *not* used: it grows all
labels into the background as a Voronoi partition, so its criterion is
"gap <= 2*distance **with** occlusion" — a Delaunay-style contact graph, which
answers a different question.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from graph_dyno.real_data.ctc import load_seg

#: Below this radius the shifted-comparison form is cheaper than the EDT form.
SHIFT_FORM_MAX_RADIUS = 5


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------

def half_disk_offsets(r: float) -> list[tuple[int, int]]:
    """Offsets (dy, dx) with 0 < dy^2 + dx^2 <= r^2, one of each +/- pair.

    Half the disk suffices: offset (dy, dx) and (-dy, -dx) produce the same
    unordered label pair.
    """
    R = int(np.floor(r))
    out = []
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            if dy == 0 and dx == 0:
                continue
            if dy * dy + dx * dx > r * r:
                continue
            if (dy, dx) < (0, 0):  # keep exactly one of each antipodal pair
                continue
            out.append((dy, dx))
    return out


def _pairs_from_flat(a: np.ndarray, b: np.ndarray, base: int) -> np.ndarray:
    """Unordered unique (lo, hi) pairs from two equal-length int64 label arrays."""
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    keys = np.unique(lo * base + hi)
    if keys.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([keys // base, keys % base], axis=1)


def frame_adjacency_shift(lbl: np.ndarray, contact_px: float) -> np.ndarray:
    """Contact pairs by shifted comparisons. Exact; label-count independent."""
    lbl = np.asarray(lbl)
    H, W = lbl.shape
    # int64 BEFORE forming the dedupe key: the SEG TIFFs are uint16 and
    # base = max_label + 1 (22216 for PhC-C2DL-PSC) overflows a uint16 multiply.
    base = int(lbl.max()) + 1 if lbl.size else 1
    if base <= 1:
        return np.zeros((0, 2), dtype=np.int64)

    a_parts, b_parts = [], []
    for dy, dx in half_disk_offsets(contact_px):
        ys0, ys1 = max(0, dy), H + min(0, dy)
        xs0, xs1 = max(0, dx), W + min(0, dx)
        if ys0 >= ys1 or xs0 >= xs1:
            continue
        a = lbl[ys0:ys1, xs0:xs1]
        b = lbl[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx]
        m = (a != b) & (a > 0) & (b > 0)
        if m.any():
            a_parts.append(a[m].astype(np.int64))
            b_parts.append(b[m].astype(np.int64))

    if not a_parts:
        return np.zeros((0, 2), dtype=np.int64)
    return _pairs_from_flat(
        np.concatenate(a_parts), np.concatenate(b_parts), base)


def frame_adjacency_edt(lbl: np.ndarray, contact_px: float) -> np.ndarray:
    """Contact pairs by per-label EDT inside a padded bbox. Radius-independent."""
    lbl = np.asarray(lbl)
    labs = np.unique(lbl)
    labs = labs[labs > 0]
    if labs.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    objs = ndimage.find_objects(lbl.astype(np.int64))
    H, W = lbl.shape
    R = int(np.ceil(contact_px))
    out: set[tuple[int, int]] = set()

    for raw_label in labs:
        lab = int(raw_label)
        sl = objs[lab - 1]
        if sl is None:
            continue
        ys, xs = sl
        y0, y1 = max(0, ys.start - R), min(H, ys.stop + R)
        x0, x1 = max(0, xs.start - R), min(W, xs.stop + R)
        sub = lbl[y0:y1, x0:x1]
        near = ndimage.distance_transform_edt(sub != lab) <= contact_px
        for raw_other in np.unique(sub[near]):
            other = int(raw_other)
            if other and other != lab:
                out.add((min(lab, other), max(lab, other)))

    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(sorted(out), dtype=np.int64).reshape(-1, 2)


def frame_adjacency(lbl: np.ndarray, contact_px: float) -> np.ndarray:
    """Contact pairs, picking whichever exact form is cheaper for this radius."""
    if contact_px <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    if contact_px <= SHIFT_FORM_MAX_RADIUS:
        return frame_adjacency_shift(lbl, contact_px)
    return frame_adjacency_edt(lbl, contact_px)


# ---------------------------------------------------------------------------
# Per-frame geometry
# ---------------------------------------------------------------------------

@dataclass
class FrameScan:
    """Everything one SEG frame contributes, from a single read of the TIFF."""

    labels: np.ndarray          # (n,) int64, sorted
    centroids: np.ndarray       # (n, 2) float64, (row, col) centre of mass
    areas: np.ndarray           # (n,) int64, foreground pixel count
    bbox_sides: np.ndarray      # (n,) int64, max(bbox_h, bbox_w)
    aspect_ratios: np.ndarray   # (n,) float64, long side / short side (>= 1)
    half_extents: np.ndarray    # (n,) float64, max(|row-cy|, |col-cx|)
    touches_border: np.ndarray  # (n,) bool
    pairs: np.ndarray           # (m, 2) int64, unordered label pairs

    @staticmethod
    def empty() -> "FrameScan":
        return FrameScan(
            labels=np.zeros(0, dtype=np.int64),
            centroids=np.zeros((0, 2)),
            areas=np.zeros(0, dtype=np.int64),
            bbox_sides=np.zeros(0, dtype=np.int64),
            aspect_ratios=np.zeros(0),
            half_extents=np.zeros(0),
            touches_border=np.zeros(0, dtype=bool),
            pairs=np.zeros((0, 2), dtype=np.int64),
        )


def frame_geometry(lbl: np.ndarray) -> FrameScan:
    """Per-label geometry from one pass over the foreground pixels."""
    lbl = np.asarray(lbl)
    H, W = lbl.shape
    ys, xs = np.nonzero(lbl)
    if ys.size == 0:
        return FrameScan.empty()

    vals = lbl[ys, xs].astype(np.int64)
    order = np.argsort(vals, kind="stable")
    vals, ys, xs = vals[order], ys[order].astype(np.int64), xs[order].astype(np.int64)

    labels, starts, counts = np.unique(vals, return_index=True, return_counts=True)
    cy = np.add.reduceat(ys.astype(np.float64), starts) / counts
    cx = np.add.reduceat(xs.astype(np.float64), starts) / counts

    y0 = np.minimum.reduceat(ys, starts)
    y1 = np.maximum.reduceat(ys, starts)
    x0 = np.minimum.reduceat(xs, starts)
    x1 = np.maximum.reduceat(xs, starts)
    bbox_h = (y1 - y0 + 1).astype(np.int64)
    bbox_w = (x1 - x0 + 1).astype(np.int64)

    ext = np.maximum(np.abs(ys - np.repeat(cy, counts)),
                     np.abs(xs - np.repeat(cx, counts)))
    half_extents = np.maximum.reduceat(ext, starts)

    long_side = np.maximum(bbox_h, bbox_w).astype(np.float64)
    short_side = np.maximum(np.minimum(bbox_h, bbox_w), 1).astype(np.float64)

    return FrameScan(
        labels=labels,
        centroids=np.stack([cy, cx], axis=1),
        areas=counts.astype(np.int64),
        bbox_sides=np.maximum(bbox_h, bbox_w),
        aspect_ratios=long_side / short_side,
        half_extents=half_extents,
        touches_border=(y0 == 0) | (x0 == 0) | (y1 == H - 1) | (x1 == W - 1),
        pairs=np.zeros((0, 2), dtype=np.int64),
    )


def scan_frame(seg_path, contact_px: float) -> FrameScan:
    """Read one SEG frame and return geometry + adjacency. `None` -> empty."""
    if seg_path is None:
        return FrameScan.empty()
    # np.asarray inside the worker: a PIL Image holds an unpicklable file handle.
    lbl = np.asarray(load_seg(seg_path))
    scan = frame_geometry(lbl)
    scan.pairs = frame_adjacency(lbl, contact_px)
    return scan


def _scan_frame_worker(args):
    """Top-level picklable worker for ProcessPoolExecutor.

    Takes a *path*, not an array — a 1010x1010 uint16 frame is 2 MB and
    pickling it in would dominate the real work.
    """
    seg_path, contact_px = args
    return scan_frame(seg_path, contact_px)


def scan_video(seg_paths, contact_px: float, workers: int = 32,
               desc: str | None = None, show_progress: bool = True
               ) -> list[FrameScan]:
    """Scan every frame of one video. One TIFF read per frame.

    Serial when `workers <= 1`; otherwise the house idiom
    (`ProcessPoolExecutor` + top-level worker + `map(chunksize=...)`). Frames
    cost 20-95 ms each, so chunksize stays small — the SDF path's 64 would
    starve workers on an 84-frame video.
    """
    items = [(p, contact_px) for p in seg_paths]
    n = len(items)

    if workers is None or workers <= 1 or n <= 1:
        iterator = map(_scan_frame_worker, items)
        return list(_maybe_tqdm(iterator, n, desc, show_progress))

    workers = min(int(workers), max(1, os.cpu_count() or 1), n)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        iterator = pool.map(_scan_frame_worker, items, chunksize=4)
        return list(_maybe_tqdm(iterator, n, desc, show_progress))


def _maybe_tqdm(iterator, total, desc, show_progress):
    if not show_progress or desc is None:
        return iterator
    try:
        from tqdm import tqdm
        return tqdm(iterator, total=total, desc=desc, leave=False)
    except ImportError:
        return iterator


# ---------------------------------------------------------------------------
# Contact radius
# ---------------------------------------------------------------------------

def estimate_median_bbox_side(seg_paths, n_sample: int = 12) -> float:
    """Median cell bbox side over a deterministic sample of frames.

    The contact radius must be *size-relative* — a couple of fixed pixels
    yields an essentially empty graph on all but the most confluent videos, and
    a fixed pixel radius would make graph density a per-video artifact of cell
    size, i.e. a trivial fingerprint that would fake the separation result.

    A median is robust to subsampling (unlike r_max, which is a max and so must
    be scanned exhaustively), so a handful of evenly spaced frames suffices.
    Sampling is deterministic, keeping the derived `contact_px` reproducible.
    """
    present = [(i, p) for i, p in enumerate(seg_paths) if p is not None]
    if not present:
        raise ValueError("no SEG frames available to estimate cell size")
    idx = np.unique(np.linspace(0, len(present) - 1, num=min(n_sample, len(present))
                                ).round().astype(int))
    sides = []
    for i in idx:
        scan = frame_geometry(np.asarray(load_seg(present[int(i)][1])))
        if scan.bbox_sides.size:
            sides.append(scan.bbox_sides)
    if not sides:
        raise ValueError("no cells found while estimating cell size")
    return float(np.median(np.concatenate(sides)))


def resolve_contact_px(seg_paths, contact_frac: float,
                       contact_px: float | None = None,
                       n_sample: int = 12) -> tuple[int, float]:
    """Resolve the per-video contact radius in pixels.

    Returns:
        (contact_px, median_bbox_side). An explicit `contact_px` overrides
        `contact_frac` but the median is still reported for diagnostics.
    """
    median_side = estimate_median_bbox_side(seg_paths, n_sample=n_sample)
    if contact_px is not None:
        return int(round(float(contact_px))), median_side
    return int(round(float(contact_frac) * median_side)), median_side
