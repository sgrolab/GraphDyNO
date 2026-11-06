"""Pixel-level observation-error operators. Pure numpy/scipy, worker-safe.

Everything here runs inside a `ProcessPoolExecutor`, so no torch, no CUDA and
no module-level state.

Canvas conventions (getting either wrong silently transposes every merge)
------------------------------------------------------------------------
`polyhoop_simulation._rasterize_polygon_from_snapshot` builds the per-cell
crop with

    scale = (image_size - 1) / (2 * r_max)
    center = (image_size - 1) / 2
    polygon2mask((S, S), [v[:, 0] * scale + center, v[:, 1] * scale + center])

and `skimage.draw.polygon2mask` maps the **first** vertex column to the **row**
axis. So:

    row axis = x,  col axis = y,  origin = canvas centre,  1 unit = `scale` px

and masks are object-centred (the centroid is subtracted before rasterising),
which is exactly why misses, ID swaps and position noise need no GPU work at
all: the stored mask does not depend on where the cell is.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, shift as ndi_shift


def scale_px(image_size: int, r_max: float) -> float:
    """Pixels per simulation unit on a per-cell crop."""
    return (image_size - 1) / (2.0 * r_max)


# ---------------------------------------------------------------------------
# Partial mask — the real shape workhorse
# ---------------------------------------------------------------------------

def half_plane_cut(mask, axis: float, area_frac: float, side: float = 0.0):
    """Keep `area_frac` of the ON pixels on one side of a line through the centroid.

    Exact, with no bisection: project the ON-pixel coordinates onto a random
    axis and take the k-th order statistic via `np.argpartition`. O(P), and
    P is only ~4500 for an 87 px cell.

    A half-plane cut of a (near-)convex polygon mask is a single connected
    component, which is what makes this a plausible under-segmentation rather
    than a speckle pattern.

    Args:
        mask: (H, W) binary.
        axis: cut-normal angle in radians.
        area_frac: fraction of ON pixels to keep, in (0, 1].
        side: 0 or 1 — which side of the line survives.

    Returns:
        (H, W) uint8 mask with exactly `round(area_frac * P)` ON pixels
        (clipped to at least 1), or an all-zero mask if the input was empty.
    """
    m = np.asarray(mask)
    rows, cols = np.nonzero(m > 0.5)
    out = np.zeros(m.shape, dtype=np.uint8)
    P = rows.size
    if P == 0:
        return out
    k = int(round(float(area_frac) * P))
    k = int(np.clip(k, 1, P))
    if k == P:
        out[rows, cols] = 1
        return out

    cr = rows.mean()
    cc = cols.mean()
    proj = (rows - cr) * np.cos(axis) + (cols - cc) * np.sin(axis)
    if side >= 0.5:
        proj = -proj
    keep = np.argpartition(proj, k - 1)[:k]
    out[rows[keep], cols[keep]] = 1
    return out


# ---------------------------------------------------------------------------
# Erode / dilate — a scale perturbation, not a shape one
# ---------------------------------------------------------------------------

def morph_mask(mask, delta_px: float):
    """Sub-pixel morphological offset via EDT thresholding.

    `delta_px > 0` dilates, `< 0` erodes, by a level set of the signed distance
    field — so fractional offsets are meaningful, which `binary_dilation` with
    a structuring element cannot express.

    The comparison is `>=` for dilation and `>` for erosion, and that asymmetry
    is deliberate. `sdf = edt(m) - edt(~m)` never takes a value in (-1, 1):
    inside pixels are at >= +1 and outside pixels at <= -1. A single `>` on both
    sides therefore makes `delta = +1` add nothing while `delta = -1` removes a
    whole ring, so an episode that sampled a positive sign would be a silent
    no-op while its negative twin was a real perturbation. With the split, both
    move exactly `round(|delta|)` rings.
    """
    m = np.asarray(mask) > 0.5
    d = float(delta_px)
    if d == 0:
        return m.astype(np.uint8)
    if not m.any():
        return np.zeros(m.shape, dtype=np.uint8)
    sdf = distance_transform_edt(m) - distance_transform_edt(~m)
    keep = (sdf >= -d) if d > 0 else (sdf > -d)
    return keep.astype(np.uint8)


def crop_side_of(mask) -> int:
    """The square crop side `_mask_to_sdf` would derive from this mask.

    `precompute_dino_embeddings._mask_to_sdf:75` takes
    `max(bbox_h, bbox_w)` *before* any clamping to the canvas, and returns 0
    for an empty mask (via `_mask_to_sdf_worker`'s fallback). Mirrored exactly
    — `crop_sides` is a *training target* (`scale_mse`), so an off-by-one here
    is an off-by-one in the loss.
    """
    m = np.asarray(mask) > 0.5
    rows, cols = np.nonzero(m)
    if rows.size == 0:
        return 0
    return int(max(rows.max() - rows.min() + 1, cols.max() - cols.min() + 1))


def morph_crop_side(mask, delta_px: float) -> int:
    """crop_side after a `delta_px` morphological offset.

    Convenience for diagnostics only. The builder re-encodes the morphed mask
    (v2) rather than applying a crop_side override: `_mask_to_sdf`'s tight-bbox
    crop + resize to 224 does normalise a uniform offset almost away in SDF
    terms (measured MAE 0.00000 at 1 px, 0.010 at 2 px, 0.019 at 4 px), but the
    DINO embedding still moves 0.33-0.43 of the between-cell distance — not a
    no-op. See SEG_ABLATION_CODE_VERSION.

    Note `delta_px <= 1` is exactly a no-op at 128 px resolution: `sdf >
    -delta` excludes the distance-1 ring, so no pixel changes.
    """
    return crop_side_of(morph_mask(mask, delta_px))


# ---------------------------------------------------------------------------
# Merge — two cells segmented as one blob
# ---------------------------------------------------------------------------

def close_mask(mask, radius_px: float):
    """Morphological closing: dilate by `radius_px`, then erode by it.

    Fuses a hairline gap without moving the outline. Used on merged pairs: a
    segmentation merge *is* a lost membrane between two cells, so the union
    should be one connected region. Measured on the live cache, the raw union
    of two contacting cells is already connected 99.7% of the time; 1 px of
    closing takes that to 100% and changes the blob area by 0.1%.
    """
    if radius_px <= 0:
        return np.asarray(mask).astype(np.uint8)
    return morph_mask(morph_mask(mask, radius_px), -radius_px)


def merge_pair(mask_i, mask_j, offset_px, pad_factor: int = 2,
               close_px: float = 1.0):
    """Paste cell j's raster into cell i's canvas at a sub-pixel offset.

    Args:
        mask_i, mask_j: (S, S) binary crops, both object-centred.
        offset_px: (d_row, d_col) = (dx, dy) * scale_px, i.e.
            `(states[t, j] - states[t, i]) * scale_px`. Row axis is
            x — see the module docstring.
        pad_factor: canvas is `pad_factor * S` on a side. Measured geometry:
            `crop_sides` median 87, p99 121, max 128 (0.30% saturated) and
            median nearest-neighbour centroid distance 60.4 px, so a merged
            pair spans 147-181 px and a 128 canvas would clip it. 2 is the
            minimum that fits; `scale_px` is unchanged so `crop_side` stays
            comparable to an unmerged cell. Measured ratio
            crop_side(merged)/crop_side(single) = 1.58-1.73.
        close_px: closing radius applied to the union (see `close_mask`).

    Returns:
        (blob, clipped) — uint8 (pad_factor*S, pad_factor*S), and True when
        any ON pixel fell outside the padded canvas or touches its border.
    """
    mi = np.asarray(mask_i) > 0.5
    mj = np.asarray(mask_j) > 0.5
    S = mi.shape[0]
    C = int(pad_factor) * S
    pad = (C - S) // 2

    canvas = np.zeros((C, C), dtype=np.float32)
    canvas[pad:pad + S, pad:pad + S] = mi.astype(np.float32)

    d_r, d_c = float(offset_px[0]), float(offset_px[1])
    i_r, i_c = int(np.floor(d_r)), int(np.floor(d_c))
    f_r, f_c = d_r - i_r, d_c - i_c

    # Shift the small crop sub-pixel first (the technique at
    # polyhoop_simulation.generate_combined_image:118), then paste at the
    # integer offset.
    shifted = ndi_shift(mj.astype(np.float32), (f_r, f_c), order=1,
                        mode="constant", cval=0.0)

    r0, c0 = pad + i_r, pad + i_c
    dr0, dc0 = max(r0, 0), max(c0, 0)
    dr1, dc1 = min(r0 + S, C), min(c0 + S, C)
    clipped = False
    if dr0 >= dr1 or dc0 >= dc1:
        clipped = True
    else:
        sub = shifted[dr0 - r0:dr1 - r0, dc0 - c0:dc1 - c0]
        outside = shifted.sum() - sub.sum()
        if outside > 0.5:
            clipped = True
        canvas[dr0:dr1, dc0:dc1] = np.maximum(canvas[dr0:dr1, dc0:dc1], sub)

    blob = close_mask(canvas > 0.5, close_px)
    if blob.any():
        if blob[0].any() or blob[-1].any() or blob[:, 0].any() or blob[:, -1].any():
            clipped = True
    return blob, clipped


def pad_to(mask, pad_factor: int = 2):
    """Centre a (S, S) mask on a `pad_factor*S` canvas.

    `_mask_to_sdf` crops to the tight bbox before resizing, so padding a
    non-border mask is a no-op on the SDF — the load-bearing property that
    makes merged and unmerged cells comparable across arms.
    """
    m = np.asarray(mask)
    S = m.shape[0]
    C = int(pad_factor) * S
    pad = (C - S) // 2
    out = np.zeros((C, C), dtype=m.dtype)
    out[pad:pad + S, pad:pad + S] = m
    return out


# ---------------------------------------------------------------------------
# Observed centroid
# ---------------------------------------------------------------------------

def observed_centroid_units(mask, px_per_unit: float):
    """Centre-of-mass of an observed mask, in simulation units off canvas centre.

    Used differentially (corrupted minus clean) so the sub-pixel bias between a
    polygon's true centroid and its raster's pixel centroid cancels exactly.

    Returns None for an empty mask.
    """
    m = np.asarray(mask) > 0.5
    rows, cols = np.nonzero(m)
    if rows.size == 0:
        return None
    H, W = m.shape
    cr = (H - 1) / 2.0
    cc = (W - 1) / 2.0
    return np.array([(rows.mean() - cr) / px_per_unit,
                     (cols.mean() - cc) / px_per_unit], dtype=np.float64)
