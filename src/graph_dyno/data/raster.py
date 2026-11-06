"""Batched, GIL-releasing replica of ``skimage.draw.polygon2mask``.

``polygon2mask`` calls ``draw.polygon``, which scans the clipped bounding box
applying the even-odd crossing test one polygon at a time. Reproducing that test
in float64 **with the same expression order**, evaluated for all polygons at
once, makes each numpy op large enough that the GIL is genuinely released — so
a ``ThreadPoolExecutor`` over polygon subsets actually scales. The output is
bit-identical to skimage (see ``tests/test_raster.py``).

Measured on real polyhoop cells (600 cells, 128 px):

    skimage serial          1.226 s/frame   1.0x
    batched numpy, 1 thread 0.663 s/frame   1.9x
    batched numpy, 16 thr   0.132 s/frame   9.3x

Two invariants keep the output exact and must not be "cleaned up":

* Everything stays **float64** and the crossing test keeps the expression order
  ``X < (xj - xi) * (Y - yi) / (yj - yi) + xi``. Any reassociation (or float32)
  changes the rounding on boundary pixels.
* Padding to a common vertex count **repeats the last vertex**, which only ever
  creates degenerate edges (``yi == yj``, so ``c1`` is False and they never
  cross) and preserves the closing edge.
"""

import numpy as np


def _pad(polys):
    """Stack ragged polygons into ``(P, Nmax, 2)`` float64 by repeating the last vertex."""
    P = len(polys)
    Nmax = max(len(p) for p in polys)
    out = np.empty((P, Nmax, 2), dtype=np.float64)
    for i, p in enumerate(polys):
        out[i, : len(p)] = p
        if len(p) < Nmax:
            out[i, len(p) :] = p[-1]
    return out


def rasterize_batch(polys, shape, chunk=64, out=None):
    """Rasterize many polygons at once, bit-identically to ``polygon2mask``.

    Args:
        polys: list of ``(n_vertices, 2)`` arrays of ``(row, col)`` coordinates.
            Polygons may have differing vertex counts.
        shape: ``(H, W)`` output image shape.
        chunk: polygons processed per numpy op. 64 keeps temporaries around
            8 MB, which is cache-friendly and small enough to thread well.
        out: optional ``(P, H, W)`` float32 array to fill in place.

    Returns:
        ``(P, H, W)`` float32 array of binary masks.
    """
    H, W = shape
    P = len(polys)
    if out is None:
        out = np.zeros((P, H, W), dtype=np.float32)
    if P == 0:
        return out

    V = _pad(polys)
    Y = np.arange(H, dtype=np.float64)[None, :, None]
    X = np.arange(W, dtype=np.float64)[None, None, :]

    for s in range(0, P, chunk):
        v = V[s : s + chunk]
        yp = v[:, :, 0]
        xp = v[:, :, 1]
        n = yp.shape[1]
        inside = np.zeros((v.shape[0], H, W), dtype=bool)
        # Loop over *edge index*, not polygon, so every op spans the chunk.
        for i in range(n):
            j = i - 1
            yi = yp[:, i][:, None, None]
            yj = yp[:, j][:, None, None]
            xi = xp[:, i][:, None, None]
            xj = xp[:, j][:, None, None]
            c1 = (yi > Y) != (yj > Y)
            with np.errstate(divide="ignore", invalid="ignore"):
                c2 = X < (xj - xi) * (Y - yi) / (yj - yi) + xi
            inside ^= c1 & c2
        out[s : s + chunk] = inside

    return out


def rasterize_batch_torch(polys, shape, device="cpu", chunk=64):
    """Torch port of :func:`rasterize_batch`; also bit-exact.

    Kept available for ``device="cuda"`` (~60x over skimage), but **off by
    default**: once rasterization is threaded it sits well below the physics
    step in the pipeline, so the GPU buys nothing end-to-end and would make
    generation jobs compete for scarcer resources. Only relevant if
    ``image_size`` or ``n_objects`` grows a lot.
    """
    import torch

    H, W = shape
    P = len(polys)
    out = torch.zeros((P, H, W), dtype=torch.float32, device=device)
    if P == 0:
        return out

    V = torch.from_numpy(_pad(polys)).to(device)
    Y = torch.arange(H, dtype=torch.float64, device=device)[None, :, None]
    X = torch.arange(W, dtype=torch.float64, device=device)[None, None, :]

    for s in range(0, P, chunk):
        v = V[s : s + chunk]
        yp = v[:, :, 0]
        xp = v[:, :, 1]
        n = yp.shape[1]
        inside = torch.zeros((v.shape[0], H, W), dtype=torch.bool, device=device)
        for i in range(n):
            j = i - 1
            yi = yp[:, i][:, None, None]
            yj = yp[:, j][:, None, None]
            xi = xp[:, i][:, None, None]
            xj = xp[:, j][:, None, None]
            c1 = (yi > Y) != (yj > Y)
            c2 = X < (xj - xi) * (Y - yi) / (yj - yi) + xi
            inside ^= c1 & c2
        out[s : s + chunk] = inside.to(torch.float32)

    return out
