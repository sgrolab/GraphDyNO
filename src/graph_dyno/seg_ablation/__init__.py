"""Segmentation / tracking ablations for the paper's polyhoop datasets.

The paper trains on *perfect* observations: polygon-exact masks, exact
centroids, permanent cell identity, no missed frames. This package injects the
errors a real segmentation + tracking pipeline makes — partial masks, merged
neighbours, dropped detections, centroid noise, ID swaps — **without touching a
single byte of the 485-569 GB trajectory caches**.

The insight that makes it cheap: the shape embedding for cell `(t, n)` is an
independent function of exactly one 128x128 mask, so corrupting a few percent
of cells means re-encoding a few percent of the crops. Misses, position noise,
ID swaps and erode/dilate need no GPU work at all.

Everything lands in two small **sidecars** per trajectory:

    segabl_{abl_id}_{cache_id}_{traj:04d}.pt                     geometry
    segabl_{abl_id}_{suffix}_{cache_id}_{traj:04d}.pt            embeddings

`build.build_sidecar` writes them; `apply.apply_sidecar` grafts them onto a
loaded trajectory at train time. The spec lives OUTSIDE `cfg.data` (see
`configs/seg_ablation/`) precisely so `generate_config_hash` is unchanged and
the caches are never regenerated.
"""

from graph_dyno.seg_ablation.plan import (  # noqa: F401
    SEG_ABLATION_CODE_VERSION,
    Event,
    KINDS,
    OP_MERGE_ABSORB,
    OP_MERGE_GONE,
    OP_MISS,
    OP_NONE,
    OP_PARTIAL,
    OP_SCALE,
    abl_id,
    build_index_maps,
    normalize_spec,
    plan_events,
)

__all__ = [
    "SEG_ABLATION_CODE_VERSION",
    "Event",
    "KINDS",
    "OP_NONE",
    "OP_PARTIAL",
    "OP_MERGE_ABSORB",
    "OP_MERGE_GONE",
    "OP_MISS",
    "OP_SCALE",
    "abl_id",
    "normalize_spec",
    "plan_events",
    "build_index_maps",
]
