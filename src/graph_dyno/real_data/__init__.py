"""Cell Tracking Challenge (CTC) real-microscopy data source.

Emits the existing trajectory-cache schema (plus `valid` / `next_valid`) from
CTC instance masks + lineage tracks, so DINO precompute, decoder finetuning,
GPU caching and the movie scripts all work unchanged.

Modules:
    ctc       — TIFF/lineage loading and dataset discovery
    neighbors — dilation-based contact graph + parallel frame scan
    tracks    — global track table, slot allocation, validity, per-slot metadata
    raster    — object-centred rasterization onto the [-r_max, r_max] grid
    generator — CTCTrajectoryGenerator (TrajectoryGenerator subclass)
"""
