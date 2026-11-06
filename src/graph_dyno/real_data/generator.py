"""CTCTrajectoryGenerator — real microscopy videos in the existing cache schema.

Emits `<hash>_<traj>.pt` with the six standard keys plus `valid` / `next_valid`,
so DINO precompute, decoder finetuning, GPU caching, rollout and the movie
scripts all work unchanged. One video (dataset+sequence, optionally
time-windowed) is one *source*; each source is cut into non-overlapping windows
of `trajectory_length` frames, and the windows are ordered **round-robin across
videos** so any prefix of `n_trajectories` covers every video.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from graph_dyno.real_data.ctc import discover_video
from graph_dyno.real_data.neighbors import resolve_contact_px, scan_video
from graph_dyno.real_data.raster import rasterize_window
from graph_dyno.real_data.tracks import (
    TrackFilters,
    build_global_tables,
    build_slot_metadata,
    n_windows_for,
    window_order,
)
from graph_dyno.data.trajectory_generator import TrajectoryGenerator

#: Grid-pixels per cell below which the upsample to 224 inside `_mask_to_sdf`
#: cannot recover the shape. Measured median cell size spans 9x across the
#: corpus, so one global r_max sized for the largest cell renders the smallest
#: as a handful of grid pixels.
GRID_PX_PER_CELL_FLOOR = 24.0


class CTCTrajectoryGenerator(TrajectoryGenerator):
    """Trajectory generator over Cell Tracking Challenge videos."""

    def __init__(self, cfg: DictConfig, device: str | None = None):
        super().__init__(cfg, device)
        self.rp = self.data.params

        # `generate_config_hash` pops the whole `generator` block, so
        # `trajectory_length` is NOT hashed — yet for real data it determines
        # the windowing, i.e. which frames land in which trajectory. Two runs
        # with different values would collide in one cache dir with silently
        # mismatched contents. Mirror it into the hashed block and assert.
        window_length = int(self.rp.window_length)
        if window_length != int(self.trajectory_length):
            raise ValueError(
                f"data.params.window_length ({window_length}) must equal "
                f"data.generator.trajectory_length ({self.trajectory_length}). "
                f"window_length is the hashed mirror of the unhashed "
                f"trajectory_length — see real_data/generator.py."
            )
        self.window_length = window_length
        self.r_max = float(self.rp.r_max_px)
        self.image_size = int(self.data.image_size)
        self.domain_size = float(self.data.domain_size)
        self.dt = float(self.data.dt)
        self.pad_pos = -4.0 * self.r_max

        self._prepared = False
        self._videos = []
        self._tracks = None
        self._plan: list[tuple[int, int]] = []
        self._pos: list[np.ndarray] = []
        self._vel: list[np.ndarray] = []
        self._sim_metadata: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Preparation (lazy — a fully-cached run never scans a TIFF)
    # ------------------------------------------------------------------

    def _prepare(self) -> None:
        if self._prepared:
            return

        filters = TrackFilters(
            min_track_length=int(self.rp.min_track_length),
            min_area_px=int(self.rp.min_area_px),
            border_policy=str(self.rp.border_policy),
            max_tracks_per_video=_opt_int(self.rp.get("max_tracks_per_video", None)),
        )
        specs = list(self.rp.videos)
        if not specs:
            raise ValueError("data.params.videos is empty")

        root = self.rp.root
        videos, contact_px, median_sides, scans_per_video = [], [], [], []
        override = self.rp.get("contact_px", None)
        contact_frac = float(self.rp.get("contact_frac", 1.0))

        for i, spec in enumerate(specs):
            video = discover_video(
                root, spec["dataset"], str(spec["seq"]),
                frame_start=int(spec.get("frame_start", 0) or 0),
                frame_end=_opt_int(spec.get("frame_end", None)),
                roi=spec.get("roi", None),
            )
            px, median_side = resolve_contact_px(
                video.seg_paths, contact_frac,
                contact_px=_per_video(override, i),
            )
            print(f"[ctc] {video.name}: {video.n_frames} frames "
                  f"{video.height}x{video.width}"
                  + (f" (roi {list(video.roi)})" if video.roi else "")
                  + f", seg={video.seg_kind}, "
                  f"median cell {median_side:.0f} px -> contact_px={px}")
            scans = scan_video(
                video.seg_paths, px,
                workers=int(self.rp.get("scan_workers", 32)),
                desc=f"  scan {video.name}",
            )
            videos.append(video)
            contact_px.append(px)
            median_sides.append(median_side)
            scans_per_video.append(scans)

        tracks = build_global_tables(
            videos, scans_per_video, filters, contact_px, median_sides,
            pseudo_split=_opt_int(self.rp.get("pseudo_split", None)),
        )
        self._videos = videos
        self._tracks = tracks

        self._check_geometry(tracks)

        # ---- window plan, round-robin across videos
        max_windows = _opt_int(self.rp.get("max_windows_per_video", None))
        n_windows = [n_windows_for(t.n_frames, self.window_length, max_windows)
                     for t in tracks.tables]
        for table, n_w in zip(tracks.tables, n_windows):
            if n_w == 0:
                raise ValueError(
                    f"{table.video.name}: {table.n_frames} frames is too short for "
                    f"window_length={self.window_length} (need >= "
                    f"{self.window_length + 1})"
                )
        self._plan = window_order(n_windows)
        if self.n_trajectories > len(self._plan):
            raise ValueError(
                f"n_trajectories={self.n_trajectories} exceeds the "
                f"{len(self._plan)} windows available "
                f"({dict(zip((t.video.name for t in tracks.tables), n_windows))})"
            )

        self._build_states()
        self._sim_metadata = self._build_sim_metadata()
        self._record_resolved(contact_px, median_sides, n_windows)

        centre = self.domain_size / 2.0
        print(f"[ctc] N={tracks.N} slots over {len(tracks.segments)} segments, "
              f"{len(self._plan)} windows, r_max={self.r_max:g}, "
              f"domain centre={centre:g}")
        self._prepared = True

    def _check_geometry(self, tracks) -> None:
        """Hard asserts on the three values that live in the hashed config."""
        if tracks.N != int(self.data.n_objects):
            raise ValueError(
                f"data.n_objects={self.data.n_objects} but the track table has "
                f"{tracks.N} slots "
                f"({[f'{t.video.name}:{t.n_slots}' for t in tracks.tables]}). "
                f"Set data.n_objects={tracks.N}."
            )

        for table in tracks.tables:
            need = table.required_r_max
            if need > self.r_max:
                where = _worst_cell(table)
                raise ValueError(
                    f"data.params.r_max_px={self.r_max:g} is too small for "
                    f"{table.video.name}: needs >= {need:.1f} px ({where}). "
                    f"Clipping the mask at the grid edge would corrupt "
                    f"_mask_to_sdf's bbox and hence crop_side. Raise r_max_px "
                    f"(and image_size with it)."
                )
            span = max(table.video.height, table.video.width)
            if span > self.domain_size:
                raise ValueError(
                    f"data.domain_size={self.domain_size:g} is smaller than "
                    f"{table.video.name}'s {table.video.height}x"
                    f"{table.video.width} field of view"
                )
            gpc = self.grid_px_per_cell(table)
            if gpc < GRID_PX_PER_CELL_FLOOR:
                print(f"[ctc] WARNING: {table.video.name} renders at "
                      f"{gpc:.0f} grid px/cell (floor {GRID_PX_PER_CELL_FLOOR:g}) — "
                      f"its cells are too small relative to r_max_px="
                      f"{self.r_max:g} for the 224 upsample to recover shape.")

    def grid_px_per_cell(self, table) -> float:
        """Median cell bbox side measured in rasterization grid pixels."""
        dx = 2 * self.r_max / (self.image_size - 1)
        return table.median_bbox_side / dx

    def _build_states(self) -> None:
        """Per-video centred positions and finite-difference velocities."""
        self._pos, self._vel = [], []
        centre = self.domain_size / 2.0
        center_fov = bool(self.rp.get("center_fov", True))

        for table in self._tracks.tables:
            valid = table.valid
            if center_fov:
                off = np.array([centre - table.video.height / 2.0,
                                centre - table.video.width / 2.0], dtype=np.float64)
            else:
                off = np.zeros(2, dtype=np.float64)

            # PAD_POS must be FINITE: masked MSE is ((pred-tgt)^2 * mask).sum()
            # and NaN * 0 = NaN would poison the whole loss. Off-canvas is free
            # correctness for plotting — generate_combined_image_dino guards
            # with `if x_start < x_end and y_start < y_end`, so padded slots
            # render nothing.
            pos = np.full(table.centroids.shape, self.pad_pos, dtype=np.float32)
            real = table.centroids.astype(np.float64) + off[None, None, :]
            pos[valid] = real[valid].astype(np.float32)

            vel = np.zeros_like(pos)
            if pos.shape[0] > 1:
                step_ok = valid[1:] & valid[:-1]
                diff = (real[1:] - real[:-1]) / self.dt
                vel[1:][step_ok] = diff[step_ok].astype(np.float32)

            self._pos.append(pos)
            self._vel.append(vel)

    def _build_sim_metadata(self) -> dict[str, torch.Tensor]:
        """Per-slot metadata — a pure function of the GLOBAL track table.

        Every value is a tensor or number, never a string: `train.py` calls
        `model.set_metadata(**sim_metadata)`, which does
        `torch.tensor(v, dtype=torch.float32)` for non-tensors, and a string
        raises *after* cache gen, SDF precompute, DINO precompute and MLflow
        start. So video/dataset/sequence identity is integer-coded and the
        code->name mapping lives in metadata.yaml.
        """
        arrays = build_slot_metadata(self._tracks, self.dt)
        meta = {k: torch.from_numpy(v.astype(np.float32)) for k, v in arrays.items()}
        meta["r_max"] = torch.tensor(float(self.r_max))
        meta["px_per_grid_unit"] = torch.tensor(
            2.0 * self.r_max / (self.image_size - 1))
        return meta

    def _record_resolved(self, contact_px, median_sides, n_windows) -> None:
        """Stash derived values in metadata.yaml, inside the UNHASHED block.

        `generate_config_hash` pops `data.generator`, so writing here records
        the code->name mappings and the resolved contact radii for the analysis
        scripts without perturbing the cache key.
        """
        info = {
            "n_objects": int(self._tracks.N),
            "r_max_px": float(self.r_max),
            "domain_size": float(self.domain_size),
            "window_length": int(self.window_length),
            "n_windows_total": len(self._plan),
            "segments": [
                {
                    "name": seg.name,
                    "video_id": seg.video_id,
                    "dataset_id": seg.dataset_id,
                    "sequence_id": seg.sequence_id,
                    "slot_start": seg.start,
                    "slot_stop": seg.stop,
                }
                for seg in self._tracks.segments
            ],
            "videos": [
                {
                    "name": t.video.name,
                    "dataset": t.video.dataset,
                    "seq": t.video.seq,
                    "seg_kind": t.video.seg_kind,
                    "n_frames": int(t.n_frames),
                    "height": int(t.video.height),
                    "width": int(t.video.width),
                    "roi": None if t.video.roi is None else list(t.video.roi),
                    "n_raw_tracks": int(t.n_raw_tracks),
                    "n_slots": int(t.n_slots),
                    "slot_offset": int(t.offset),
                    "n_windows": int(nw),
                    "contact_px": int(px),
                    "median_bbox_side": float(ms),
                    "mean_degree": float(t.mean_degree),
                    "grid_px_per_cell": float(self.grid_px_per_cell(t)),
                    "required_r_max": float(t.required_r_max),
                }
                for t, px, ms, nw in zip(self._tracks.tables, contact_px,
                                         median_sides, n_windows)
            ],
            "window_plan": [[int(v), int(w)] for v, w in self._plan],
        }
        was_struct = OmegaConf.is_struct(self.cfg)
        was_readonly = OmegaConf.is_readonly(self.cfg)
        OmegaConf.set_struct(self.cfg, False)
        OmegaConf.set_readonly(self.cfg, False)
        try:
            self.cfg.data.generator.ctc_resolved = info
        finally:
            OmegaConf.set_readonly(self.cfg, was_readonly)
            OmegaConf.set_struct(self.cfg, was_struct)

    # ------------------------------------------------------------------
    # Trajectory emission
    # ------------------------------------------------------------------

    def generate_all(self, show_progress: bool = True):
        # Resolve everything before the base class writes metadata.yaml, so the
        # ctc_resolved block lands in the file even on a resume.
        dataset_exists, _ = self._check_existing_dataset()
        if not self.overwrite_existing and dataset_exists:
            print(f"Dataset already exists in '{self.cache_dir}' with matching "
                  f"parameters. Skipping generation.")
            return
        self._prepare()
        super().generate_all(show_progress=show_progress)

    def _generate_trajectory(self, trajectory_idx: int) -> Dict[str, torch.Tensor]:
        self._prepare()
        video_idx, window_idx = self._plan[trajectory_idx]
        table = self._tracks.tables[video_idx]
        video = self._videos[video_idx]

        T = self.window_length
        N = self._tracks.N
        S = self.image_size
        f0 = window_idx * T
        lo, hi = table.offset, table.offset + table.n_slots
        print(f"  Trajectory {trajectory_idx}: {video.name} window {window_idx} "
              f"frames [{f0}, {f0 + T}) -> slots [{lo}, {hi})")

        # ---- validity. `next_valid` is sliced from the FULL-video array, not
        # derived in the dataloader: next_states[T-1] describes frame f0+T,
        # which the window's own `valid` does not cover and which the
        # `next_t = min(t+1, T-1)` clamp cannot reach. Deriving it would mark a
        # cell that dies at f0+T as valid against a PAD_POS target at every
        # window seam (~1/T of samples) — and seams are where divisions land.
        valid = np.zeros((T, N), dtype=bool)
        next_valid = np.zeros((T, N), dtype=bool)
        valid[:, lo:hi] = table.valid[f0:f0 + T]
        next_valid[:, lo:hi] = table.valid[f0 + 1:f0 + T + 1]

        # Trajectory v is all-False outside its own slot range.
        outside = np.ones(N, dtype=bool)
        outside[lo:hi] = False
        assert not valid[:, outside].any() and not next_valid[:, outside].any()

        # ---- states
        states = np.zeros((T, N, 4), dtype=np.float32)
        next_states = np.zeros((T, N, 4), dtype=np.float32)
        states[..., :2] = self.pad_pos
        next_states[..., :2] = self.pad_pos
        states[:, lo:hi, :2] = self._pos[video_idx][f0:f0 + T]
        states[:, lo:hi, 2:] = self._vel[video_idx][f0:f0 + T]
        next_states[:, lo:hi, :2] = self._pos[video_idx][f0 + 1:f0 + T + 1]
        next_states[:, lo:hi, 2:] = self._vel[video_idx][f0 + 1:f0 + T + 1]

        # ---- images
        frame_slots, frame_labels, frame_centroids = [], [], []
        for t in range(T):
            local = np.nonzero(table.valid[f0 + t])[0]
            frame_slots.append(local)
            frame_labels.append(table.labels[local])
            frame_centroids.append(table.centroids[f0 + t, local])

        masks = rasterize_window(
            video.seg_paths[f0:f0 + T], frame_labels, frame_centroids,
            self.r_max, S, workers=int(self.rp.get("raster_workers", 32)),
        )
        channeled_images = np.zeros((T, N, S, S), dtype=np.float32)
        for t, (local, m) in enumerate(zip(frame_slots, masks)):
            if local.size:
                channeled_images[t, lo + local] = m.astype(np.float32)

        # ---- interaction graph
        graphs = [self._frame_graph(table, f0 + t, N) for t in range(T)]

        times = (np.arange(f0, f0 + T, dtype=np.float32) * self.dt).reshape(T, 1)

        return {
            "times": torch.from_numpy(times),
            "states": torch.from_numpy(states),
            "next_states": torch.from_numpy(next_states),
            # Unread by the model, but dataloader.py's single-step path indexes it.
            "forces": torch.zeros(T, N * 2, dtype=torch.float32),
            "channeled_images": torch.from_numpy(channeled_images),
            "W_interactions": torch.stack(graphs),
            "valid": torch.from_numpy(valid),
            "next_valid": torch.from_numpy(next_valid),
            "sim_metadata": self._sim_metadata,
        }

    def _frame_graph(self, table, frame: int, N: int) -> torch.Tensor:
        """Symmetric (N, N) bool sparse COO adjacency for one frame."""
        pairs = table.pairs[frame]
        if pairs.size == 0:
            idx = torch.zeros((2, 0), dtype=torch.long)
            vals = torch.zeros(0, dtype=torch.bool)
        else:
            s = torch.from_numpy(pairs[:, 0] + table.offset)
            d = torch.from_numpy(pairs[:, 1] + table.offset)
            idx = torch.stack([torch.cat([s, d]), torch.cat([d, s])])
            vals = torch.ones(idx.shape[1], dtype=torch.bool)
        return torch.sparse_coo_tensor(idx, vals, (N, N), dtype=torch.bool).coalesce()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _opt_int(value) -> int | None:
    return None if value is None else int(value)


def _per_video(value, idx: int):
    """Read a config field that may be a scalar, a per-video list, or None."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) or OmegaConf.is_list(value):
        item = list(value)[idx]
        return None if item is None else float(item)
    return float(value)


def _worst_cell(table) -> str:
    flat = np.where(table.valid, table.half_extents, -np.inf)
    t, i = np.unravel_index(int(np.argmax(flat)), flat.shape)
    return (f"frame {t}, label {int(table.labels[i])}, "
            f"half-extent {math.ceil(flat[t, i])} px")
