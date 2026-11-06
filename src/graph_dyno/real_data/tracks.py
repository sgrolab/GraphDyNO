"""Global track table: slot allocation, validity, filters, per-slot metadata.

Slots are **disjoint per video** — video *v* owns `[offset_v, offset_v + n_v)`.
That is not an implementation detail, it is the experiment: if slot *i* were
shared across videos, `parameter_embedding` row *i* would be shared too and
could never separate them.

Divisions need no special code under the `new_slots` policy, and that is the
point. Parent `P` has `end = b - 1` where `b` is its daughters' birth, so
`valid[b:, slot_P]` is False straight out of the presence table; the daughters
get fresh slots so `valid[:b, slot_C]` is False. The parent's last transition is
masked by `next_valid`, each daughter's first by `prev_valid`, and no cell is
ever asked to predict "I turn into half of myself" nor has a velocity computed
across a mitosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from graph_dyno.real_data.ctc import Video
from graph_dyno.real_data.neighbors import FrameScan

#: Fraction of the observed lifetime a cell may sit on the FOV border before
#: `border_policy: drop_track` discards it. Only used by that policy.
BORDER_POLICIES = ("invalidate_frame", "drop_track", "keep")


@dataclass
class TrackFilters:
    """Which tracks and which frames survive into the dataset."""

    #: Observed frames a track needs to keep its slot. The main lever on N —
    #: a slot with 3 valid frames gets ~3 gradient updates on its `b`, i.e.
    #: noise. PhC-C2DL-PSC/01 has 1168 divisions across 1370 tracks, which is
    #: exactly why this matters under `new_slots`.
    min_track_length: int = 10
    #: Kills specks that would rasterize to `crop_side ~ 1`.
    min_area_px: int = 8
    #: A cell clipped by the FOV edge has a wrong centroid, wrong crop_side and
    #: wrong shape — but dropping the whole track is far too lossy where cells
    #: continuously enter and leave, so the default invalidates just that frame.
    border_policy: str = "invalidate_frame"
    max_tracks_per_video: int | None = None

    def __post_init__(self):
        if self.border_policy not in BORDER_POLICIES:
            raise ValueError(
                f"border_policy must be one of {BORDER_POLICIES}, "
                f"got {self.border_policy!r}"
            )


@dataclass
class Segment:
    """Slot-owning unit: a contiguous sub-block of one video's slots.

    Normally one segment per video. `pseudo_split` gives a video two segments
    over disjoint halves of its tracks — the null control for Risk 1.
    """

    name: str
    video_idx: int
    video_id: int
    dataset_id: int
    sequence_id: int
    start: int   # global slot, inclusive
    stop: int    # global slot, exclusive

    @property
    def n_slots(self) -> int:
        return self.stop - self.start


@dataclass
class VideoTable:
    """Everything one video contributes, indexed by local slot 0..n_v-1."""

    video: Video
    offset: int                    # global slot of local slot 0
    labels: np.ndarray             # (n_v,) track label per local slot
    label_to_slot: np.ndarray      # (max_label+1,) int32 -> LOCAL slot, -1 if dropped
    birth: np.ndarray              # (n_v,) int32
    end: np.ndarray                # (n_v,) int32
    parent_label: np.ndarray       # (n_v,) int32, 0 = no parent
    parent_slot: np.ndarray        # (n_v,) int32, GLOBAL slot or -1
    n_children: np.ndarray         # (n_v,) int32, from the raw lineage table
    valid: np.ndarray              # (T_v, n_v) bool
    centroids: np.ndarray          # (T_v, n_v, 2) float32, (row, col) px
    areas: np.ndarray              # (T_v, n_v) float32
    aspect_ratios: np.ndarray      # (T_v, n_v) float32
    half_extents: np.ndarray       # (T_v, n_v) float32
    n_neighbors: np.ndarray        # (T_v, n_v) float32
    pairs: list[np.ndarray] = field(repr=False, default_factory=list)  # local slots
    contact_px: int = 0
    median_bbox_side: float = 0.0
    n_raw_tracks: int = 0

    @property
    def n_slots(self) -> int:
        return len(self.labels)

    @property
    def n_frames(self) -> int:
        return self.valid.shape[0]

    @property
    def required_r_max(self) -> float:
        """Smallest r_max that renders every valid cell without clipping.

        The half-extent *from the centroid*, not half the bbox — they differ
        for a concave cell. A silent violation clips the mask at the grid edge,
        corrupting `_mask_to_sdf`'s bbox and hence `crop_side`.
        """
        if not self.valid.any():
            return 0.0
        return float(self.half_extents[self.valid].max()) + 1.0

    @property
    def mean_degree(self) -> float:
        if not self.valid.any():
            return 0.0
        return float(self.n_neighbors[self.valid].mean())


def build_video_table(video: Video, scans: list[FrameScan],
                      filters: TrackFilters, contact_px: int,
                      median_bbox_side: float,
                      pseudo_split: bool = False) -> VideoTable:
    """Build one video's track table from its full-video scan.

    Track ordering is `(birth, label)`. With `pseudo_split`, the retained
    tracks are first split into two arbitrary halves (index parity — balanced
    and uncorrelated with birth time) and re-concatenated, so each half owns a
    contiguous slot sub-block.
    """
    T = video.n_frames
    if len(scans) != T:
        raise ValueError(f"{video.name}: {len(scans)} scans for {T} frames")

    # ---- candidate labels: in the lineage table AND observed in >=1 SEG frame
    observed: set[int] = set()
    for scan in scans:
        observed.update(int(x) for x in scan.labels)
    candidates = sorted(observed & set(video.tracks))
    n_raw = len(candidates)
    if not candidates:
        raise ValueError(f"{video.name}: no labels shared between SEG and man_track.txt")

    max_label = max(candidates)
    # Dense int32 index rather than a dict: labels are sparse and
    # non-contiguous (PhC-C2DL-PSC's max label is 22215 for 1370 tracks) and
    # this sits in the hot pair-mapping path. PSC's array is 89 KB.
    cand_index = np.full(max_label + 1, -1, dtype=np.int32)
    cand_index[np.asarray(candidates, dtype=np.int64)] = np.arange(len(candidates),
                                                                  dtype=np.int32)

    birth = np.array([video.tracks[L]["birth"] for L in candidates], dtype=np.int32)
    end = np.array([video.tracks[L]["end"] for L in candidates], dtype=np.int32)

    # ---- per-frame observation + geometry, gathered onto candidate columns
    n_c = len(candidates)
    present = np.zeros((T, n_c), dtype=bool)
    centroids = np.zeros((T, n_c, 2), dtype=np.float32)
    areas = np.zeros((T, n_c), dtype=np.float32)
    aspect = np.zeros((T, n_c), dtype=np.float32)
    half_ext = np.zeros((T, n_c), dtype=np.float32)
    border = np.zeros((T, n_c), dtype=bool)

    for t, scan in enumerate(scans):
        if scan.labels.size == 0:
            continue
        keep = scan.labels <= max_label
        cols = cand_index[scan.labels[keep]]
        sel = cols >= 0
        cols = cols[sel]
        if cols.size == 0:
            continue
        src = np.nonzero(keep)[0][sel]
        present[t, cols] = True
        centroids[t, cols] = scan.centroids[src]
        areas[t, cols] = scan.areas[src]
        aspect[t, cols] = scan.aspect_ratios[src]
        half_ext[t, cols] = scan.half_extents[src]
        border[t, cols] = scan.touches_border[src]

    # ---- validity: observation, clipped to [birth, end], then the filters
    frames = np.arange(T, dtype=np.int32)[:, None]
    in_life = (frames >= birth[None, :]) & (frames <= end[None, :])
    valid = present & in_life
    valid &= areas >= filters.min_area_px
    if filters.border_policy == "invalidate_frame":
        valid &= ~border

    # ---- track-level filters
    keep_track = valid.sum(axis=0) >= filters.min_track_length
    if filters.border_policy == "drop_track":
        keep_track &= ~(border & present).any(axis=0)

    if filters.max_tracks_per_video is not None:
        lengths = np.where(keep_track, valid.sum(axis=0), -1)
        order = np.argsort(-lengths, kind="stable")[:filters.max_tracks_per_video]
        mask = np.zeros(n_c, dtype=bool)
        mask[order] = True
        keep_track &= mask

    kept = np.nonzero(keep_track)[0]
    if kept.size == 0:
        raise ValueError(
            f"{video.name}: no tracks survive the filters "
            f"(min_track_length={filters.min_track_length}, "
            f"min_area_px={filters.min_area_px}, "
            f"border_policy={filters.border_policy})"
        )

    # ---- slot order: (birth, label), then optionally split into halves
    labels_arr = np.asarray(candidates, dtype=np.int64)
    order = sorted(kept.tolist(), key=lambda i: (int(birth[i]), int(labels_arr[i])))
    if pseudo_split:
        order = order[0::2] + order[1::2]
    order = np.asarray(order, dtype=np.int64)

    label_to_slot = np.full(max_label + 1, -1, dtype=np.int32)
    label_to_slot[labels_arr[order]] = np.arange(len(order), dtype=np.int32)

    # ---- lineage: children counted on the RAW table so analysis can tell
    # mitosis (2) from a single-child relink (1). A filter can orphan a child,
    # so parent_slot = -1 must be handled, not assumed away.
    child_count: dict[int, int] = {}
    for info in video.tracks.values():
        p = int(info["parent"])
        if p:
            child_count[p] = child_count.get(p, 0) + 1

    kept_labels = labels_arr[order]
    parent_label = np.array(
        [int(video.tracks[int(L)]["parent"]) for L in kept_labels], dtype=np.int32)
    n_children = np.array([child_count.get(int(L), 0) for L in kept_labels],
                          dtype=np.int32)
    # Local for now; build_global_tables shifts the non-negative entries by the
    # video's slot offset once every table exists.
    parent_slot = np.full(len(order), -1, dtype=np.int32)
    for i, p_label in enumerate(parent_label):
        p = int(p_label)
        if p and p <= max_label and label_to_slot[p] >= 0:
            parent_slot[i] = int(label_to_slot[p])

    # ---- per-frame neighbour degree and pairs, remapped to local slots
    n_v = len(order)
    valid_kept = valid[:, order]
    n_neighbors = np.zeros((T, n_v), dtype=np.float32)
    local_pairs: list[np.ndarray] = []
    for t, scan in enumerate(scans):
        pairs = scan.pairs
        if pairs.size == 0:
            local_pairs.append(np.zeros((0, 2), dtype=np.int64))
            continue
        ok = (pairs[:, 0] <= max_label) & (pairs[:, 1] <= max_label)
        pairs = pairs[ok]
        if pairs.size == 0:
            local_pairs.append(np.zeros((0, 2), dtype=np.int64))
            continue
        s = label_to_slot[pairs[:, 0]]
        d = label_to_slot[pairs[:, 1]]
        vt = valid_kept[t]
        # Dropping any edge that touches an invalid slot is what guarantees
        # "padded slots have no incident edges", which is what makes them
        # harmless in gnn.py (never a src/tgt, so scatter_add_ leaves their
        # message at exactly zero).
        s_safe = np.where(s >= 0, s, 0)
        d_safe = np.where(d >= 0, d, 0)
        keep_edge = (s >= 0) & (d >= 0) & vt[s_safe] & vt[d_safe]
        p = np.stack([s[keep_edge], d[keep_edge]], axis=1).astype(np.int64)
        local_pairs.append(p)
        if p.size:
            np.add.at(n_neighbors[t], p[:, 0], 1.0)
            np.add.at(n_neighbors[t], p[:, 1], 1.0)

    return VideoTable(
        video=video,
        offset=0,  # filled in by build_global_tables
        labels=kept_labels.astype(np.int64),
        label_to_slot=label_to_slot,
        birth=birth[order].astype(np.int32),
        end=end[order].astype(np.int32),
        parent_label=parent_label,
        parent_slot=parent_slot,
        n_children=n_children,
        valid=valid_kept,
        centroids=centroids[:, order],
        areas=areas[:, order],
        aspect_ratios=aspect[:, order],
        half_extents=half_ext[:, order],
        n_neighbors=n_neighbors,
        pairs=local_pairs,
        contact_px=int(contact_px),
        median_bbox_side=float(median_bbox_side),
        n_raw_tracks=n_raw,
    )


@dataclass
class GlobalTracks:
    """All videos assembled into one disjoint global slot space."""

    tables: list[VideoTable]
    segments: list[Segment]
    N: int

    @property
    def n_videos(self) -> int:
        return len(self.segments)

    @property
    def required_r_max(self) -> float:
        return max((t.required_r_max for t in self.tables), default=0.0)

    def assert_slot_ranges_disjoint(self) -> None:
        covered = np.zeros(self.N, dtype=bool)
        for seg in self.segments:
            block = covered[seg.start:seg.stop]
            if block.any():
                raise AssertionError(f"segment {seg.name} overlaps an earlier one")
            covered[seg.start:seg.stop] = True
        if not covered.all():
            raise AssertionError("slot space has gaps — segments do not tile [0, N)")


def build_global_tables(videos: list[Video], scans_per_video: list[list[FrameScan]],
                        filters: TrackFilters, contact_px: list[int],
                        median_sides: list[float],
                        pseudo_split: int | None = None) -> GlobalTracks:
    """Assemble per-video tables into one global, disjoint slot space.

    Videos keep their config-declared order; within a video, retained tracks are
    sorted by `(birth, label)`. `slot = offset_v + i`, `N = sum_v n_v`.
    """
    tables: list[VideoTable] = []
    segments: list[Segment] = []
    dataset_codes: dict[str, int] = {}
    sequence_codes: dict[str, int] = {}

    offset = 0
    for v_idx, (video, scans) in enumerate(zip(videos, scans_per_video)):
        split = (pseudo_split is not None and pseudo_split == v_idx)
        table = build_video_table(
            video, scans, filters, contact_px[v_idx], median_sides[v_idx],
            pseudo_split=split,
        )
        table.offset = offset

        ds_code = dataset_codes.setdefault(video.dataset, len(dataset_codes))
        seq_code = sequence_codes.setdefault(video.seq, len(sequence_codes))

        n_v = table.n_slots
        if split:
            half = (n_v + 1) // 2
            bounds = [(0, half), (half, n_v)]
            names = [f"{video.name}#A", f"{video.name}#B"]
        else:
            bounds = [(0, n_v)]
            names = [video.name]

        for name, (lo, hi) in zip(names, bounds):
            segments.append(Segment(
                name=name,
                video_idx=v_idx,
                video_id=len(segments),
                dataset_id=ds_code,
                sequence_id=seq_code,
                start=offset + lo,
                stop=offset + hi,
            ))

        tables.append(table)
        offset += n_v

    # Shift the local parent slots into the global space. Done after every
    # table exists so a division that straddles a pseudo-split still points at
    # the right slot.
    for table in tables:
        table.parent_slot = np.where(table.parent_slot >= 0,
                                     table.parent_slot + table.offset,
                                     -1).astype(np.int32)

    global_tracks = GlobalTracks(tables=tables, segments=segments, N=offset)
    global_tracks.assert_slot_ranges_disjoint()
    return global_tracks


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def n_windows_for(n_frames: int, window_length: int,
                  max_windows: int | None = None) -> int:
    """Number of non-overlapping windows, leaving room for `next_states`.

    Window `w` reads states from `[wT, wT+T)` and next_states from
    `[wT+1, wT+T+1)`, so it needs `wT + T + 1 <= n_frames`.
    """
    n = max(0, (n_frames - 1) // window_length)
    if max_windows is not None:
        n = min(n, int(max_windows))
    return n


def window_order(n_windows_per_video: list[int]) -> list[tuple[int, int]]:
    """Round-robin (video_idx, window_idx) ordering.

    Any prefix of `n_trajectories` then covers every video. Video-major
    ordering would make `n_trajectories=4` yield four windows of video 0 only.
    """
    out: list[tuple[int, int]] = []
    if not n_windows_per_video:
        return out
    for w in range(max(n_windows_per_video)):
        for v, n_w in enumerate(n_windows_per_video):
            if w < n_w:
                out.append((v, w))
    return out


# ---------------------------------------------------------------------------
# Per-slot metadata
# ---------------------------------------------------------------------------

def _safe_stats(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise mean and coefficient of variation over masked entries."""
    m = mask.astype(np.float64)
    n = m.sum(axis=0)
    denom = np.maximum(n, 1.0)
    mean = (values * m).sum(axis=0) / denom
    var = ((values - mean[None, :]) ** 2 * m).sum(axis=0) / denom
    std = np.sqrt(np.maximum(var, 0.0))
    nonzero = np.abs(mean) > 1e-12
    cv = np.divide(std, np.abs(mean), out=np.zeros_like(std), where=nonzero)
    return np.where(n > 0, mean, 0.0), np.where(n > 0, cv, 0.0)


def build_slot_metadata(tracks: GlobalTracks, dt: float) -> dict[str, np.ndarray]:
    """Per-slot (N,) statistics, computed over each video's WHOLE timeline.

    Never a function of the window: `verify_metadata_consistency` compares every
    key across all trajectories with `torch.allclose`, so any per-window
    quantity would raise. Because slots are globally disjoint and every
    statistic is computed over the whole video, the (N,) vectors are identical
    in every trajectory file — including trajectories from other videos, where
    those entries just describe never-valid slots.
    """
    N = tracks.N
    out = {
        key: np.zeros(N, dtype=np.float64)
        for key in ("video_id", "dataset_id", "sequence_id", "rs", "mean_area_px",
                    "area_cv", "mean_speed_px", "speed_cv", "mean_n_neighbors",
                    "neighbor_cv", "mean_aspect_ratio", "track_length_frames",
                    "birth_frame", "n_children", "parent_slot")
    }
    out["parent_slot"][:] = -1.0

    for seg in tracks.segments:
        out["video_id"][seg.start:seg.stop] = seg.video_id
        out["dataset_id"][seg.start:seg.stop] = seg.dataset_id
        out["sequence_id"][seg.start:seg.stop] = seg.sequence_id

    for table in tracks.tables:
        lo, hi = table.offset, table.offset + table.n_slots
        valid = table.valid

        mean_area, area_cv = _safe_stats(table.areas.astype(np.float64), valid)
        mean_aspect, _ = _safe_stats(table.aspect_ratios.astype(np.float64), valid)
        mean_nb, nb_cv = _safe_stats(table.n_neighbors.astype(np.float64), valid)

        # Speed only where both endpoints of the step are valid — a velocity
        # across a gap or a mitosis is meaningless.
        pos = table.centroids.astype(np.float64)
        step_ok = valid[1:] & valid[:-1]
        speed = np.zeros_like(valid, dtype=np.float64)
        if speed.shape[0] > 1:
            speed[1:] = np.linalg.norm(pos[1:] - pos[:-1], axis=-1) / dt
        speed_mask = np.zeros_like(valid)
        speed_mask[1:] = step_ok
        mean_speed, speed_cv = _safe_stats(speed, speed_mask)

        out["rs"][lo:hi] = np.sqrt(mean_area / np.pi)
        out["mean_area_px"][lo:hi] = mean_area
        out["area_cv"][lo:hi] = area_cv
        out["mean_speed_px"][lo:hi] = mean_speed
        out["speed_cv"][lo:hi] = speed_cv
        out["mean_n_neighbors"][lo:hi] = mean_nb
        out["neighbor_cv"][lo:hi] = nb_cv
        out["mean_aspect_ratio"][lo:hi] = mean_aspect
        out["track_length_frames"][lo:hi] = valid.sum(axis=0)
        out["birth_frame"][lo:hi] = table.birth
        out["n_children"][lo:hi] = table.n_children
        out["parent_slot"][lo:hi] = table.parent_slot

    return out
