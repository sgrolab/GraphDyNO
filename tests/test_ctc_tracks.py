"""Tests for the global track table (real_data/tracks.py).

Builds synthetic CTC trees in tmp_path so lineage edge cases (relinks, mitosis,
ST gaps, orphaned children) are exercised without touching real_cell_data.

Run with: pytest tests/test_ctc_tracks.py -v
"""

import pickle

import numpy as np
import pytest
from PIL import Image

from graph_dyno.real_data.ctc import SegFrame, discover_video, load_seg, parse_man_track
from graph_dyno.real_data.neighbors import scan_video
from graph_dyno.real_data.tracks import (
    TrackFilters,
    build_global_tables,
    build_slot_metadata,
    build_video_table,
    n_windows_for,
    window_order,
)

SIZE = 64
RADIUS = 4


# ---------------------------------------------------------------------------
# Synthetic CTC dataset builder
# ---------------------------------------------------------------------------

def _stamp(img, label, cy, cx, radius=RADIUS):
    ys, xs = np.ogrid[:img.shape[0], :img.shape[1]]
    img[(ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2] = label
    return img


def make_ctc_video(root, dataset, seq, n_frames, man_track_rows, presence,
                   positions=None, size=SIZE, radius=RADIUS, seg_kind="ST"):
    """Write a minimal CTC dataset/sequence to disk.

    Args:
        man_track_rows: list of (label, birth, end, parent).
        presence: {label: set of frames the SEG mask actually contains it}.
        positions: {label: (row, col)} or {label: {frame: (row, col)}}.
    """
    root = root / dataset
    raw = root / seq
    tra = root / f"{seq}_GT" / "TRA"
    seg = root / (f"{seq}_ST" if seg_kind == "ST" else f"{seq}_GT") / "SEG"
    for d in (raw, tra, seg):
        d.mkdir(parents=True, exist_ok=True)

    (tra / "man_track.txt").write_text(
        "".join(f"{L} {B} {E} {P}\n" for L, B, E, P in man_track_rows))

    labels = [row[0] for row in man_track_rows]
    if positions is None:
        positions = {L: (size // 2, 8 + 12 * i) for i, L in enumerate(labels)}

    for t in range(n_frames):
        Image.fromarray(np.zeros((size, size), np.uint8)).save(raw / f"t{t:03d}.tif")
        img = np.zeros((size, size), dtype=np.uint16)
        for L in labels:
            if t in presence.get(L, set()):
                p = positions[L]
                cy, cx = p[t] if isinstance(p, dict) else p
                _stamp(img, L, cy, cx, radius)
        Image.fromarray(img).save(seg / f"man_seg{t:03d}.tif")
    return root


def build(tmp_path, dataset="Synth-A", seq="01", n_frames=20, rows=None,
          presence=None, positions=None, filters=None, contact_px=6,
          pseudo_split=False):
    """Discover + scan + table, in one call."""
    make_ctc_video(tmp_path, dataset, seq, n_frames, rows, presence, positions)
    video = discover_video(tmp_path, dataset, seq)
    scans = scan_video(video.seg_paths, contact_px, workers=1, show_progress=False)
    table = build_video_table(video, scans, filters or TrackFilters(),
                              contact_px, 2 * RADIUS + 1, pseudo_split=pseudo_split)
    return video, scans, table


def _all_present(rows, n_frames):
    return {L: set(range(B, E + 1)) for L, B, E, _ in rows}


# ---------------------------------------------------------------------------

class TestManTrackParsing:
    def test_unsorted_and_sparse_labels(self, tmp_path):
        p = tmp_path / "man_track.txt"
        p.write_text("22215 5 19 0\n3 0 19 0\n\n900 0 19 3\n")
        tracks = parse_man_track(p)
        assert set(tracks) == {3, 900, 22215}
        assert tracks[22215] == {"birth": 5, "end": 19, "parent": 0}
        assert tracks[900]["parent"] == 3


class TestLineage:
    def test_single_child_relink_is_not_a_mitosis(self, tmp_path):
        """A nonzero parent does not imply two children (`14 39 91 12`)."""
        rows = [(1, 0, 9, 0), (2, 10, 19, 1)]
        _, _, table = build(tmp_path, rows=rows,
                            presence=_all_present(rows, 20))
        by_label = {int(L): i for i, L in enumerate(table.labels)}
        assert table.n_children[by_label[1]] == 1     # relink, not mitosis
        assert table.n_children[by_label[2]] == 0
        # The child still gets its own slot under new_slots.
        assert by_label[1] != by_label[2]
        assert table.parent_slot[by_label[2]] == by_label[1]

    def test_mitosis_masks_parent_and_daughters(self, tmp_path):
        """Parent False from the daughters' birth; daughters False before it."""
        rows = [(1, 0, 9, 0), (2, 10, 19, 1), (3, 10, 19, 1)]
        pos = {1: (32, 20), 2: (24, 20), 3: (40, 20)}
        _, _, table = build(tmp_path, rows=rows,
                            presence=_all_present(rows, 20), positions=pos)
        by_label = {int(L): i for i, L in enumerate(table.labels)}
        p, c1, c2 = by_label[1], by_label[2], by_label[3]

        assert table.n_children[p] == 2
        assert table.valid[:10, p].all() and not table.valid[10:, p].any()
        for c in (c1, c2):
            assert not table.valid[:10, c].any() and table.valid[10:, c].all()
            assert table.parent_slot[c] == p

        # No frame has both the parent and a daughter valid — so no velocity is
        # ever computed across the mitosis.
        assert not (table.valid[:, p] & table.valid[:, c1]).any()

    def test_orphaned_child_gets_parent_slot_minus_one(self, tmp_path):
        """A filter can orphan a child; parent_slot = -1 must be handled."""
        rows = [(1, 0, 3, 0), (2, 4, 19, 1)]     # parent lives only 4 frames
        _, _, table = build(tmp_path, rows=rows, presence=_all_present(rows, 20),
                            filters=TrackFilters(min_track_length=10))
        assert table.labels.tolist() == [2]
        assert table.parent_slot.tolist() == [-1]
        assert table.n_children.tolist() == [0]


class TestValidity:
    def test_interior_seg_gap_invalidates_frames_without_splitting(self, tmp_path):
        """ST occasionally misses a cell TRA has. That is a gap, not a split.

        Splitting would give a slot whose parameter_embedding row sees only a
        fraction of its data.
        """
        rows = [(1, 0, 19, 0)]
        presence = {1: set(range(20)) - {7, 8, 9}}
        _, _, table = build(tmp_path, rows=rows, presence=presence)
        assert table.n_slots == 1
        assert not table.valid[7:10, 0].any()
        assert table.valid[:7, 0].all() and table.valid[10:, 0].all()

    def test_min_track_length_drops_short_tracks(self, tmp_path):
        rows = [(1, 0, 19, 0), (2, 0, 4, 0)]
        _, _, table = build(tmp_path, rows=rows, presence=_all_present(rows, 20),
                            filters=TrackFilters(min_track_length=10))
        assert table.labels.tolist() == [1]

    def test_min_area_px_invalidates_specks(self, tmp_path):
        rows = [(1, 0, 19, 0), (2, 0, 19, 0)]
        pos = {1: (20, 20), 2: (44, 44)}
        # Label 2 is a 1-px speck; label 1 is a normal disc.
        make_ctc_video(tmp_path, "Synth-A", "01", 20, rows,
                       _all_present(rows, 20), positions=pos, radius=RADIUS)
        # Re-stamp label 2 as a single pixel in every frame.
        seg = tmp_path / "Synth-A" / "01_ST" / "SEG"
        for t in range(20):
            img = np.array(Image.open(seg / f"man_seg{t:03d}.tif"))
            img[img == 2] = 0
            img[44, 44] = 2
            Image.fromarray(img).save(seg / f"man_seg{t:03d}.tif")

        video = discover_video(tmp_path, "Synth-A", "01")
        scans = scan_video(video.seg_paths, 6, workers=1, show_progress=False)
        table = build_video_table(
            video, scans, TrackFilters(min_track_length=1, min_area_px=8),
            6, 9)
        assert table.labels.tolist() == [1]

    def test_all_tracks_filtered_out_raises_clearly(self, tmp_path):
        rows = [(1, 0, 19, 0)]
        with pytest.raises(ValueError, match="no tracks survive the filters"):
            build(tmp_path, rows=rows, presence=_all_present(rows, 20),
                  filters=TrackFilters(min_track_length=1, min_area_px=10_000))

    def test_border_policy_invalidate_frame(self, tmp_path):
        rows = [(1, 0, 19, 0)]
        # Sits on the border for the first 5 frames, then moves inside.
        pos = {1: {t: ((1, 32) if t < 5 else (32, 32)) for t in range(20)}}
        _, _, table = build(tmp_path, rows=rows, presence=_all_present(rows, 20),
                            positions=pos,
                            filters=TrackFilters(min_track_length=1))
        assert not table.valid[:5, 0].any()
        assert table.valid[5:, 0].all()

    def test_border_policy_drop_track(self, tmp_path):
        rows = [(1, 0, 19, 0), (2, 0, 19, 0)]
        pos = {1: (1, 20), 2: (32, 40)}
        with pytest.raises(ValueError):
            build(tmp_path / "a", rows=[rows[0]],
                  presence={1: set(range(20))}, positions={1: (1, 20)},
                  filters=TrackFilters(min_track_length=1,
                                       border_policy="drop_track"))
        _, _, table = build(tmp_path / "b", rows=rows,
                            presence=_all_present(rows, 20), positions=pos,
                            filters=TrackFilters(min_track_length=1,
                                                 border_policy="drop_track"))
        assert table.labels.tolist() == [2]

    def test_bad_border_policy_rejected(self):
        with pytest.raises(ValueError, match="border_policy"):
            TrackFilters(border_policy="nope")


class TestRoi:
    """`roi` crops the label image, so its edge behaves exactly like the FOV's."""

    ROI = (8, 64, 0, 32)   # (y0, y1, x0, x1): rows 8.., left half

    def _build(self, tmp_path):
        rows = [(1, 0, 19, 0), (2, 0, 19, 0), (3, 0, 19, 0)]
        # 1 inside, 2 straddling the x=32 crop edge, 3 entirely outside.
        pos = {1: (32, 12), 2: (32, 32), 3: (32, 50)}
        make_ctc_video(tmp_path, "Synth-A", "01", 20, rows,
                       _all_present(rows, 20), pos)
        video = discover_video(tmp_path, "Synth-A", "01", roi=self.ROI)
        scans = scan_video(video.seg_paths, 6, workers=1, show_progress=False)
        table = build_video_table(video, scans, TrackFilters(min_track_length=1),
                                  6, 2 * RADIUS + 1)
        return video, table

    def test_fov_is_the_crop(self, tmp_path):
        video, _ = self._build(tmp_path)
        assert (video.height, video.width) == (56, 32)
        assert video.roi == self.ROI
        assert all(isinstance(p, SegFrame) for p in video.seg_paths)
        assert load_seg(video.seg_paths[0]).shape == (56, 32)

    def test_crop_edge_is_a_border_and_outside_cells_vanish(self, tmp_path):
        _, table = self._build(tmp_path)
        # 3 never appears; 2 is clipped by the crop edge on every frame, so
        # invalidate_frame leaves it no valid frames and it is dropped.
        assert table.labels.tolist() == [1]
        # Centroids are in cropped coordinates.
        np.testing.assert_allclose(table.centroids[0, 0], (24, 12))

    def test_seg_frame_survives_pickling_into_workers(self, tmp_path):
        video, _ = self._build(tmp_path)
        ref = pickle.loads(pickle.dumps(video.seg_paths[3]))
        np.testing.assert_array_equal(load_seg(ref), load_seg(video.seg_paths[3]))

    def test_roi_outside_frame_rejected(self, tmp_path):
        rows = [(1, 0, 19, 0)]
        make_ctc_video(tmp_path, "Synth-A", "01", 20, rows, _all_present(rows, 20))
        with pytest.raises(ValueError, match="roi"):
            discover_video(tmp_path, "Synth-A", "01", roi=(0, 65, 0, 32))


class TestGlobalAssembly:
    def _two_videos(self, tmp_path):
        rows_a = [(1, 0, 19, 0), (2, 0, 19, 0), (5, 0, 19, 0)]
        rows_b = [(7, 0, 19, 0), (9, 0, 19, 0)]
        make_ctc_video(tmp_path, "Synth-A", "01", 20, rows_a,
                       _all_present(rows_a, 20))
        make_ctc_video(tmp_path, "Synth-B", "01", 20, rows_b,
                       _all_present(rows_b, 20))
        videos = [discover_video(tmp_path, "Synth-A", "01"),
                  discover_video(tmp_path, "Synth-B", "01")]
        scans = [scan_video(v.seg_paths, 6, workers=1, show_progress=False)
                 for v in videos]
        return videos, scans

    def test_slot_ranges_are_disjoint(self, tmp_path):
        videos, scans = self._two_videos(tmp_path)
        tracks = build_global_tables(videos, scans, TrackFilters(), [6, 6], [9, 9])
        assert tracks.N == 5
        assert [(s.start, s.stop) for s in tracks.segments] == [(0, 3), (3, 5)]
        tracks.assert_slot_ranges_disjoint()          # also runs inside build
        assert tracks.tables[0].offset == 0
        assert tracks.tables[1].offset == 3

    def test_deterministic_across_two_builds(self, tmp_path):
        videos, scans = self._two_videos(tmp_path)
        a = build_global_tables(videos, scans, TrackFilters(), [6, 6], [9, 9])
        b = build_global_tables(videos, scans, TrackFilters(), [6, 6], [9, 9])
        for ta, tb in zip(a.tables, b.tables):
            np.testing.assert_array_equal(ta.labels, tb.labels)
            np.testing.assert_array_equal(ta.valid, tb.valid)
        assert [s.start for s in a.segments] == [s.start for s in b.segments]

    def test_metadata_identical_across_windows(self, tmp_path):
        """sim_metadata is a pure function of the GLOBAL table, never the window.

        verify_metadata_consistency compares every key across all trajectories
        with torch.allclose, so any per-window quantity would raise.
        """
        videos, scans = self._two_videos(tmp_path)
        tracks = build_global_tables(videos, scans, TrackFilters(), [6, 6], [9, 9])
        meta = build_slot_metadata(tracks, dt=1.0)
        for key, arr in meta.items():
            assert arr.shape == (tracks.N,), key
        # Video identity is integer-coded and disjoint by slot range.
        assert meta["video_id"].tolist() == [0, 0, 0, 1, 1]
        assert meta["dataset_id"].tolist() == [0, 0, 0, 1, 1]
        assert (meta["rs"] > 0).all()

    def test_pseudo_split_makes_two_segments_over_one_video(self, tmp_path):
        rows = [(i, 0, 19, 0) for i in range(1, 7)]
        make_ctc_video(tmp_path, "Synth-A", "01", 20, rows,
                       _all_present(rows, 20),
                       positions={i: (10 + 8 * (i % 3), 10 + 15 * (i // 3))
                                  for i in range(1, 7)})
        video = discover_video(tmp_path, "Synth-A", "01")
        scans = scan_video(video.seg_paths, 6, workers=1, show_progress=False)
        tracks = build_global_tables([video], [scans], TrackFilters(), [6], [9],
                                     pseudo_split=0)
        assert len(tracks.segments) == 2
        assert [(s.start, s.stop) for s in tracks.segments] == [(0, 3), (3, 6)]
        # Same source video -> same dataset/sequence codes, different video_id.
        assert {s.dataset_id for s in tracks.segments} == {0}
        assert [s.video_id for s in tracks.segments] == [0, 1]


class TestWindowing:
    def test_n_windows_leaves_room_for_next_states(self):
        assert n_windows_for(33, 32) == 1     # needs wT + T + 1 <= n_frames
        assert n_windows_for(32, 32) == 0
        assert n_windows_for(65, 32) == 2
        assert n_windows_for(65, 32, max_windows=1) == 1

    def test_window_order_is_round_robin(self):
        """Any prefix of n_trajectories must cover every video."""
        order = window_order([2, 3, 1])
        assert order[:3] == [(0, 0), (1, 0), (2, 0)]
        assert order == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (1, 2)]
        assert {v for v, _ in order[:3]} == {0, 1, 2}
