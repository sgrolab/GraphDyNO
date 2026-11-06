"""Tests for the dilation contact graph (real_data/neighbors.py).

The criterion under test: dilate each cell's mask by `contact_px`, draw an edge
to every *other* mask it overlaps. Equivalently, two cells are neighbours iff
the gap between their pixel sets is <= contact_px. There is NO occlusion.

Run with: pytest tests/test_ctc_neighbors.py -v
"""

import numpy as np
import pytest
from PIL import Image

from graph_dyno.real_data.neighbors import (
    frame_adjacency,
    frame_adjacency_edt,
    frame_adjacency_shift,
    frame_geometry,
    half_disk_offsets,
    scan_video,
)


# ---- fixtures ------------------------------------------------------------

def _disc(img, label, cy, cx, radius):
    """Stamp a filled disc of `label` into `img`."""
    H, W = img.shape
    ys, xs = np.ogrid[:H, :W]
    img[(ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2] = label
    return img


def _two_discs(gap, radius=6, size=120):
    """Two discs whose closest pixels are exactly `gap` apart (along a row)."""
    img = np.zeros((size, size), dtype=np.uint16)
    cy = size // 2
    cx1 = size // 3
    cx2 = cx1 + 2 * radius + gap
    _disc(img, 1, cy, cx1, radius)
    _disc(img, 2, cy, cx2, radius)
    return img


def _as_set(pairs):
    return {(int(a), int(b)) for a, b in pairs}


# ---- the criterion -------------------------------------------------------

class TestContactBoundary:
    @pytest.mark.parametrize("gap", [1, 2, 3, 5, 8])
    def test_connect_disconnect_boundary_at_contact_px(self, gap):
        """Edge appears exactly when contact_px reaches the surface gap.

        Allow +/-1 px of discretisation slop: a rasterized disc's true nearest
        pixel pair is not exactly `gap` apart in every direction.
        """
        img = _two_discs(gap)
        # Well below the gap -> no edge.
        assert _as_set(frame_adjacency(img, gap - 2)) == set()
        # Comfortably above -> edge.
        assert _as_set(frame_adjacency(img, gap + 1)) == {(1, 2)}

    def test_overlapping_masks_always_connect(self):
        img = np.zeros((60, 60), dtype=np.uint16)
        _disc(img, 1, 30, 25, 10)
        _disc(img, 2, 30, 32, 10)  # centres 7 apart, radii 10 -> touching
        assert _as_set(frame_adjacency(img, 1)) == {(1, 2)}

    def test_three_collinear_discs_yield_the_end_pair(self):
        """No occlusion: a cell between two others does not block their edge.

        This is the test that pins the chosen semantics against the rejected
        `skimage.segmentation.expand_labels` one, whose Voronoi growth *is*
        occluded and never produces the end-to-end pair at any distance.
        """
        img = np.zeros((60, 200), dtype=np.uint16)
        r, cy = 6, 30
        for i, cx in enumerate((30, 100, 170), start=1):
            _disc(img, i, cy, cx, r)

        # Small radius: only the adjacent pairs are within reach... of nothing.
        assert _as_set(frame_adjacency(img, 5)) == set()
        # Reaching neighbour-to-neighbour (gap 70 - 2*6 = 58).
        assert _as_set(frame_adjacency(img, 60)) == {(1, 2), (2, 3)}
        # Reaching end-to-end (gap 140 - 12 = 128) — the middle disc does NOT
        # suppress the (1, 3) edge.
        assert _as_set(frame_adjacency(img, 130)) == {(1, 2), (1, 3), (2, 3)}

    def test_chain_of_five(self):
        img = np.zeros((60, 260), dtype=np.uint16)
        for i, cx in enumerate(range(20, 260, 50), start=1):
            _disc(img, i, 30, cx, 6)
        pairs = _as_set(frame_adjacency(img, 40))  # gap between neighbours = 38
        assert pairs == {(1, 2), (2, 3), (3, 4), (4, 5)}

    def test_single_cell_and_empty_image(self):
        single = np.zeros((40, 40), dtype=np.uint16)
        _disc(single, 7, 20, 20, 5)
        assert frame_adjacency(single, 20).shape == (0, 2)
        assert frame_adjacency(np.zeros((40, 40), np.uint16), 10).shape == (0, 2)

    def test_border_clipped_disc(self):
        """A disc cut by the image edge still participates normally."""
        img = np.zeros((60, 60), dtype=np.uint16)
        _disc(img, 1, 3, 30, 8)     # clipped at the top
        _disc(img, 2, 30, 30, 8)
        assert _as_set(frame_adjacency(img, 30)) == {(1, 2)}
        geom = frame_geometry(img)
        assert geom.touches_border.tolist() == [True, False]


class TestImplementationsAgree:
    def test_half_disk_offsets_are_a_half_disk(self):
        offs = half_disk_offsets(3)
        assert (0, 0) not in offs
        for dy, dx in offs:
            assert dy * dy + dx * dx <= 9
            assert (-dy, -dx) not in offs   # exactly one of each antipodal pair
        # Full disk minus origin, halved.
        full = sum(1 for dy in range(-3, 4) for dx in range(-3, 4)
                   if (dy or dx) and dy * dy + dx * dx <= 9)
        assert len(offs) == full // 2

    @pytest.mark.parametrize("r", [2, 3, 5])
    def test_edt_matches_shift(self, r):
        """The production EDT form and the vectorized shift form are identical."""
        rng = np.random.default_rng(0)
        for _ in range(30):
            lbl = rng.integers(0, 6, size=(40, 40)).astype(np.uint16)
            a = _as_set(frame_adjacency_edt(lbl, r))
            b = _as_set(frame_adjacency_shift(lbl, r))
            assert a == b

    def test_sparse_noncontiguous_labels_survive_int64_key(self):
        """PhC-C2DL-PSC's max label is 22215; base*label overflows uint16."""
        img = np.zeros((60, 60), dtype=np.uint16)
        _disc(img, 3, 30, 20, 6)
        _disc(img, 22215, 30, 33, 6)   # surface gap 1 px
        assert _as_set(frame_adjacency_shift(img, 5)) == {(3, 22215)}
        assert _as_set(frame_adjacency_edt(img, 5)) == {(3, 22215)}


class TestGeometry:
    def test_centroid_area_and_half_extent(self):
        img = np.zeros((80, 80), dtype=np.uint16)
        _disc(img, 4, 40, 30, 7)
        g = frame_geometry(img)
        assert g.labels.tolist() == [4]
        assert g.centroids[0] == pytest.approx([40.0, 30.0], abs=0.01)
        assert g.areas[0] == int((img == 4).sum())
        assert g.bbox_sides[0] == 15
        assert g.half_extents[0] == pytest.approx(7.0, abs=0.01)
        assert g.aspect_ratios[0] == pytest.approx(1.0)

    def test_half_extent_is_from_the_centroid_not_half_the_bbox(self):
        """They differ for a concave/off-centre shape — r_max needs the former."""
        img = np.zeros((60, 60), dtype=np.uint16)
        img[20:24, 10:50] = 1        # long thin bar
        img[20:50, 10:14] = 1        # plus a leg -> centroid pulled off centre
        g = frame_geometry(img)
        cy, cx = g.centroids[0]
        ys, xs = np.nonzero(img)
        expected = max(np.abs(ys - cy).max(), np.abs(xs - cx).max())
        assert g.half_extents[0] == pytest.approx(expected)
        # The two really do differ — using half the bbox would under-size r_max.
        assert abs(g.half_extents[0] - g.bbox_sides[0] / 2) > 1.0


class TestScanVideo:
    def test_serial_matches_parallel(self, tmp_path):
        rng = np.random.default_rng(3)
        paths = []
        for t in range(12):
            img = np.zeros((64, 64), dtype=np.uint16)
            for label in range(1, 5):
                cy, cx = rng.integers(8, 56, size=2)
                _disc(img, label, cy, cx, 5)
            p = tmp_path / f"man_seg{t:03d}.tif"
            Image.fromarray(img).save(p)
            paths.append(p)

        serial = scan_video(paths, 6, workers=1, show_progress=False)
        parallel = scan_video(paths, 6, workers=8, show_progress=False)

        assert len(serial) == len(parallel) == 12
        for a, b in zip(serial, parallel):
            np.testing.assert_array_equal(a.labels, b.labels)
            np.testing.assert_allclose(a.centroids, b.centroids)
            np.testing.assert_array_equal(a.areas, b.areas)
            np.testing.assert_array_equal(a.pairs, b.pairs)

    def test_missing_frames_scan_as_empty(self, tmp_path):
        img = np.zeros((32, 32), dtype=np.uint16)
        _disc(img, 1, 16, 16, 4)
        p = tmp_path / "man_seg000.tif"
        Image.fromarray(img).save(p)
        out = scan_video([p, None], 3, workers=1, show_progress=False)
        assert out[0].labels.tolist() == [1]
        assert out[1].labels.size == 0
