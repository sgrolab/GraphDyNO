"""
Tests for precompute_dino_embeddings.py and dataloader DINO integration.

Tests are organized into:
  - SDF conversion tests (no DINO model needed, fast)
  - _SimMaskDataset tests
  - Cache path tests
  - Multiprocess SDF batch tests
  - End-to-end precompute + dataloader integration (uses mock trajectory data)
  - Visual verification plots (saved to tests/artifacts/)

Run with: pytest tests/test_precompute_dino.py -v -s
"""

import os
import tempfile

import cv2
import numpy as np
import pytest
import torch
from scipy.ndimage import distance_transform_edt
from omegaconf import OmegaConf

from graph_dyno.precompute.precompute_dino_embeddings import (
    _mask_to_sdf,
    _mask_to_sdf_worker,
    _compute_sdfs_multiprocess,
    _get_dino_cache_path,
    _SimMaskDataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def circle_mask():
    """64x64 binary mask with a centered circle of radius 15."""
    H, W = 64, 64
    mask = np.zeros((H, W), dtype=np.float32)
    cy, cx, r = H // 2, W // 2, 15
    ys, xs = np.ogrid[:H, :W]
    mask[((ys - cy) ** 2 + (xs - cx) ** 2) <= r ** 2] = 1.0
    return mask


@pytest.fixture
def off_center_mask():
    """64x64 binary mask with a circle near the top-left corner."""
    H, W = 64, 64
    mask = np.zeros((H, W), dtype=np.float32)
    cy, cx, r = 10, 12, 8
    ys, xs = np.ogrid[:H, :W]
    mask[((ys - cy) ** 2 + (xs - cx) ** 2) <= r ** 2] = 1.0
    return mask


@pytest.fixture
def ellipse_mask():
    """64x64 binary mask with an ellipse (non-square bbox)."""
    H, W = 64, 64
    mask = np.zeros((H, W), dtype=np.float32)
    cy, cx, ry, rx = H // 2, W // 2, 8, 20
    ys, xs = np.ogrid[:H, :W]
    mask[((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2 <= 1] = 1.0
    return mask


@pytest.fixture
def empty_mask():
    """64x64 all-zeros mask."""
    return np.zeros((64, 64), dtype=np.float32)


@pytest.fixture
def full_mask():
    """64x64 all-ones mask."""
    return np.ones((64, 64), dtype=np.float32)


@pytest.fixture
def mock_trajectory_cache():
    """Create a temporary cache directory with mock trajectory files and metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        T, N, H, W = 5, 3, 64, 64
        n_trajectories = 2

        # Build metadata matching the real format
        metadata = OmegaConf.create({
            "data": {
                "type": "simulation",
                "n_objects": N,
                "image_size": H,
                "dt": 0.01,
                "domain_size": 10.0,
                "seed": 0,
                "generator": {
                    "n_trajectories": n_trajectories,
                    "trajectory_length": T,
                    "seed_offset": 0,
                    "cache_dir": tmpdir,
                    "overwrite_existing": False,
                },
                "params": {
                    "name": "sphere_interaction",
                    "periodic_boundary": False,
                    "drag": 0.5,
                    "r_cutoff": 5,
                    "potential_D_e": 10,
                    "potential_a": 1.5,
                    "brownian_motion_std": 0.0,
                    "r_min": 0.5,
                    "r_max": 1.5,
                    "r_steps": 3,
                    "r_repeats": 1,
                    "growth_rate_min": 0.0,
                    "growth_rate_max": 0.0,
                    "growth_rate_steps": 3,
                    "growth_rate_repeats": 1,
                    "relaxation_steps": 0,
                    "relaxation_dt": 0.01,
                    "relaxation_drag": 5.0,
                },
            },
        })

        # Save metadata
        metadata_path = os.path.join(tmpdir, "metadata.yaml")
        with open(metadata_path, "w") as f:
            f.write(OmegaConf.to_yaml(metadata, resolve=True))

        # Compute cache_id
        from graph_dyno.data.trajectory_generator import generate_config_hash
        cache_id = generate_config_hash(metadata.data)

        # Generate mock trajectory files with circle masks
        for traj_idx in range(n_trajectories):
            channeled_images = torch.zeros(T, N, H, W)
            for t in range(T):
                for n in range(N):
                    # Place a circle with varying position/size per object
                    cy = 20 + n * 10 + t
                    cx = 20 + n * 8 + t * 2
                    r = 5 + n * 2
                    ys, xs = torch.meshgrid(
                        torch.arange(H, dtype=torch.float32),
                        torch.arange(W, dtype=torch.float32),
                        indexing="ij",
                    )
                    channeled_images[t, n] = (((ys - cy) ** 2 + (xs - cx) ** 2) <= r ** 2).float()

            # Build sparse W_interactions
            w_list = []
            for _ in range(T):
                indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
                values = torch.ones(2)
                w_sparse = torch.sparse_coo_tensor(indices, values, (N, N))
                w_list.append(w_sparse)
            W_interactions = torch.stack(w_list)

            trajectory_data = {
                "times": torch.linspace(0, 1, T),
                "states": torch.randn(T, N, 4),
                "next_states": torch.randn(T, N, 4),
                "forces": torch.randn(T, N, 4),
                "channeled_images": channeled_images,
                "W_interactions": W_interactions,
                "sim_metadata": {
                    "rs": torch.ones(N),
                    "growth_rates": torch.zeros(N),
                    "r_max": torch.tensor(1.5),
                },
            }

            filename = f"{cache_id}_{traj_idx:04d}.pt"
            torch.save(trajectory_data, os.path.join(tmpdir, filename))

        yield {
            "tmpdir": tmpdir,
            "cache_id": cache_id,
            "T": T,
            "N": N,
            "H": H,
            "W": W,
            "n_trajectories": n_trajectories,
            "metadata": metadata,
        }


# ---------------------------------------------------------------------------
# _mask_to_sdf tests
# ---------------------------------------------------------------------------

class TestMaskToSdf:
    """Tests for the SDF conversion function."""

    def test_circle_mask_produces_valid_sdf(self, circle_mask):
        result = _mask_to_sdf(circle_mask, target_size=224)
        assert result is not None
        sdf, crop_side = result

        assert sdf.shape == (224, 224)
        assert sdf.dtype == np.float16
        assert sdf.min() >= -1.0
        assert sdf.max() <= 1.0

    def test_circle_sdf_positive_inside(self, circle_mask):
        """Center of circle should have positive SDF (inside)."""
        sdf, _ = _mask_to_sdf(circle_mask, target_size=224)
        center = sdf[112, 112]
        assert center > 0, f"Center SDF should be positive (inside), got {center}"

    def test_circle_sdf_negative_outside(self, circle_mask):
        """Corner of image should have negative SDF (outside)."""
        sdf, _ = _mask_to_sdf(circle_mask, target_size=224)
        corner = sdf[0, 0]
        assert corner < 0, f"Corner SDF should be negative (outside), got {corner}"

    def test_crop_side_is_square(self, circle_mask):
        """crop_side should equal max(bbox_h, bbox_w)."""
        _, crop_side = _mask_to_sdf(circle_mask, target_size=224)
        ys, xs = np.nonzero(circle_mask)
        bbox_h = ys.max() - ys.min() + 1
        bbox_w = xs.max() - xs.min() + 1
        assert crop_side == max(bbox_h, bbox_w)

    def test_ellipse_crop_side_is_max_dim(self, ellipse_mask):
        """For ellipse (non-square bbox), crop_side should be the larger dimension."""
        _, crop_side = _mask_to_sdf(ellipse_mask, target_size=224)
        ys, xs = np.nonzero(ellipse_mask)
        bbox_h = ys.max() - ys.min() + 1
        bbox_w = xs.max() - xs.min() + 1
        assert bbox_h != bbox_w, "Ellipse bbox should not be square"
        assert crop_side == max(bbox_h, bbox_w)

    def test_ellipse_aspect_ratio_preserved(self, ellipse_mask):
        """Square crop should preserve aspect ratio — ellipse should not be distorted."""
        sdf, _ = _mask_to_sdf(ellipse_mask, target_size=224)
        # The object in the SDF should be wider than tall (matching original ellipse)
        # Check by finding the extent of positive SDF region
        positive = sdf > 0
        rows_with_positive = np.any(positive, axis=1)
        cols_with_positive = np.any(positive, axis=0)
        sdf_height = rows_with_positive.sum()
        sdf_width = cols_with_positive.sum()
        # Original ellipse: ry=8, rx=20, so width > height
        assert sdf_width > sdf_height, (
            f"Ellipse should be wider than tall in SDF, got {sdf_width}x{sdf_height}"
        )

    def test_off_center_crop_side(self, off_center_mask):
        """Off-center mask should still report correct square crop side."""
        _, crop_side = _mask_to_sdf(off_center_mask, target_size=224)
        ys, xs = np.nonzero(off_center_mask)
        bbox_h = ys.max() - ys.min() + 1
        bbox_w = xs.max() - xs.min() + 1
        assert crop_side == max(bbox_h, bbox_w)

    def test_empty_mask_returns_none(self, empty_mask):
        result = _mask_to_sdf(empty_mask, target_size=224)
        assert result is None

    def test_full_mask_positive_everywhere(self, full_mask):
        """A full mask should produce SDF > 0 everywhere (all inside)."""
        sdf, crop_side = _mask_to_sdf(full_mask, target_size=224)
        assert (sdf >= 0).all(), "Full mask should have non-negative SDF everywhere"
        assert crop_side == 64

    def test_small_target_size(self, circle_mask):
        """SDF works with target sizes other than 224."""
        sdf, _ = _mask_to_sdf(circle_mask, target_size=56)
        assert sdf.shape == (56, 56)
        assert sdf.dtype == np.float16


# ---------------------------------------------------------------------------
# _mask_to_sdf_worker tests
# ---------------------------------------------------------------------------

class TestMaskToSdfWorker:
    """Tests for the multiprocessing worker wrapper."""

    def test_normal_mask(self, circle_mask):
        sdf, crop_side = _mask_to_sdf_worker((circle_mask, 224))
        assert sdf.shape == (224, 224)
        assert crop_side > 0

    def test_empty_mask_fallback(self, empty_mask):
        """Empty mask should return all-outside SDF with crop_side=0."""
        sdf, crop_side = _mask_to_sdf_worker((empty_mask, 224))
        assert sdf.shape == (224, 224)
        assert (sdf == np.float16(-1.0)).all()
        assert crop_side == 0


# ---------------------------------------------------------------------------
# _compute_sdfs_multiprocess tests
# ---------------------------------------------------------------------------

class TestComputeSdfsMultiprocess:
    """Tests for the batched multiprocess SDF computation."""

    def test_batch_shapes(self, circle_mask, off_center_mask):
        masks = torch.tensor(np.stack([circle_mask, off_center_mask]))
        sdf_masks, crop_sides = _compute_sdfs_multiprocess(masks, target_size=224, num_workers=2)

        assert sdf_masks.shape == (2, 224, 224)
        assert crop_sides.shape == (2,)
        assert crop_sides.dtype == torch.int32

    def test_batch_with_empty_mask(self, circle_mask, empty_mask):
        masks = torch.tensor(np.stack([circle_mask, empty_mask]))
        sdf_masks, crop_sides = _compute_sdfs_multiprocess(masks, target_size=224, num_workers=2)

        # First mask should have valid SDF
        assert sdf_masks[0].max() > 0
        # Empty mask should be all -1
        assert (sdf_masks[1] == torch.tensor(-1.0, dtype=torch.float16)).all()
        # Empty mask crop_side should be 0
        assert crop_sides[1] == 0

    def test_single_worker_matches_multi(self, circle_mask, off_center_mask):
        masks = torch.tensor(np.stack([circle_mask, off_center_mask]))
        sdf_1, crops_1 = _compute_sdfs_multiprocess(masks, target_size=224, num_workers=1)
        sdf_2, crops_2 = _compute_sdfs_multiprocess(masks, target_size=224, num_workers=4)

        torch.testing.assert_close(sdf_1, sdf_2)
        torch.testing.assert_close(crops_1, crops_2)


# ---------------------------------------------------------------------------
# _SimMaskDataset tests
# ---------------------------------------------------------------------------

class TestSimMaskDataset:
    """Tests for the DINO-ready dataset wrapper."""

    def test_output_shape(self):
        sdf_masks = torch.randn(10, 224, 224).clamp(-1, 1)
        ds = _SimMaskDataset(sdf_masks)

        assert len(ds) == 10
        (rgb,) = ds[0]
        assert rgb.shape == (3, 224, 224)

    def test_output_range(self):
        """RGB output should be in [0, 1] when input SDF is in [-1, 1]."""
        sdf_masks = torch.tensor([[[-1.0, 0.0], [0.5, 1.0]]])  # (1, 2, 2)
        ds = _SimMaskDataset(sdf_masks)
        (rgb,) = ds[0]
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_channels_identical(self):
        """All 3 RGB channels should be identical (grayscale expansion)."""
        sdf_masks = torch.randn(1, 224, 224).clamp(-1, 1)
        ds = _SimMaskDataset(sdf_masks)
        (rgb,) = ds[0]
        torch.testing.assert_close(rgb[0], rgb[1])
        torch.testing.assert_close(rgb[1], rgb[2])


# ---------------------------------------------------------------------------
# _get_dino_cache_path tests
# ---------------------------------------------------------------------------

class TestGetDinoCachePath:

    def test_format(self):
        path = _get_dino_cache_path("/data/cache", "abc123", 0, "SP")
        assert path == "/data/cache/abc123_0000_dino_SP.pt"

    def test_large_index(self):
        path = _get_dino_cache_path("/data/cache", "abc123", 9999, "B")
        assert path == "/data/cache/abc123_9999_dino_B.pt"

    def test_different_models(self):
        for key in ["S", "SP", "B", "L", "H", "F"]:
            path = _get_dino_cache_path("/tmp", "hash", 5, key)
            assert f"_dino_{key}.pt" in path


# ---------------------------------------------------------------------------
# End-to-end precompute + dataloader integration
# ---------------------------------------------------------------------------

class TestPrecomputeAndDataloader:
    """Integration tests using mock trajectory data (no real DINO model)."""

    def _create_mock_dino_companion_files(self, info, D=384):
        """Create fake DINO companion files for dataloader testing."""
        for traj_idx in range(info["n_trajectories"]):
            T, N = info["T"], info["N"]
            companion = {
                "cls_tokens": torch.randn(T, N, D),
                "patch_means": torch.randn(T, N, D),
                "crop_sides": torch.randint(10, 30, (T, N), dtype=torch.int32),
            }
            path = _get_dino_cache_path(info["tmpdir"], info["cache_id"], traj_idx, "SP")
            torch.save(companion, path)
        return D

    def test_sdf_conversion_on_mock_masks(self, mock_trajectory_cache):
        """Verify SDF conversion works on the mock trajectory channeled_images."""
        info = mock_trajectory_cache

        # Load a trajectory and convert its masks
        from graph_dyno.data.trajectory_generator import generate_config_hash
        cache_id = generate_config_hash(info["metadata"].data)
        traj_path = os.path.join(info["tmpdir"], f"{cache_id}_0000.pt")
        traj_data = torch.load(traj_path, map_location="cpu")

        channeled_images = traj_data["channeled_images"]  # (T, N, H, W)
        T, N, H, W = channeled_images.shape

        flat_masks = channeled_images.reshape(T * N, H, W)
        sdf_masks, crop_sides = _compute_sdfs_multiprocess(flat_masks, target_size=224, num_workers=2)

        assert sdf_masks.shape == (T * N, 224, 224)
        assert crop_sides.shape == (T * N,)

        # All masks have circles, so all should have positive crop sides
        assert (crop_sides > 0).all(), "All mock masks contain circles, crop_sides should be > 0"

        # SDFs should have both positive and negative values (inside/outside)
        for i in range(T * N):
            assert sdf_masks[i].max() > 0, f"Mask {i} SDF has no positive values (no inside)"
            assert sdf_masks[i].min() < 0, f"Mask {i} SDF has no negative values (no outside)"


# ---------------------------------------------------------------------------
# Visual verification
# ---------------------------------------------------------------------------

def _make_star(H, W, cy, cx, outer_r, inner_r, n_points=5):
    """Binary star mask."""
    angles = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False) - np.pi / 2
    pts = []
    for i, a in enumerate(angles):
        r = outer_r if i % 2 == 0 else inner_r
        pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    pts = np.array(pts, dtype=np.int32)
    mask = np.zeros((H, W), dtype=np.float32)
    cv2.fillPoly(mask, [pts], 1.0)
    return mask


def _make_blob(H, W, cy, cx, base_r, seed=42):
    """Binary irregular blob via random radial perturbation."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, 360)
    radii = np.ones_like(angles) * base_r
    for k in range(2, 10):
        amp = rng.uniform(0.05, 0.25) * base_r
        phase = rng.uniform(0, 2 * np.pi)
        radii += amp * np.sin(k * angles + phase)
    pts = np.stack([
        cx + radii * np.cos(angles),
        cy + radii * np.sin(angles),
    ], axis=-1).astype(np.int32)
    mask = np.zeros((H, W), dtype=np.float32)
    cv2.fillPoly(mask, [pts], 1.0)
    return mask


def _make_ellipse(H, W, cy, cx, ry, rx):
    """Binary ellipse mask."""
    ys, xs = np.ogrid[:H, :W]
    return (((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2 <= 1).astype(np.float32)


def _make_circle(H, W, cy, cx, r):
    """Binary circle mask."""
    ys, xs = np.ogrid[:H, :W]
    return (((ys - cy) ** 2 + (xs - cx) ** 2) <= r ** 2).astype(np.float32)


class TestSdfVisualization:
    """Generate verification plots for visual inspection.

    Plots are saved to tests/artifacts/ and NOT checked programmatically.
    """

    ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

    def _ensure_artifact_dir(self):
        os.makedirs(self.ARTIFACT_DIR, exist_ok=True)

    def test_visualize_various_shapes(self):
        """Produce a grid showing mask → SDF for circles, ellipse, star, blobs."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        self._ensure_artifact_dir()

        H, W = 64, 64
        c = H // 2
        masks = {
            "circle r=5": _make_circle(H, W, c, c, 5),
            "circle r=10": _make_circle(H, W, c, c, 10),
            "circle r=20": _make_circle(H, W, c, c, 20),
            "circle r=28": _make_circle(H, W, c, c, 28),
            "small off-center": _make_circle(H, W, 10, 12, 6),
            "ellipse 8x20": _make_ellipse(H, W, c, c, 8, 20),
            "star": _make_star(H, W, c, c, 22, 10),
            "blob seed=0": _make_blob(H, W, c, c, 18, seed=0),
            "blob seed=7": _make_blob(H, W, c, c, 15, seed=7),
            "blob seed=99": _make_blob(H, W, c, c, 20, seed=99),
        }

        n = len(masks)
        fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
        fig.suptitle("Binary Mask -> Square Crop -> SDF", fontsize=14, y=1.0)

        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        for row, (name, mask) in enumerate(masks.items()):
            result = _mask_to_sdf(mask, target_size=224)

            # Column 0: original mask
            axes[row, 0].imshow(mask, cmap="gray", origin="lower", vmin=0, vmax=1)
            axes[row, 0].set_title(f"{name} ({H}x{W})", fontsize=9)
            axes[row, 0].axis("off")

            if result is None:
                for c_idx in (1, 2):
                    axes[row, c_idx].text(0.5, 0.5, "empty", ha="center", va="center",
                                          transform=axes[row, c_idx].transAxes)
                    axes[row, c_idx].axis("off")
                continue

            sdf, crop_side = result

            # Column 1: square crop region centered (visualize what gets resized)
            ys, xs = np.nonzero(mask)
            y0, y1_b = int(ys.min()), int(ys.max()) + 1
            x0, x1_b = int(xs.min()), int(xs.max()) + 1
            side = max(y1_b - y0, x1_b - x0)
            cy, cx = (y0 + y1_b) // 2, (x0 + x1_b) // 2
            y0 = max(0, min(cy - side // 2, H - side))
            x0 = max(0, min(cx - side // 2, W - side))
            sq_crop = np.zeros((side, side), dtype=np.float32)
            ch = min(side, H - y0)
            cw = min(side, W - x0)
            sq_crop[:ch, :cw] = mask[y0:y0 + ch, x0:x0 + cw]

            axes[row, 1].imshow(sq_crop, cmap="gray", origin="lower", vmin=0, vmax=1)
            axes[row, 1].set_title(f"square crop ({side}x{side})", fontsize=9)
            axes[row, 1].axis("off")

            # Column 2: SDF
            sdf_f32 = sdf.astype(np.float32)
            axes[row, 2].imshow(sdf_f32, cmap="RdBu", norm=norm, origin="lower")
            axes[row, 2].contour(sdf_f32, levels=[0], colors="black", linewidths=0.8)
            axes[row, 2].set_title(
                f"SDF (224x224) [{sdf_f32.min():.2f}, {sdf_f32.max():.2f}]", fontsize=9
            )
            axes[row, 2].axis("off")

        plt.tight_layout()
        out_path = os.path.join(self.ARTIFACT_DIR, "sdf_shapes.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved: {out_path}")

    def test_visualize_different_circle_sizes(self):
        """Show that circles of different sizes all produce round SDFs (aspect preserved)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        self._ensure_artifact_dir()

        H, W = 64, 64
        radii = [3, 5, 8, 12, 18, 25]
        fig, axes = plt.subplots(len(radii), 3, figsize=(10, 2.8 * len(radii)))
        fig.suptitle("Circle Radius Sweep — Square Crop SDF", fontsize=13, y=1.0)
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        for row, r in enumerate(radii):
            mask = _make_circle(H, W, H // 2, W // 2, r)
            sdf, crop_side = _mask_to_sdf(mask, target_size=224)

            axes[row, 0].imshow(mask, cmap="gray", origin="lower", vmin=0, vmax=1)
            axes[row, 0].set_title(f"r={r} ({H}x{W})", fontsize=9)
            axes[row, 0].axis("off")

            # Show the square crop (centered)
            ys, xs = np.nonzero(mask)
            y0, y1_b = int(ys.min()), int(ys.max()) + 1
            x0, x1_b = int(xs.min()), int(xs.max()) + 1
            side = crop_side
            cy, cx = (y0 + y1_b) // 2, (x0 + x1_b) // 2
            y0 = max(0, min(cy - side // 2, H - side))
            x0 = max(0, min(cx - side // 2, W - side))
            sq_crop = np.zeros((side, side), dtype=np.float32)
            ch = min(side, H - y0)
            cw = min(side, W - x0)
            sq_crop[:ch, :cw] = mask[y0:y0 + ch, x0:x0 + cw]

            axes[row, 1].imshow(sq_crop, cmap="gray", origin="lower", vmin=0, vmax=1)
            axes[row, 1].set_title(f"crop ({side}x{side})", fontsize=9)
            axes[row, 1].axis("off")

            sdf_f32 = sdf.astype(np.float32)
            axes[row, 2].imshow(sdf_f32, cmap="RdBu", norm=norm, origin="lower")
            axes[row, 2].contour(sdf_f32, levels=[0], colors="black", linewidths=0.8)
            axes[row, 2].set_title(f"SDF [{sdf_f32.min():.2f}, {sdf_f32.max():.2f}]", fontsize=9)
            axes[row, 2].axis("off")

        plt.tight_layout()
        out_path = os.path.join(self.ARTIFACT_DIR, "sdf_circle_sizes.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved: {out_path}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
