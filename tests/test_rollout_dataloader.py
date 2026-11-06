"""
Tests for multi-step rollout dataloader changes.

Tests:
- rollout_steps=1 backward compatibility (same format as before)
- rollout_steps>1 returns sequence keys with correct shapes
- None returned for timesteps without enough room
- sparse_collate_fn filters out None samples
- rollout_steps mutability
- Sequence data values match single-step lookups

Run with: pytest tests/test_rollout_dataloader.py -v -s
"""

import pytest
import os
import torch
from omegaconf import OmegaConf
from graph_dyno.data.dataloader import TrajectoryDatasetFromFolder, sparse_collate_fn
from graph_dyno.data.trajectory_generator import generate_config_hash


# ---- Test constants ----
N_TRAJECTORIES = 3
TRAJECTORY_LENGTH = 20
N_OBJECTS = 5
FEATURE_DIM = 4
IMAGE_SIZE = 8
SPATIAL_DIM = 2


def _make_mock_data(tmp_path):
    """Create mock trajectory .pt files and metadata.yaml in tmp_path."""
    data_cfg = OmegaConf.create({
        "type": "simulation",
        "n_objects": N_OBJECTS,
        "image_size": IMAGE_SIZE,
        "domain_size": 10.0,
        "dt": 0.01,
        "seed": 0,
        "params": {
            "name": "mock",
            "periodic_boundary": False,
        },
        "generator": {
            "n_trajectories": N_TRAJECTORIES,
            "trajectory_length": TRAJECTORY_LENGTH,
            "seed_offset": 0,
            "cache_dir": str(tmp_path),
            "overwrite_existing": False,
        },
    })

    cache_id = generate_config_hash(data_cfg)

    # Save metadata.yaml
    metadata = OmegaConf.create({"data": data_cfg})
    OmegaConf.save(metadata, os.path.join(tmp_path, "metadata.yaml"))

    # Create trajectory .pt files with deterministic data
    for traj_idx in range(N_TRAJECTORIES):
        torch.manual_seed(traj_idx)
        T, N, F = TRAJECTORY_LENGTH, N_OBJECTS, FEATURE_DIM
        H = W = IMAGE_SIZE

        trajectory_data = {
            "times": torch.arange(T, dtype=torch.float32),
            "states": torch.randn(T, N, F),
            "next_states": torch.randn(T, N, F),
            "forces": torch.randn(T, N, F),
            "channeled_images": torch.rand(T, N, H, W),
            "W_interactions": torch.zeros(T, N, N),  # dense for simplicity
            "sim_metadata": {
                "rs": torch.ones(N),
                "growth_rates": torch.zeros(N),
                "r_max": torch.tensor(1.5),
            },
        }

        filename = f"{cache_id}_{traj_idx:04d}.pt"
        torch.save(trajectory_data, os.path.join(tmp_path, filename))

    return data_cfg, cache_id


@pytest.fixture
def mock_dataset(tmp_path):
    """Create a TrajectoryDatasetFromFolder with mock data."""
    _make_mock_data(tmp_path)
    ds = TrajectoryDatasetFromFolder(
        folder_path=str(tmp_path),
        enable_gpu_cache=False,
    )
    return ds


# ===========================================================================
# Tests for backward compatibility (rollout_steps=1)
# ===========================================================================


class TestSingleStep:
    """Verify rollout_steps=1 returns the original format exactly."""

    def test_default_rollout_steps_is_one(self, mock_dataset):
        assert mock_dataset.rollout_steps == 1

    def test_single_step_keys(self, mock_dataset):
        sample = mock_dataset[0]
        expected_keys = {
            "timestep_idx", "time", "states", "next_states",
            "forces", "channeled_images", "W_interactions",
        }
        assert set(sample.keys()) == expected_keys

    def test_single_step_shapes(self, mock_dataset):
        sample = mock_dataset[0]
        assert sample["states"].shape == (N_OBJECTS, FEATURE_DIM)
        assert sample["next_states"].shape == (N_OBJECTS, FEATURE_DIM)
        assert sample["channeled_images"].shape == (N_OBJECTS, IMAGE_SIZE, IMAGE_SIZE)
        assert sample["W_interactions"].shape == (N_OBJECTS, N_OBJECTS)
        assert sample["timestep_idx"].shape == ()

    def test_single_step_no_none(self, mock_dataset):
        """Every index should return a valid sample when rollout_steps=1."""
        for idx in range(len(mock_dataset)):
            assert mock_dataset[idx] is not None


# ===========================================================================
# Tests for multi-step rollout (rollout_steps > 1)
# ===========================================================================


class TestMultiStep:
    """Verify rollout_steps > 1 returns correct sequence format."""

    def test_sequence_keys(self, mock_dataset):
        mock_dataset.rollout_steps = 4
        sample = mock_dataset[0]
        expected_keys = {
            "timestep_idx", "W_interactions",
            "states_seq", "next_states_seq", "channeled_images_seq",
        }
        assert set(sample.keys()) == expected_keys

    def test_sequence_shapes(self, mock_dataset):
        k = 4
        mock_dataset.rollout_steps = k
        sample = mock_dataset[0]
        assert sample["states_seq"].shape == (k, N_OBJECTS, FEATURE_DIM)
        assert sample["next_states_seq"].shape == (k, N_OBJECTS, FEATURE_DIM)
        assert sample["channeled_images_seq"].shape == (k, N_OBJECTS, IMAGE_SIZE, IMAGE_SIZE)
        assert sample["W_interactions"].shape == (N_OBJECTS, N_OBJECTS)
        assert sample["timestep_idx"].shape == ()

    def test_w_interactions_is_single_timestep(self, mock_dataset):
        """W_interactions should always be 2D (N, N), not a sequence."""
        mock_dataset.rollout_steps = 4
        sample = mock_dataset[0]
        assert sample["W_interactions"].ndim == 2

    def test_timestep_idx_is_scalar(self, mock_dataset):
        mock_dataset.rollout_steps = 4
        sample = mock_dataset[0]
        assert sample["timestep_idx"].ndim == 0


# ===========================================================================
# Tests for None filtering at trajectory boundaries
# ===========================================================================


class TestBoundaryFiltering:
    """Verify None is returned for timesteps without enough room."""

    def test_valid_timesteps_return_dict(self, mock_dataset):
        k = 4
        mock_dataset.rollout_steps = k
        # First timestep of trajectory 0 should always be valid
        sample = mock_dataset[0]
        assert sample is not None
        assert isinstance(sample, dict)

    def test_last_timestep_returns_none(self, mock_dataset):
        """The very last timestep in a trajectory can't fit a 2-step window."""
        mock_dataset.rollout_steps = 2
        # Last timestep of trajectory 0
        last_idx = TRAJECTORY_LENGTH - 1
        sample = mock_dataset[last_idx]
        assert sample is None

    def test_boundary_exact_fit(self, mock_dataset):
        """Timestep at exactly trajectory_length - k should be valid."""
        k = 4
        mock_dataset.rollout_steps = k
        # timestep_idx = T - k = 16, needs indices 16..19 → all valid
        valid_idx = TRAJECTORY_LENGTH - k
        sample = mock_dataset[valid_idx]
        assert sample is not None
        assert sample["timestep_idx"].item() == valid_idx

    def test_boundary_one_past(self, mock_dataset):
        """Timestep at trajectory_length - k + 1 should be None."""
        k = 4
        mock_dataset.rollout_steps = k
        invalid_idx = TRAJECTORY_LENGTH - k + 1
        sample = mock_dataset[invalid_idx]
        assert sample is None

    def test_none_count_per_trajectory(self, mock_dataset):
        """Exactly k-1 timesteps per trajectory should return None."""
        k = 4
        mock_dataset.rollout_steps = k
        # Check first trajectory only (indices 0..TRAJECTORY_LENGTH-1)
        none_count = sum(
            1 for i in range(TRAJECTORY_LENGTH)
            if mock_dataset[i] is None
        )
        assert none_count == k - 1


# ===========================================================================
# Tests for sparse_collate_fn with None filtering
# ===========================================================================


class TestCollateFiltering:
    """Verify collate function handles None samples correctly."""

    def test_collate_filters_none(self, mock_dataset):
        mock_dataset.rollout_steps = 2
        # Grab a mix of valid and None samples
        samples = [mock_dataset[0], None, mock_dataset[1]]
        result = sparse_collate_fn(samples)
        assert result is not None
        assert result["states_seq"].shape[0] == 2  # only 2 valid samples

    def test_collate_all_none_returns_none(self):
        result = sparse_collate_fn([None, None, None])
        assert result is None

    def test_collate_no_none_unchanged(self, mock_dataset):
        mock_dataset.rollout_steps = 2
        samples = [mock_dataset[0], mock_dataset[1], mock_dataset[2]]
        result = sparse_collate_fn(samples)
        assert result is not None
        assert result["states_seq"].shape[0] == 3

    def test_collate_single_step_unchanged(self, mock_dataset):
        """Collate with rollout_steps=1 should work as before."""
        samples = [mock_dataset[0], mock_dataset[1]]
        result = sparse_collate_fn(samples)
        assert result is not None
        assert result["states"].shape == (2, N_OBJECTS, FEATURE_DIM)


# ===========================================================================
# Tests for rollout_steps mutability
# ===========================================================================


class TestMutability:
    """Verify rollout_steps can be changed and affects output format."""

    def test_switch_from_single_to_multi(self, mock_dataset):
        # Start single-step
        sample1 = mock_dataset[0]
        assert "states" in sample1
        assert "states_seq" not in sample1

        # Switch to multi-step
        mock_dataset.rollout_steps = 3
        sample2 = mock_dataset[0]
        assert "states_seq" in sample2
        assert "states" not in sample2

    def test_switch_back_to_single(self, mock_dataset):
        mock_dataset.rollout_steps = 4
        sample1 = mock_dataset[0]
        assert "states_seq" in sample1

        # Switch back
        mock_dataset.rollout_steps = 1
        sample2 = mock_dataset[0]
        assert "states" in sample2
        assert "states_seq" not in sample2

    def test_different_k_values(self, mock_dataset):
        for k in [2, 4, 8]:
            if k > TRAJECTORY_LENGTH:
                continue
            mock_dataset.rollout_steps = k
            sample = mock_dataset[0]
            assert sample["states_seq"].shape[0] == k


# ===========================================================================
# Tests for data correctness
# ===========================================================================


class TestDataCorrectness:
    """Verify sequence data values match single-step lookups."""

    def test_states_seq_matches_single_step(self, mock_dataset):
        """Each slice of states_seq should match the single-step states."""
        k = 4
        t_start = 5  # arbitrary starting timestep

        # Get single-step samples for comparison
        single_step_states = []
        mock_dataset.rollout_steps = 1
        for step in range(k):
            sample = mock_dataset[t_start + step]
            single_step_states.append(sample["states"])

        # Get multi-step sample
        mock_dataset.rollout_steps = k
        seq_sample = mock_dataset[t_start]

        for step in range(k):
            assert torch.allclose(
                seq_sample["states_seq"][step],
                single_step_states[step],
            ), f"states_seq[{step}] doesn't match single-step states at t={t_start + step}"

    def test_next_states_seq_matches_single_step(self, mock_dataset):
        k = 4
        t_start = 5

        single_step_next = []
        mock_dataset.rollout_steps = 1
        for step in range(k):
            sample = mock_dataset[t_start + step]
            single_step_next.append(sample["next_states"])

        mock_dataset.rollout_steps = k
        seq_sample = mock_dataset[t_start]

        for step in range(k):
            assert torch.allclose(
                seq_sample["next_states_seq"][step],
                single_step_next[step],
            ), f"next_states_seq[{step}] mismatch"

    def test_channeled_images_seq_matches_single_step(self, mock_dataset):
        k = 3
        t_start = 2

        single_step_imgs = []
        mock_dataset.rollout_steps = 1
        for step in range(k):
            sample = mock_dataset[t_start + step]
            single_step_imgs.append(sample["channeled_images"])

        mock_dataset.rollout_steps = k
        seq_sample = mock_dataset[t_start]

        for step in range(k):
            assert torch.allclose(
                seq_sample["channeled_images_seq"][step],
                single_step_imgs[step],
            ), f"channeled_images_seq[{step}] mismatch"

    def test_w_interactions_from_starting_timestep(self, mock_dataset):
        """W_interactions should always come from the starting timestep."""
        k = 4
        t_start = 5

        mock_dataset.rollout_steps = 1
        single_W = mock_dataset[t_start]["W_interactions"]

        mock_dataset.rollout_steps = k
        seq_W = mock_dataset[t_start]["W_interactions"]

        assert torch.allclose(single_W, seq_W)

    def test_timestep_idx_is_starting_index(self, mock_dataset):
        k = 4
        t_start = 7
        mock_dataset.rollout_steps = k
        sample = mock_dataset[t_start]
        assert sample["timestep_idx"].item() == t_start

    def test_cross_trajectory_isolation(self, mock_dataset):
        """Sequences should not cross trajectory boundaries."""
        k = 4
        mock_dataset.rollout_steps = k

        # Sample from trajectory 0 and trajectory 1
        sample_traj0 = mock_dataset[0]
        sample_traj1 = mock_dataset[TRAJECTORY_LENGTH]

        # They should have different data (different random seeds)
        assert not torch.allclose(
            sample_traj0["states_seq"],
            sample_traj1["states_seq"],
        )


# ===========================================================================
# Tests for large rollout_steps edge case
# ===========================================================================


class TestEdgeCases:
    """Edge cases for rollout behavior."""

    def test_rollout_equals_trajectory_length(self, mock_dataset):
        """When k == T, only timestep 0 is valid."""
        k = TRAJECTORY_LENGTH
        mock_dataset.rollout_steps = k
        # Only index 0 should be valid
        assert mock_dataset[0] is not None
        assert mock_dataset[0]["states_seq"].shape[0] == k
        # All others should be None
        for i in range(1, TRAJECTORY_LENGTH):
            assert mock_dataset[i] is None

    def test_rollout_steps_two(self, mock_dataset):
        """Minimal multi-step case."""
        mock_dataset.rollout_steps = 2
        sample = mock_dataset[0]
        assert sample is not None
        assert sample["states_seq"].shape == (2, N_OBJECTS, FEATURE_DIM)

    def test_collated_batch_shape(self, mock_dataset):
        """Test that collated sequence batch has correct 4D shape."""
        k = 3
        mock_dataset.rollout_steps = k
        batch_size = 4
        samples = [mock_dataset[i] for i in range(batch_size)]
        # All should be valid (timesteps 0-3 with k=3, T=20)
        assert all(s is not None for s in samples)

        result = sparse_collate_fn(samples)
        assert result["states_seq"].shape == (batch_size, k, N_OBJECTS, FEATURE_DIM)
        assert result["next_states_seq"].shape == (batch_size, k, N_OBJECTS, FEATURE_DIM)
        assert result["channeled_images_seq"].shape == (batch_size, k, N_OBJECTS, IMAGE_SIZE, IMAGE_SIZE)
        assert result["W_interactions"].shape == (batch_size, N_OBJECTS, N_OBJECTS)
        assert result["timestep_idx"].shape == (batch_size,)
