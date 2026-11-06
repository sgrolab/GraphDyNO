"""
Pytest suite for testing dataloader caching behavior and GPU memory usage.

Run with: pytest test_dataloader_caching.py -v -s
"""

import pytest
import torch
from graph_dyno.data.dataloader import create_dataloader_from_folder
import os


@pytest.fixture
def dataloader_path():
    """Path to test data."""
    return "./data/train_cache"


@pytest.fixture
def check_data_exists(dataloader_path):
    """Skip tests if data doesn't exist."""
    if not os.path.exists(dataloader_path):
        pytest.skip(f"Test data not found at {dataloader_path}")


class TestDataloaderCaching:
    """Test suite for dataloader caching functionality."""

    def test_batch_device_placement(self, dataloader_path, check_data_exists):
        """Test that batches are on the correct device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dataloader = create_dataloader_from_folder(
            folder_path=dataloader_path,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            cpu_cache_size=-1,
            enable_gpu_cache=False,
        )

        # Get first batch
        batch = next(iter(dataloader))

        # All tensors should be on CPU
        assert batch["states"].device.type == "cpu", "states should be on CPU"
        assert batch["next_states"].device.type == "cpu", "next_states should be on CPU"
        assert batch["channeled_images"].device.type == "cpu", "channeled_images should be on CPU"
        assert batch["W_interactions"].device.type == "cpu", "W_interactions should be on CPU"

        print("\n✓ Batch data correctly placed on CPU")

    def test_no_gpu_memory_accumulation(self, dataloader_path, check_data_exists):
        """Test that GPU memory doesn't accumulate when GPU cache is disabled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Clear GPU memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        initial_allocated = torch.cuda.memory_allocated()
        print(f"\n\nInitial GPU allocated: {initial_allocated / 1024**3:.2f} GB")

        dataloader = create_dataloader_from_folder(
            folder_path=dataloader_path,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            cpu_cache_size=-1,
            enable_gpu_cache=False,
        )

        after_loader_creation = torch.cuda.memory_allocated()
        print(f"After dataloader creation: {after_loader_creation / 1024**3:.2f} GB")

        # Iterate through several batches
        batch_memories = []
        for i, batch in enumerate(dataloader):
            if i >= 5:  # Test first 5 batches
                break

            # Force batch to be used (simulate training)
            _ = batch["states"].sum()

            # Delete batch and collect garbage
            del batch
            torch.cuda.empty_cache()

            current_allocated = torch.cuda.memory_allocated()
            batch_memories.append(current_allocated)
            print(f"After batch {i+1}: {current_allocated / 1024**3:.2f} GB")

        # Memory should return to baseline after processing batches
        final_allocated = torch.cuda.memory_allocated()
        print(f"Final GPU allocated: {final_allocated / 1024**3:.2f} GB")

        # Check that memory didn't grow significantly
        # Allow some tolerance for PyTorch's caching allocator
        memory_growth = final_allocated - after_loader_creation
        print(f"Memory growth: {memory_growth / 1024**3:.2f} GB")

        # Memory shouldn't grow by more than 1GB (generous tolerance)
        assert memory_growth < 1 * 1024**3, \
            f"GPU memory grew by {memory_growth / 1024**3:.2f} GB - possible memory leak!"

        print("\n✓ No significant GPU memory accumulation detected")

    def test_cpu_cache_population(self, dataloader_path, check_data_exists):
        """Test that CPU cache gets populated correctly."""
        dataloader = create_dataloader_from_folder(
            folder_path=dataloader_path,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            cpu_cache_size=-1,
            enable_gpu_cache=False,
        )

        # Initially CPU cache should be empty
        initial_cpu_cache_size = len(dataloader.dataset._cpu_cache)
        print(f"\n\nInitial CPU cache size: {initial_cpu_cache_size}")

        # Process some batches
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Process a few batches
                break

        # CPU cache should now contain some trajectories
        final_cpu_cache_size = len(dataloader.dataset._cpu_cache)
        print(f"CPU cache size after {i+1} batches: {final_cpu_cache_size}")

        assert final_cpu_cache_size > initial_cpu_cache_size, \
            "CPU cache should populate as batches are loaded"

        assert len(dataloader.dataset._gpu_cache) == 0, "GPU cache should remain empty"

        print("\n✓ CPU cache populating correctly")

    def test_gpu_cache_stays_empty(self, dataloader_path, check_data_exists):
        """Test that GPU cache never gets populated when disabled."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dataloader = create_dataloader_from_folder(
            folder_path=dataloader_path,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            cpu_cache_size=-1,
            enable_gpu_cache=False,
        )

        # Process entire dataset
        for batch in dataloader:
            pass

        # GPU cache should still be empty
        assert len(dataloader.dataset._gpu_cache) == 0, \
            "GPU cache should remain empty throughout entire dataset iteration"

        print("\n✓ GPU cache remained empty throughout iteration")

    def test_trajectory_device_in_cache(self, dataloader_path, check_data_exists):
        """Test that trajectories in CPU cache are actually on CPU."""
        dataloader = create_dataloader_from_folder(
            folder_path=dataloader_path,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            cpu_cache_size=-1,
            enable_gpu_cache=False,
        )

        # Load some batches to populate CPU cache
        for i, batch in enumerate(dataloader):
            if i >= 2:
                break

        # Check that cached trajectories are on CPU
        cpu_cache = dataloader.dataset._cpu_cache
        print(f"\n\nCPU cache contains {len(cpu_cache)} trajectories")

        for traj_idx, trajectory in cpu_cache.items():
            for key, tensor in trajectory.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                assert tensor.device.type == "cpu", \
                    f"Trajectory {traj_idx}, key {key} is on {tensor.device}, expected CPU"

        print("✓ All cached trajectories are on CPU")

    def test_memory_with_multiple_epochs(self, dataloader_path, check_data_exists):
        """Test GPU memory doesn't accumulate over multiple epochs."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        dataloader = create_dataloader_from_folder(
            folder_path=dataloader_path,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            cpu_cache_size=-1,
            enable_gpu_cache=False,
        )

        baseline_memory = torch.cuda.memory_allocated()
        print(f"\n\nBaseline GPU memory: {baseline_memory / 1024**3:.2f} GB")

        epoch_memories = []

        # Simulate 3 epochs
        for epoch in range(3):
            for batch in dataloader:
                # Simulate some work
                _ = batch["states"].sum()

            torch.cuda.empty_cache()
            epoch_mem = torch.cuda.memory_allocated()
            epoch_memories.append(epoch_mem)
            print(f"After epoch {epoch+1}: {epoch_mem / 1024**3:.2f} GB")

        # Memory should be stable across epochs
        final_memory = epoch_memories[-1]
        memory_growth = final_memory - baseline_memory

        print(f"Total memory growth: {memory_growth / 1024**3:.2f} GB")

        assert memory_growth < 1 * 1024**3, \
            f"GPU memory grew by {memory_growth / 1024**3:.2f} GB across epochs"

        print("\n✓ No memory accumulation across epochs")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
