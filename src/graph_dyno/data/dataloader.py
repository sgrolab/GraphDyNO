# %%
from torch.utils.data import Dataset, DataLoader
import torch
import os
import hashlib
import dataclasses
from typing import Dict, Optional, Tuple
from omegaconf import DictConfig, OmegaConf
from graph_dyno.data.trajectory_generator import generate_config_hash


def get_tensor_memory_bytes(tensor: torch.Tensor) -> int:
    """Calculate memory usage of a tensor in bytes.

    Sparse COO tensors are charged for their indices + values, not for the
    dense element count: `nelement()` on a sparse (T, N, N) tensor returns
    T*N*N, which charged W_interactions 720 MB against a true 102 MB on the
    membrane_stiffness cache. Correcting it only lowers the GPU-cache estimate, which is
    the safe direction.

    Args:
        tensor: PyTorch tensor

    Returns:
        Memory usage in bytes
    """
    if tensor.is_sparse:
        idx = tensor._indices()
        vals = tensor._values()
        return (idx.element_size() * idx.nelement()
                + vals.element_size() * vals.nelement())
    return tensor.element_size() * tensor.nelement()


def get_trajectory_memory_bytes(trajectory_data: Dict) -> int:
    """Calculate total memory usage of a trajectory in bytes.

    Args:
        trajectory_data: Dictionary containing trajectory tensors (non-tensor values are skipped)

    Returns:
        Total memory usage in bytes
    """
    total_bytes = 0
    for value in trajectory_data.values():
        if isinstance(value, torch.Tensor):
            total_bytes += get_tensor_memory_bytes(value)
    return total_bytes


def parse_memory_string(memory_str: str) -> int:
    """Parse memory string to bytes.

    Args:
        memory_str: String like "10GB", "512MB", "1.5GB"

    Returns:
        Memory in bytes
    """
    memory_str = memory_str.strip().upper()

    if memory_str.endswith('GB'):
        return int(float(memory_str[:-2]) * 1024**3)
    elif memory_str.endswith('MB'):
        return int(float(memory_str[:-2]) * 1024**2)
    elif memory_str.endswith('KB'):
        return int(float(memory_str[:-2]) * 1024)
    else:
        # Assume bytes
        return int(memory_str)


def format_bytes(bytes_val: int) -> str:
    """Format bytes as human-readable string.

    Args:
        bytes_val: Number of bytes

    Returns:
        Formatted string (e.g., "4.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


@dataclasses.dataclass
class TrajectoryData:
    """
    Standard trajectory data format:
        - positions: (trajectory_length, n_entities * n_dims)
        - next_states: (trajectory_length, n_entities * n_dims)
        - forces: (trajectory_length, n_entities * n_dims)
        - combined_images: (trajectory_length, H, W)
        - channeled_images: (trajectory_length, n_entities, H, W)
        - W_interactions: (trajectory_length, n_entities, n_entities)
    """
    positions: torch.Tensor
    next_states: torch.Tensor
    forces: torch.Tensor
    combined_images: torch.Tensor
    channeled_images: torch.Tensor
    W_interactions: torch.Tensor


def sparse_collate_fn(batch) -> Optional[Dict]:
    """Custom collate function to handle sparse tensors in batches.

    Filters out None samples (from rollout boundary skipping).

    Args:
        batch: List of dictionaries (or None) containing sample data

    Returns:
        Dictionary with batched tensors, or None if all samples were filtered out
    """
    batch = [s for s in batch if s is not None]
    if len(batch) == 0:
        return None

    batched = {}
    for key in batch[0].keys():
        stacked = torch.stack([sample[key] for sample in batch])
        batched[key] = stacked

    return batched


class TrajectoryDatasetFromFolder(Dataset):
    """PyTorch Dataset that loads pre-generated trajectory data from a folder.

    This dataset automatically reads metadata.yaml to discover all parameters,
    making it easy to load datasets with just a folder path.

    Implements simplified caching with two modes:
    1. Full GPU cache: Entire dataset preloaded to GPU memory (fastest)
    2. CPU cache: Entire dataset in RAM, batches transferred to GPU on-demand

    """

    def __init__(
        self,
        folder_path: str,
        cpu_cache_size: int = 10,
        gpu_cache_memory: Optional[str] = None,
        enable_gpu_cache: bool = True,
        embeddings: Optional[dict] = None,
        n_trajectories: Optional[int] = None,
        history_window: int = 0,
    ):
        """
        Args:
            folder_path: Directory containing metadata.yaml and trajectory files
            cpu_cache_size: Maximum number of trajectories to keep in CPU memory.
                           Set to 0 to always load from disk. Set to -1 to cache all trajectories.
            gpu_cache_memory: GPU memory to reserve for caching (e.g., "10GB", "512MB").
                             If None, automatically determines based on available GPU memory.
                             Can also be a float representing GB (e.g., 10.0 for 10GB).
            enable_gpu_cache: Whether to attempt GPU caching (requires CUDA available)
            embeddings: Pre-loaded bottleneck dict mapping traj_idx to
                       {"L": (T,N,bn), "crop_sides": (T,N)}. Caller is expected
                       to have run shape_decoder.get_bottleneck once; the
                       dataloader ships L verbatim so the training loop avoids
                       per-step encoder calls. If provided, images are excluded
                       from GPU cache.
            n_trajectories: Override number of trajectories to use. If None, uses
                          the value from metadata.yaml. Useful when the cache folder
                          contains more trajectories than needed.
        """
        self.folder_path = folder_path
        self.cpu_cache_size = cpu_cache_size
        self.enable_gpu_cache = enable_gpu_cache
        self._embeddings = embeddings
        self._has_embeddings = embeddings is not None
        self.rollout_steps = 1  # mutable; training loop sets this for curriculum
        self.include_images = True  # mutable; toggle False to exclude images from batches
        self.history_window = int(history_window)  # past K frames of embeddings (0 = off)

        # Verify folder exists
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(
                f"Folder '{self.folder_path}' does not exist."
            )

        # Load metadata
        metadata_path = os.path.join(self.folder_path, "metadata.yaml")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"No metadata.yaml found in '{self.folder_path}'. "
                "This folder does not appear to contain trajectory data generated by a TrajectoryGenerator."
            )

        self.metadata = OmegaConf.load(metadata_path)

        # Extract key parameters from metadata (Hydra config structure)
        metadata_n_trajectories = self.metadata.data.generator.n_trajectories
        if n_trajectories is not None:
            if n_trajectories > metadata_n_trajectories:
                raise ValueError(
                    f"Requested n_trajectories={n_trajectories} but only "
                    f"{metadata_n_trajectories} exist in '{folder_path}'"
                )
            if n_trajectories < metadata_n_trajectories:
                print(f"Using {n_trajectories} of {metadata_n_trajectories} "
                      f"available trajectories")
            self.n_trajectories = n_trajectories
        else:
            self.n_trajectories = metadata_n_trajectories
        self.trajectory_length = self.metadata.data.generator.trajectory_length

        # Compute cache_id from data config (excluding generator)
        self.cache_id = generate_config_hash(self.metadata.data)

        # Verify that at least the first trajectory exists
        first_cache_path = self._get_cache_path(0)
        if not os.path.exists(first_cache_path):
            raise FileNotFoundError(
                f"No trajectory files found in '{self.folder_path}'. "
                f"Expected file: {first_cache_path}"
            )

        # Parse gpu_cache_memory parameter
        if gpu_cache_memory is not None:
            if isinstance(gpu_cache_memory, (int, float)):
                # Assume GB
                self.gpu_cache_memory_bytes = int(gpu_cache_memory * 1024**3)
            else:
                # Parse string
                self.gpu_cache_memory_bytes = parse_memory_string(gpu_cache_memory)
        else:
            self.gpu_cache_memory_bytes = None

        # CPU and GPU caches (FIFO)
        self._cpu_cache = {}
        self._gpu_cache = {}

        # Determine cache strategy and initialize caches
        self._determine_cache_strategy()

    def _get_cache_path(self, trajectory_idx):
        """Get the file path for a cached trajectory."""
        filename = f"{self.cache_id}_{trajectory_idx:04d}.pt"
        return os.path.join(self.folder_path, filename)

    @property
    def has_embeddings(self) -> bool:
        """Whether embeddings are available (pre-loaded or already merged into cache)."""
        return self._has_embeddings

    # Image keys kept on CPU (not moved to GPU) when using shape embeddings
    _IMAGE_KEYS = {"combined_images", "channeled_images"}

    @classmethod
    def _graft(cls, trajectory_data: Dict, entry: Dict) -> None:
        """Overlay a companion entry onto a loaded trajectory dict.

        Without a seg-ablation sidecar `entry` is exactly {L, crop_sides},
        neither of which exists on disk, so this is byte-identical to the two
        explicit assignments it replaces. With one, it also overrides
        states / next_states / W_interactions and adds valid / next_valid.
        """
        for key, value in entry.items():
            if key in cls._IMAGE_KEYS:
                raise ValueError(
                    f"companion entry carries image key {key!r}; images are "
                    f"deliberately kept off the GPU cache and reloaded via "
                    f"load_images()")
            trajectory_data[key] = value

    def _calculate_dataset_memory(self) -> Tuple[int, int]:
        """Calculate GPU memory usage of the entire dataset.

        When using DINO, image tensors are excluded from the GPU estimate
        (they stay on CPU) and DINO companion data is included.

        Returns:
            Tuple of (total_gpu_bytes, gpu_bytes_per_trajectory)
        """
        # Memory-map first trajectory to estimate sizes without loading data into RAM
        first_trajectory = torch.load(self._get_cache_path(0), map_location='cpu', mmap=True)

        exclude_keys = self._IMAGE_KEYS if self.has_embeddings else set()
        # Keys the companion dict supplies are *replaced*, not added: the graft
        # in _preload_gpu_cache / _load_trajectory overwrites the on-disk value.
        # Counting both would double-charge states/next_states/W_interactions
        # under a seg-ablation sidecar.
        overridden = set(self._embeddings[0]) if self.has_embeddings else set()

        bytes_per_trajectory = 0
        excluded_bytes = 0
        for key, value in first_trajectory.items():
            if isinstance(value, torch.Tensor):
                size = get_tensor_memory_bytes(value)
                if key in exclude_keys:
                    excluded_bytes += size
                elif key in overridden:
                    continue
                else:
                    bytes_per_trajectory += size

        # Include embedding data in GPU estimate
        if self.has_embeddings:
            for value in self._embeddings[0].values():
                if isinstance(value, torch.Tensor):
                    bytes_per_trajectory += get_tensor_memory_bytes(value)

        if excluded_bytes > 0:
            print(f"Embedding mode: excluding images from GPU memory estimate "
                  f"({format_bytes(excluded_bytes)}/traj, "
                  f"{format_bytes(excluded_bytes * self.n_trajectories)} total)")

        total_bytes = bytes_per_trajectory * self.n_trajectories
        return total_bytes, bytes_per_trajectory

    def _get_available_gpu_memory(self) -> int:
        """Get available GPU memory in bytes.

        Returns:
            Available GPU memory in bytes, or 0 if CUDA not available
        """
        if not torch.cuda.is_available():
            return 0

        torch.cuda.empty_cache()
        return torch.cuda.mem_get_info()[0]  # Returns (free_memory, total_memory)

    def _determine_cache_strategy(self):
        """Determine caching strategy based on available memory and dataset size."""
        # Calculate dataset memory requirements
        total_bytes, bytes_per_trajectory = self._calculate_dataset_memory()

        print(f"\n{'='*60}")
        print(f"Dataset Memory Analysis")
        print(f"{'='*60}")
        print(f"Total trajectories: {self.n_trajectories}")
        print(f"Memory per trajectory: {format_bytes(bytes_per_trajectory)}")
        print(f"Total dataset size: {format_bytes(total_bytes)}")

        # Determine GPU caching strategy
        self.use_gpu_cache = False
        self.gpu_cache_capacity = 0

        if self.enable_gpu_cache and torch.cuda.is_available():
            available_gpu_mem = self._get_available_gpu_memory()
            print(f"Available GPU memory: {format_bytes(available_gpu_mem)}")

            # Determine GPU cache budget
            if self.gpu_cache_memory_bytes is not None:
                gpu_budget = min(self.gpu_cache_memory_bytes, available_gpu_mem)
                print(f"GPU cache budget (user-specified): {format_bytes(gpu_budget)}")
            else:
                # Use 75% of available GPU memory by default
                gpu_budget = int(available_gpu_mem * 0.75)
                print(f"GPU cache budget (auto, 75% of available): {format_bytes(gpu_budget)}")

            # Only enable GPU cache if ENTIRE dataset fits
            # Partial GPU caching is too slow due to transfer overhead
            if total_bytes <= gpu_budget:
                print(f"Entire dataset fits in GPU cache ({self.n_trajectories} trajectories)")
                self.use_gpu_cache = True
                self.gpu_cache_capacity = self.n_trajectories
                # Preload all trajectories to GPU
                self._preload_gpu_cache()
            else:
                print(f"GPU cache disabled (dataset too large: {format_bytes(total_bytes)} > {format_bytes(gpu_budget)})")
                print(f"   Partial GPU caching is not supported (too slow)")
                print(f"   Will use CPU cache with batch-level GPU transfers")
                self.use_gpu_cache = False
        else:
            if not torch.cuda.is_available():
                print(f"GPU cache disabled (CUDA not available)")
            elif not self.enable_gpu_cache:
                print(f"GPU cache disabled (enable_gpu_cache=False)")
            else:
                print(f"GPU cache disabled (device is CPU)")

        # Determine CPU caching strategy
        # Simplified: either use GPU cache OR CPU cache, not both
        if self.use_gpu_cache:
            # When using GPU cache, disable CPU cache to save RAM
            self.cpu_cache_size = 0
            print(f"CPU cache disabled (using GPU cache)")
        else:
            # Force full CPU cache when not using GPU cache
            self.cpu_cache_size = self.n_trajectories
            print(f"CPU cache enabled for all trajectories ({self.n_trajectories})")

        print(f"{'='*60}\n")

    def _preload_gpu_cache(self):
        """Preload all trajectories to GPU cache if entire dataset fits."""
        if self.has_embeddings:
            print(f"Embedding mode: keeping images on CPU ({', '.join(sorted(self._IMAGE_KEYS))})")
        print(f"Preloading {self.n_trajectories} trajectories to GPU...")

        try:
            from tqdm import tqdm
            iterator = tqdm(range(self.n_trajectories), desc="Loading to GPU")
        except ImportError:
            iterator = range(self.n_trajectories)

        for traj_idx in iterator:
            # Load from disk
            # mmap=True avoids reading image data into RAM when images are skipped
            trajectory_data = torch.load(self._get_cache_path(traj_idx), map_location='cpu', mmap=True)

            # Merge pre-loaded companion data if available. Generalised from
            # the fixed {L, crop_sides} pair so a seg-ablation sidecar can also
            # override states / next_states / W_interactions and add
            # valid / next_valid. Byte-identical without a sidecar (the dict
            # holds exactly those two keys, neither of which is on disk).
            if self.has_embeddings:
                self._graft(trajectory_data, self._embeddings[traj_idx])

            # Move to GPU (skip sim_metadata and image keys entirely)
            skip_keys = self._IMAGE_KEYS if self.has_embeddings else set()
            gpu_trajectory = {}
            for key, value in trajectory_data.items():
                if key in skip_keys:
                    continue  # images loaded on-demand via load_images()
                elif key == "sim_metadata":
                    gpu_trajectory[key] = value
                elif isinstance(value, torch.Tensor):
                    gpu_trajectory[key] = value.to("cuda", non_blocking=True)
                else:
                    gpu_trajectory[key] = value

            self._gpu_cache[traj_idx] = gpu_trajectory

        # Free pre-loaded embeddings — now stored in GPU cache
        if self.has_embeddings:
            self._embeddings = None

        print(f"Preloaded {len(self._gpu_cache)} trajectories to GPU\n")

    def _load_trajectory(self, trajectory_idx):
        """Load a trajectory from GPU cache, CPU cache, or disk.

        Simplified logic:
        - If using GPU cache: all trajectories are preloaded, just return from GPU cache
        - If using CPU cache: check CPU cache first, otherwise load from disk

        Args:
            trajectory_idx: Index of trajectory to load

        Returns:
            Dictionary containing trajectory data
        """
        # Check GPU cache (all trajectories preloaded if enabled)

        if self.use_gpu_cache:
            if trajectory_idx in self._gpu_cache:
                return self._gpu_cache[trajectory_idx]
            # This should never happen if GPU cache is properly preloaded
            raise RuntimeError(
                f"Trajectory {trajectory_idx} not in GPU cache, but GPU cache is enabled. "
                "This indicates a bug in the caching logic."
            )

        # Check CPU cache
        if trajectory_idx in self._cpu_cache:
            return self._cpu_cache[trajectory_idx]

        # Load from disk
        cache_path = self._get_cache_path(trajectory_idx)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Trajectory {trajectory_idx} not found in cache. "
                f"Expected file: {cache_path}"
            )

        # Force load to CPU to ensure CPU cache actually stores CPU tensors
        trajectory_data = torch.load(cache_path, map_location='cpu')

        # Skip image keys when using embeddings (loaded on-demand via load_images())
        if self.has_embeddings:
            for key in list(self._IMAGE_KEYS):
                trajectory_data.pop(key, None)

        # Merge pre-loaded companion data if available (see _preload_gpu_cache)
        if self.has_embeddings:
            self._graft(trajectory_data, self._embeddings[trajectory_idx])
            # Free this entry — now stored in CPU cache
            del self._embeddings[trajectory_idx]

        # Store in CPU cache (always enabled when not using GPU cache)
        if self.cpu_cache_size > 0:
            # FIFO eviction for CPU cache
            if len(self._cpu_cache) >= self.cpu_cache_size:
                oldest_key = min(self._cpu_cache.keys())
                del self._cpu_cache[oldest_key]

            self._cpu_cache[trajectory_idx] = trajectory_data

        return trajectory_data

    def load_images(self):
        """Load channeled_images from disk into all cache entries."""
        cache = self._gpu_cache if self.use_gpu_cache else self._cpu_cache
        for traj_idx in sorted(cache.keys()):
            disk_data = torch.load(self._get_cache_path(traj_idx), map_location='cpu', mmap=True)
            for key in self._IMAGE_KEYS:
                if key in disk_data:
                    cache[traj_idx][key] = disk_data[key]

    def drop_images(self):
        """Delete channeled_images from all cache entries to free memory."""
        cache = self._gpu_cache if self.use_gpu_cache else self._cpu_cache
        for traj_data in cache.values():
            for key in list(self._IMAGE_KEYS):
                traj_data.pop(key, None)
        import gc; gc.collect()
        if self.use_gpu_cache:
            torch.cuda.empty_cache()

    def __len__(self):
        """Total number of samples across all trajectories."""
        return self.n_trajectories * self.trajectory_length

    def get_simulation_metadata(self) -> Optional[dict]:
        """Get simulation-specific metadata from trajectory 0.

        This metadata (rs, growth_rates, r_max) is computed during simulation
        and should be used instead of recalculating from config.

        Returns:
            Dict with rs, growth_rates, r_max tensors, or None if not found.
        """
        trajectory_data = self._load_trajectory(0)
        return trajectory_data.get("sim_metadata", None)

    def verify_metadata_consistency(self) -> bool:
        """Verify that all trajectories have matching simulation metadata.

        Should be called after data is loaded to ensure consistency.

        Returns:
            True if all metadata matches, False if no metadata found.

        Raises:
            ValueError: If metadata values differ between trajectories.
        """
        reference_metadata = self.get_simulation_metadata()
        if reference_metadata is None:
            print("Warning: No simulation metadata found (old cache format)")
            return False

        for traj_idx in range(1, self.n_trajectories):
            trajectory_data = self._load_trajectory(traj_idx)
            sim_metadata = trajectory_data.get("sim_metadata", None)

            if sim_metadata is None:
                raise ValueError(f"Trajectory {traj_idx} missing sim_metadata")

            # Oscillation phases are randomized per-trajectory (each
            # trajectory gets a different seed) so they're allowed to differ.
            phase_fields = {
                "area_fluct_phases",
                "gam_fluct_phases",
                "kl_fluct_phases",
            }

            for key in reference_metadata.keys():
                if key in phase_fields:
                    continue
                val_ref = reference_metadata[key]
                val_cur = sim_metadata[key]

                if isinstance(val_ref, torch.Tensor):
                    if not torch.allclose(val_ref.cpu(), val_cur.cpu(), rtol=1e-5, atol=1e-5):
                        max_diff = (val_ref.cpu() - val_cur.cpu()).abs().max().item()
                        raise ValueError(
                            f"Metadata '{key}' mismatch in trajectory {traj_idx}. "
                            f"Max difference from trajectory 0: {max_diff}"
                        )
                else:
                    if val_ref != val_cur:
                        raise ValueError(
                            f"Metadata '{key}' mismatch in trajectory {traj_idx}. "
                            f"Expected {val_ref}, got {val_cur}"
                        )

        print(f"Verified simulation metadata consistency across {self.n_trajectories} trajectories")
        return True

    def __getitem__(self, idx):
        """Get a single sample from the dataset.

        Args:
            idx: Global index across all trajectories

        Returns:
            dict or None: Dictionary containing trajectory data.
                When rollout_steps > 1, returns sequence keys (*_seq) instead
                of single-step keys, plus single-step W_interactions (reused).
                Returns None if the timestep doesn't have enough room for the
                rollout window (filtered out by collate_fn).
        """
        # Determine which trajectory and timestep this index corresponds to
        trajectory_idx = idx // self.trajectory_length
        timestep_idx = idx % self.trajectory_length

        # Load the trajectory from cache
        trajectory_data = self._load_trajectory(trajectory_idx)

        # ---- Multi-step rollout path ----
        if self.rollout_steps > 1:
            k = self.rollout_steps
            # Skip timesteps that don't have enough room for the full window
            if timestep_idx + k > self.trajectory_length:
                return None
            # Also skip when there isn't enough backward room for the history
            if self.history_window > 0 and timestep_idx < self.history_window:
                return None

            sample = {
                "timestep_idx": torch.tensor(timestep_idx, dtype=torch.long),
                "W_interactions": trajectory_data["W_interactions"][timestep_idx],
                "states_seq": trajectory_data["states"][timestep_idx:timestep_idx + k],
                "next_states_seq": trajectory_data["next_states"][timestep_idx:timestep_idx + k],
            }
            if self.include_images:
                sample["channeled_images_seq"] = trajectory_data["channeled_images"][timestep_idx:timestep_idx + k]

            if self.has_embeddings:
                # No previous timestep available for velocity computation
                if timestep_idx == 0:
                    return None

                L_seq = trajectory_data["L"][timestep_idx:timestep_idx + k]

                # Next-step bottlenecks: indices [t+1, ..., t+k], clamped at boundary
                next_end = min(timestep_idx + k + 1, self.trajectory_length)
                next_L_seq = trajectory_data["L"][timestep_idx + 1:next_end]
                if next_L_seq.shape[0] < k:
                    pad = k - next_L_seq.shape[0]
                    next_L_seq = torch.cat([next_L_seq, next_L_seq[-1:].expand(pad, -1, -1)], dim=0)

                sample["L_seq"] = L_seq
                sample["next_L_seq"] = next_L_seq

                crop_seq = trajectory_data["crop_sides"][timestep_idx:timestep_idx + k]
                next_crop_seq = trajectory_data["crop_sides"][timestep_idx + 1:next_end]
                if next_crop_seq.shape[0] < k:
                    pad = k - next_crop_seq.shape[0]
                    next_crop_seq = torch.cat([next_crop_seq, next_crop_seq[-1:].expand(pad, -1)], dim=0)

                sample["crop_sides_seq"] = crop_seq
                sample["next_crop_sides_seq"] = next_crop_seq

                # Previous-step bottleneck for velocity at step 0 of rollout
                prev_t = timestep_idx - 1
                sample["prev_L"] = trajectory_data["L"][prev_t]
                sample["prev_crop_sides"] = trajectory_data["crop_sides"][prev_t]

                # Past-K-frame bottleneck history (K, N, bn), oldest-first.
                if self.history_window > 0:
                    h0 = timestep_idx - self.history_window
                    sample["L_history"] = \
                        trajectory_data["L"][h0:timestep_idx]

                # Per-slot validity (real data only; sim caches lack the key).
                # Plain slices of the stored keys — next_valid needs no
                # clamp-and-repeat now that it is cached, which also avoids the
                # boundary defect the next_L_seq pad above would inherit.
                if "valid" in trajectory_data:
                    sample["valid_seq"] = trajectory_data["valid"][timestep_idx:timestep_idx + k]
                    sample["next_valid_seq"] = trajectory_data["next_valid"][timestep_idx:timestep_idx + k]
                    sample["prev_valid"] = trajectory_data["valid"][prev_t]

            return sample

        # ---- Original single-step path ----
        sample = {
            "timestep_idx": torch.tensor(timestep_idx, dtype=torch.long),
            "time": trajectory_data["times"][timestep_idx],
            "states": trajectory_data["states"][timestep_idx],
            "next_states": trajectory_data["next_states"][timestep_idx],
            "forces": trajectory_data["forces"][timestep_idx],
            "W_interactions": trajectory_data["W_interactions"][timestep_idx],
        }
        if self.include_images:
            sample["channeled_images"] = trajectory_data["channeled_images"][timestep_idx]

        if self.has_embeddings:
            next_t = min(timestep_idx + 1, self.trajectory_length - 1)

            sample["L"] = trajectory_data["L"][timestep_idx]          # (N, bn)
            sample["next_L"] = trajectory_data["L"][next_t]

            sample["crop_sides"] = trajectory_data["crop_sides"][timestep_idx]          # (N,)
            sample["next_crop_sides"] = trajectory_data["crop_sides"][next_t]

            # Previous-step bottleneck for velocity computation
            if timestep_idx == 0:
                return None  # no previous timestep available
            # History window requires K frames of past room
            if self.history_window > 0 and timestep_idx < self.history_window:
                return None
            prev_t = timestep_idx - 1
            sample["prev_L"] = trajectory_data["L"][prev_t]
            sample["prev_crop_sides"] = trajectory_data["crop_sides"][prev_t]

            # Past-K-frame bottleneck history (K, N, bn), oldest-first.
            if self.history_window > 0:
                h0 = timestep_idx - self.history_window
                sample["L_history"] = \
                    trajectory_data["L"][h0:timestep_idx]

            # Per-slot validity (real data only; sim caches lack the key, so
            # those paths stay byte-identical). next_valid is indexed from the
            # cache's OWN next_valid, not derived as valid[next_t]: next_states
            # is sliced from the full-video array, so next_states[T-1] describes
            # a frame the window's `valid` does not cover and the
            # `next_t = min(t+1, T-1)` clamp cannot reach.
            if "valid" in trajectory_data:
                sample["valid"] = trajectory_data["valid"][timestep_idx]
                sample["next_valid"] = trajectory_data["next_valid"][timestep_idx]
                sample["prev_valid"] = trajectory_data["valid"][prev_t]

            # Uncorrupted targets, present only under a seg-ablation sidecar
            # that asked for them. Inputs stay corrupted — the model is blind;
            # these exist so `val_loss` can be read against the truth, which is
            # the honest checkpoint-selection diagnostic when the observed
            # val loss is dominated by unpredictable corrupted transitions.
            if "L_clean" in trajectory_data:
                sample["next_L_clean"] = trajectory_data["L_clean"][next_t]
                sample["next_crop_sides_clean"] = \
                    trajectory_data["crop_sides_clean"][next_t]
                sample["next_states_clean"] = \
                    trajectory_data["next_states_clean"][timestep_idx]

        return sample


class CUDAPrefetcher:
    """Wraps a DataLoader to overlap CPU->GPU copies with compute.

    Uses a side CUDA stream so the next batch is being transferred while the
    current batch is processed on the default stream.

    Handles both dict batches (main training loop) and tuple batches (DINO
    fine-tuning). Sparse tensors are transferred directly; dense tensors are
    pinned first for optimal DMA transfer.

    Usage:
        loader = DataLoader(ds, batch_size=64, pin_memory=False)
        for batch in CUDAPrefetcher(loader, device):
            ...
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()

    def _transfer(self, tensor):
        if tensor.is_sparse:
            return tensor.to(self.device, non_blocking=True)
        if not tensor.is_pinned():
            tensor = tensor.pin_memory()
        return tensor.to(self.device, non_blocking=True)

    def _preload(self):
        try:
            self._next = next(self._iter)
        except StopIteration:
            self._next = None
            return
        with torch.cuda.stream(self.stream):
            if isinstance(self._next, dict):
                self._next = {k: self._transfer(v) for k, v in self._next.items()}
            else:
                self._next = tuple(self._transfer(t) for t in self._next)

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self._next
        if batch is None:
            raise StopIteration
        self._preload()
        return batch

    def __len__(self):
        return len(self.loader)



def split_dataset(
    dataset: TrajectoryDatasetFromFolder,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    val_split_mode: str = "trajectory",
):
    """Split a dataset into train/validation/test subsets based on trajectories.

    This function splits at the trajectory level (not individual samples) to prevent
    data leakage between splits.

    Args:
        dataset: TrajectoryDatasetFromFolder instance to split
        train_ratio: Proportion of trajectories for training (default: 0.7)
        val_ratio: Proportion of trajectories for validation (default: 0.15)
        test_ratio: Proportion of trajectories for testing (default: 0.15)
        seed: Random seed for reproducibility
        val_split_mode: How to split validation data. Options:
            - "trajectory": Split by whole trajectories (default). With a single
              trajectory, falls back to "temporal".
            - "temporal": Within each trajectory, use the last val_ratio% of
              timesteps as validation and the first (1-val_ratio)% as training.
              Test set is empty in this mode.

    Returns:
        Tuple of (train_indices, val_indices, test_indices) where each is a list of sample indices

    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Get total number of trajectories
    n_trajectories = dataset.n_trajectories
    trajectory_length = dataset.trajectory_length

    # Temporal split: use last val_ratio% of timesteps in each trajectory as validation
    if val_split_mode == "temporal" or n_trajectories == 1:
        n_val_per_traj = int(trajectory_length * val_ratio)
        n_train_per_traj = trajectory_length - n_val_per_traj

        train_indices = []
        val_indices = []
        for traj_idx in range(n_trajectories):
            offset = traj_idx * trajectory_length
            train_indices.extend(range(offset, offset + n_train_per_traj))
            val_indices.extend(range(offset + n_train_per_traj, offset + trajectory_length))
        test_indices = []

        print(f"\nDataset split (temporal - last {100*val_ratio:.0f}% of each trajectory):")
        print(f"  Trajectories: {n_trajectories}")
        print(f"  Train: {len(train_indices)} samples (first {n_train_per_traj} timesteps per trajectory)")
        print(f"  Val:   {len(val_indices)} samples (last {n_val_per_traj} timesteps per trajectory)")
        print(f"  Test:  {len(test_indices)} samples (none)")

        return train_indices, val_indices, test_indices

    # Trajectory split: split by whole trajectories
    trajectory_indices = torch.randperm(n_trajectories).tolist()

    # Calculate split points
    n_train = int(n_trajectories * train_ratio)
    n_val = int(n_trajectories * val_ratio)

    # Split trajectory indices
    train_trajectories = trajectory_indices[:n_train]
    val_trajectories = trajectory_indices[n_train:n_train + n_val]
    test_trajectories = trajectory_indices[n_train + n_val:]

    # Convert trajectory indices to sample indices
    def trajectory_to_sample_indices(traj_indices):
        sample_indices = []
        for traj_idx in traj_indices:
            start_idx = traj_idx * trajectory_length
            end_idx = start_idx + trajectory_length
            sample_indices.extend(range(start_idx, end_idx))
        return sample_indices

    train_indices = trajectory_to_sample_indices(train_trajectories)
    val_indices = trajectory_to_sample_indices(val_trajectories)
    test_indices = trajectory_to_sample_indices(test_trajectories)

    print(f"\nDataset split:")
    print(f"  Train: {len(train_trajectories)} trajectories, {len(train_indices)} samples")
    print(f"  Val:   {len(val_trajectories)} trajectories, {len(val_indices)} samples")
    print(f"  Test:  {len(test_trajectories)} trajectories, {len(test_indices)} samples")

    return train_indices, val_indices, test_indices


def create_dataloader_from_folder(
    folder_path: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    cpu_cache_size: int = -1,
    enable_gpu_cache: bool = False,
    gpu_cache_memory: Optional[str] = None,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    indices: Optional[list] = None,
):
    """Convenience function to create a DataLoader from a folder containing trajectory data.

    This function creates a dataloader that loads pre-generated trajectory data with
    optimized caching and prefetching for maximum performance.

    Args:
        folder_path: Directory containing metadata.yaml and trajectory files
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading (default: 4).
                    Set to 0 to disable multiprocessing.
        cpu_cache_size: Maximum number of trajectories to keep in CPU memory.
                       Set to 0 to always load from disk. Set to -1 to cache all trajectories.
        enable_gpu_cache: Whether to attempt GPU caching (requires CUDA available)
        gpu_cache_memory: GPU memory to reserve for caching (e.g., "10GB", "512MB").
                         If None, automatically uses 75% of available GPU memory.
                         Can also be a float representing GB (e.g., 10.0 for 10GB).
        persistent_workers: Keep worker processes alive between epochs (requires num_workers > 0).
                           Reduces startup overhead between epochs.
        prefetch_factor: Number of batches to prefetch per worker (default: 2).
                        Higher values use more memory but can improve throughput.
        indices: Optional list of indices to use for a subset of the dataset (for train/val/test splits)

    Returns:
        DataLoader: PyTorch DataLoader instance

    """

    # Create dataset from folder with dual-layer caching
    dataset = TrajectoryDatasetFromFolder(
        folder_path=folder_path,
        cpu_cache_size=cpu_cache_size,
        gpu_cache_memory=gpu_cache_memory,
        enable_gpu_cache=enable_gpu_cache,
    )

    # If indices are provided, create a Subset
    if indices is not None:
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)

    # Determine pin_memory setting
    # Only use pin_memory if data is on CPU and target is GPU
    # If GPU cache is enabled, data may already be on GPU, so pin_memory provides no benefit
    use_pin_memory = (
        torch.cuda.is_available() and
        not dataset.use_gpu_cache if hasattr(dataset, 'use_gpu_cache') else True
    )

    # Create DataLoader
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
        "collate_fn": lambda batch: sparse_collate_fn(batch),
    }

    # Add persistent_workers and prefetch_factor if num_workers > 0
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    dataloader = DataLoader(dataset, **dataloader_kwargs)

    return dataloader


def create_split_dataloaders_from_folder(
    folder_path: str,
    cfg: Optional[DictConfig] = None,
    embeddings: Optional[dict] = None,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    cpu_cache_size: int = -1,
    enable_gpu_cache: bool = False,
    gpu_cache_memory: Optional[str] = None,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    seed: int = 42,
    shuffle: bool = True,
    val_split_mode: str = "trajectory",
):
    """Create train/val/test dataloaders from a single dataset folder.

    This function loads the dataset once and creates three dataloaders with appropriate splits.
    More efficient than calling create_dataloader_from_folder three times.

    Can be called either with a Hydra dataloader config or with individual parameters.

    Args:
        folder_path: Directory containing metadata.yaml and trajectory files
        dataloader_cfg: Optional Hydra dataloader config (cfg.dataloader). If provided,
                       extracts all parameters from it and ignores individual params.
        batch_size: Number of samples per batch
        train_ratio: Proportion of trajectories for training (default: 0.7)
        val_ratio: Proportion of trajectories for validation (default: 0.15)
        test_ratio: Proportion of trajectories for testing (default: 0.15)
        num_workers: Number of worker processes for data loading
        cpu_cache_size: Maximum number of trajectories to keep in CPU memory
        enable_gpu_cache: Whether to attempt GPU caching
        gpu_cache_memory: GPU memory to reserve for caching
        persistent_workers: Keep worker processes alive between epochs
        prefetch_factor: Number of batches to prefetch per worker
        seed: Random seed for reproducible splits
        val_split_mode: "trajectory" or "temporal" (see split_dataset)

    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    from torch.utils.data import Subset

    # If dataloader config provided, extract all parameters from it
    if cfg is not None:
        batch_size = cfg.dataloader.batch_size
        train_ratio = cfg.dataloader.train_ratio
        val_ratio = cfg.dataloader.val_ratio
        test_ratio = cfg.dataloader.test_ratio
        num_workers = cfg.dataloader.num_workers
        cpu_cache_size = cfg.dataloader.cpu_cache_size
        enable_gpu_cache = cfg.dataloader.enable_gpu_cache
        gpu_cache_memory = cfg.dataloader.gpu_cache_memory
        seed = cfg.dataloader.seed
        val_split_mode = cfg.dataloader.get("val_split_mode", "trajectory")
    if folder_path == '' or folder_path is None:
        # find the folder path by getting the cache from the param config
        config_hash = generate_config_hash(cfg.data)
        folder_path = "./data/cache_" + config_hash

    n_trajectories = None
    if cfg is not None:
        n_trajectories = cfg.data.generator.n_trajectories

    history_window = 0
    if cfg is not None:
        history_window = int(cfg.get("model", {}).get("history_window", 0))

    # Load dataset once
    dataset = TrajectoryDatasetFromFolder(
        folder_path=folder_path,
        cpu_cache_size=cpu_cache_size,
        gpu_cache_memory=gpu_cache_memory,
        enable_gpu_cache=enable_gpu_cache,
        embeddings=embeddings,
        n_trajectories=n_trajectories,
        history_window=history_window,
    )

    # Split the dataset
    train_indices, val_indices, test_indices = split_dataset(
        dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        val_split_mode=val_split_mode,
    )

    # pin_memory=False in both paths:
    # - GPU cache ON: data already on GPU, pinning is pointless
    # - GPU cache OFF: CUDAPrefetcher pins per-tensor (avoids sparse tensor errors)
    use_pin_memory = False

    # Worker processes cannot access GPU tensors from the parent process,
    # so force num_workers=0 when the dataset is cached on GPU.
    if dataset.use_gpu_cache and num_workers > 0:
        print(f"GPU cache active — overriding num_workers={num_workers} → 0")
        num_workers = 0

    # Create batch multiplicity augmentation for training if configured
    train_collate_fn = sparse_collate_fn

    # Common dataloader kwargs 
    base_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
        "collate_fn": sparse_collate_fn,
    }

    if num_workers > 0:
        base_kwargs["persistent_workers"] = persistent_workers
        base_kwargs["prefetch_factor"] = prefetch_factor

    # Create train dataloader with augmented collate function
    train_kwargs = base_kwargs.copy()
    train_kwargs["collate_fn"] = train_collate_fn
    train_dataset = Subset(dataset, train_indices)
    train_dataloader = DataLoader(train_dataset, shuffle=shuffle, **train_kwargs)

    # Create val dataloader (no augmentation)
    val_dataset = Subset(dataset, val_indices)
    val_dataloader = DataLoader(val_dataset, shuffle=False, **base_kwargs)

    # Create test dataloader (no augmentation)
    test_dataset = Subset(dataset, test_indices)
    test_dataloader = DataLoader(test_dataset, shuffle=False, **base_kwargs)

    return train_dataloader, val_dataloader, test_dataloader


def load_data(
    folder_path: str,
    cfg: DictConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load trajectory data from cache folder.

    Simple API: pass folder path and full Hydra config.

    Args:
        folder_path: Path to cached trajectory data
        cfg: Full Hydra config (uses cfg.dataloader for dataloader parameters)

    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    return create_split_dataloaders_from_folder(
        folder_path=folder_path,
        cfg=cfg,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from graph_dyno.data.trajectory_generator import SphereInteractionGenerator
    from sphere_interaction import generate_combined_image  # pre-existing missing module — leave untouched

    # --- Test DataLoader ---
    print("\n" + "=" * 50)
    print("Testing DataLoader")
    print("=" * 50)

    # Create test config using OmegaConf (mimics Hydra config structure)
    test_cfg = OmegaConf.create({
        "data": {
            "type": "simulation",
            "n_objects": 20,
            "image_size": 64,
            "dt": 0.01,
            "domain_size": 10.0,
            "seed": 0,
            "generator": {
                "n_trajectories": 10,
                "trajectory_length": 20,
                "seed_offset": 0,
                "cache_dir": "./data/test_cache",
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
                "r_steps": 20,
                "r_repeats": 1,
                "growth_rate_min": 0.0,
                "growth_rate_max": 0.0,
                "growth_rate_steps": 20,
                "growth_rate_repeats": 1,
                "relaxation_steps": 50,
                "relaxation_dt": 0.01,
                "relaxation_drag": 5.0,
            },
        },
        "dataloader": {
            "batch_size": 8,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "num_workers": 0,
            "cpu_cache_size": -1,
            "enable_gpu_cache": False,
            "gpu_cache_memory": None,
            "seed": 42,
        },
    })

    # Step 1: Generate and save trajectory data
    print("Generating trajectory data...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = SphereInteractionGenerator(cfg=test_cfg, device=device)
    generator.generate_all(show_progress=True)

    print(f"Data saved to: {generator.cache_dir}")
    print(f"Cache ID: {generator.config_hash}")

    # Step 2: Load data from folder using the simple API
    print("\nLoading data from folder...")
    train_dl, val_dl, test_dl = load_data(
        folder_path=test_cfg.data.generator.cache_dir,
        cfg=test_cfg,
    )

    print(f"Total samples in train dataset: {len(train_dl.dataset)}")
    print(f"Number of train batches: {len(train_dl)}")

    # Iterate through a few batches
    for batch_idx, batch in enumerate(train_dl):
        if batch_idx >= 3:
            break
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  States shape: {batch['states'].shape}")
        print(f"  Next states shape: {batch['next_states'].shape}")
        print(f"  Forces shape: {batch['forces'].shape}")
        print(f"  Channeled image shape: {batch['channeled_images'].shape}")
        print(f"  W_interactions type: {type(batch['W_interactions'])}")
        print(f"  W_interactions shape: {batch['W_interactions'].shape}")
        print(f"  W_interactions is sparse: {batch['W_interactions'].is_sparse}")
        print(f"  W_interactions sparse_dim: {batch['W_interactions'].sparse_dim()}")
        print(f"  W_interactions dtype: {batch['W_interactions'].dtype}")
        # Convert first sparse tensor to dense for statistics
        first_W_dense = batch["W_interactions"][0].to_dense()
        print(f"  Avg interactions in first sample: {first_W_dense.sum():.2f}")

    # Visualize a batch
    print("\nVisualizing a sample batch...")
    batch = next(iter(train_dl))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Sample Batch from DataLoader")

    for i in range(min(8, batch["states"].shape[0])):
        ax = axes[i // 4, i % 4]
        states = batch["states"][i]
        positions = states[:, :2]
        channeled_image = batch["channeled_images"][i]
        combined_image = generate_combined_image(
            positions, channeled_image,
            test_cfg.data.domain_size,
            test_cfg.data.params.r_max
        )
        img = combined_image.cpu().numpy()
        ax.imshow(img, origin="lower", cmap="viridis")
        ax.set_title(f"Sample {i+1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print("\nDataLoader test complete!")
