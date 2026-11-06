# %%
import gc
import torch
import os
from graph_dyno.data.polyhoop_simulation import PolyhoopSimulation
import hashlib
from abc import ABC, abstractmethod
from typing import Dict
from omegaconf import DictConfig, OmegaConf
from graph_dyno.data.param_utils import resolve_sweep_param, is_sweep_param

# Execution-resource knobs that live in data.params but cannot change a single
# output value — only how fast it is produced. They must not enter the cache key,
# or tuning threads for a node would silently orphan an existing cache.
# Absent from every committed config, so excluding them leaves hashes unchanged.
_NON_PHYSICAL_PARAMS = ("omp_threads", "raster_threads")


def generate_config_hash(data_cfg: DictConfig) -> str:
    """Generate a unique hash based on the data configuration.

    Excludes generator-specific properties so the same cache can be reused with different
    numbers of trajectories, and thread-count knobs so it can be reused across hosts.
    """
    # Create a copy and remove generator config before hashing
    cfg_for_hash = OmegaConf.to_container(data_cfg, resolve=True)
    cfg_for_hash.pop("generator", None)
    params = cfg_for_hash.get("params")
    if isinstance(params, dict):
        for key in _NON_PHYSICAL_PARAMS:
            params.pop(key, None)
    config_yaml = OmegaConf.to_yaml(OmegaConf.create(cfg_for_hash))
    return hashlib.md5(config_yaml.encode()).hexdigest()[:8]

class TrajectoryGenerator(ABC):
    """Abstract base class for trajectory data generators.

    All trajectory generators must produce data in the standard trajectory_data format:
    {
        "positions": Tensor of shape (trajectory_length, n_entities * n_dims),
        "next_states": Tensor of shape (trajectory_length, n_entities * n_dims),
        "forces": Tensor of shape (trajectory_length, n_entities * n_dims),
        "channeled_images": Tensor of shape (trajectory_length, n_entities, H, W),
        "W_interactions": Tensor of shape (trajectory_length, n_entities, n_entities)
    }
    """

    def __init__(self, cfg: DictConfig, device: str = None):
        """
        Args:
            cfg: Full Hydra config containing data config with generation parameters
            device: Device to use for tensors ('cuda', 'cpu', or None for auto-detect)
        """
        self.cfg = cfg
        self.data = cfg.data

        # Extract generation params from generator config
        gen = cfg.data.generator
        self.n_trajectories = gen.n_trajectories
        self.trajectory_length = gen.trajectory_length
        self.seed_offset = gen.seed_offset
        self.overwrite_existing = gen.overwrite_existing

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Generate a unique hash for this generator configuration
        self.config_hash = self._generate_config_hash()

        # Auto-generate cache_dir if null
        if gen.cache_dir == "":
            self.cache_dir = f"./data/cache_{self.config_hash}"
        else:
            self.cache_dir = gen.cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

    def _generate_config_hash(self) -> str:
        """Generate a unique hash based on the data configuration.

        Excludes generator config so the same cache can be reused with different
        numbers of trajectories.
        """
        return generate_config_hash(self.data)

    @abstractmethod
    def _generate_trajectory(self, trajectory_idx: int) -> Dict[str, torch.Tensor]:
        """Generate a single trajectory of simulation data.

        Args:
            trajectory_idx: Index of the trajectory to generate

        Returns:
            Dict containing trajectory_data with keys:
                - times: (trajectory_length)
                - positions: (trajectory_length, n_entities * n_dims)
                - next_states: (trajectory_length, n_entities * n_dims)
                - forces: (trajectory_length, n_entities * n_dims)
                - channeled_images: (trajectory_length, n_entities, H, W)
                - W_interactions: (trajectory_length, n_entities, n_entities)
        """
        pass

    def save_metadata(self):
        """Save full Hydra config as YAML file in the cache directory."""
        metadata_path = os.path.join(self.cache_dir, "metadata.yaml")
        with open(metadata_path, "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg, resolve=True))

    def save_metadata_atomic(self):
        """Write metadata.yaml atomically (tmp file + os.replace).

        save_metadata() truncates then writes, so concurrent cluster jobs — which
        all target the same path — can observe or leave a partial file. os.replace
        is atomic within a filesystem, and every job writes identical content.
        """
        metadata_path = os.path.join(self.cache_dir, "metadata.yaml")
        tmp_path = f"{metadata_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg, resolve=True))
        os.replace(tmp_path, metadata_path)

    def generate_trajectory(self, trajectory_idx: int) -> bool:
        """Generate and save exactly one trajectory, skipping it if it exists.

        This is the unit of work for the cluster fan-out (one job per index).
        It deliberately does not go through generate_all(), whose resume is
        contiguous — _check_existing_dataset returns min(missing_trajectories),
        so a single hole at index 0 would regenerate the whole dataset.

        Returns:
            True if the trajectory was generated, False if it already existed.
        """
        if not 0 <= trajectory_idx < self.n_trajectories:
            raise IndexError(
                f"trajectory_idx={trajectory_idx} out of range for "
                f"n_trajectories={self.n_trajectories}"
            )
        self._verify_metadata_hash()

        cache_path = self._get_cache_path(trajectory_idx)
        if os.path.exists(cache_path) and not self.overwrite_existing:
            print(f"Trajectory {trajectory_idx} already exists at {cache_path}. Skipping.")
            return False

        # Written by every job; identical content, so an atomic write is enough.
        self.save_metadata_atomic()

        print(f"Generating trajectory {trajectory_idx} "
              f"(of {self.n_trajectories}) -> {cache_path}")
        trajectory_data = self._generate_trajectory(trajectory_idx)
        torch.save(trajectory_data, cache_path)

        del trajectory_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    def _get_cache_path(self, trajectory_idx: int) -> str:
        """Get the file path for a cached trajectory."""
        filename = f"{self.config_hash}_{trajectory_idx:04d}.pt"
        return os.path.join(self.cache_dir, filename)

    def _verify_metadata_hash(self):
        """Raise if an existing metadata.yaml describes a different dataset.

        The per-index path (generate_trajectory) needs this guard on its own —
        it never calls _check_existing_dataset, whose contiguous-resume logic it
        deliberately bypasses.
        """
        metadata_path = os.path.join(self.cache_dir, "metadata.yaml")
        if not os.path.exists(metadata_path):
            return
        existing_hash = generate_config_hash(OmegaConf.load(metadata_path).data)
        if existing_hash != self.config_hash:
            raise ValueError(
                f"A different dataset already exists in '{self.cache_dir}'.\n"
                f"Config hash mismatch: existing={existing_hash}, current={self.config_hash}\n"
                f"\nPlease choose a different cache_dir or delete the existing data."
            )

    def _check_existing_dataset(self) -> tuple[bool, None | int]:
        """Check if dataset exists and matches current parameters.

        Returns:
            (True, None) if dataset exists and matches,
            (False, None) if no dataset exists,
            (False, new_start_index) if dataset is incomplete and needs regeneration from new_start_index

        Raises:
            ValueError: If dataset exists but config hash doesn't match
        """
        metadata_path = os.path.join(self.cache_dir, "metadata.yaml")

        # If no metadata exists, dataset doesn't exist
        if not os.path.exists(metadata_path):
            return False, None

        # Load existing metadata and compute hash (excluding generator config)
        existing_cfg = OmegaConf.load(metadata_path)
        existing_hash = generate_config_hash(existing_cfg.data)

        # Check if config hash matches
        if existing_hash != self.config_hash:
            raise ValueError(
                f"A different dataset already exists in '{self.cache_dir}'.\n"
                f"Config hash mismatch: existing={existing_hash}, current={self.config_hash}\n"
                f"\nPlease choose a different cache_dir or delete the existing data."
            )

        # Check if all trajectory files exist
        missing_trajectories = []
        for i in range(self.n_trajectories):
            if not os.path.exists(self._get_cache_path(i)):
                missing_trajectories.append(i)

        if missing_trajectories:
            print(
                f"Warning: {len(missing_trajectories)} trajectory files are missing. "
                f"Regenerating from index {min(missing_trajectories)}."
            )
            new_start_index = min(missing_trajectories)
            return False, new_start_index

        return True, None

    def generate_all(self, show_progress: bool = True):
        """Generate and save all trajectories.

        If a dataset with matching parameters already exists and is complete,
        generation is skipped. If a dataset with different parameters exists,
        an error is raised.

        Args:
            show_progress: Whether to show a progress bar during generation

        Raises:
            ValueError: If a different dataset already exists in cache_dir
        """
        # Check if dataset already exists
        dataset_exists, new_start_index = self._check_existing_dataset()
        if not self.overwrite_existing and dataset_exists:
            print(f"Dataset already exists in '{self.cache_dir}' with matching parameters. Skipping generation.")
            return

        # Save metadata before generating so interrupted runs can resume
        self.save_metadata()

        print(f"Generating {self.n_trajectories} trajectories...")

        start_idx = 0
        if new_start_index is not None and not self.overwrite_existing:
            print(f"Resuming generation from trajectory index {new_start_index}...")
            start_idx = new_start_index

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(start_idx, self.n_trajectories), desc="Generating trajectories")
            except ImportError:
                iterator = range(start_idx, self.n_trajectories)
        else:
            iterator = range(start_idx, self.n_trajectories)

        for trajectory_idx in iterator:
            trajectory_data = self._generate_trajectory(trajectory_idx)

            # Save to disk
            cache_path = self._get_cache_path(trajectory_idx)
            torch.save(trajectory_data, cache_path)

            # Free memory between trajectories
            del trajectory_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def generate_data(cfg: DictConfig, device: str = None) -> str:
    """Generate simulation data based on config.

    Simple API: pass full Hydra config and device.

    Args:
        cfg: Full Hydra config (uses cfg.data for generation parameters)
        device: Device to use for tensors ('cuda', 'cpu', or None for auto-detect)

    Returns:
        Path to the cache directory where data was saved
    """
    generator = create_generator(cfg, device=device)
    generator.generate_all()
    return generator.cache_dir


def create_generator(cfg: DictConfig, device: str = None) -> TrajectoryGenerator:
    """Build the generator for cfg without running any generation."""
    source_type = cfg.data.get("type", "simulation")
    if source_type == "real":
        # Cell Tracking Challenge videos. Imported lazily so the simulation
        # paths never pull in PIL/scipy-ndimage or the real_data package.
        from graph_dyno.real_data.generator import CTCTrajectoryGenerator

        return CTCTrajectoryGenerator(cfg=cfg, device=device)
    if source_type != "simulation":
        raise ValueError(f"Unknown data source type: {source_type}")

    simulation_type = cfg.data.params.name
    if simulation_type == "sphere_interaction":
        return SphereInteractionGenerator(cfg=cfg, device=device)
    elif simulation_type == "polyhoop":
        return PolyhoopGenerator(cfg=cfg, device=device)
    else:
        raise ValueError(f"Unknown simulation type: {simulation_type}")


def generate_one(cfg: DictConfig, trajectory_idx: int, device: str = None) -> str:
    """Generate a single trajectory by index — the unit of work for a cluster wave.

    Unlike generate_data(), this generates exactly one index and skips it if the
    file already exists, so N independent jobs can fill a cache in any order.

    IMPORTANT: cfg must carry the *full* data.generator.n_trajectories, never a
    truncated override. Sweep params resolve against it (resolve_sweep_param(...,
    trajectory_idx, n_trajectories)) and dataloader.py treats it as ground truth.
    Only the index varies between jobs.

    SIMULATION CACHES ONLY. CTCTrajectoryGenerator overrides generate_all() to
    run _prepare() *before* the base class writes metadata.yaml, so the resolved
    `ctc_resolved` block lands in the file. This path writes metadata first and
    only reaches _prepare() inside _generate_trajectory, which would leave a real
    cache's metadata.yaml without it. Use generate_data() for `type: real`.

    Args:
        cfg: Full Hydra config
        trajectory_idx: Index of the trajectory to generate
        device: Device to use for tensors

    Returns:
        Path to the cache directory.
    """
    generator = create_generator(cfg, device=device)
    generator.generate_trajectory(trajectory_idx)
    return generator.cache_dir


class SphereInteractionGenerator(TrajectoryGenerator):
    """Generator for sphere interaction simulation trajectories.

    Generates data from multiple independent simulation trajectories.
    Caches trajectories to disk to avoid memory issues.
    """

    def __init__(self, cfg: DictConfig, device: str = None):
        """
        Args:
            cfg: Full Hydra config containing data config with simulation parameters
            device: Device to use for tensors ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__(cfg, device)
        # Shorthand for simulation-specific params (sphere interaction physics)
        self.si = self.data.params

    def _generate_trajectory(self, trajectory_idx: int) -> Dict[str, torch.Tensor]:
        """Generate a single trajectory of simulation data."""
        # Create a config copy with a unique seed for this trajectory
        trajectory_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        trajectory_cfg.data.seed = self.seed_offset + trajectory_idx

        # Resolve sweep parameters for this trajectory
        params = trajectory_cfg.data.params
        for key in list(params.keys()):
            if is_sweep_param(params[key]):
                params[key] = resolve_sweep_param(
                    params[key], trajectory_idx, self.n_trajectories
                )
                print(f"Resolved param '{key}' to {params[key]}")

        from sphere_interaction import SphereInteractionSimulation  # pre-existing missing module
        generator = SphereInteractionSimulation(trajectory_cfg, device=self.device)
        print(f"  Trajectory {trajectory_idx}: drag={generator.si.drag:.4f}")

        # Relax the system
        if self.si.relaxation_steps > 0:
            generator.relax()
            # Add some initial velocity after relaxation
            generator.velocities = torch.randn_like(generator.velocities) * 0.1

        # Generate the trajectory
        times, states, next_states, forces, channeled_images, interaction_graphs = (
            generator.generate()
        )

        # Get simulation metadata (rs, growth_rates, r_max)
        sim_metadata = generator.get_simulation_metadata()

        trajectory_data = {
            "times": times,
            "states": states,
            "next_states": next_states,
            "forces": forces,
            "channeled_images": channeled_images,
            "W_interactions": interaction_graphs,
            # Simulation metadata for model initialization
            "sim_metadata": sim_metadata,
        }

        return trajectory_data


class PolyhoopGenerator(TrajectoryGenerator):
    """Generator for polyhoop polygon simulation trajectories.

    Generates data from multiple independent polygon simulation trajectories.
    Caches trajectories to disk to avoid memory issues.
    """

    def __init__(self, cfg: DictConfig, device: str = None):
        """
        Args:
            cfg: Full Hydra config containing data config with simulation parameters
            device: Device to use for tensors ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__(cfg, device)
        # Shorthand for simulation-specific params (polyhoop physics)
        self.ph = self.data.params

    def _generate_trajectory(self, trajectory_idx: int) -> Dict[str, torch.Tensor]:
        """Generate a single trajectory of simulation data."""
        # Create a config copy with a unique seed for this trajectory
        trajectory_cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        trajectory_cfg.data.seed = self.seed_offset + trajectory_idx

        # Resolve sweep parameters for this trajectory
        params = trajectory_cfg.data.params
        for key in list(params.keys()):
            if is_sweep_param(params[key]):
                params[key] = resolve_sweep_param(
                    params[key], trajectory_idx, self.n_trajectories
                )
                print(f"Resolved param '{key}' to {params[key]}")

        sim = PolyhoopSimulation(trajectory_cfg, device="cpu")

        # Relax the system if configured
        sim.relax()

        # Generate the trajectory
        times, states, next_states, forces, channeled_images, interaction_graphs = (
            sim.generate()
        )

        # Get simulation metadata (rs, growth_rates, r_max)
        sim_metadata = sim.get_simulation_metadata()

        trajectory_data = {
            "times": times,
            "states": states,
            "next_states": next_states,
            "forces": forces,
            "channeled_images": channeled_images,
            "W_interactions": interaction_graphs,
            # Simulation metadata for model initialization
            "sim_metadata": sim_metadata,
        }

        return trajectory_data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sphere_interaction import generate_combined_image

    # --- Test Generator ---
    print("\n" + "=" * 50)
    print("Testing Trajectory Generator")
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
                "n_trajectories": 5,
                "trajectory_length": 20,
                "seed_offset": 0,
                "cache_dir": "./data/test_generator_cache",
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
                "r_steps": 10,
                "r_repeats": 2,
                "growth_rate_min": 0.0,
                "growth_rate_max": 0.5,
                "growth_rate_steps": 2,
                "growth_rate_repeats": 10,
                "relaxation_steps": 50,
                "relaxation_dt": 0.01,
                "relaxation_drag": 5.0,
            },
        },
    })

    # Generate trajectory data
    print("Generating trajectory data...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = SphereInteractionGenerator(cfg=test_cfg, device=device)
    generator.generate_all(show_progress=True)

    print(f"Data saved to: {generator.cache_dir}")
    print(f"Cache ID: {generator.config_hash}")

    # Load and visualize a trajectory
    print("\nLoading and visualizing first trajectory...")
    trajectory_path = generator._get_cache_path(0)
    trajectory_data = torch.load(trajectory_path)

    print(f"Trajectory data keys: {trajectory_data.keys()}")
    print(f"States shape: {trajectory_data['states'].shape}")
    print(f"Channeled images shape: {trajectory_data['channeled_images'].shape}")

    # Visualize first few frames
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("First 8 Frames of Trajectory")

    for i in range(min(8, trajectory_data["states"].shape[0])):
        ax = axes[i // 4, i % 4]
        states = trajectory_data["states"][i]
        positions = states[:, :2]
        channeled_image = trajectory_data["channeled_images"][i]
        combined_image = generate_combined_image(
            positions, channeled_image,
            test_cfg.data.domain_size,
            test_cfg.data.params.r_max
        )
        img = combined_image.cpu().numpy()
        ax.imshow(img, origin="lower", cmap="viridis")
        ax.set_title(f"Frame {i+1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    #%%
    print("\nGenerator test complete!")

    # --- Test Sweep Settings ---
    print("\n" + "=" * 50)
    print("Testing Sweep Parameters (varying drag across trajectories)")
    print("=" * 50)

    # Create config with drag as a sweep parameter and growth_rate as dist
    sweep_cfg = OmegaConf.create({
        "data": {
            "type": "simulation",
            "n_objects": 20,
            "image_size": 64,
            "dt": 0.01,
            "domain_size": 10.0,
            "seed": 0,
            "generator": {
                "n_trajectories": 6,
                "trajectory_length": 100,
                "seed_offset": 0,
                "cache_dir": "./data/test_sweep_cache",
                "overwrite_existing": True,
            },
            "params": {
                "name": "sphere_interaction",
                "periodic_boundary": False,
                # Sweep drag from 0.0 to 2.0 across 5 trajectories
                "drag": {
                    "sweep": "choice",
                    "values": [0, 0.5, 1.0]
                },
                "r_cutoff": 5,
                "potential_D_e": 10,
                "potential_a": 1.5,
                "brownian_motion_std": 0.0,
                # Per-object radius distribution
                "radius": {
                    "dist": "linspace",
                    "min": 0.5,
                    "max": 1.0,
                    "steps": 10,
                    "repeats": 2,
                    "interleaved": True,
                },
                # Per-object growth rate distribution
                "growth_rate": {
                    "dist": "linspace",
                    "min": -0.01,
                    "max": 0.01,
                    "steps": 10,
                    "repeats": 2,
                    "interleaved": True,
                },
                "relaxation_steps": 50,
                "relaxation_dt": 0.01,
                "relaxation_drag": 5.0,
            },
        },
    })

    n_traj = sweep_cfg.data.generator.n_trajectories
    print(f"Generating {n_traj} trajectories with:")
    print(f"  - drag sweep from 0.1 to 2.0 (per-trajectory)")
    print(f"  - radius linspace [0.5, 1.0] with 10 steps x 2 repeats (per-object)")
    print(f"  - growth_rate linspace [-0.01, 0.01] with 10 steps x 2 repeats (per-object)")

    # Verify sweep resolution
    from graph_dyno.data.param_utils import resolve_sweep_param
    # print("\nResolved drag values per trajectory:")
    # for i in range(n_traj):
    #     resolved_drag = resolve_sweep_param(sweep_cfg.data.params.drag, i, n_traj)
    #     print(f"  Trajectory {i}: drag = {resolved_drag:.3f}")

    # Generate the data
    sweep_generator = SphereInteractionGenerator(cfg=sweep_cfg, device=device)
    sweep_generator.generate_all(show_progress=True)

    # Load trajectories and compare velocities (higher drag = slower movement)
    print("\nComparing average speeds across trajectories:")
    avg_speeds = []
    for i in range(n_traj):
        traj_path = sweep_generator._get_cache_path(i)
        traj_data = torch.load(traj_path)
        velocities = traj_data["states"][:, :, 2:4]  # (time, n_objects, 2)
        speeds = torch.norm(velocities, dim=2)  # (time, n_objects)
        avg_speed = speeds.mean().item()
        avg_speeds.append(avg_speed)
        resolved_drag = resolve_sweep_param(sweep_cfg.data.params.drag, i, n_traj)
        print(f"  Trajectory {i} (drag={resolved_drag:.2f}): avg speed = {avg_speed:.4f}")

    # Plot comparison
    fig, axes = plt.subplots(1, n_traj, figsize=(3 * n_traj, 3))
    fig.suptitle("Final frame for each drag value (with growth rate variation)")

    for i in range(n_traj):
        traj_path = sweep_generator._get_cache_path(i)
        traj_data = torch.load(traj_path)
        states = traj_data["states"][-1]  # Last frame
        positions = states[:, :2]
        channeled_image = traj_data["channeled_images"][-1]
        combined_image = generate_combined_image(
            positions, channeled_image,
            sweep_cfg.data.domain_size,
            1.5  # account for growth
        )
        resolved_drag = resolve_sweep_param(sweep_cfg.data.params.drag, i, n_traj)
        axes[i].imshow(combined_image.cpu().numpy(), origin="lower", cmap="viridis")
        axes[i].set_title(f"drag={resolved_drag:.2f}\nspeed={avg_speeds[i]:.3f}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig("sweep_test_output.png", dpi=150)
    print("\nSaved sweep test output to sweep_test_output.png")

    print("\nSweep test complete!")
