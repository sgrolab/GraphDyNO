"""Generate the data cache for a hydra config without creating a hydra output directory.

Usage:
    python scripts/generate_cache.py +experiment=oscillating_cell_dino
    python scripts/generate_cache.py data=polyhoop data.generator.n_trajectories=5
"""

import sys
import os


import torch
import hydra
from omegaconf import DictConfig
from graph_dyno.data.trajectory_generator import generate_data


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Generating trajectory data...")
    cache_dir = generate_data(cfg, device=device)
    print(f"\nData saved to: {cache_dir}")


if __name__ == "__main__":
    sys.argv.extend([
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra/job_logging=none",
    ])
    main()
