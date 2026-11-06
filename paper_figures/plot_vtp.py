#!/usr/bin/env python3
#%%
"""Plot polygons from a PolyHoop simulation using Hydra configs.

Usage:
    # Run simulation, save frames, and render video
    python paper_figures/plot_vtp.py --save-dir runs/my_sim data=polyhoop

    # Re-render video from saved frames with different style
    python paper_figures/plot_vtp.py --load-dir runs/my_sim --color-by perimeter --cmap plasma

    # Quick run without saving
    python paper_figures/plot_vtp.py data=polyhoop data.generator.trajectory_length=100
"""

import sys
import os
import math
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation
from pathlib import Path
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

# Add parent dir so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_polyhoop"))


def extract_frame(ensemble, static_metadata=None):
    """Extract polygon vertices and cell data from a PolyHoop ensemble.

    Returns:
        polygons: list of (N, 2) arrays of vertex positions
        cell_data: dict with 'area', 'perimeter', 'phase' arrays plus any
            keys passed in static_metadata (per-cell numpy arrays).
    """
    n = ensemble.num_polygons()
    polygons = []
    areas = np.zeros(n)
    perimeters = np.zeros(n)
    phases = np.zeros(n)

    for i in range(n):
        poly = ensemble.get_polygon(i)
        verts = np.array(poly.get_vertex_positions())
        # Swap (x, y) → (y, x) to match the (row, col) convention used by
        # polyhoop_simulation's combined images rendered with imshow
        polygons.append(verts[:, ::-1])
        areas[i] = poly.A if poly.A > 0 else poly.area()
        perimeters[i] = poly.perimeter()
        phases[i] = float(poly.phase)

    cell_data = {'area': areas, 'perimeter': perimeters, 'phase': phases}
    if static_metadata:
        cell_data.update(static_metadata)
    return polygons, cell_data


def save_frames(save_dir, frames, domain_size, config_yaml):
    """Save simulation frames to disk.

    Saves:
        {save_dir}/config.yaml  - data config used to generate
        {save_dir}/frames.pkl   - list of (polygons, cell_data) + domain_size
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "config.yaml", "w") as f:
        f.write(config_yaml)

    data = {
        'domain_size': float(domain_size),
        'frames': frames,
    }
    with open(save_dir / "frames.pkl", "wb") as f:
        pickle.dump(data, f)

    print(f"Saved {len(frames)} frames to {save_dir}")


def load_frames(load_dir):
    """Load saved simulation frames.

    Returns:
        frames: list of (polygons, cell_data)
        domain_size: float
    """
    load_dir = Path(load_dir)
    with open(load_dir / "frames.pkl", "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {len(data['frames'])} frames from {load_dir}")
    return data['frames'], data['domain_size']


def run_init_steps(sim, record_frames=False, static_metadata=None):
    """Run the init_steps warm-up that PolyhoopSimulation.generate() runs
    before data capture. Optionally record frames during init.

    Mirrors the init-steps + compression + gam/kl ramp activation from
    polyhoop_simulation.PolyhoopSimulation.generate().
    """
    import polyhoop

    init_steps = sim.gen.get("init_steps", 0)
    frames = []
    if init_steps > 0:
        init_compress_force = sim.gen.get("init_compress_force", 0)
        cv = sim.ph.get("cv", 0)
        v_thresh = 0.1
        if init_compress_force > 0 and cv > 0:
            terminal_v = init_compress_force / cv
            if terminal_v > v_thresh:
                t_settle = math.log(terminal_v / v_thresh) / cv
                settle_steps = math.ceil(t_settle / sim.dt)
            else:
                settle_steps = 0
        else:
            settle_steps = 0
        compress_steps = max(0, init_steps - settle_steps)

        params = polyhoop.get_params()
        prev_cf = params.compress_force
        prev_cx = params.compress_cx
        prev_cy = params.compress_cy
        force_on = init_compress_force > 0 and compress_steps > 0
        if force_on:
            params.compress_force = float(init_compress_force)
            params.compress_cx = sim.data.domain_size / 2
            params.compress_cy = sim.data.domain_size / 2
            polyhoop.set_params(params)

        try:
            for i in tqdm(range(init_steps), desc="init"):
                if force_on and i == compress_steps:
                    params.compress_force = prev_cf
                    params.compress_cx = prev_cx
                    params.compress_cy = prev_cy
                    polyhoop.set_params(params)
                    force_on = False
                sim.step()
                if record_frames:
                    frames.append(extract_frame(sim.ensemble, static_metadata))
        finally:
            if force_on:
                params.compress_force = prev_cf
                params.compress_cx = prev_cx
                params.compress_cy = prev_cy
                polyhoop.set_params(params)

    # Activate per-cell exponential gam/kl/kh ramps for the recording phase.
    if (torch.any(sim.gam_rates != 0)
            or torch.any(sim.kl_rates != 0)
            or torch.any(sim.kh_rates != 0)):
        params = polyhoop.get_params()
        params.simulation_start_time = sim.ensemble.t
        polyhoop.set_params(params)
        for i in range(sim.n_polygons):
            poly = sim.ensemble.get_polygon(i)
            g_rate = float(sim.gam_rates[i])
            k_rate = float(sim.kl_rates[i])
            h_rate = float(sim.kh_rates[i])
            if g_rate != 0:
                poly.gam_rate = g_rate
            if k_rate != 0:
                poly.kl_rate = k_rate
            if h_rate != 0:
                poly.kh_rate = h_rate

    return frames


def run_simulation(cfg, trajectory_idx=0, include_init=True):
    """Run a PolyHoop simulation and collect frames.

    Mirrors PolyhoopGenerator._generate_trajectory so that the same
    trajectory_idx produces identical cell positions and dynamics.

    Args:
        include_init: If True, also record frames during the init_steps
            warm-up. If False, run init_steps silently (matching training
            data) and only record the trajectory_length recording phase.

    Returns:
        frames: list of (polygons, cell_data)
        domain_size: float
    """
    from graph_dyno.data.polyhoop_simulation import PolyhoopSimulation
    from graph_dyno.data.param_utils import is_sweep_param, resolve_sweep_param

    # Resolve config the same way trajectory_generator does
    trajectory_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    trajectory_cfg.data.seed = cfg.data.generator.seed_offset + trajectory_idx

    # Resolve sweep parameters for this trajectory
    n_trajectories = trajectory_cfg.data.generator.n_trajectories
    params = trajectory_cfg.data.params
    for key in list(params.keys()):
        if is_sweep_param(params[key]):
            params[key] = resolve_sweep_param(
                params[key], trajectory_idx, n_trajectories
            )
            print(f"Resolved param '{key}' to {params[key]}")

    domain_size = trajectory_cfg.data.domain_size
    n_steps = trajectory_cfg.data.generator.trajectory_length

    sim = PolyhoopSimulation(trajectory_cfg, device="cpu")
    sim.relax()

    # Snapshot the static per-cell metadata (kh, gam, kl, growth_rates, ...)
    # once and merge into every frame's cell_data so plot_polygons can color
    # by any of these keys via --color-by.
    metadata = sim.get_simulation_metadata()
    static_metadata = {
        k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v))
        for k, v in metadata.items()
        if hasattr(v, "shape") and tuple(v.shape) == (sim.n_polygons,)
    }

    frames = run_init_steps(
        sim, record_frames=include_init, static_metadata=static_metadata,
    )

    for _ in tqdm(range(n_steps), desc="Simulating"):
        sim.step()
        frames.append(extract_frame(sim.ensemble, static_metadata))

    return frames, domain_size


def _make_norm(values, log_color):
    """Build a (Log)Normalize for the colormap. Falls back to linear if
    log requested but values aren't strictly positive."""
    if log_color:
        positive = values[values > 0]
        if positive.size > 0:
            from matplotlib.colors import LogNorm
            return LogNorm(vmin=positive.min(), vmax=positive.max())
        print("Warning: --log-color requested but no positive values; using linear")
    return plt.Normalize(vmin=values.min(), vmax=values.max())


def plot_polygons(polygons, cell_data, domain_size, color_by='area',
                  cmap='viridis', figsize=(10, 10), show_edges=True,
                  edge_color='black', edge_width=0.5, title=None,
                  colorbar=True, ax=None, log_color=False):
    """Plot polygons with the domain as axis limits.

    Returns:
        fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    norm = None
    if color_by is None:
        facecolors = 'lightblue'
    elif color_by in cell_data:
        values = np.asarray(cell_data[color_by])
        norm = _make_norm(values, log_color)
        facecolors = plt.colormaps[cmap](norm(values))
    else:
        print(f"Warning: '{color_by}' not found. Available: {list(cell_data.keys())}")
        facecolors = 'lightblue'

    collection = PolyCollection(
        polygons,
        facecolors=facecolors,
        edgecolors=edge_color if show_edges else 'none',
        linewidths=edge_width
    )
    ax.add_collection(collection)

    margin = 0.02 * domain_size
    ax.set_xlim(-margin, domain_size + margin)
    ax.set_ylim(-margin, domain_size + margin)
    ax.set_aspect('equal')

    if colorbar and norm is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label(color_by)

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return fig, ax


def animate_frames(frames, domain_size, color_by='area', cmap='viridis',
                   figsize=(10, 10), show_edges=True, edge_color='black',
                   edge_width=0.5, filename='animation.mp4', writer='ffmpeg',
                   fps=20, dpi=150, log_color=False):
    """Animate pre-collected frames.

    Args:
        frames: list of (polygons, cell_data)
        domain_size: simulation domain size for axis limits
    """
    # Global color normalization
    global_norm = None
    if color_by is not None:
        all_values = []
        for polygons, cell_data in frames:
            if color_by in cell_data:
                all_values.extend(cell_data[color_by])
        if all_values:
            global_norm = _make_norm(np.asarray(all_values), log_color)

    fig, ax = plt.subplots(figsize=figsize)
    collection = None

    def get_facecolors(cell_data):
        if color_by is None:
            return 'lightblue'
        elif color_by in cell_data:
            values = np.asarray(cell_data[color_by])
            norm = global_norm if global_norm else _make_norm(values, log_color)
            return plt.colormaps[cmap](norm(values))
        return 'lightblue'

    def init():
        margin = 0.02 * domain_size
        ax.set_xlim(-margin, domain_size + margin)
        ax.set_ylim(-margin, domain_size + margin)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return []

    def update(frame):
        nonlocal collection
        polygons, cell_data = frames[frame]

        if collection is not None:
            collection.remove()

        facecolors = get_facecolors(cell_data)
        collection = PolyCollection(
            polygons,
            facecolors=facecolors,
            edgecolors=edge_color if show_edges else 'none',
            linewidths=edge_width
        )
        ax.add_collection(collection)
        ax.set_title(f'Step {frame}/{len(frames) - 1}')
        return [collection]

    anim = FuncAnimation(
        fig, update, frames=len(frames),
        init_func=init, blit=False
    )

    pbar = tqdm(total=len(frames), desc=f"Saving {filename}")
    anim.save(filename, writer=writer, fps=fps, dpi=dpi,
              progress_callback=lambda *_: pbar.update(1))
    pbar.close()

    return anim, fig, ax


def load_config(overrides=None):
    """Load hydra config without creating experiment directories."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs")
    with hydra.initialize_config_dir(version_base=None, config_dir=os.path.abspath(config_path)):
        cfg = hydra.compose(config_name="config", overrides=overrides or [])
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot PolyHoop simulation polygons',
        epilog='Any extra arguments are passed as Hydra overrides (e.g. data=polyhoop data.n_objects=100)')
    parser.add_argument('--save-dir', '-s', help='Directory to save frames for later replotting')
    parser.add_argument('--load-dir', '-l', help='Directory to load previously saved frames from (skips simulation)')
    parser.add_argument('--color-by', '-c', default='area',
                        help='Field to color by: area, perimeter, phase, or any '
                             'per-cell metadata key (kh, khs, gams, kls, '
                             'growth_rates, area_fluct_periods, ...). '
                             "Use 'none' for a flat color (default: area)")
    parser.add_argument('--log-color', action='store_true',
                        help='Use log-scaled color norm (useful for kh/kl/gam '
                             'parameters sampled on a logspace dist)')
    parser.add_argument('--cmap', default='viridis', help='Colormap name (default: viridis)')
    parser.add_argument('--trajectory-idx', '-t', type=int, default=0,
                        help='Trajectory index (sets seed and resolves sweep params, default: 0)')
    parser.add_argument('--no-edges', action='store_true', help='Hide polygon edges')
    parser.add_argument('--output', '-o', default='simulation.mp4', help='Output video filename (default: simulation.mp4)')
    parser.add_argument('--fps', type=int, default=20, help='Video FPS (default: 20)')
    parser.add_argument('--dpi', type=int, default=150, help='Video DPI (default: 150)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[10, 10], help='Figure size (default: 10 10)')

    args, hydra_overrides = parser.parse_known_args()
    return args, hydra_overrides


def main():
    args, hydra_overrides = parse_args()
    color_by = None if args.color_by.lower() == 'none' else args.color_by

    if args.load_dir:
        # Load saved frames - no simulation needed
        frames, domain_size = load_frames(args.load_dir)
    else:
        # Run simulation
        cfg = load_config(hydra_overrides)
        print(OmegaConf.to_yaml(cfg.data))
        frames, domain_size = run_simulation(cfg, trajectory_idx=args.trajectory_idx)

        if args.save_dir:
            save_frames(args.save_dir, frames, domain_size, OmegaConf.to_yaml(cfg.data))

    # Plot initial state
    polygons, cell_data = frames[0]
    fig, ax = plot_polygons(
        polygons, cell_data, domain_size,
        color_by=color_by, cmap=args.cmap,
        figsize=tuple(args.figsize),
        show_edges=not args.no_edges,
        title='Initial state',
        log_color=args.log_color,
    )
    fig.savefig("initial_state.png", dpi=args.dpi, bbox_inches='tight')
    print("Saved initial_state.png")
    plt.close(fig)

    # Animate
    animate_frames(
        frames, domain_size,
        color_by=color_by, cmap=args.cmap,
        figsize=tuple(args.figsize),
        show_edges=not args.no_edges,
        filename=args.output, fps=args.fps, dpi=args.dpi,
        log_color=args.log_color,
    )
    print(f"Saved {args.output}")


if __name__ == '__main__':
    main()
