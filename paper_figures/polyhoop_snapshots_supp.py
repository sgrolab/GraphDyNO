"""Supplementary figure: 1x5 grid of equally-spaced PolyHoop snapshots.

Runs (or loads) a PolyHoop trajectory using the same machinery as
``paper_figures/plot_vtp.py``, then renders five equally-spaced frames
(including the first and last) side-by-side.

Usage:
    # Run a fresh trajectory and save the figure
    python paper_figures/polyhoop_snapshots_supp.py data=polyhoop

    # Re-render from frames previously saved by plot_vtp.py
    python paper_figures/polyhoop_snapshots_supp.py --load-dir runs/my_sim
"""
import argparse
import sys
from pathlib import Path


import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from omegaconf import OmegaConf

from plot_vtp import _make_norm, load_config, load_frames, run_simulation

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
matplotlib.rcParams["font.family"] = "DejaVu Sans"


def _pick_snapshot_indices(n_frames, n_panels=5):
    """Return n_panels indices equally spaced over the [0, n_frames] interval
    (i.e. trajectory fractions 0, 1/(n-1), ..., 1), clipped to [0, n_frames-1].

    Using ``linspace(0, n_frames - 1, ...)`` gives spacings of (n-1)/(k-1),
    which off-by-ones the "natural" round numbers (e.g. 1499 instead of
    1500 for n=2000, k=5). This version uses the exact trajectory fraction
    so the labels land on T/4, T/2, 3T/4, etc.
    """
    fractions = np.linspace(0.0, 1.0, n_panels)
    return np.clip((fractions * n_frames).astype(int), 0, n_frames - 1)


def _augment_with_derived(frames):
    """Inject derived color-by keys (e.g. amplitude * frequency) into each
    frame's cell_data dict in place. Keys are added only if their inputs
    are present."""
    for _, cell_data in frames:
        amps = cell_data.get("area_fluct_amps")
        periods = cell_data.get("area_fluct_periods")
        if amps is not None and periods is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                omega = np.where(periods > 0, 2.0 * np.pi / periods, 0.0)
            cell_data["area_fluct_amp_x_freq"] = np.asarray(amps) * omega
            cell_data["area_fluct_amp_x_period"] = (
                np.asarray(amps) * np.asarray(periods)
            )


def _resolve_cmap_and_norm(values, cmap_name, log_color):
    """Build a (Colormap, Normalize) pair for the given values.

    Recognizes paper-figure-specific styles in addition to standard
    matplotlib colormap names:
        - 'colorblind_pair' — bimodal 2-color discrete cmap (matches the kl
          and kh scatter in figure3)
        - 'truncated_blues' / 'truncated_greens' — perceptual cmaps with the
          low-end truncated, matching figure3's period/amplitude panels
    """
    if cmap_name == "colorblind_pair":
        cb = sns.color_palette("colorblind")
        cmap = ListedColormap([cb[0], cb[1]], name="colorblind_pair")
        lo = float(np.min(values))
        hi = float(np.max(values))
        if lo == hi:
            hi = lo + 1.0
        mid = 0.5 * (lo + hi)
        norm = BoundaryNorm([lo, mid, hi], cmap.N)
        return cmap, norm
    if cmap_name in ("truncated_blues", "truncated_greens"):
        base = "Blues" if cmap_name.endswith("blues") else "Greens"
        base_cmap = plt.get_cmap(base)
        cmap = LinearSegmentedColormap.from_list(
            f"trunc_{base.lower()}", base_cmap(np.linspace(0.25, 1.0, 256)),
        )
        return cmap, _make_norm(np.asarray(values), log_color)
    return plt.get_cmap(cmap_name), _make_norm(np.asarray(values), log_color)


def _global_cmap_norm(frames, color_by, cmap_name, log_color):
    if color_by is None:
        return None, None
    values = []
    for _, cell_data in frames:
        if color_by in cell_data:
            values.extend(cell_data[color_by])
    if not values:
        return None, None
    return _resolve_cmap_and_norm(np.asarray(values), cmap_name, log_color)


def _draw_panel(ax, polygons, cell_data, domain_size, color_by, cmap, norm,
                show_edges, edge_color, edge_width):
    if color_by is None or color_by not in cell_data or cmap is None:
        facecolors = "lightblue"
    else:
        values = np.asarray(cell_data[color_by])
        facecolors = cmap(norm(values))

    collection = PolyCollection(
        polygons,
        facecolors=facecolors,
        edgecolors=edge_color if show_edges else "none",
        linewidths=edge_width,
    )
    ax.add_collection(collection)
    margin = 0.02 * domain_size
    ax.set_xlim(-margin, domain_size + margin)
    ax.set_ylim(-margin, domain_size + margin)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_figure(frames, domain_size, output_path,
                color_by="area", cmap="viridis", log_color=False,
                show_edges=True, edge_color="black", edge_width=0.4,
                n_panels=5, dpi=300, panel_size=1.6, colorbar=True,
                colorbar_label=None, colorbar_ticks=None,
                colorbar_ticklabels=None):
    indices = _pick_snapshot_indices(len(frames), n_panels)
    cmap_obj, norm = _global_cmap_norm(frames, color_by, cmap, log_color)

    fig_w = panel_size * n_panels + 0.2 + (0.45 if colorbar else 0.05)
    left_m = 0.05
    right_m = 0.45 if colorbar else 0.05
    col_gap = 0.08
    top_m = 0.25
    bottom_m = 0.25
    panel_w = (fig_w - left_m - right_m - (n_panels - 1) * col_gap) / n_panels
    panel_h = panel_w
    fig_h = top_m + panel_h + bottom_m
    wspace = col_gap / panel_w

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = fig.add_gridspec(
        1, n_panels,
        wspace=wspace, hspace=0.0,
        left=left_m / fig_w, right=1 - right_m / fig_w,
        top=1 - top_m / fig_h, bottom=bottom_m / fig_h,
    )

    axes = []
    for col, idx in enumerate(indices):
        ax = fig.add_subplot(gs[0, col])
        polygons, cell_data = frames[idx]
        _draw_panel(ax, polygons, cell_data, domain_size,
                    color_by=color_by, cmap=cmap_obj, norm=norm,
                    show_edges=show_edges, edge_color=edge_color,
                    edge_width=edge_width)
        ax.set_title(f"step {int(idx)}", fontsize=8, pad=2)
        axes.append(ax)

    if colorbar and norm is not None:
        last_bbox = axes[-1].get_position()
        cax = fig.add_axes([
            last_bbox.x1 + 0.015,
            last_bbox.y0,
            0.012,
            last_bbox.height,
        ])
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar_kwargs = {}
        if colorbar_ticks is not None:
            cbar_kwargs["ticks"] = colorbar_ticks
        cbar = fig.colorbar(sm, cax=cax, **cbar_kwargs)
        if colorbar_ticklabels is not None:
            cbar.set_ticklabels(colorbar_ticklabels)
        cbar.set_label(colorbar_label or color_by, fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.patch.set_alpha(0.0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    print(f"Saved {output_path}")
    return fig


def _parse_args():
    p = argparse.ArgumentParser(
        description="1x5 grid of equally-spaced PolyHoop trajectory snapshots",
        epilog="Extra arguments are passed as Hydra overrides (e.g. data=polyhoop).",
    )
    p.add_argument("--load-dir", "-l",
                   help="Directory of frames saved by paper_figures/plot_vtp.py "
                        "(skips simulation).")
    p.add_argument("--trajectory-idx", "-t", type=int, default=0,
                   help="Trajectory index when running a fresh simulation.")
    p.add_argument("--color-by", "-c", default="area",
                   help="Field to color cells by, or 'none' for flat. "
                        "Defaults to 'area'.")
    p.add_argument("--log-color", action="store_true",
                   help="Use log-scaled color norm.")
    p.add_argument("--cmap", default="viridis",
                   help="Standard matplotlib name, or 'colorblind_pair' for "
                        "the bimodal 2-color cmap from figure3.")
    p.add_argument("--colorbar-label", default=None,
                   help="Override the colorbar axis label (defaults to "
                        "the --color-by key).")
    p.add_argument("--colorbar-ticks", type=float, nargs="+", default=None,
                   help="Explicit numeric tick locations on the colorbar.")
    p.add_argument("--colorbar-ticklabels", nargs="+", default=None,
                   help="Tick labels paired 1:1 with --colorbar-ticks "
                        "(supports LaTeX, e.g. '$10^3$').")
    p.add_argument("--no-edges", action="store_true")
    p.add_argument("--n-panels", type=int, default=5)
    p.add_argument("--panel-size", type=float, default=1.6,
                   help="Side length of each square panel in inches.")
    p.add_argument("--no-colorbar", action="store_true")
    p.add_argument("--output", "-o",
                   default="./figures/polyhoop_snapshots_supp.pdf")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_known_args()


def main():
    args, hydra_overrides = _parse_args()
    color_by = None if args.color_by.lower() == "none" else args.color_by

    if args.load_dir:
        frames, domain_size = load_frames(args.load_dir)
    else:
        cfg = load_config(hydra_overrides)
        print(OmegaConf.to_yaml(cfg.data))
        frames, domain_size = run_simulation(
            cfg, trajectory_idx=args.trajectory_idx, include_init=False,
        )

    _augment_with_derived(frames)

    make_figure(
        frames, domain_size,
        output_path=args.output,
        color_by=color_by,
        cmap=args.cmap,
        log_color=args.log_color,
        show_edges=not args.no_edges,
        n_panels=args.n_panels,
        panel_size=args.panel_size,
        colorbar=not args.no_colorbar,
        colorbar_label=args.colorbar_label,
        colorbar_ticks=args.colorbar_ticks,
        colorbar_ticklabels=args.colorbar_ticklabels,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
