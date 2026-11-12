"""Supplementary figure: 4x5 grid of equally-spaced PolyHoop snapshots
across the four trajectory experiments used in the paper.

Each row is one experiment. Frames are sampled at trajectory fractions
0, 1/4, 1/2, 3/4, 1. Each row uses its own colormap matched to the
relevant figures in ``paper_figures/`` (oscillation rows reuse the
categorical colorblind palette from figure4v2).

Loads pre-saved frames from ``${POLYHOOP_SNAPSHOTS_DIR:-./data/polyhoop_snapshots}/<exp>/``
(produced by ``paper_figures/plot_vtp.py --save-dir``).
"""
import argparse
import os
import sys
from pathlib import Path


import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap

from plot_vtp import load_frames
from polyhoop_snapshots_supp import (
    _augment_with_derived,
    _global_cmap_norm,
    _pick_snapshot_indices,
)

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
matplotlib.rcParams["font.family"] = "DejaVu Sans"


DEFAULT_VTP_ROOT = Path(os.environ.get(
    "POLYHOOP_SNAPSHOTS_DIR", "./data/polyhoop_snapshots"
))

# Categorical colorblind palette shared with paper_figures/figure4v2.py.
_CB = sns.color_palette("colorblind", 10)
PALETTE_OSC = _CB[2:10]      # 8 colors — figure4v2 SPECS["oscillation_paper"]
PALETTE_MEMBRANE_STIFFNESS = _CB[0:2]   # 2 colors — figure4v2 SPECS["membrane_stiffness_clusters"]
PALETTE_ADHESION = _CB[6:8]  # 2 colors — figure4v2 SPECS["diff_adhesion_ablations"]


# Row order: osc continuous, osc clusters, membrane_stiffness, adhesion.
ROWS = [
    dict(
        experiment="paper_oscillating_continuous",
        row_label="osc. growth (continuous)",
        color_by="area_fluct_amp_x_freq",
        cmap="viridis",
        log_color=False,
        colorbar_label=r"$\alpha\omega$",
        start_frame=200,
    ),
    dict(
        experiment="more_osc_clusters",
        row_label="osc. growth (cluster)",
        color_by="area_fluct_amps",
        palette_slice=PALETTE_OSC,
        colorbar_label=r"$\alpha$",
    ),
    dict(
        experiment="membrane_stiffness_two_clusters",
        row_label="stiffness",
        color_by="kls",
        palette_slice=PALETTE_MEMBRANE_STIFFNESS,
        colorbar_label=r"$k_\ell$",
        colorbar_ticklabels=[r"$10^3$", r"$10^6$"],
    ),
    dict(
        experiment="differential_adhesion_clusters",
        row_label="adhesion",
        color_by="khs",
        palette_slice=PALETTE_ADHESION,
        colorbar_label=r"$k_h$",
        colorbar_ticklabels=[r"$10^4$", r"$5{\times}10^4$"],
    ),
]


def _categorical_cmap_setup(frames, color_by, palette_slice):
    """Set up the categorical colormap for a row matching figure4v2.

    Returns (cmap, norm, transform, tick_positions, tick_vals).
    ``transform`` maps raw values to integer category indices and is
    applied per-frame at draw time. The norm is the linear [-0.5, N-0.5]
    range so the resulting ListedColormap lookups work via cmap(norm(cat)).
    """
    all_vals = np.concatenate([
        np.asarray(cd[color_by]) for _, cd in frames if color_by in cd
    ])
    cap = len(palette_slice)
    unique = np.sort(np.unique(all_vals))

    if len(unique) <= cap:
        cmap = ListedColormap(palette_slice[:len(unique)])
        tick_vals = unique
        tick_positions = np.arange(len(unique))
        unique_arr = unique

        def transform(values):
            return np.searchsorted(unique_arr, np.asarray(values))
    else:
        edges = np.linspace(all_vals.min(), all_vals.max(), cap + 1)
        cmap = ListedColormap(palette_slice[:cap])
        tick_vals = 0.5 * (edges[:-1] + edges[1:])
        tick_positions = np.arange(cap)

        def transform(values):
            return np.clip(
                np.digitize(np.asarray(values), edges[1:-1]), 0, cap - 1,
            )

    n_cats = cmap.N
    norm = matplotlib.colors.Normalize(vmin=-0.5, vmax=n_cats - 0.5)
    return cmap, norm, transform, tick_positions, tick_vals


def _draw_panel(ax, polygons, cell_data, domain_size, color_by, cmap, norm,
                show_edges, edge_color, edge_width, transform=None):
    if color_by is None or color_by not in cell_data or cmap is None:
        facecolors = "lightblue"
    else:
        values = np.asarray(cell_data[color_by])
        if transform is not None:
            values = transform(values)
        facecolors = cmap(norm(values))

    coll = PolyCollection(
        polygons,
        facecolors=facecolors,
        edgecolors=edge_color if show_edges else "none",
        linewidths=edge_width,
    )
    ax.add_collection(coll)
    margin = 0.02 * domain_size
    ax.set_xlim(-margin, domain_size + margin)
    ax.set_ylim(-margin, domain_size + margin)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_color("black")


def _format_tick(v):
    return f"{v:.1f}".rstrip("0").rstrip(".") or "0"


def make_combined_figure(rows, output_path, vtp_root=DEFAULT_VTP_ROOT,
                         n_panels=5, fig_width=5.5, dpi=300,
                         show_edges=True, edge_color="black", edge_width=0.3):
    n_rows = len(rows)
    n_cols = n_panels

    left_m = 0.3
    right_m = 0.55
    col_gap = 0.04
    row_gap = 0.08
    top_m = 0.2
    bottom_m = 0.05

    panel_size = (fig_width - left_m - right_m - (n_cols - 1) * col_gap) / n_cols

    fig_w = fig_width
    fig_h = top_m + n_rows * panel_size + (n_rows - 1) * row_gap + bottom_m

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    panel_w = panel_size
    panel_h = panel_size
    wspace = col_gap / panel_w
    hspace = row_gap / panel_h

    gs = fig.add_gridspec(
        n_rows, n_cols,
        wspace=wspace, hspace=hspace,
        left=left_m / fig_w, right=1 - right_m / fig_w,
        top=1 - top_m / fig_h, bottom=bottom_m / fig_h,
    )

    fraction_titles = [r"$0$", r"$T/4$", r"$T/2$", r"$3T/4$", r"$T$"]

    for row_idx, row in enumerate(rows):
        load_dir = vtp_root / row["experiment"]
        frames, domain_size = load_frames(load_dir)
        _augment_with_derived(frames)

        is_categorical = bool(row.get("palette_slice"))
        if is_categorical:
            cmap_obj, norm, transform, tick_positions, tick_vals = (
                _categorical_cmap_setup(frames, row["color_by"],
                                        row["palette_slice"])
            )
        else:
            cmap_obj, norm = _global_cmap_norm(
                frames, row["color_by"], row.get("cmap", "viridis"),
                row.get("log_color", False),
            )
            transform = None

        start_frame = int(row.get("start_frame", 0))
        n_remaining = len(frames) - start_frame
        indices = start_frame + _pick_snapshot_indices(n_remaining, n_panels)

        row_axes = []
        for col_idx, frame_idx in enumerate(indices):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            polygons, cell_data = frames[frame_idx]
            _draw_panel(ax, polygons, cell_data, domain_size,
                        color_by=row["color_by"], cmap=cmap_obj, norm=norm,
                        show_edges=show_edges, edge_color=edge_color,
                        edge_width=edge_width, transform=transform)
            if row_idx == 0:
                ax.set_title(fraction_titles[col_idx], fontsize=9, pad=3)
            row_axes.append(ax)

        first_bbox = row_axes[0].get_position()
        fig.text(
            first_bbox.x0 - 0.008,
            0.5 * (first_bbox.y0 + first_bbox.y1),
            row["row_label"],
            fontsize=6, ha="center", va="center", rotation=90,
        )

        last_bbox = row_axes[-1].get_position()
        cax = fig.add_axes([
            last_bbox.x1 + 0.012,
            last_bbox.y0,
            0.010,
            last_bbox.height,
        ])
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])

        if is_categorical:
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_ticks(tick_positions)
            tick_labels = (
                row.get("colorbar_ticklabels")
                or [_format_tick(v) for v in tick_vals]
            )
            cbar.set_ticklabels(tick_labels)
        else:
            cbar_kwargs = {}
            if row.get("colorbar_ticks") is not None:
                cbar_kwargs["ticks"] = row["colorbar_ticks"]
            cbar = fig.colorbar(sm, cax=cax, **cbar_kwargs)
            if row.get("colorbar_ticklabels") is not None:
                cbar.set_ticklabels(row["colorbar_ticklabels"])

        cbar.set_label(row["colorbar_label"], fontsize=8, labelpad=3)
        cbar.ax.tick_params(labelsize=7)

    fig.patch.set_alpha(0.0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", transparent=True)
    print(f"Saved {output_path}")
    return fig


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vtp-root", default=str(DEFAULT_VTP_ROOT),
                   help="Root directory containing per-experiment frame "
                        "folders saved by paper_figures/plot_vtp.py")
    p.add_argument("--output", "-o",
                   default="./figures/polyhoop_snapshots_combined.pdf")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--fig-width", type=float, default=5.5,
                   help="Total figure width in inches.")
    p.add_argument("--no-edges", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    make_combined_figure(
        ROWS,
        output_path=args.output,
        vtp_root=Path(args.vtp_root).expanduser(),
        fig_width=args.fig_width,
        dpi=args.dpi,
        show_edges=not args.no_edges,
    )


if __name__ == "__main__":
    main()
