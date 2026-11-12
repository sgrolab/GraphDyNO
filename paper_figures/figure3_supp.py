"""Supplementary figure for figure3.

Both rows are one H = 64 run (`run_training.sh grok-h64`), the same run and
snapshots as figure3's row 2 (_grok_common):
Row 0: alpha and omega vs their best linear axis of the epoch-40 embedding.
Row 1: the same for the epoch-200 embedding.
"""
import argparse
import sys
from pathlib import Path


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

from figure3 import _projection_correlation
from _ckpt_store import add_source_args, apply_source_args
from _grok_common import EARLY, LATE, pick_snapshots
from _run_lookup import setup_mlflow_uri


def _best_linear_proj(emb, target):
    w = LinearRegression().fit(emb, target).coef_
    axis = w / np.linalg.norm(w)
    return emb @ axis

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
matplotlib.rcParams["font.family"] = "DejaVu Sans"


def make_figure(grok_run_name=None,
                grok_experiment=None,
                output_path="./figures/figure3_supp.png",
                dpi=300,
                early=EARLY,
                late=LATE):
    sns.set_context("paper")
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8

    snaps = pick_snapshots(grok_experiment, grok_run_name)
    print(f"run {snaps.run.info.run_name}, epochs {early} / {late}")

    osc_emb = snaps[early]
    osc_alpha = snaps.targets["alpha"]
    osc_omega = snaps.targets["omega"]
    osc_alpha_proj = _best_linear_proj(osc_emb, osc_alpha)
    osc_omega_proj = _best_linear_proj(osc_emb, osc_omega)

    grok_emb = snaps[late]
    grok_alpha, grok_omega = osc_alpha, osc_omega
    grok_alpha_proj = _best_linear_proj(grok_emb, grok_alpha)
    grok_omega_proj = _best_linear_proj(grok_emb, grok_omega)

    fig_w = 5.5
    n_rows, n_cols = 2, 2
    left_m = 0.35
    right_m = 0.1
    col_gap = 0.55
    top_m = 0.15
    bottom_m = 0.3
    row_gap = 0.55
    panel_w = (fig_w - left_m - right_m - (n_cols - 1) * col_gap) / n_cols
    panel_h = panel_w
    fig_h = top_m + n_rows * panel_h + (n_rows - 1) * row_gap + bottom_m
    wspace = col_gap / panel_w
    hspace = row_gap / panel_h

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = fig.add_gridspec(
        n_rows, n_cols,
        wspace=wspace, hspace=hspace,
        left=left_m / fig_w, right=1 - right_m / fig_w,
        top=1 - top_m / fig_h, bottom=bottom_m / fig_h,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    _projection_correlation(ax_a, osc_alpha_proj, osc_alpha, r"$\alpha$",
                            axis_name=r"\hat{\alpha}\text{-axis}",
                            axis_label_color="black",
                            text_loc="upper left", embedding=osc_emb)
    ax_b = fig.add_subplot(gs[0, 1])
    _projection_correlation(ax_b, osc_omega_proj, osc_omega, r"$\omega$",
                            axis_name=r"\hat{\omega}\text{-axis}",
                            axis_label_color="black",
                            text_loc="upper left", embedding=osc_emb)
    ax_c = fig.add_subplot(gs[1, 0])
    _projection_correlation(ax_c, grok_alpha_proj, grok_alpha, r"$\alpha$",
                            axis_name=r"\hat{\alpha}\text{-axis}",
                            axis_label_color="black",
                            text_loc="upper left", embedding=grok_emb)
    ax_d = fig.add_subplot(gs[1, 1])
    _projection_correlation(ax_d, grok_omega_proj, grok_omega, r"$\omega$",
                            axis_name=r"\hat{\omega}\text{-axis}",
                            axis_label_color="black",
                            text_loc="upper left", embedding=grok_emb)

    axes_grid = [(ax_a, ax_b), (ax_c, ax_d)]
    labels_grid = [("a", "b"), ("c", "d")]
    for row_axes, row_labels in zip(axes_grid, labels_grid):
        for ax, label in zip(row_axes, row_labels):
            bbox = ax.get_position()
            fig.text(bbox.x0 - 0.015, bbox.y1 + 0.005, label,
                     fontweight="bold", va="bottom", ha="left")

    fig.patch.set_alpha(0.0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    svg_path = output_path.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", format="svg", transparent=True)
    pdf_path = output_path.with_name("figure3_supp.pdf")
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf", transparent=True)
    print(f"Saved {output_path}")
    print(f"Saved {svg_path}")
    print(f"Saved {pdf_path}")
    return fig


def _parse_args():
    p = argparse.ArgumentParser(description="Supplementary figure for figure3")
    p.add_argument("--grok-experiment", default=None,
                   help="H = 64 experiment (default: grok_h64 in experiments.yaml)")
    p.add_argument("--grok-run", default=None,
                   help="Explicit MLflow run name (default: grok_h64.figure_seed).")
    p.add_argument("--early", type=int, default=EARLY, help="snapshot epoch for row 0")
    p.add_argument("--late", type=int, default=LATE, help="snapshot epoch for row 1")
    p.add_argument("--output", default="./figures/figure3_supp.png")
    p.add_argument("--dpi", type=int, default=300)
    add_source_args(p)
    return apply_source_args(p.parse_args())


if __name__ == "__main__":
    setup_mlflow_uri()
    args = _parse_args()
    make_figure(
        grok_run_name=args.grok_run,
        grok_experiment=args.grok_experiment,
        output_path=args.output,
        dpi=args.dpi,
        early=args.early,
        late=args.late,
    )
