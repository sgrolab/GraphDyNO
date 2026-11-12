"""Supplementary figure: scale-only ablation across three datasets.

Layout (2x2):
    (a) osc. growth — embedding scatter colored by α
    (b) membrane stiffness — embedding scatter colored by k_ℓ
    (c) differential adhesion — embedding scatter colored by k_h
    (d) ARI bar chart comparing kMeans recovery across the three datasets
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial import KDTree
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
from _combo_common import TEX_IQR_DEFS, fmt_iqr, tex_iqr  # noqa: E402
from _ckpt_store import CkptRun, add_source_args, apply_source_args, ckpt_path  # noqa: E402
from _run_lookup import (load_emb_meta, load_experiments_yaml,  # noqa: E402
                         search_runs, setup_mlflow_uri)

setup_mlflow_uri()
_SCALE_CFG = load_experiments_yaml().get("scale_only", {})
SCALE_EXPERIMENT = _SCALE_CFG.get("experiment", "scale")

N_BINS = 20
K_NEIGHBORS = 10

METRIC_KEYS = ["NC", "ARI_kmeans", "NMI_kmeans", "ARI_hdbscan", "NMI_hdbscan"]
METRIC_LABELS = {
    "NC": "NC",
    "ARI_kmeans": r"ARI$_{\text{km}}$",
    "NMI_kmeans": r"NMI$_{\text{km}}$",
    "ARI_hdbscan": r"ARI$_{\text{hdb}}$",
    "NMI_hdbscan": r"NMI$_{\text{hdb}}$",
}
EMB_LIM = (-0.6, 0.6)

_CB = sns.color_palette("colorblind", 10)

SPECS = [
    dict(
        cache_dir_suffix="more_oscillating_clusters_new/",
        primary="area_fluct_amps",
        colorbar_label=r"$\alpha$",
        palette_slice=_CB[2:10],
        panel_title="Osc. growth",
        bar_label="Osc.",
    ),
    dict(
        cache_dir_suffix="membrane_stiffness_two_clusters_cache/",
        primary="kls",
        colorbar_label=r"$k_\ell$",
        colorbar_tick_override=[r"$10^3$", r"$10^6$"],
        palette_slice=_CB[0:2],
        panel_title="Membrane stiffness",
        bar_label="Stiffness",
    ),
    dict(
        cache_dir_suffix="differential_adhesion_clusters_cache/",
        primary="khs",
        colorbar_label=r"$k_h$",
        colorbar_tick_override=[r"$1\times 10^4$", r"$5\times 10^4$"],
        palette_slice=_CB[6:8],
        panel_title="Differential adhesion",
        bar_label="Adhesion",
    ),
]


def _categorical_cmap(vals, palette_slice, max_categories=None):
    cap = max_categories or len(palette_slice)
    unique = np.sort(np.unique(vals))
    if len(unique) <= cap:
        cat = np.searchsorted(unique, vals)
        cmap = ListedColormap(palette_slice[: len(unique)])
        tick_vals = unique
        tick_positions = np.arange(len(unique))
    else:
        edges = np.linspace(vals.min(), vals.max(), cap + 1)
        cat = np.clip(np.digitize(vals, edges[1:-1]), 0, cap - 1)
        cmap = ListedColormap(palette_slice[:cap])
        tick_vals = 0.5 * (edges[:-1] + edges[1:])
        tick_positions = np.arange(cap)
    return cat, cmap, tick_positions, tick_vals


def _normalize_emb(emb):
    centered = emb - emb.mean(axis=0, keepdims=True)
    span = centered.max(axis=0) - centered.min(axis=0)
    scale = span.max()
    if scale > 0:
        centered = centered / scale
    return centered


def _make_labels(target):
    unique_vals = np.unique(target)
    if len(unique_vals) <= N_BINS:
        return np.searchsorted(unique_vals, target), len(unique_vals)
    bins = np.linspace(target.min() - 1e-9, target.max() + 1e-9, N_BINS + 1)
    return np.digitize(target, bins), N_BINS


def _compute_metrics(emb, target):
    mask = np.isfinite(target) & np.isfinite(emb).all(axis=1)
    emb, target = emb[mask], target[mask]
    nans = {m: np.nan for m in METRIC_KEYS}
    if len(target) < K_NEIGHBORS + 2 or np.std(target) < 1e-12:
        return nans

    bins = np.linspace(target.min() - 1e-9, target.max() + 1e-9, N_BINS + 1)
    binned = np.digitize(target, bins)
    _, idx = KDTree(emb).query(emb, k=K_NEIGHBORS + 1)
    nc = float((binned[idx[:, 1:]] == binned[:, None]).mean())

    labels, n_clusters = _make_labels(target)
    out = {"NC": nc}

    if n_clusters < 2:
        out.update({m: np.nan for m in METRIC_KEYS if m != "NC"})
        return out

    km_ids = KMeans(n_clusters=n_clusters, n_init=10,
                    random_state=0).fit_predict(emb)
    out["ARI_kmeans"] = float(adjusted_rand_score(labels, km_ids))
    out["NMI_kmeans"] = float(normalized_mutual_info_score(labels, km_ids))

    min_cs = max(20, len(emb) // (2 * n_clusters))
    hdb_ids = HDBSCAN(min_cluster_size=min_cs).fit_predict(emb)
    valid = hdb_ids >= 0
    if valid.sum() >= 2:
        out["ARI_hdbscan"] = float(
            adjusted_rand_score(labels[valid], hdb_ids[valid]))
        out["NMI_hdbscan"] = float(
            normalized_mutual_info_score(labels[valid], hdb_ids[valid]))
    else:
        out["ARI_hdbscan"] = np.nan
        out["NMI_hdbscan"] = np.nan

    return out


def _fetch_runs_for_spec(spec):
    runs = search_runs(SCALE_EXPERIMENT)
    assert runs is not None, f"No experiment named {SCALE_EXPERIMENT!r}"
    out = []
    for r in runs:
        if r.info.status != "FINISHED":
            continue
        if r.data.tags.get("ablation") != "scale_only":
            continue
        cd = r.data.params.get("data.generator.cache_dir", "")
        if not cd.endswith(spec["cache_dir_suffix"]):
            continue
        out.append(r)
    return out


def _fig_seed(runs, primary):
    """The scatter panel's (ARI, embedding, values) from an export's fig_seed.pt, or None."""
    paths = {ckpt_path(r, "fig_seed") for r in runs if isinstance(r, CkptRun)} - {None}
    if not paths:
        return None
    sd = torch.load(paths.pop(), map_location="cpu", weights_only=False)
    emb = sd["parameter_embedding"].numpy()
    vals = sd[f"_meta_{primary}"].numpy()
    return _compute_metrics(emb, vals)["ARI_kmeans"], _normalize_emb(emb), vals


def _build_spec_data(spec):
    runs = _fetch_runs_for_spec(spec)
    primary = spec["primary"]

    seeds = []
    rows = []
    for r in runs:
        emb, meta = load_emb_meta(r)
        if emb is None or primary not in meta:
            continue
        metrics = _compute_metrics(emb, meta[primary])
        row = {"run_id": r.info.run_id}
        row.update(metrics)
        rows.append(row)
        ari = metrics["ARI_kmeans"]
        if np.isfinite(ari):
            seeds.append((ari, _normalize_emb(emb), meta[primary]))

    chosen = None
    if seeds:
        aris = np.array([s[0] for s in seeds])
        idx = int(np.argmin(np.abs(aris - np.mean(aris))))
        chosen = seeds[idx]
    pinned = _fig_seed(runs, primary)
    if pinned is not None:
        chosen = pinned

    df = pd.DataFrame(rows)
    return chosen, df


def main():
    sns.set_theme(style="ticks", context="paper")
    mpl.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    payloads = []
    for spec in SPECS:
        chosen, df = _build_spec_data(spec)
        n_runs = len(df)
        print(
            f"[{spec['panel_title']}] {n_runs} runs, ARI median [IQR] "
            f"{fmt_iqr(df['ARI_kmeans'])} (mean {df['ARI_kmeans'].mean():.3f})"
        )
        payloads.append((spec, chosen, df))

    fig = plt.figure(figsize=(5.5, 1.9))
    outer = fig.add_gridspec(
        nrows=1,
        ncols=3,
        wspace=0.75,
        left=0.08,
        right=0.92,
        top=0.90,
        bottom=0.20,
    )

    letters = ["a", "b", "c"]
    scatter_axes = []
    for i, (spec, chosen, df) in enumerate(payloads):
        ax = fig.add_subplot(outer[0, i])
        scatter_axes.append(ax)

        if chosen is None:
            ax.set_visible(False)
            continue

        ari_val, emb_n, color_vals = chosen
        cat, cmap, tick_positions, tick_vals = _categorical_cmap(
            color_vals, spec["palette_slice"]
        )
        vmin, vmax = -0.5, cmap.N - 0.5
        sc = ax.scatter(
            emb_n[:, 0],
            emb_n[:, 1],
            c=cat,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=2.5,
            edgecolors="face",
            linewidths=0.2,
            alpha=0.9,
        )
        ax.set_aspect("equal")
        ax.set_xlim(*EMB_LIM)
        ax.set_ylim(*EMB_LIM)
        ax.set_xlabel(r"$\lambda_0$", labelpad=2)
        ax.set_ylabel(r"$\lambda_1$", labelpad=2, rotation=0)
        ax.set_title(spec["panel_title"], pad=4)
        ax.tick_params(axis="both", length=2, pad=1)
        ax.text(
            -0.42,
            1.02,
            letters[i],
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        for s in ax.spines.values():
            s.set_edgecolor("black")
            s.set_linewidth(0.8)
            s.set_visible(True)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.04)
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label(spec["colorbar_label"], rotation=0, va="center")
        cb.ax.yaxis.set_label_coords(3.5, 0.5)
        cb.ax.tick_params(pad=1, labelsize=6, length=2)
        cb.set_ticks(tick_positions)
        tick_override = spec.get("colorbar_tick_override")
        if tick_override and len(tick_override) == len(tick_positions):
            cb.set_ticklabels(tick_override)
        else:
            cb.set_ticklabels(
                [
                    f"{v:.1f}".rstrip("0").rstrip(".") or "0"
                    for v in tick_vals
                ]
            )

    # Print full LaTeX metrics table (median [IQR] across runs)
    metric_headers = " & ".join(METRIC_LABELS[m] for m in METRIC_KEYS)
    col_spec = "l" + "c" * len(METRIC_KEYS)
    lines = [
        TEX_IQR_DEFS,
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        "Dataset & " + metric_headers + r" \\",
        r"\midrule",
    ]
    for spec, chosen, df in payloads:
        cells = [spec["panel_title"]]
        for m in METRIC_KEYS:
            cells.append(tex_iqr(df[m]))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    table_tex = "\n".join(lines)
    print("\n% --- LaTeX table ---")
    print(table_tex)

    save_dir = Path(_args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tex_path = save_dir / "scale_only_metrics.tex"
    tex_path.write_text(table_tex + "\n")
    print(f"saved {tex_path}")

    save_dir = Path(_args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = save_dir / "scale_only_supp.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    print(f"saved {out_pdf}")
    out_svg = save_dir / "scale_only_supp.svg"
    fig.savefig(out_svg, format="svg", bbox_inches="tight",
                pad_inches=0.02, transparent=True)
    print(f"saved {out_svg}")


def _parse_args():
    global SCALE_EXPERIMENT
    p = argparse.ArgumentParser(
        description="Supplementary figure — scale-only ablation"
    )
    p.add_argument("--save-dir", default="./figures")
    p.add_argument("--experiment", default=SCALE_EXPERIMENT,
                   help="MLflow experiment name (default: from experiments.yaml).")
    add_source_args(p)
    args = apply_source_args(p.parse_args())
    SCALE_EXPERIMENT = args.experiment
    return args


if __name__ == "__main__":
    _args = _parse_args()
    main()
