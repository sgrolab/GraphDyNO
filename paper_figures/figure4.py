"""Figure 4: full-model comparison (DINO / Conv / Fourier) across three datasets.

Layout (4 rows):
    Rows 1-3 — a tightly packed 3x3 scatter block. Each row is one dataset
               and shows the median-ARI seed's parameter embedding per encoder
               (DINO / Conv / Fourier). Rows share x/y limits; encoder
               column titles appear only above row 1; the λ₀ x-axis label
               and tick labels appear only on row 3.
        Row 1: oscillation_paper (dim=2, embedding_noise=0.1), color = α
        Row 2: tension_clusters (L_dropout=0.5, n_coeffs=32), color = k̃_L
        Row 3: diff_adhesion_ablations, color = k_h
    Row 4   — three ARI bar panels (one per dataset) with chance lines,
              comparing the three full models: median bar, IQR whiskers.

Rendered natively in matplotlib so the output is fully vector.
"""

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
from _combo_common import iqr_errorbar  # noqa: E402
from _run_lookup import load_experiments_yaml, setup_mlflow_uri  # noqa: E402

setup_mlflow_uri()
client = mlflow.tracking.MlflowClient()
_EXPERIMENTS_CFG = load_experiments_yaml().get("sweeps", {})

EMBEDDINGS = ["dino", "fourier", "conv"]
EMBEDDING_LABELS = {"dino": "DINO", "conv": "Conv", "fourier": "Fourier"}
FULL_COLOR = "#785ef0"
N_BINS = 20
EMB_LIM = (-0.6, 0.6)

# Both rows use the seaborn `colorblind` categorical palette. We slice it
# so rows 1 (a-c) and 2 (d-f) use disjoint hues:
#   Row 2 (kls, 2 clusters) → colorblind indices 0..1
#   Row 1 (α·ω, remaining clusters) → colorblind indices 2..9
_CB = sns.color_palette("colorblind", 10)

SPECS = [
    dict(experiment=_EXPERIMENTS_CFG.get("oscillation", "oscillation_paper"),
         filters={"model.embedding_noise_std": "0.1",
                  "model.interaction_dim": "2"},
         primary="area_fluct_amps",
         colorbar_label=r"$\alpha$",
         palette_slice=_CB[2:10],
         row_label="osc. growth"),
    dict(experiment=_EXPERIMENTS_CFG.get("tension", "tension_clusters"),
         filters={},
         primary="kls",
         colorbar_label=r"$k_\ell$",
         colorbar_tick_override=[r"$10^3$", r"$10^6$"],
         palette_slice=_CB[0:2],
         row_label="membrane stiffness"),
    dict(experiment=_EXPERIMENTS_CFG.get("adhesion", "diff_adhesion_ablations"),
         filters={},
         primary="khs",
         colorbar_label=r"$k_h$",
         palette_slice=_CB[6:8],
         colorbar_tick_override=[r"$1\times 10^4$", r"$5\times 10^4$"],
         row_label="adhesion"),
]


def _categorical_cmap(vals, palette_slice, max_categories=None):
    """Map per-cell values to categorical indices + a ListedColormap.

    If there are <= len(palette_slice) unique values, each unique value
    gets its own color. Otherwise values are binned into equal-width
    bins, one color per bin.
    """
    cap = max_categories or len(palette_slice)
    unique = np.sort(np.unique(vals))
    if len(unique) <= cap:
        cat = np.searchsorted(unique, vals)
        cmap = ListedColormap(palette_slice[:len(unique)])
        tick_vals = unique
        tick_positions = np.arange(len(unique))
    else:
        edges = np.linspace(vals.min(), vals.max(), cap + 1)
        # Clip so the max value lands in the last bin.
        cat = np.clip(np.digitize(vals, edges[1:-1]), 0, cap - 1)
        cmap = ListedColormap(palette_slice[:cap])
        tick_vals = 0.5 * (edges[:-1] + edges[1:])
        tick_positions = np.arange(cap)
    return cat, cmap, tick_positions, tick_vals


def _run_dir(run):
    art = Path(urlparse(run.info.artifact_uri).path)
    if (art / "graph_decoder_model.pth").exists():
        return art
    hdir = run.data.tags.get("hydra_output_dir")
    if hdir and (Path(hdir) / "graph_decoder_model.pth").exists():
        return Path(hdir)
    return None


def _load_emb_meta(run):
    rdir = _run_dir(run)
    if rdir is None:
        return None, None
    sd = torch.load(rdir / "graph_decoder_model.pth",
                    map_location="cpu", weights_only=False)
    emb = sd["parameter_embedding"].numpy()
    meta = {k[len("_meta_"):]: sd[k].numpy()
            for k in sd if k.startswith("_meta_")}
    if "area_fluct_periods" in meta and "area_fluct_amps" in meta:
        meta["area_fluct_amp_x_period"] = (
            meta["area_fluct_periods"] * meta["area_fluct_amps"])
    return emb, meta


def _normalize_emb(emb):
    centered = emb - emb.mean(axis=0, keepdims=True)
    span = centered.max(axis=0) - centered.min(axis=0)
    scale = span.max()
    if scale > 0:
        centered = centered / scale
    return centered


def _recovery_ari(emb, target):
    mask = np.isfinite(target) & np.isfinite(emb).all(axis=1)
    emb, target = emb[mask], target[mask]
    if len(target) < 4 or np.std(target) < 1e-12:
        return np.nan
    unique_vals = np.unique(target)
    if len(unique_vals) <= N_BINS:
        labels = np.searchsorted(unique_vals, target)
    else:
        bins = np.linspace(target.min() - 1e-9, target.max() + 1e-9, N_BINS + 1)
        labels = np.digitize(target, bins)
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        return np.nan
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    km_ids = km.fit_predict(emb)
    return float(adjusted_rand_score(labels, km_ids))


def _fetch_full_runs(experiment_name, param_filters):
    exps = [e.experiment_id for e in client.search_experiments()
            if e.name == experiment_name]
    if not exps:
        print(f"  [{experiment_name}] not found in MLflow — skipping row.")
        return {e: [] for e in EMBEDDINGS}
    runs = mlflow.search_runs(experiment_ids=exps, output_format="list")
    out = {e: [] for e in EMBEDDINGS}
    _has_conv_fix = False
    for r in runs:
        if r.info.status != "FINISHED":
            continue
        e = r.data.tags.get("embedding")
        # `full_velocity` is the standard tag; diff_adhesion_ablations uses
        # the `_int` suffix variant `full_int`.
        if r.data.tags.get("ablation") not in ("full_velocity", "full_int"):
            continue
        if e == "conv_fix":
            e = "conv"
            _has_conv_fix = True
        if e not in EMBEDDINGS:
            continue
        if any(r.data.params.get(k) is not None
               and str(r.data.params.get(k)) != str(v)
               for k, v in param_filters.items()):
            continue
        out[e].append(r)
    if _has_conv_fix:
        filtered = [r for r in out["conv"]
                    if r.data.tags.get("embedding") == "conv_fix"]
        out["conv"] = filtered
        print(f"  [{experiment_name}] using conv_fix runs only (dropping plain conv)")
    return out


def _build_spec_data(spec):
    runs_by_emb = _fetch_full_runs(spec["experiment"], spec["filters"])
    primary = spec["primary"]

    # Pick the median-ARI seed for that encoder. Embeddings can't be averaged
    # directly across seeds because each is in a separately rotated/reflected
    # coordinate frame.
    seeds_per_emb = {e: [] for e in EMBEDDINGS}
    rows = []
    for e in EMBEDDINGS:
        for r in runs_by_emb[e]:
            emb, meta = _load_emb_meta(r)
            if emb is None or primary not in meta:
                continue
            ari = _recovery_ari(emb, meta[primary])
            rows.append({
                "embedding": e,
                "embedding_label": EMBEDDING_LABELS[e],
                "run_id": r.info.run_id,
                "val_loss_no_shape": (
                    r.data.metrics.get("val_position_mse", np.nan)
                    + r.data.metrics.get("val_velocity_mse", np.nan)
                    + r.data.metrics.get("val_scale_mse", np.nan)),
                "ARI": ari,
            })
            if np.isfinite(ari):
                seeds_per_emb[e].append((ari, _normalize_emb(emb), meta[primary]))

    chosen_per_emb = {}
    for e, seeds in seeds_per_emb.items():
        if not seeds:
            continue
        # With an even number of seeds the median falls between the two middle
        # seeds; take the lower one so the panel never looks better than the median.
        order = np.argsort([s[0] for s in seeds], kind="stable")
        chosen_per_emb[e] = seeds[order[(len(seeds) - 1) // 2]]

    scatter_cache = {e: (emb_n, vals) for e, (_, emb_n, vals) in chosen_per_emb.items()}
    scatter_ari = {e: ari for e, (ari, _, _) in chosen_per_emb.items()}

    df = pd.DataFrame(rows)
    if df.empty:
        return scatter_cache, df, scatter_ari
    df["embedding"] = pd.Categorical(df["embedding"],
                                     categories=EMBEDDINGS, ordered=True)
    df["embedding_label"] = pd.Categorical(
        df["embedding_label"],
        categories=[EMBEDDING_LABELS[e] for e in EMBEDDINGS], ordered=True)
    return scatter_cache, df, scatter_ari


def draw_scatter_row(fig, row_slot, scatter_cache, df, colorbar_label,
                     palette_slice, scatter_ari=None,
                     tick_override=None,
                     panel_letters=("a", "b", "c"),
                     show_column_titles=False,
                     show_xlabel=False):
    """Render one row: 3 embedding scatters + thin shared colorbar.

    `palette_slice` is a list of RGB tuples — each unique data value gets
    one color. `scatter_ari` is the ARI of the displayed (median-ARI) seed
    per encoder. `show_column_titles` puts DINO/Conv/Fourier above row-0
    only; `show_xlabel` puts the λ₀ label and x tick labels on the bottom
    row only.
    """
    sub = row_slot.subgridspec(1, 4, width_ratios=[1, 1, 1, 0.04],
                               wspace=0.02)
    axes = []
    for j in range(3):
        sharex = axes[0] if j else None
        sharey = axes[0] if j else None
        axes.append(fig.add_subplot(sub[0, j], sharex=sharex, sharey=sharey))
    cax = fig.add_subplot(sub[0, 3])
    scatter_ari = scatter_ari or {}
    ref_vals = next(iter(scatter_cache.values()))[1]
    _, cmap, tick_positions, tick_vals = _categorical_cmap(ref_vals, palette_slice)
    vmin = -0.5
    vmax = cmap.N - 0.5
    sc = None
    for j, e in enumerate(EMBEDDINGS):
        ax = axes[j]
        if e not in scatter_cache:
            ax.set_visible(False)
            continue
        emb_n, color_vals = scatter_cache[e]
        cat, _, _, _ = _categorical_cmap(color_vals, palette_slice)
        sc = ax.scatter(emb_n[:, 0], emb_n[:, 1], c=cat,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        s=2.5, edgecolors="face", linewidths=0.2, alpha=0.9)
        ax.set_box_aspect(1)
        ax.set_xlim(*EMB_LIM)
        ax.set_ylim(*EMB_LIM)
        ax.tick_params(axis="both", length=2, pad=1)
        ari = scatter_ari.get(e, np.nan)
        ax.text(0.04, 0.04, f"ARI={ari:.2f}",
                transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", edgecolor="none", alpha=0.75))
        if show_column_titles:
            ax.set_title(EMBEDDING_LABELS[e], pad=2, fontweight="bold")
        ax.text(0.04, 0.96, panel_letters[j], transform=ax.transAxes,
                fontsize=10, fontweight="bold", ha="left", va="top")
        if j == 0:
            ax.set_ylabel(r"$\lambda_1$", labelpad=2)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        if show_xlabel:
            ax.set_xlabel(r"$\lambda_0$", labelpad=2)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        for s in ax.spines.values():
            s.set_edgecolor("black"); s.set_linewidth(0.8); s.set_visible(True)

    if sc is not None:
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label(colorbar_label, labelpad=2)
        cb.ax.yaxis.set_label_coords(5.0, 0.5)
        cb.ax.tick_params(pad=1, labelsize=6, length=2)
        cb.set_ticks(tick_positions)
        if tick_override is not None and len(tick_override) == len(tick_positions):
            cb.set_ticklabels(tick_override)
        else:
            cb.set_ticklabels([f"{v:.1f}".rstrip("0").rstrip(".") or "0"
                               for v in tick_vals])
    return axes


_DATASET_LABELS = {
    "oscillation_paper": "Osc. growth",
    "tension_clusters": "Membrane stiffness",
    "diff_adhesion_ablations": "Differential adhesion",
}
_LATEX_ENCODER_ORDER = ["dino", "conv", "fourier"]


def _clamped_sd_cell(aris):
    arr = np.asarray(aris, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "--"
    m = float(arr.mean())
    s = float(arr.std())
    err = min(s, m + 1.0, 1.0 - m)
    return f"${m:.3f} \\pm {err:.3f}$"


def draw_ari_row(fig, row_slot, dfs, specs, chance_levels, panel_letters,
                 ymin=-0.05):
    """One ARI bar panel per dataset (variable count), y from `ymin` to 1."""
    sub = row_slot.subgridspec(1, max(len(dfs), 1), wspace=0.75)
    order = [EMBEDDING_LABELS[e] for e in EMBEDDINGS]
    axes = []
    for col, (df, spec, chance, letter) in enumerate(
            zip(dfs, specs, chance_levels, panel_letters)):
        ax = fig.add_subplot(sub[0, col])
        sns.barplot(data=df, x="embedding_label", y="ARI",
                    order=order, color=FULL_COLOR,
                    edgecolor="none", estimator="median",
                    errorbar=iqr_errorbar,
                    err_kws={"linewidth": 1.6}, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(f"ARI ({spec['colorbar_label']})", labelpad=2)
        ax.set_ylim(ymin, 1)
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
        ax.axhline(chance, linestyle="--", color="black", linewidth=0.8)
        ax.tick_params(axis="x", rotation=40, pad=1)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
            lab.set_rotation_mode("anchor")
        sns.despine(ax=ax)
        ax.text(-0.15, 1.08, letter, transform=ax.transAxes,
                fontsize=10, fontweight="bold", ha="right", va="bottom")
        axes.append(ax)
    return axes


def main():
    sns.set_theme(style="ticks", context="paper")
    # Talk-like text sizing (bumped above paper defaults) at 5.5" width.
    mpl.rcParams.update({
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
    })

    payloads = []
    for spec in SPECS:
        scatter_cache, df, scatter_ari = _build_spec_data(spec)
        n_by = df.groupby("embedding", observed=True).size().to_dict() if not df.empty else {}
        shown_ari = {e: f"{v:.3f}" for e, v in scatter_ari.items()}
        print(f"[{spec['experiment']}] per-embedding seeds: {n_by}  "
              f"median-seed ARI: {shown_ari}")
        if df.empty and not scatter_cache:
            print(f"  [{spec['experiment']}] no matching runs — dropping row.")
            continue
        payloads.append((spec, scatter_cache, df, scatter_ari))

    if not payloads:
        raise SystemExit("No experiments with matching runs were found; "
                         "nothing to render.")

    # Measure bottom margin programmatically from the rotated tick-label
    # extent of the bottom-row ARI bars (same approach as figure5.py), then
    # recompute fig_h so there is no slack below the labels.
    fig_w = 5.5
    top_m_in = 0.275
    order = [EMBEDDING_LABELS[e] for e in EMBEDDINGS]

    # The ARI axes start just below 0 so the dashed chance line stays visible,
    # or lower if an IQR whisker dips below that.
    ari_ymin = float(np.nanmin([-0.05] + [
        iqr_errorbar(g)[0]
        for _, _, df, _ in payloads if not df.empty
        for _, g in df.groupby("embedding", observed=True)["ARI"]]))

    # Row heights in inches. The bar row keeps the inches per ARI unit it had
    # with a -0.5..1 axis, so a shorter ARI range shortens the figure.
    scatter_h_in = 3.2
    gap_in = 0.5133
    bar_h_in = 0.9067 * (1 - ari_ymin) / 1.5
    content_h_in = scatter_h_in + gap_in + bar_h_in

    _tmp_fig = plt.figure(figsize=(fig_w, 2.0), dpi=300)
    _tmp_ax = _tmp_fig.add_axes([0.2, 0.5, 0.2, 0.3])
    sns.barplot(data=payloads[0][2], x="embedding_label", y="ARI",
                order=order, color=FULL_COLOR, edgecolor="none",
                estimator="median", errorbar=iqr_errorbar, ax=_tmp_ax)
    _tmp_ax.tick_params(axis="x", rotation=40, pad=1)
    for lab in _tmp_ax.get_xticklabels():
        lab.set_ha("right")
        lab.set_rotation_mode("anchor")
    _tmp_fig.canvas.draw()
    _renderer = _tmp_fig.canvas.get_renderer()
    _ax_bb = _tmp_ax.get_window_extent(_renderer)
    _min_label_y = _ax_bb.y0
    for _lbl in _tmp_ax.get_xticklabels():
        _bb = _lbl.get_window_extent(_renderer)
        _min_label_y = min(_min_label_y, _bb.y0)
    _label_drop = (_ax_bb.y0 - _min_label_y) / _tmp_fig.dpi
    plt.close(_tmp_fig)
    bottom_m_in = _label_drop + 0.03

    fig_h = top_m_in + content_h_in + bottom_m_in
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[scatter_h_in, bar_h_in],
        hspace=gap_in / ((scatter_h_in + bar_h_in) / 2),
        left=0.21, right=0.83,
        top=1 - top_m_in / fig_h,
        bottom=bottom_m_in / fig_h,
    )
    scatter_block = outer[0].subgridspec(len(payloads), 1, hspace=0.08)

    scatter_letters = [("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i")][:len(payloads)]
    left_axes = []
    for i, (spec, scatter_cache, df, scatter_ari) in enumerate(payloads):
        row_axes = draw_scatter_row(
            fig, scatter_block[i], scatter_cache, df,
            colorbar_label=spec["colorbar_label"],
            palette_slice=spec["palette_slice"],
            scatter_ari=scatter_ari,
            tick_override=spec.get("colorbar_tick_override"),
            panel_letters=scatter_letters[i],
            show_column_titles=(i == 0),
            show_xlabel=(i == len(payloads) - 1))
        left_axes.append((row_axes[0], spec.get("row_label")))

    chance_levels = []
    dfs = []
    specs_for_ari = []
    for spec, scatter_cache, df, _ari in payloads:
        chance_levels.append(0.0)
        dfs.append(df)
        specs_for_ari.append(spec)
    bar_letters = ("j", "k", "l")[:len(payloads)]
    draw_ari_row(fig, outer[1], dfs, specs_for_ari, chance_levels,
                 panel_letters=bar_letters, ymin=ari_ymin)

    # Row labels along the left margin, just outside each row's λ₁ ylabel.
    # Mirrors notebooks/more_oscillation_analysis.py:813-826.
    fig.canvas.draw()
    for left_ax, row_label in left_axes:
        if not row_label:
            continue
        ylabel_bbox = left_ax.yaxis.get_label().get_window_extent(
            renderer=fig.canvas.get_renderer())
        inv = fig.transFigure.inverted()
        x_fig = inv.transform((ylabel_bbox.x0, 0))[0] - 0.01
        pos = left_ax.get_position()
        y_fig = (pos.y0 + pos.y1) / 2
        fig.text(x_fig, y_fig, row_label,
                 rotation=90, fontweight="bold",
                 ha="right", va="center")

    save_dir = Path(_fig4_args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = save_dir / "figure4v2.pdf"
    fig.savefig(out_pdf)
    print(f"saved {out_pdf}")


def _parse_args():
    p = argparse.ArgumentParser(description="Figure 4 — full-model comparison")
    p.add_argument("--save-dir", default="./figures")
    p.add_argument("--oscillation-experiment",
                   default=_EXPERIMENTS_CFG.get("oscillation", "oscillation_paper"))
    p.add_argument("--tension-experiment",
                   default=_EXPERIMENTS_CFG.get("tension", "tension_clusters"))
    p.add_argument("--adhesion-experiment",
                   default=_EXPERIMENTS_CFG.get("adhesion", "diff_adhesion_ablations"))
    args = p.parse_args()
    SPECS[0]["experiment"] = args.oscillation_experiment
    SPECS[1]["experiment"] = args.tension_experiment
    SPECS[2]["experiment"] = args.adhesion_experiment
    return args


if __name__ == "__main__":
    _fig4_args = _parse_args()
    main()
