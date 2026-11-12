"""Figure 5 — DINO ablation neighbor-consistency comparison.

Top row:    schematic icons for the four ablations (static, temporal, spatial,
            full), embedded as nested <svg> elements at composite time so they
            render exactly as authored in paper_figures/icons/.
Bottom row: ARI-vs-ablation bar plots (median bar, IQR whiskers) for the
            three encoder×ablation sweep experiments (oscillating / membrane_stiffness /
            adhesion). Experiment names come from paper_figures/experiments.yaml.
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cairosvg

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from _ckpt_store import add_source_args, apply_source_args  # noqa: E402
from _combo_common import iqr_errorbar  # noqa: E402
from _run_lookup import (load_emb_meta, load_experiments_yaml,  # noqa: E402
                         search_runs, setup_mlflow_uri)

setup_mlflow_uri()
_EXPERIMENTS_CFG = load_experiments_yaml().get("sweeps", {})

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
matplotlib.rcParams["font.family"] = "DejaVu Sans"

ABLATIONS = ["static", "temporal", "spatial", "full"]
ABLATION_PALETTE = {
    "static": "#ffb000",
    "temporal": "#fe6100",
    "spatial": "#dc267f",
    "full": "#785ef0",
}
N_BINS = 20

_SEED_PARAM_NAMES = ("seed", "training.seed", "model.seed", "p.seed",
                     "data.seed", "dataloader.seed")


def _seed_of(run):
    for k in _SEED_PARAM_NAMES:
        v = run.data.params.get(k) or run.data.tags.get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                return v
    return None


def _recovery_ari(emb, target, n_bins=N_BINS):
    target = np.asarray(target, dtype=float)
    valid = np.isfinite(target) & np.isfinite(emb).all(axis=1)
    emb = emb[valid]
    target = target[valid]
    if len(target) < 4 or np.std(target) < 1e-12:
        return np.nan
    unique_vals = np.unique(target)
    if len(unique_vals) <= n_bins:
        labels = np.searchsorted(unique_vals, target)
    else:
        bins = np.linspace(target.min() - 1e-9, target.max() + 1e-9, n_bins + 1)
        labels = np.digitize(target, bins)
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        return np.nan
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    km_ids = km.fit_predict(emb)
    return float(adjusted_rand_score(labels, km_ids))


def _load_dino_nc_df(experiment_name, param_filters, target_meta):
    """One row per DINO run matching the filters: ablation, seed, NC."""
    runs = search_runs(experiment_name)
    if runs is None:
        raise RuntimeError(f"No MLflow experiment named {experiment_name!r}")
    rows = []
    for r in runs:
        if r.info.status != "FINISHED":
            continue
        ablation = r.data.tags.get("ablation")
        if ablation == "full_velocity":
            ablation = "full"
        elif ablation == "full":
            continue
        # diff_adhesion_ablations runs use the "_int" suffix on every tag.
        if isinstance(ablation, str) and ablation.endswith("_int"):
            ablation = ablation[: -len("_int")]
        embedding = r.data.tags.get("embedding")
        if embedding != "dino" or ablation not in ABLATIONS:
            continue
        # "Applicable-only" filter: a run that doesn't carry the param passes.
        skip = False
        for k, v in param_filters.items():
            p = r.data.params.get(k)
            if p is not None and str(p) != str(v):
                skip = True
                break
        if skip:
            continue
        emb, meta = load_emb_meta(r)
        if emb is None or target_meta not in meta:
            continue
        ari = _recovery_ari(emb, meta[target_meta])
        rows.append({"ablation": ablation, "seed": _seed_of(r), "ARI": ari})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ablation"] = pd.Categorical(df["ablation"], categories=ABLATIONS,
                                        ordered=True)
    return df


def _plot_ari_bars(ax, df, title=None, var_label=None):
    sns.barplot(data=df, x="ablation", y="ARI", order=ABLATIONS,
                palette=[ABLATION_PALETTE[a] for a in ABLATIONS],
                edgecolor="none", estimator="median",
                errorbar=iqr_errorbar, err_kws={"linewidth": 1.6},
                ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel(f"ARI ({var_label})" if var_label else "ARI (k-Means)",
                  labelpad=2, fontsize=9)
    if title:
        ax.set_title(title, fontsize=7, pad=2)
    ax.tick_params(axis="x", rotation=30, pad=1, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def _embed_icons_into_svg(matplotlib_svg, icon_paths, icon_rects, output_path):
    """Append each icon as a nested <svg> at the given (x, y, w, h) in pt.

    Each icon SVG keeps its own viewBox, so its content is preserved verbatim.
    """
    SVG_NS = "http://www.w3.org/2000/svg"
    XLINK_NS = "http://www.w3.org/1999/xlink"
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", XLINK_NS)

    tree = ET.parse(matplotlib_svg)
    root = tree.getroot()

    for icon_path, (x, y, w, h) in zip(icon_paths, icon_rects):
        icon_root = ET.parse(icon_path).getroot()
        icon_root.set("x", f"{x:.3f}")
        icon_root.set("y", f"{y:.3f}")
        icon_root.set("width", f"{w:.3f}")
        icon_root.set("height", f"{h:.3f}")
        # The original viewBox sets the icon's internal coordinate system; we
        # preserve aspect explicitly so the icon centers in its placement box.
        icon_root.set("preserveAspectRatio", "xMidYMid meet")
        root.append(icon_root)

    tree.write(output_path, xml_declaration=True, encoding="utf-8")


def make_figure(
    output_path="./figures/figure5.svg",
    icons_dir="./paper_figures/icons",
    osc_experiment=None,
    osc_filters=(("model.embedding_noise_std", "0.1"),
                 ("model.interaction_dim", "2")),
    osc_target="area_fluct_amp_x_period",
    osc_title="osc. growth",
    osc_var_label=r"$\alpha$",
    kl_experiment=None,
    kl_filters=(),
    kl_target="kls",
    kl_title="membrane stiffness",
    kl_var_label=r"$k_\ell$",
    da_experiment=None,
    da_filters=(),
    da_target="khs",
    da_title="differential adhesion",
    da_var_label=r"$k_h$",
):
    osc_experiment = osc_experiment or _EXPERIMENTS_CFG.get(
        "oscillation", "oscillation_paper")
    kl_experiment = kl_experiment or _EXPERIMENTS_CFG.get(
        "membrane_stiffness", "membrane_stiffness_clusters")
    da_experiment = da_experiment or _EXPERIMENTS_CFG.get(
        "adhesion", "diff_adhesion_ablations")
    sns.set_context("paper")
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8

    df_osc = _load_dino_nc_df(osc_experiment, dict(osc_filters), osc_target)
    df_kl = _load_dino_nc_df(kl_experiment, dict(kl_filters), kl_target)
    df_da = _load_dino_nc_df(da_experiment, dict(da_filters), da_target)
    if df_osc.empty:
        raise RuntimeError(f"No matching DINO runs for {osc_experiment}")
    if df_kl.empty:
        raise RuntimeError(f"No matching DINO runs for {kl_experiment}")
    if df_da.empty:
        raise RuntimeError(f"No matching DINO runs for {da_experiment}")

    for name, df in (("osc. growth", df_osc),
                     ("membrane stiffness", df_kl),
                     ("differential adhesion", df_da)):
        counts = (df.groupby("ablation", observed=True).size()
                  .reindex(ABLATIONS, fill_value=0).to_dict())
        print(f"[{name}] DINO seeds per ablation: {counts}")

    # ---- Layout ----
    fig_w = 5.5
    left_m = 0.60
    right_m = 0.08
    top_m = 0.10
    title_h = 0.08
    title_pad = 0.02
    icon_strip_m = 0.30
    icon_aspect = 200.0 / 140.0
    icon_h = (fig_w - 2 * icon_strip_m) / (4 * icon_aspect)
    row_gap = 0.10
    plot_h = 0.70
    col_gap = 0.75
    n_cols = 3
    plot_w = (fig_w - left_m - right_m - (n_cols - 1) * col_gap) / n_cols

    # Measure bottom_m programmatically from tick-label extent
    _tmp_fig = plt.figure(figsize=(fig_w, 2.0), dpi=300)
    _tmp_ax = _tmp_fig.add_axes([left_m / fig_w, 0.5, plot_w / fig_w, 0.4])
    _plot_ari_bars(_tmp_ax, df_osc, var_label=osc_var_label)
    _tmp_fig.canvas.draw()
    _renderer = _tmp_fig.canvas.get_renderer()
    _ax_bb = _tmp_ax.get_window_extent(_renderer)
    _min_label_y = _ax_bb.y0
    for _lbl in _tmp_ax.get_xticklabels():
        _bb = _lbl.get_window_extent(_renderer)
        _min_label_y = min(_min_label_y, _bb.y0)
    _label_drop = (_ax_bb.y0 - _min_label_y) / _tmp_fig.dpi
    plt.close(_tmp_fig)
    bottom_m = _label_drop + 0.03

    fig_h = top_m + title_h + title_pad + icon_h + row_gap + plot_h + bottom_m
    print(f"Figure size: {fig_w:.2f} x {fig_h:.2f} inches")

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

    bot_y = bottom_m / fig_h
    plot_h_frac = plot_h / fig_h
    plot_w_frac = plot_w / fig_w
    col_x_frac = [(left_m + i * (plot_w + col_gap)) / fig_w
                  for i in range(n_cols)]
    left_x_frac = col_x_frac[0]

    axes = [fig.add_axes([x, bot_y, plot_w_frac, plot_h_frac])
            for x in col_x_frac]
    ax_l, ax_m, ax_r = axes
    _plot_ari_bars(ax_l, df_osc, title=osc_title, var_label=osc_var_label)
    _plot_ari_bars(ax_m, df_kl, title=kl_title, var_label=kl_var_label)
    _plot_ari_bars(ax_r, df_da, title=da_title, var_label=da_var_label)
    sns.despine(fig=fig)

    # ---- Icon strip placement (SVG points, top-down y origin) ----
    icons_dir = Path(icons_dir)
    icon_paths = [icons_dir / f"ablation_{a}.svg" for a in ABLATIONS]
    for ip in icon_paths:
        if not ip.exists():
            raise FileNotFoundError(f"Missing icon: {ip}")

    pt = 72.0
    strip_x_pt = icon_strip_m * pt
    strip_w_pt = (fig_w - 2 * icon_strip_m) * pt
    strip_y_pt = (top_m + title_h + title_pad) * pt
    strip_h_pt = icon_h * pt
    fig_h_pt = fig_h * pt

    # Width-driven packing: 4 icons tile the strip exactly, no gap.
    icon_w_pt = strip_w_pt / 4
    icon_h_pt = icon_w_pt / icon_aspect
    gap = 0.0

    icon_rects = []
    icon_centers_x_frac = []
    for i in range(4):
        x = strip_x_pt + i * (icon_w_pt + gap)
        y = strip_y_pt + (strip_h_pt - icon_h_pt) / 2
        icon_rects.append((x, y, icon_w_pt, icon_h_pt))
        icon_centers_x_frac.append((x + icon_w_pt / 2) / (fig_w * pt))

    # ---- Per-icon titles (matplotlib text in figure-fraction coords) ----
    title_y_frac = 1 - (strip_y_pt - title_pad * pt) / fig_h_pt
    for cx, name in zip(icon_centers_x_frac, ABLATIONS):
        fig.text(cx, title_y_frac, name.capitalize(),
                 ha="center", va="bottom", fontsize=7)

    # ---- Panel labels — sit above and slightly left of each panel's content.
    label_kwargs = dict(fontweight="bold", va="bottom", ha="left", fontsize=8)
    label_dx = -0.035    # figure-fraction offset left of x0
    plot_label_dy = 0.045
    label_x = axes[0].get_position().x0 + label_dx
    fig.text(label_x, title_y_frac, "a", **label_kwargs)
    for ax, letter in zip(axes, ("b", "c", "d")):
        bbox = ax.get_position()
        fig.text(bbox.x0 + label_dx, bbox.y1 + plot_label_dy,
                 letter, **label_kwargs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_alpha(0.0)
    tmp_bare = output_path.with_name(output_path.stem + "__bare.svg")
    tmp_icons = output_path.with_name(output_path.stem + "__icons.svg")
    fig.savefig(tmp_bare, format="svg", transparent=True)

    _embed_icons_into_svg(tmp_bare, icon_paths, icon_rects, tmp_icons)
    tmp_bare.unlink()

    pdf_path = output_path.with_suffix(".pdf")
    cairosvg.svg2pdf(url=str(tmp_icons), write_to=str(pdf_path))
    tmp_icons.unlink()
    print(f"Saved {pdf_path}")
    return fig


def _parse_args():
    p = argparse.ArgumentParser(description="Figure 5 — DINO ablation NC")
    p.add_argument("--save-dir", default="./figures")
    p.add_argument("--icons-dir", default="./paper_figures/icons")
    p.add_argument("--oscillation-experiment",
                   default=_EXPERIMENTS_CFG.get("oscillation", "oscillation_paper"))
    p.add_argument("--membrane-stiffness-experiment",
                   default=_EXPERIMENTS_CFG.get("membrane_stiffness", "membrane_stiffness_clusters"))
    p.add_argument("--adhesion-experiment",
                   default=_EXPERIMENTS_CFG.get("adhesion", "diff_adhesion_ablations"))
    add_source_args(p)
    return apply_source_args(p.parse_args())


if __name__ == "__main__":
    args = _parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    make_figure(output_path=str(save_dir / "figure5.svg"),
                icons_dir=args.icons_dir,
                osc_experiment=args.oscillation_experiment,
                kl_experiment=args.membrane_stiffness_experiment,
                da_experiment=args.adhesion_experiment)
