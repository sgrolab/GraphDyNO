"""Supplementary figure: embedding scatter grids (encoder × ablation) for each dataset.

Generates one PDF per dataset with the same layout as the embedding_scatter
plot from notebooks/more_oscillation_analysis.py — rows are encoders
(DINO / Conv / Fourier), columns are ablations (static / temporal / spatial /
full). Color schemes match figure4v2.py.
"""
import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parents[1]
from _run_lookup import load_experiments_yaml, setup_mlflow_uri  # noqa: E402

setup_mlflow_uri()
client = mlflow.tracking.MlflowClient()
_EXPERIMENTS_CFG = load_experiments_yaml().get("sweeps", {})

EMBEDDINGS = ["dino", "conv", "fourier"]
EMBEDDING_LABELS = {"dino": "DINO", "conv": "Conv", "fourier": "Fourier"}
ABLATIONS = ["static", "temporal", "spatial", "full"]
K_NEIGHBORS = 10
N_BINS = 20
EMB_LIM = (-0.6, 0.6)
TARGET_WIDTH_IN = 5.5

_CB = sns.color_palette("colorblind", 10)

SPECS = [
    dict(experiment=_EXPERIMENTS_CFG.get("oscillation", "oscillation_paper"),
         filters={"model.embedding_noise_std": "0.1",
                  "model.interaction_dim": "2"},
         primary="area_fluct_amps",
         colorbar_label=r"$\alpha$",
         palette_slice=_CB[2:10],
         colorbar_tick_override=None),
    dict(experiment=_EXPERIMENTS_CFG.get("tension", "tension_clusters"),
         filters={},
         primary="kls",
         colorbar_label=r"$k_\ell$",
         palette_slice=_CB[0:2],
         colorbar_tick_override=[r"$10^3$", r"$10^6$"]),
    dict(experiment=_EXPERIMENTS_CFG.get("adhesion", "diff_adhesion_ablations"),
         filters={},
         primary="khs",
         colorbar_label=r"$k_h$",
         palette_slice=_CB[6:8],
         colorbar_tick_override=[r"$1\times 10^4$", r"$5\times 10^4$"]),
]

_SEED_PARAM_NAMES = ("seed", "training.seed", "model.seed", "p.seed",
                     "data.seed", "dataloader.seed")


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


def _seed_key(r):
    for k in _SEED_PARAM_NAMES:
        v = r.data.params.get(k) or r.data.tags.get(k)
        if v is not None:
            try:
                return (0, int(v))
            except (TypeError, ValueError):
                return (0, v)
    return (1, r.info.start_time)


def _normalize_emb(emb):
    centered = emb - emb.mean(axis=0, keepdims=True)
    span = centered.max(axis=0) - centered.min(axis=0)
    scale = span.max()
    if scale > 0:
        centered = centered / scale
    return centered


def _categorical_cmap(vals, palette_slice, max_categories=None):
    cap = max_categories or len(palette_slice)
    unique = np.sort(np.unique(vals))
    if len(unique) <= cap:
        cat = np.searchsorted(unique, vals)
        cmap = ListedColormap(palette_slice[:len(unique)])
        tick_vals = unique
        tick_positions = np.arange(len(unique))
    else:
        edges = np.linspace(vals.min(), vals.max(), cap + 1)
        cat = np.clip(np.digitize(vals, edges[1:-1]), 0, cap - 1)
        cmap = ListedColormap(palette_slice[:cap])
        tick_vals = 0.5 * (edges[:-1] + edges[1:])
        tick_positions = np.arange(cap)
    return cat, cmap, tick_positions, tick_vals


def _recovery_nc(emb, target):
    mask = np.isfinite(target) & np.isfinite(emb).all(axis=1)
    emb, target = emb[mask], target[mask]
    if len(target) < K_NEIGHBORS + 2 or np.std(target) < 1e-12:
        return np.nan
    bins = np.linspace(target.min() - 1e-9, target.max() + 1e-9, N_BINS + 1)
    binned = np.digitize(target, bins)
    _, idx = KDTree(emb).query(emb, k=K_NEIGHBORS + 1)
    return float((binned[idx[:, 1:]] == binned[:, None]).mean())


def _fetch_all_runs(experiment_name, param_filters):
    """Returns dict[(embedding, ablation)] -> representative run."""
    exps = [e.experiment_id for e in client.search_experiments()
            if e.name == experiment_name]
    assert exps, f"No experiment named {experiment_name!r}"
    runs = mlflow.search_runs(experiment_ids=exps, output_format="list")

    by_key: dict[tuple[str, str], list] = {}
    has_conv_fix = False
    for r in runs:
        if r.info.status != "FINISHED":
            continue
        embedding = r.data.tags.get("embedding")
        ablation = r.data.tags.get("ablation")
        if ablation == "full_velocity":
            ablation = "full"
        elif ablation == "full":
            continue
        if isinstance(ablation, str) and ablation.endswith("_int"):
            ablation = ablation[: -len("_int")]
        if embedding == "conv_fix":
            embedding = "conv"
            has_conv_fix = True
        elif embedding == "conv" and has_conv_fix:
            pass
        if embedding not in EMBEDDINGS or ablation not in ABLATIONS:
            continue
        if any(r.data.params.get(k) is not None
               and str(r.data.params.get(k)) != str(v)
               for k, v in param_filters.items()):
            continue
        by_key.setdefault((embedding, ablation), []).append(r)

    # Need a second pass to detect conv_fix since order isn't guaranteed.
    if not has_conv_fix:
        has_conv_fix = any(r.data.tags.get("embedding") == "conv_fix"
                          for r in runs if r.info.status == "FINISHED")
    if has_conv_fix:
        for key in list(by_key):
            if key[0] == "conv":
                filtered = [r for r in by_key[key]
                            if r.data.tags.get("embedding") == "conv_fix"]
                if filtered:
                    by_key[key] = filtered
                else:
                    del by_key[key]
        print(f"  [{experiment_name}] using conv_fix runs only")

    for k in by_key:
        by_key[k].sort(key=_seed_key)

    return {k: v[0] for k, v in by_key.items()}


def make_scatter_grid(spec, save_dir):
    experiment = spec["experiment"]
    primary = spec["primary"]
    palette_slice = spec["palette_slice"]
    colorbar_label = spec["colorbar_label"]
    tick_override = spec["colorbar_tick_override"]

    print(f"\n=== {experiment} ===")
    runs_index = _fetch_all_runs(experiment, spec["filters"])

    embeddings_cache = {}
    for (emb_type, ablation), run in runs_index.items():
        emb, meta = _load_emb_meta(run)
        if emb is None or primary not in meta:
            print(f"  skipping ({emb_type}, {ablation}): no checkpoint or meta")
            continue
        embeddings_cache[(emb_type, ablation)] = (emb, meta)

    if not embeddings_cache:
        print(f"  no embeddings found, skipping")
        return

    all_primary_vals = np.concatenate([
        np.asarray(meta[primary]).ravel()
        for _, meta in embeddings_cache.values()
    ])
    ref_cat, cmap, tick_positions, tick_vals = _categorical_cmap(
        all_primary_vals, palette_slice)
    vmin = -0.5
    vmax = cmap.N - 0.5

    spacing = 0.05
    fig, axes = plt.subplots(
        len(EMBEDDINGS), len(ABLATIONS),
        figsize=(2.4 * len(ABLATIONS) + 0.6, 2.4 * len(EMBEDDINGS)),
        sharex=True, sharey=True,
        gridspec_kw={"wspace": spacing, "hspace": spacing},
    )
    sc = None
    for i, emb_type in enumerate(EMBEDDINGS):
        for j, ablation in enumerate(ABLATIONS):
            ax = axes[i, j]
            key = (emb_type, ablation)
            if key not in embeddings_cache:
                ax.set_visible(False)
                continue
            emb, meta = embeddings_cache[key]
            emb_n = _normalize_emb(emb)
            cat, _, _, _ = _categorical_cmap(meta[primary], palette_slice)
            sc = ax.scatter(emb_n[:, 0], emb_n[:, 1], c=cat,
                            cmap=cmap, vmin=vmin, vmax=vmax,
                            s=3, edgecolors="face", linewidths=0.2, alpha=0.9)
            ax.set_box_aspect(1)
            ax.set_xlim(*EMB_LIM)
            ax.set_ylim(*EMB_LIM)
            if i == len(EMBEDDINGS) - 1:
                ax.set_xlabel(r"$\lambda_0$")
            if j == 0:
                ax.set_ylabel(r"$\lambda_1$", labelpad=0)
            if i == 0:
                ax.set_title(ablation.capitalize(), fontweight="bold",
                             fontsize=mpl.rcParams["axes.labelsize"])

    # Resize to target width.
    w, h = fig.get_size_inches()
    if abs(w - TARGET_WIDTH_IN) > 0.01:
        fig.set_size_inches(TARGET_WIDTH_IN, TARGET_WIDTH_IN * h / w)
        fig.tight_layout()

    # Encoder row labels on the left margin.
    fig.canvas.draw()
    for i, emb_type in enumerate(EMBEDDINGS):
        left_ax = axes[i, 0]
        ylabel_bbox = left_ax.yaxis.get_label().get_window_extent(
            renderer=fig.canvas.get_renderer())
        inv = fig.transFigure.inverted()
        x_fig = inv.transform((ylabel_bbox.x0, 0))[0] - 0.01
        pos = left_ax.get_position()
        y_fig = (pos.y0 + pos.y1) / 2
        fig.text(x_fig, y_fig, EMBEDDING_LABELS[emb_type],
                 rotation=90, fontweight="bold", ha="right", va="center")

    if sc is not None:
        fig.canvas.draw()
        mid_row = len(EMBEDDINGS) // 2
        right_pos = axes[mid_row, -1].get_position()
        cb_x = right_pos.x1 + 0.02
        cb_y = right_pos.y0
        cb_w = 0.018
        cb_h = right_pos.height
        cax = fig.add_axes([cb_x, cb_y, cb_w, cb_h])
        cbar = fig.colorbar(sc, cax=cax, label=colorbar_label, ticks=tick_positions)
        if tick_override is not None and len(tick_override) == len(tick_positions):
            cbar.ax.set_yticklabels(tick_override)
        else:
            cbar.ax.set_yticklabels([
                f"{v:.1f}".rstrip("0").rstrip(".") or "0"
                for v in tick_vals
            ])

    out_pdf = save_dir / f"embedding_scatter_{experiment}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"  saved {out_pdf}")
    plt.close(fig)


def main():
    sns.set_theme(style="ticks", context="paper")
    mpl.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 8,
    })

    p = argparse.ArgumentParser(
        description="Supplementary embedding scatter grids")
    p.add_argument("--save-dir",
                   default="./figures")
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for spec in SPECS:
        make_scatter_grid(spec, save_dir)


if __name__ == "__main__":
    main()
