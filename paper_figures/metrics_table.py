"""Extended recovery-metrics table: NC, k-Means ARI/NMI, HDBSCAN ARI/NMI.

Compares DINO / Conv / Fourier (full model only) across three datasets,
matching the scope of figure4.py but reporting five embedding-recovery
metrics instead of just NC. Cells are median [IQR] across runs; one row per
metric and one column per encoder, so the table fits the NeurIPS text width.

Run from the repo root:
    python paper_figures/metrics_table.py
    python paper_figures/metrics_table.py --save-dir /some/path
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import figure4 as fig4  # noqa: E402
from _ckpt_store import add_source_args, apply_source_args  # noqa: E402
from _combo_common import TEX_IQR_DEFS, tex_iqr  # noqa: E402
from _run_lookup import load_emb_meta, load_experiments_yaml  # noqa: E402

K_NEIGHBORS = 10
N_BINS = 20

_SWEEPS = load_experiments_yaml().get("sweeps", {})
DATASETS = [
    ("oscillating",
     _SWEEPS.get("oscillation", "oscillation_paper"),
     {"model.embedding_noise_std": "0.1", "model.interaction_dim": "2"},
     "area_fluct_amps"),
    ("membrane stiffness",
     _SWEEPS.get("membrane_stiffness", "membrane_stiffness_clusters"),
     {},
     "kls"),
    ("differential adhesion",
     _SWEEPS.get("adhesion", "diff_adhesion_ablations"),
     {},
     "khs"),
]

ENCODERS = ["dino", "conv", "fourier"]

METRICS = [
    ("NC", "NC"),
    ("ARI_kmeans", r"ARI$_{\text{km}}$"),
    ("NMI_kmeans", r"NMI$_{\text{km}}$"),
    ("ARI_hdbscan", r"ARI$_{\text{hdb}}$"),
    ("NMI_hdbscan", r"NMI$_{\text{hdb}}$"),
]


def _make_labels(target):
    unique_vals = np.unique(target)
    if len(unique_vals) <= N_BINS:
        return np.searchsorted(unique_vals, target), len(unique_vals)
    bins = np.linspace(target.min() - 1e-9, target.max() + 1e-9, N_BINS + 1)
    return np.digitize(target, bins), N_BINS


def compute_metrics(emb, target):
    mask = np.isfinite(target) & np.isfinite(emb).all(axis=1)
    emb, target = emb[mask], target[mask]
    nans = {m: np.nan for m, _ in METRICS}
    if len(target) < K_NEIGHBORS + 2 or np.std(target) < 1e-12:
        return nans

    bins = np.linspace(target.min() - 1e-9, target.max() + 1e-9, N_BINS + 1)
    binned = np.digitize(target, bins)
    _, idx = KDTree(emb).query(emb, k=K_NEIGHBORS + 1)
    nc = float((binned[idx[:, 1:]] == binned[:, None]).mean())

    labels, n_clusters = _make_labels(target)
    out = {"NC": nc}

    if n_clusters < 2:
        out.update({m: np.nan for m in
                    ["ARI_kmeans", "NMI_kmeans", "ARI_hdbscan", "NMI_hdbscan"]})
        return out

    try:
        km_ids = KMeans(n_clusters=n_clusters, n_init=10,
                        random_state=0).fit_predict(emb)
        out["ARI_kmeans"] = float(adjusted_rand_score(labels, km_ids))
        out["NMI_kmeans"] = float(normalized_mutual_info_score(labels, km_ids))
    except Exception:
        out["ARI_kmeans"] = np.nan
        out["NMI_kmeans"] = np.nan

    min_cs = max(20, len(emb) // (2 * n_clusters))
    try:
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
    except Exception:
        out["ARI_hdbscan"] = np.nan
        out["NMI_hdbscan"] = np.nan

    return out


def _load_all_runs():
    rows = []
    for ds_label, exp_name, filters, target in DATASETS:
        runs_by_emb = fig4._fetch_full_runs(exp_name, filters)
        for enc in ENCODERS:
            for r in runs_by_emb[enc]:
                emb_arr, meta = load_emb_meta(r)
                if emb_arr is None or target not in meta:
                    continue
                rec = {
                    "dataset": ds_label,
                    "encoder": fig4.EMBEDDING_LABELS[enc],
                }
                rec.update(compute_metrics(emb_arr, meta[target]))
                rows.append(rec)
    return pd.DataFrame(rows)


def emit_table(df):
    encoders = [fig4.EMBEDDING_LABELS[e] for e in ENCODERS]
    n_cols = 1 + len(encoders)

    lines = [
        TEX_IQR_DEFS,
        r"\begin{tabular}{l" + "c" * len(encoders) + "}",
        r"\toprule",
        "Metric & " + " & ".join(encoders) + r" \\",
        r"\midrule",
    ]

    for ds_idx, (ds_label, *_) in enumerate(DATASETS):
        if ds_idx > 0:
            lines.append(r"\midrule")
        lines.append(
            r"\multicolumn{" + str(n_cols) + r"}{l}{\textit{"
            + ds_label.capitalize() + r"}} \\")

        sub = df[df["dataset"] == ds_label]
        for m, header in METRICS:
            cells = [header]
            for enc in encoders:
                cells.append(tex_iqr(sub[sub["encoder"] == enc][m]))
            lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save-dir",
                    default="./figures")
    add_source_args(ap)
    args = apply_source_args(ap.parse_args())

    df = _load_all_runs()

    print("\n=== Per-seed values ===")
    print(df.to_string(index=False))

    table = emit_table(df)
    print("\n" + table)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics_table.tex"
    out_path.write_text(table + "\n")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
