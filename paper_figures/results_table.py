"""Supplement LaTeX tables for the runs plotted in figure4.py and figure5.py.

Two supplement tables are emitted (Table S2, the losses, is loss_table.py):

  Table S1 — Neighbor consistency for every DINO ablation and the full Conv /
             Fourier models. Combines the figure-5 ablation NC with the
             figure-4 full-encoder NC. Median [IQR] across seeds, 3 decimals.

  Table S3 — Neighbor consistency for the Conv / Fourier ablations, as S1.

Run from the repo root:
    python paper_figures/results_table.py
    python paper_figures/results_table.py -o paper_figures/tables/results.tex
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import figure4  # noqa: E402
from _ckpt_store import add_source_args, apply_source_args  # noqa: E402
from _combo_common import TEX_IQR_DEFS, tex_iqr  # noqa: E402
from _run_lookup import load_emb_meta, load_experiments_yaml, search_runs  # noqa: E402
from all_embeddings_supp import _recovery_nc  # noqa: E402

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

ABLATIONS = ["static", "temporal", "spatial", "full"]

# Conditions to display, in order. (encoder display label, ablation tag.)
# DINO is the only encoder with sub-ablations; Conv/Fourier appear only as
# the full-velocity model (i.e. the figure-4 baseline).
CONDITIONS = [
    ("DINO",    "static"),
    ("DINO",    "temporal"),
    ("DINO",    "spatial"),
    ("DINO",    "full"),
    ("Conv",    "full"),
    ("Fourier", "full"),
]

# Conv/Fourier ablations — used for the third NC table.
NON_DINO_ABLATION_CONDITIONS = [
    ("Conv",    "static"),
    ("Conv",    "temporal"),
    ("Conv",    "spatial"),
    ("Conv",    "full"),
    ("Fourier", "static"),
    ("Fourier", "temporal"),
    ("Fourier", "spatial"),
    ("Fourier", "full"),
]

# Metrics loaded from each MLflow run.
LOSS_COMPONENTS = [
    ("val_position_mse", "Position"),
    ("val_velocity_mse", "Velocity"),
    ("val_scale_mse",    "Scale"),
    ("val_dino_mse",     "Shape"),
]

# Components summed into Total — matches figure4's `val_loss_no_shape`.
TOTAL_KEYS = ("val_position_mse", "val_velocity_mse", "val_scale_mse")


def _load_all_runs():
    """One row per matching MLflow run across both datasets / conditions.

    Filter semantics match figure4 and figure5: a parameter that's missing
    on the run does *not* exclude the run ("applicable-only" filtering).
    `full_velocity` ablation tags are normalized to `full`; legacy literal
    `full` tags are dropped to match figure5.
    """
    rows = []
    for ds_label, exp_name, filters, target in DATASETS:
        runs = search_runs(exp_name)
        if runs is None:
            continue
        has_conv_fix = any(r.data.tags.get("embedding") == "conv_fix"
                          for r in runs if r.info.status == "FINISHED")
        for r in runs:
            if r.info.status != "FINISHED":
                continue
            emb = r.data.tags.get("embedding")
            abl = r.data.tags.get("ablation")
            if abl == "full_velocity":
                abl = "full"
            elif abl == "full":
                continue
            if isinstance(abl, str) and abl.endswith("_int"):
                abl = abl[: -len("_int")]
            if emb == "conv_fix":
                emb = "conv"
            elif emb == "conv" and has_conv_fix:
                continue
            if emb not in figure4.EMBEDDINGS or abl not in ABLATIONS:
                continue
            if any(r.data.params.get(k) is not None
                   and str(r.data.params.get(k)) != str(v)
                   for k, v in filters.items()):
                continue
            emb_arr, meta = load_emb_meta(r)
            nc = (_recovery_nc(emb_arr, meta[target])
                  if emb_arr is not None and target in meta else np.nan)
            rec = {
                "dataset": ds_label,
                "encoder": figure4.EMBEDDING_LABELS[emb],
                "ablation": abl,
                "NC": nc,
            }
            for k, _ in LOSS_COMPONENTS:
                rec[k] = r.data.metrics.get(k, np.nan)
            comp_vals = [rec[k] for k in TOTAL_KEYS if np.isfinite(rec[k])]
            rec["val_total"] = sum(comp_vals) if comp_vals else np.nan
            rows.append(rec)
        if has_conv_fix:
            print(f"  [{exp_name}] using conv_fix runs only (dropping plain conv)")
    return pd.DataFrame(rows)


def _subset(df, dataset, encoder, ablation):
    return df[(df["dataset"] == dataset)
              & (df["encoder"] == encoder)
              & (df["ablation"] == ablation)]


def _nc_table(df, conditions=CONDITIONS):
    """NC median [IQR] across seeds; 3 decimals, since most cells sit near 1."""
    ds_headers = [label.capitalize() for label, *_ in DATASETS]
    col_spec = "ll" + "c" * len(DATASETS)
    lines = [
        TEX_IQR_DEFS,
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        "Encoder & Ablation & " + " & ".join(ds_headers) + r" \\",
        r"\midrule",
    ]
    last_enc = None
    for enc, abl in conditions:
        cells = []
        for ds_label, *_ in DATASETS:
            cells.append(tex_iqr(_subset(df, ds_label, enc, abl)["NC"], prec=3))
        enc_cell = enc if enc != last_enc else ""
        last_enc = enc
        lines.append(" & ".join([enc_cell, abl.capitalize(), *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


DEFAULT_OUTPUT = ROOT / "figures" / "results_table.tex"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT),
                    help=f"Where to write the LaTeX (default: {DEFAULT_OUTPUT}).")
    add_source_args(ap)
    args = apply_source_args(ap.parse_args())

    df = _load_all_runs()

    parts = [
        r"% Table S1 — Neighbor consistency: DINO ablations + full encoders "
        r"(median [IQR] across seeds)",
        _nc_table(df, conditions=CONDITIONS),
        "",
        r"% Table S3 — Neighbor consistency: Conv/Fourier ablations "
        r"(median [IQR] across seeds)",
        _nc_table(df, conditions=NON_DINO_ABLATION_CONDITIONS),
    ]
    out = "\n".join(parts) + "\n"
    print(out)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out)
    print(f"# saved {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
