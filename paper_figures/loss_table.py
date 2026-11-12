"""Supplement Table S2: validation-loss components for the runs in results_table.py.

Validation-loss components (Position / Velocity / Scale / Shape if present,
plus a Total) for the conditions of Table S1. Mean ± std across seeds; mean
and std share the same scientific exponent when scientific notation is used.

Reads MLflow metrics, so it has no checkpoint-only mode.

Run from the repo root:
    python paper_figures/loss_table.py
    python paper_figures/loss_table.py -o paper_figures/tables/loss_table.tex
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from _ckpt_store import add_source_args, apply_source_args, ckpt_root
from results_table import CONDITIONS, DATASETS, ROOT, _load_all_runs, _subset

# Subset shown in the table (Position omitted — it scales 1:1 with Velocity
# in these runs, so it's redundant).
DISPLAY_COMPONENTS = [
    ("val_velocity_mse", "Velocity"),
    ("val_scale_mse",    "Scale"),
    ("val_dino_mse",     "Shape"),
]


def _agg(series):
    vals = series.dropna()
    if len(vals) == 0:
        return np.nan, np.nan
    if len(vals) == 1:
        return float(vals.iloc[0]), np.nan
    return float(vals.mean()), float(vals.std())


def _column_exp(values):
    """Median floor(log10|v|) over finite, nonzero values."""
    arr = np.asarray([v for v in values if np.isfinite(v) and v != 0],
                     dtype=float)
    if arr.size == 0:
        return 0
    return int(np.median(np.floor(np.log10(np.abs(arr)))))


def _format_loss_column(pairs, decimals=2):
    """Format already-scaled (mean, std) pairs with fixed decimal places."""
    out = []
    for m, s in pairs:
        if not np.isfinite(m):
            out.append("--")
        elif np.isfinite(s):
            out.append(f"${m:.{decimals}f} \\pm {s:.{decimals}f}$")
        else:
            out.append(f"${m:.{decimals}f}$")
    return out


def _loss_table(df):
    """Single combined table; each dataset becomes a sub-section with a
    centered italic header row plus its own column-multiplier row.
    """
    cols_meta = [(k, lab) for k, lab in DISPLAY_COMPONENTS
                 if df[k].notna().any()]
    cols_meta.append(("val_total", "Total"))
    n_total_cols = 2 + len(cols_meta)

    cols_spec = "ll" + "c" * len(cols_meta)
    lines = [r"\begin{tabular}{" + cols_spec + "}", r"\toprule"]
    lines.append("Encoder & Ablation & "
                 + " & ".join(lab for _, lab in cols_meta) + r" \\")
    lines.append(r"\midrule")

    for ds_idx, (ds_label, *_rest) in enumerate(DATASETS):
        if ds_idx > 0:
            lines.append(r"\midrule")

        sub_all = df[df["dataset"] == ds_label]
        col_exps = {k: _column_exp(sub_all[k].values) for k, _ in cols_meta}

        col_cells = {k: [] for k, _ in cols_meta}
        for enc, abl in CONDITIONS:
            sub = _subset(df, ds_label, enc, abl)
            for k, _ in cols_meta:
                # Shape (val_dino_mse) isn't comparable across encoders;
                # mask it for non-DINO rows.
                if k == "val_dino_mse" and enc != "DINO":
                    col_cells[k].append((np.nan, np.nan))
                    continue
                m, s = _agg(sub[k])
                E = col_exps[k]
                ms = m / 10 ** E if np.isfinite(m) else m
                ss = s / 10 ** E if np.isfinite(s) else s
                col_cells[k].append((ms, ss))
        formatted = {k: _format_loss_column(col_cells[k])
                     for k, _ in cols_meta}

        # Italic dataset section header in the first cell.
        lines.append(r"\textit{" + ds_label.capitalize() + r"} & \\")
        # Per-section exponent row.
        lines.append(
            "& & "
            + " & ".join(f"$(\\times 10^{{{col_exps[k]}}})$"
                         for k, _ in cols_meta)
            + r" \\"
        )

        last_enc = None
        for i, (enc, abl) in enumerate(CONDITIONS):
            cells = [enc if enc != last_enc else "", abl.capitalize()]
            last_enc = enc
            for k, _ in cols_meta:
                cells.append(formatted[k][i])
            lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


DEFAULT_OUTPUT = ROOT / "figures" / "loss_table.tex"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT),
                    help=f"Where to write the LaTeX (default: {DEFAULT_OUTPUT}).")
    add_source_args(ap)
    args = apply_source_args(ap.parse_args())
    if ckpt_root() is not None:
        raise SystemExit("loss_table.py needs MLflow metrics; the exported checkpoints have none.")

    df = _load_all_runs()

    out = "\n".join([
        r"% Table S2 — Validation-loss components across all conditions "
        r"(mean $\pm$ std across seeds; column exponents per section)",
        _loss_table(df),
    ]) + "\n"
    print(out)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out)
    print(f"# saved {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
