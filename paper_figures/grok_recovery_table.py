"""Table tab:grok_recovery — recovery of alpha, omega and alpha*omega from the
parameter embedding at epoch 40 and epoch 200.

Both columns come from the same H = 64 oscillation runs
(`run_training.sh grok-h64`), read from their per-epoch embedding snapshots
(_grok_common) rather than from a best or last checkpoint, so the early and
late embeddings are the same model at two points in training. These are the
snapshots Figure 3 d–f and figure3_supp draw.

For each run, measure how recoverable alpha, omega and alpha*omega are from
the snapshot (in-sample linear R^2 and 5-fold cross-validated kernel-ridge
R^2, figure3._compute_lin_nl_r2), then report median [IQR] over runs per
epoch. The experiment, filters and epochs come from `grok_h64` in
experiments.yaml; set MLFLOW_URI to the tracking store.

    python paper_figures/grok_recovery_table.py [--early 40] [--late 200]
    python paper_figures/grok_recovery_table.py --all-epochs  # + every snapshot to CSV
"""
import argparse
from pathlib import Path

import pandas as pd

from figure3 import _compute_lin_nl_r2
from _ckpt_store import add_source_args, apply_source_args
from _combo_common import TEX_IQR_DEFS, fmt_iqr, tex_iqr
from _grok_common import EARLY, LATE, TARGETS, Snapshots, find_runs
from _run_lookup import setup_mlflow_uri  # noqa: E402


def score(emb, targets):
    """R^2_lin (in-sample linear fit) and R^2_nl (5-fold CV kernel ridge) per target."""
    out = {}
    for name in TARGETS:
        out[f"{name}_r2_lin"], out[f"{name}_r2_nl"] = _compute_lin_nl_r2(emb, targets[name])
    return out


def _summarize(per_run):
    rows = []
    for epoch, group in per_run.groupby("epoch"):
        row = {"epoch": int(epoch), "n_runs": len(group)}
        for name in TARGETS:
            for kind in ("lin", "nl"):
                col = f"{name}_r2_{kind}"
                row[col] = fmt_iqr(group[col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)


_TARGET_LATEX = {
    "alpha":       r"$\alpha$",
    "omega":       r"$\omega$",
    "alpha_omega": r"$\alpha\omega$",
}


def _to_latex(per_run):
    """Render the summary as a LaTeX table grouped by snapshot epoch."""
    epoch_groups = sorted(per_run["epoch"].unique())
    n_runs = {e: int((per_run["epoch"] == e).sum()) for e in epoch_groups}

    def cell(epoch, name, kind):
        return tex_iqr(per_run[per_run["epoch"] == epoch][f"{name}_r2_{kind}"])

    n_col_groups = len(epoch_groups)
    col_spec = "l" + "cc" * n_col_groups

    super_headers = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{epoch} epochs ($n={n_runs[epoch]}$)}}"
        for epoch in epoch_groups
    )
    cmidrules = " ".join(
        f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
        for i in range(n_col_groups)
    )
    sub_headers = " & ".join(
        [r"$R^2_{\mathrm{lin}}$", r"$R^2_{\mathrm{nl}}$"] * n_col_groups
    )

    body_rows = []
    for name in TARGETS:
        cells = [_TARGET_LATEX[name]]
        for epoch in epoch_groups:
            cells.append(cell(epoch, name, "lin"))
            cells.append(cell(epoch, name, "nl"))
        body_rows.append(" & ".join(cells) + r" \\")

    lines = [
        TEX_IQR_DEFS,
        r"\begin{table}[h]",
        r"  \centering",
        r"  \caption{Recovery of $\alpha$, $\omega$, and $\alpha\omega$ from the "
        r"parameter embedding at two epochs of the same training runs "
        r"($H = 64$; median \textcolor{iqrgray}{[IQR]} across runs). $R^2_{\mathrm{lin}}$ uses an "
        r"in-sample linear fit; $R^2_{\mathrm{nl}}$ is the 5-fold "
        r"cross-validated kernel-ridge score.}",
        r"  \label{tab:grok_recovery}",
        f"  \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        f"              & {super_headers} \\\\",
        f"    {cmidrules}",
        f"    Parameter & {sub_headers} \\\\",
        r"    \midrule",
    ]
    for row in body_rows:
        lines.append(f"    {row}")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default=None,
                        help="H = 64 experiment (default: grok_h64 in experiments.yaml)")
    parser.add_argument("--early", type=int, default=EARLY)
    parser.add_argument("--late", type=int, default=LATE)
    parser.add_argument("--output", default="./figures/grok_recovery_table.csv")
    parser.add_argument("--all-epochs", action="store_true",
                        help="also score every snapshot and write <output>_all_epochs.csv")
    add_source_args(parser)
    args = apply_source_args(parser.parse_args())
    print(f"tracking uri: {setup_mlflow_uri()}")

    runs = find_runs(args.experiment)
    rows, every = [], []
    for r in runs:
        snaps = Snapshots(r)
        for epoch in (args.early, args.late):
            rows.append(dict(run=r.info.run_name, seed=r.data.params.get("model.seed"),
                             epoch=epoch, **score(snaps[epoch], snaps.targets)))
        if args.all_epochs:
            for epoch in snaps.epochs():
                every.append(dict(run=r.info.run_name, epoch=epoch,
                                  **score(snaps[epoch], snaps.targets)))

    per_run = pd.DataFrame(rows)
    summary = _summarize(per_run)

    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("\nPer-run R^2:")
    print(per_run.to_string(index=False))
    print("\nSummary by epoch (median [IQR] across runs):")
    print(summary.to_string(index=False))

    latex = _to_latex(per_run)
    print("\nLaTeX table:")
    print(latex)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(out, index=False)
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
    tex_path = out.with_suffix(".tex")
    tex_path.write_text(latex + "\n")
    print(f"\nSaved per-run table:  {out}")
    print(f"Saved summary table:  {out.with_name(out.stem + '_summary.csv')}")
    print(f"Saved LaTeX table:    {tex_path}")
    if every:
        all_path = out.with_name(out.stem + "_all_epochs.csv")
        pd.DataFrame(every).to_csv(all_path, index=False)
        print(f"Saved every snapshot: {all_path}")


if __name__ == "__main__":
    main()
