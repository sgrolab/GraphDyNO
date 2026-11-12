"""Table tab:encoder — Cell-DINO against DINOv3 (and Conv / Fourier) on the
oscillation, membrane-stiffness and differential-adhesion datasets.

Uses figure 4's metric functions verbatim (`load_emb_meta`, `_normalize_emb`,
`_recovery_ari`), so every column is directly comparable to figure 4's ARI
panels: k-means ARI at known k on each dataset's primary factor
(oscillation `area_fluct_amps`, k = 4; stiffness `kls`, k = 2; adhesion `khs`,
k = 2), FINAL checkpoint, ablation full_int, median [IQR] over seeds, plus
two-sided Welch and Mann-Whitney U tests of Cell-DINO against DINOv3.

  cell_dino          <- `cell_dino.<dataset>` in experiments.yaml
  dino/conv/fourier  <- figure 4's experiment and filters for that dataset
                        (`sweeps.<dataset>` in experiments.yaml)

n can differ per encoder, so it is reported explicitly. Writes, into --save-dir:

  cell_dino_table.tex        the tab:encoder tabular (Cell-DINO and DINO rows)
  cell_dino_per_seed.csv     dataset, encoder, seed, ARI, run_id for every run scored
  cell_dino_<dataset>.png    per dataset, each encoder's median-ARI seed (+ .pdf)

Set MLFLOW_URI to the tracking store.

    python paper_figures/cell_dino_table.py [--save-dir figures]
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
matplotlib.rcParams["font.family"] = "DejaVu Sans"
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import figure4
from _ckpt_store import add_source_args, apply_source_args
from _combo_common import TEX_IQR_DEFS, fmt_iqr, quartiles, tex_iqr
from _run_lookup import load_emb_meta, load_experiments_yaml, search_runs

ABLATION = "full_int"
ENCODERS = ["cell_dino", "dino", "conv", "fourier"]
TABLE_ENCODERS = ["cell_dino", "dino"]
LABELS = {"cell_dino": "Cell-DINO", "dino": "DINO", "conv": "Conv",
          "fourier": "Fourier"}

DATASETS = [
    ("oscillation", "Oscillation", 0, "oscillation_cell_dino"),
    ("membrane_stiffness", "Membrane stiffness", 1, "membrane_stiffness_cell_dino"),
    ("adhesion", "Differential adhesion", 2, "diff_adhesion_cell_dino"),
]


def _collect(experiment_name, param_filters, keep_embeddings):
    """FINISHED full_int runs from one experiment, grouped by encoder tag."""
    runs = search_runs(experiment_name)
    out = {e: [] for e in keep_embeddings}
    if runs is None:
        print(f"  [{experiment_name}] not found in MLflow")
        return out
    for r in runs:
        if r.info.status != "FINISHED" or r.data.tags.get("ablation") != ABLATION:
            continue
        e = r.data.tags.get("embedding")
        # figure 4 treats `conv_fix` as the canonical conv variant.
        if e == "conv_fix":
            e = "conv"
        if e not in out:
            continue
        if any(r.data.params.get(k) is not None
               and str(r.data.params.get(k)) != str(v)
               for k, v in param_filters.items()):
            continue
        out[e].append(r)
    if "conv" in out:
        fixed = [r for r in out["conv"] if r.data.tags.get("embedding") == "conv_fix"]
        if fixed:
            out["conv"] = fixed
    return out


def score(spec, cell_experiment):
    """Per encoder: the scored seeds as dicts (ari, emb, vals, seed, run_id), sorted by seed."""
    runs_by_enc = _collect(spec["experiment"], spec["filters"], ["dino", "conv", "fourier"])
    runs_by_enc.update(_collect(cell_experiment, {}, ["cell_dino"]))
    primary = spec["primary"]
    scored = {}
    for enc in ENCODERS:
        seeds = []
        for r in runs_by_enc.get(enc, []):
            emb, meta = load_emb_meta(r)
            if emb is None or primary not in meta:
                continue
            ari = figure4._recovery_ari(emb, meta[primary])
            if np.isfinite(ari):
                seed = r.data.params.get("model.seed")
                seeds.append(dict(ari=float(ari), emb=figure4._normalize_emb(emb),
                                  vals=meta[primary],
                                  seed=None if seed is None else int(seed),
                                  run_id=r.info.run_id))
        if not seeds:
            print(f"  WARNING: no usable runs for {enc}")
            continue
        scored[enc] = sorted(seeds, key=lambda s: (s["seed"] is None, s["seed"] or 0))
    return scored


def _median_seed(seeds):
    """figure 4's rule: the median-ARI seed (lower middle when n is even)."""
    order = np.argsort([s["ari"] for s in seeds], kind="stable")
    return seeds[order[(len(seeds) - 1) // 2]]


def report(key, spec, scored):
    print(f"\n[{key}] k-means ARI on `{spec['primary']}`, ablation={ABLATION}")
    print(f"{'encoder':<12} {'n':>4} {'median':>8} {'Q1':>8} {'Q3':>8} {'mean':>8} {'SD':>8}")
    for enc in ENCODERS:
        if enc not in scored:
            continue
        v = np.array([s["ari"] for s in scored[enc]])
        q1, md, q3 = quartiles(v)
        sd = v.std(ddof=1) if v.size > 1 else 0.0
        print(f"{LABELS[enc]:<12} {len(v):>4} {md:>8.3f} {q1:>8.3f} {q3:>8.3f} "
              f"{v.mean():>8.3f} {sd:>8.3f}")
    if "cell_dino" in scored:
        print("Cell-DINO seed:ARI  " + " ".join(
            f"{s['seed']}:{s['ari']:.3f}" for s in scored["cell_dino"]))
    if "cell_dino" in scored and "dino" in scored:
        a = np.array([s["ari"] for s in scored["cell_dino"]])
        b = np.array([s["ari"] for s in scored["dino"]])
        wt = stats.ttest_ind(a, b, equal_var=False)
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"Cell-DINO vs DINO: two-sided Welch t = {wt.statistic:.3f}, p = {wt.pvalue:.3g}; "
              f"Mann-Whitney U = {mw.statistic:.1f}, p = {mw.pvalue:.3g}")
        if a.std() == 0 or b.std() == 0:
            print("  (one arm has zero spread, so neither test is informative)")


def plot(key, spec, scored, save_dir):
    """One panel per encoder: its median-ARI seed, colored by the primary factor."""
    present = [e for e in ENCODERS if e in scored]
    if not present:
        return
    chosen = {e: _median_seed(scored[e]) for e in present}
    fig, axes = plt.subplots(1, len(present), figsize=(3.2 * len(present), 3.4),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    palette = spec["palette_slice"]
    _, cmap, tick_pos, tick_vals = figure4._categorical_cmap(chosen[present[0]]["vals"], palette)
    sc = None
    for ax, enc in zip(axes, present):
        s = chosen[enc]
        cat, _, _, _ = figure4._categorical_cmap(s["vals"], palette)
        sc = ax.scatter(s["emb"][:, 0], s["emb"][:, 1], c=cat, cmap=cmap,
                        vmin=-0.5, vmax=cmap.N - 0.5, s=8, linewidths=0)
        md = quartiles([x["ari"] for x in scored[enc]])[1]
        ax.set_title(f"{LABELS[enc]}\nARI={s['ari']:.3f}  "
                     f"(n={len(scored[enc])}, median={md:.3f})", fontsize=9)
        ax.set_xlabel(r"$\lambda_0$")
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"$\lambda_1$")
    cbar = fig.colorbar(sc, ax=axes.tolist(), ticks=tick_pos, fraction=0.02)
    labels = spec.get("colorbar_tick_override") or [f"{v:.3g}" for v in tick_vals]
    cbar.ax.set_yticklabels(labels)
    cbar.set_label(spec["colorbar_label"])

    out = Path(save_dir) / f"cell_dino_{key}.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Saved {out} (+ .pdf)")


def emit_table(results):
    """The tab:encoder tabular: one column per dataset, Cell-DINO and DINO rows."""
    cols = [col for _, col, _, _ in DATASETS]
    lines = [
        TEX_IQR_DEFS,
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        "Encoder & " + " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for enc in TABLE_ENCODERS:
        cells = [LABELS[enc]]
        for key, *_ in DATASETS:
            cells.append(tex_iqr([s["ari"] for s in results[key].get(enc, [])]))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    cfg = load_experiments_yaml().get("cell_dino", {})
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for key, _, _, default in DATASETS:
        ap.add_argument(f"--{key}-experiment", default=cfg.get(key, default),
                        help=f"MLflow experiment with the Cell-DINO {key} runs")
    ap.add_argument("--save-dir", default="figures")
    add_source_args(ap)
    args = apply_source_args(ap.parse_args())
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for key, _, spec_idx, _ in DATASETS:
        spec = figure4.SPECS[spec_idx]
        scored = score(spec, getattr(args, f"{key}_experiment"))
        results[key] = scored
        report(key, spec, scored)
        plot(key, spec, scored, save_dir)

    print("\nTable tab:encoder — k-means ARI, median [IQR]")
    for enc in TABLE_ENCODERS:
        print(f"  {LABELS[enc]:<10} " + "   ".join(
            f"{key}: {fmt_iqr([s['ari'] for s in results[key].get(enc, [])])}"
            for key, *_ in DATASETS))

    table = emit_table(results)
    print("\n" + table)
    tex_path = save_dir / "cell_dino_table.tex"
    tex_path.write_text(table + "\n")
    print(f"\nSaved {tex_path}")

    csv_path = save_dir / "cell_dino_per_seed.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "encoder", "seed", "ARI", "run_id"])
        for key, *_ in DATASETS:
            for enc in ENCODERS:
                for s in results[key].get(enc, []):
                    w.writerow([key, enc, s["seed"], s["ari"], s["run_id"]])
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
