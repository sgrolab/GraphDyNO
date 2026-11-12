"""Table tab:arch_sweep — parameter recovery is robust to network architecture.

The oscillation row of figure 4 is reported at one architecture (hidden width
512, 3 hidden layers, interaction dim 2, embedding dim 2). This reads the
sweep that moves one width at a time around it, four levels per axis, and
reports figure 4's own readout per arm — k-means ARI at known k on the FINAL
checkpoint — plus the delta against figure 4's published baseline runs, with
Welch t-tests Benjamini-Hochberg-corrected over all arms.

Everything load-bearing is figure 4's: `_recovery_ari`, `load_emb_meta`,
`_fetch_full_runs` and its oscillation spec (experiment, filters, target).

Caveats:
  * Error bars are initialization variance: the split and batch order are
    fixed by `dataloader.seed` for every run.
  * One axis moves at a time, so this measures robustness, not the best
    architecture; interactions between axes are out of scope.

The sweep experiment comes from `arch_sweep` in experiments.yaml, the
baseline from `sweeps.oscillation`; set MLFLOW_URI to the tracking store.

    python paper_figures/arch_sweep_table.py [--save-dir figures] [--no-plot]
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import stats

import figure4
from _ckpt_store import add_source_args, apply_source_args, ckpt_root
from _combo_common import benjamini_hochberg, fmt_iqr, summarize
from _run_lookup import load_emb_meta, load_experiments_yaml
from _seg_common import delta, mean_se

#: The published configuration this sweep varies around. `hidden_layers` is
#: not swept, but pinning it lets the baseline check catch a drifted baseline.
BASELINE = {"hidden_dim": 512, "hidden_layers": 1,
            "interaction_dim": 2, "embedding_dim": 2}

#: Axis -> the four non-baseline levels. Used for ordering and to report
#: missing levels; extra levels are reported, not dropped.
AXES = {
    "hidden_dim": [64, 128, 256, 1024],
    "interaction_dim": [4, 8, 16, 32],
    "embedding_dim": [3, 4, 8, 16],
}
AXIS_LABELS = {"hidden_dim": r"hidden width $H$",
               "interaction_dim": r"message-passing dim $M$",
               "embedding_dim": r"embedding dim $d_\lambda$"}


def _score(run, primary):
    """figure 4's ARI for one run, or nan when the checkpoint is unreadable."""
    emb, meta = load_emb_meta(run)
    if emb is None or primary not in meta:
        return np.nan
    return figure4._recovery_ari(emb, meta[primary])


def _arch(run):
    p = run.data.params
    return tuple(p.get(f"model.{k}") for k in BASELINE)


def load_baseline(primary, experiment, filters):
    runs = figure4._fetch_full_runs(experiment, filters)["dino"]
    aris = [a for a in (_score(r, primary) for r in runs) if np.isfinite(a)]
    return aris, {_arch(r) for r in runs}


def load_sweep(primary, experiment):
    """{axis: {level: [ari, ...]}} from the sweep experiment."""
    runs = figure4._fetch_full_runs(experiment, {})["dino"]
    by_axis, skipped, at_baseline = {}, 0, []
    for r in runs:
        axis = r.data.tags.get("arch_axis")
        raw = r.data.params.get(f"model.{axis}") if axis else None
        if raw is None:
            skipped += 1
            continue
        if all(str(r.data.params.get(f"model.{k}")) == str(v)
               for k, v in BASELINE.items()):
            # The baseline arm belongs to figure 4's experiment; a run here at
            # every baseline value would double-count it.
            at_baseline.append(r.info.run_name)
        a = _score(r, primary)
        if not np.isfinite(a):
            skipped += 1
            continue
        by_axis.setdefault(axis, {}).setdefault(int(raw), []).append(a)
    return by_axis, len(runs), skipped, at_baseline


def build_rows(by_axis, base_aris):
    """One row per arm: level stats, delta vs baseline, Welch t and p, BH q."""
    base = np.asarray(base_aris, dtype=float)
    base_ref = mean_se(base)
    rows = []
    for axis in AXES:
        for lvl in sorted(by_axis.get(axis, {})):
            vals = np.asarray(by_axis[axis][lvl], dtype=float)
            # `delta` takes {level: [(ari, ...), ...]} and an index.
            d, se_d, _ = delta({lvl: [(v,) for v in vals]}, lvl, base_ref, 0)
            s = summarize(vals)
            if vals.size > 1 and base.size > 1:
                res = stats.ttest_ind(vals, base, equal_var=False)
                t, p = float(res.statistic), float(res.pvalue)
            else:
                t = p = np.nan
            rows.append(dict(axis=axis, level=lvl, n=s["n"], mean=s["mean"],
                             sd=s["sd"], median=s["median"], iqr=fmt_iqr(vals),
                             delta=d, se_delta=se_d, t=t, p=p, q=np.nan))
    finite = [i for i, r in enumerate(rows) if np.isfinite(r["p"])]
    for i, qi in zip(finite, benjamini_hochberg([rows[i]["p"] for i in finite])):
        rows[i]["q"] = float(qi)
    return rows, base_ref


def _star(q, alpha):
    if not np.isfinite(q):
        return "--"
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < alpha else "ns"


def print_report(rows, base_ref, primary, alpha, base_aris):
    b_mean, b_se, b_n = base_ref
    b_sd = b_se * np.sqrt(b_n) if b_n > 1 else 0.0
    hdr = (f"{'level':>6s} {'n':>3s} {'ARI mean':>9s} {'sd':>7s} {'median':>7s} "
           f"{'Δ vs paper':>11s} {'SE(Δ)':>7s} {'t':>7s} {'p':>10s} {'q(BH)':>10s}  sig")
    print(f"\n{'=' * len(hdr)}\narchitecture sweep — figure 4 k-means ARI on "
          f"`{primary}` (final checkpoint)\n{'=' * len(hdr)}")
    for axis in AXES:
        sub = [r for r in rows if r["axis"] == axis]
        print(f"\naxis: {axis}   (baseline {BASELINE[axis]}: "
              f"ARI {b_mean:.3f} ± {b_sd:.3f} sd, n={b_n})")
        if not sub:
            print("  -- no runs found")
            continue
        print(hdr + "\n" + "-" * len(hdr))
        for r in sub:
            print(f"{r['level']:6d} {r['n']:3d} {r['mean']:9.3f} {r['sd']:7.3f} "
                  f"{r['median']:7.3f} {r['delta']:+11.3f} {r['se_delta']:7.3f} "
                  f"{r['t']:7.2f} {r['p']:10.2e} {r['q']:10.2e}  {_star(r['q'], alpha)}")
    sig = [r for r in rows if np.isfinite(r["q"]) and r["q"] < alpha]
    print(f"\n{len(sig)}/{len(rows)} arms differ from the published baseline at "
          f"q<{alpha} (BH over all {len(rows)} arms):")
    for r in sorted(sig, key=lambda r: r["delta"]):
        print(f"    {r['axis']}={r['level']:<6d} Δ={r['delta']:+.3f} q={r['q']:.2e}")

    print("\nTable tab:arch_sweep — k-means ARI, median [IQR]   (mean ± sd)")
    print(f"  baseline (H=512, M=2, d=2)    {fmt_iqr(base_aris)}   "
          f"({b_mean:.3f} ± {b_sd:.3f}, n={b_n})")
    for r in rows:
        print(f"  {r['axis']:<16} {r['level']:>5}    {r['iqr']}   "
              f"({r['mean']:.3f} ± {r['sd']:.3f}, n={r['n']})")


def make_plot(rows, base_ref, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b_mean, b_se, b_n = base_ref
    b_sd = b_se * np.sqrt(b_n) if b_n > 1 else 0.0
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, axis in zip(axes, AXES):
        sub = sorted([r for r in rows if r["axis"] == axis], key=lambda r: r["level"])
        # The baseline is drawn as a band (mean ± sd) so an arm inside it is
        # visibly indistinguishable.
        ax.axhspan(b_mean - b_sd, b_mean + b_sd, color="0.85", zorder=0,
                   label=f"paper baseline ± sd (n={b_n})")
        ax.axhline(b_mean, color="0.45", lw=1, zorder=1)
        if sub:
            ax.errorbar([r["level"] for r in sub], [r["mean"] for r in sub],
                        yerr=[r["sd"] for r in sub], fmt="o-", color="#785ef0",
                        capsize=3, ms=5, lw=1.4, zorder=3, label="swept arm ± sd")
        ax.plot([BASELINE[axis]], [b_mean], marker="*", ms=14, color="#d55e00",
                zorder=4, ls="none", label=f"baseline {BASELINE[axis]}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(AXIS_LABELS[axis])
        ax.grid(alpha=0.25, which="both", axis="y")
    axes[0].set_ylabel("k-means ARI ($\\alpha$)")
    axes[0].legend(fontsize=7, loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"\nwrote {path}")


def main():
    cfg = load_experiments_yaml().get("arch_sweep", {})
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default=cfg.get("experiment", "osc_arch_sweep"),
                    help="MLflow experiment holding the swept arms")
    ap.add_argument("--baseline-experiment", default=None,
                    help="override the baseline experiment "
                         "(default: figure 4's oscillation experiment)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--save-dir", default="figures")
    ap.add_argument("--no-plot", action="store_true")
    add_source_args(ap)
    args = apply_source_args(ap.parse_args())

    spec = figure4.SPECS[0]
    primary = spec["primary"]
    base_exp = args.baseline_experiment or spec["experiment"]
    base_aris, base_archs = load_baseline(primary, base_exp, spec["filters"])
    print(f"[{base_exp}] baseline: {len(base_aris)} scored runs; architectures "
          f"(hidden_dim, hidden_layers, interaction_dim, embedding_dim): "
          f"{sorted(base_archs)}")
    if ckpt_root() is None and base_archs != {tuple(str(v) for v in BASELINE.values())}:
        print("  !! WARNING: the baseline is not the single published "
              "architecture; the deltas below mix configurations")

    by_axis, n_runs, skipped, at_baseline = load_sweep(primary, args.experiment)
    print(f"\n[{args.experiment}] {n_runs} FINISHED DINO full-model runs, "
          f"{skipped} unscorable/untagged")
    for axis in AXES:
        got = by_axis.get(axis, {})
        missing = [lv for lv in AXES[axis] if lv not in got]
        print(f"  {axis:16s} {({lv: len(v) for lv, v in sorted(got.items())})}"
              + (f"   MISSING LEVELS {missing}" if missing else ""))
    if at_baseline:
        print(f"  !! WARNING: {len(at_baseline)} run(s) sit at the full baseline "
              f"architecture: {at_baseline[:5]}")

    rows, base_ref = build_rows(by_axis, base_aris)
    print_report(rows, base_ref, primary, args.alpha, base_aris)
    if not args.no_plot:
        out = Path(args.save_dir) / "arch_sweep.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        make_plot(rows, base_ref, out)


if __name__ == "__main__":
    main()
