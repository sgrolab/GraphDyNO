"""Tables tab:seg_single and tab:seg_joint — clustering under segmentation and
tracking error, on the oscillation dataset.

Every level is scored with the paper's metric (k-means ARI at known k on
`area_fluct_amps`, FINAL checkpoint) and compared with the matched control
(`bank_none`, tagged `none`; see _seg_common for why it, not the published
clean runs, is the zero point). Each table cell is the level's ARI, median
[IQR] over seeds, marked by a two-sided Welch test against the control; the
per-level detail adds delta-ARI = level - control with the SE of the
difference.

tab:seg_single  one row per error kind, one column per dose d1..d4. Welch p
                is reported raw and Benjamini-Hochberg-adjusted over every
                ablated level in the experiment (one control tested against
                many levels is exactly the setup that manufactures false
                positives). The BH family is whatever the experiment holds:
                the paper's was 37 levels, including centroid-noise and
                episodic joint ladders it does not report; with only the 24
                levels in configs/seg_ablation, ID swap and merged neighbor at
                d2 land just above q = 0.05.
tab:seg_joint   the balanced joint family joint_{1,2,5,10}pct, analysed in the
                conventional order on BOTH checkpoints, because the two can
                disagree: an omnibus test (one-way ANOVA, Kruskal-Wallis),
                post-hoc Tukey HSD and Holm-corrected Welch tests against the
                control, and one planned linear trend of ARI on log10(realized
                dose).

With n=10 per arm a Welch test resolves a delta of roughly 0.10 ARI, so "not
significant" means "smaller than about that, or not resolvable with 10 seeds",
never "exactly zero".

The experiment and control tag come from `seg_ablation` in experiments.yaml;
the published clean reference from `sweeps.oscillation`. Set MLFLOW_URI to the
tracking store. Reads ~400 checkpoints.

    python paper_figures/seg_ablation_tables.py [--alpha 0.05]
"""
import argparse

import numpy as np
from scipy import stats

import figure4
from _ckpt_store import add_source_args, apply_source_args, ckpt_root
from _combo_common import benjamini_hochberg, fmt_iqr, holm
from _run_lookup import load_experiments_yaml, setup_mlflow_uri
from _seg_common import (BEST, DOSES, FINAL, JOINT_PCT, KIND_LABELS, RATE_KINDS,
                         delta, dose_x, load_levels, score, stat)

PRIMARY = "area_fluct_amps"


def dose_word():
    return "requested" if ckpt_root() is not None else "realized"


def welch(scores, level, control):
    """(n, delta, se, raw p) of one level against the control."""
    a = np.array([s[0] for s in scores[level]], dtype=float)
    ctrl = np.array([s[0] for s in scores[control]], dtype=float)
    d, se, _ = delta(scores, level, control, 0)
    p = (float(stats.ttest_ind(a, ctrl, equal_var=False).pvalue)
         if a.size > 1 and ctrl.size > 1 else np.nan)
    return a.size, d, se, p


def aris(scores, level):
    return np.array([s[0] for s in scores.get(level, [])], dtype=float)


def published_clean():
    spec = figure4.SPECS[0]
    runs = figure4._fetch_full_runs(spec["experiment"], spec["filters"])["dino"]
    vals = [s[0] for s in (score(r, PRIMARY) for r in runs) if s is not None]
    return spec["experiment"], np.array(vals, dtype=float)


def single_table(scores, diag, control, alpha):
    family = sorted(l for l in scores if l != control)
    tests = {l: welch(scores, l, control) for l in family}
    finite = [l for l in family if np.isfinite(tests[l][3])]
    q = dict(zip(finite, benjamini_hochberg([tests[l][3] for l in finite])))

    print(f"\nper-level detail (delta vs matched control `{control}`; "
          f"BH over all {len(finite)} ablated levels)")
    hdr = (f"{'level':14s} {'n':>3s} {dose_word():>9s} {'delta':>8s} {'se':>7s} "
           f"{'p':>10s} {'p_BH':>10s}")
    print(hdr + "\n" + "-" * len(hdr))
    for lvl in family:
        n, d, se, p = tests[lvl]
        kind = lvl.rsplit("_d", 1)[0]
        x = dose_x(diag, lvl, "seg_rate" if lvl in JOINT_PCT else kind)
        print(f"{lvl:14s} {n:3d} {x:9.4f} {d:+8.3f} {se:7.3f} {p:10.2e} "
              f"{q.get(lvl, np.nan):10.2e}")

    print(f"\nTable tab:seg_single — k-means ARI, median [IQR]; control `{control}` "
          f"{fmt_iqr(aris(scores, control), 2)}   (* = significant, two-sided Welch, "
          f"BH q < {alpha}; raw-p verdict in brackets)")
    print(f"  {'error mode':18s}" + "".join(f"{'d' + str(d):>27s}" for d in DOSES))
    for kind in RATE_KINDS:
        cells = []
        for d in DOSES:
            lvl = f"{kind}_d{d}"
            if lvl not in tests:
                cells.append(f"{'-':>27s}")
                continue
            p = tests[lvl][3]
            bh = "ns" if q.get(lvl, np.nan) >= alpha else "* "
            raw = "ns" if p >= alpha else "* "
            cells.append(f"{fmt_iqr(aris(scores, lvl), 2)} {bh} [{raw}]")
        print(f"  {KIND_LABELS[kind]:18s}" + "".join(f"{c:>27s}" for c in cells))


def joint_table(experiment, control, alpha):
    arms = [control] + JOINT_PCT
    for name in (FINAL, BEST):
        scores, diag = load_levels(experiment, PRIMARY, name, levels=arms)
        names = [a for a in arms if a in scores]
        groups = [np.array([s[0] for s in scores[a]], dtype=float) for a in names]
        label = "FINAL" if name == FINAL else "BEST-VAL"
        print(f"\n{'=' * 78}\n{label} checkpoint — balanced joint family\n{'=' * 78}")
        if control not in names or len(names) < 2:
            print("  missing the control or every joint arm")
            continue
        for a, g in zip(names, groups):
            x = dose_x(diag, a, "seg_rate")
            print(f"  {a:13s} n={g.size:2d}  {dose_word()} {x:.4f}  mean {g.mean():.4f}  "
                  f"SD {g.std(ddof=1):.4f}  SE {g.std(ddof=1) / np.sqrt(g.size):.4f}")

        print("\n  assumption checks: Shapiro-Wilk p per group "
              + ", ".join(f"{stats.shapiro(g).pvalue:.3f}" for g in groups)
              + f";  Levene p {stats.levene(*groups).pvalue:.3f}")
        f, pf = stats.f_oneway(*groups)
        h, ph = stats.kruskal(*groups)
        print(f"  1. omnibus: one-way ANOVA F={f:.3f} p={pf:.4f};  "
              f"Kruskal-Wallis H={h:.3f} p={ph:.4f}")

        print("  2. post-hoc: Tukey HSD over all pairs")
        thsd = stats.tukey_hsd(*groups)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p = thsd.pvalue[i, j]
                print(f"       {names[i]:12s} vs {names[j]:12s} "
                      f"diff {groups[j].mean() - groups[i].mean():+.4f}  "
                      f"p={p:.4f}{' *' if p < alpha else ''}")
        joint = [a for a in names if a != control]
        raw = [welch(scores, a, control)[3] for a in joint]
        adj = holm(raw)
        print(f"     Welch vs `{control}`, Holm-corrected over the {len(joint)} arms:")
        for a, r_, a_ in zip(joint, raw, adj):
            _, d, se, _ = welch(scores, a, control)
            print(f"       {a:12s} delta {d:+.4f} ± {se:.4f}  raw p={r_:.4f}  "
                  f"Holm p={a_:.4f}{' *' if a_ < alpha else ''}")

        xs, ys = [], []
        for a in joint:
            g = groups[names.index(a)]
            xs += [np.log10(dose_x(diag, a, "seg_rate"))] * g.size
            ys += g.tolist()
        lr = stats.linregress(xs, ys)
        rho, prho = stats.spearmanr(xs, ys)
        lo, hi = lr.slope - 1.96 * lr.stderr, lr.slope + 1.96 * lr.stderr
        print(f"  3. planned trend on log10({dose_word()} dose): slope {lr.slope:+.4f}/decade "
              f"SE {lr.stderr:.4f} r={lr.rvalue:+.3f} p={lr.pvalue:.4f};  "
              f"Spearman rho={rho:+.3f} p={prho:.4f};  95% CI "
              f"[{lo:+.4f}, {hi:+.4f}]{' (excludes 0)' if lo * hi > 0 else ' (includes 0)'}")

        print(f"\n  Table tab:seg_joint ({label}) — k-means ARI, median [IQR]; control "
              f"`{control}` {fmt_iqr(aris(scores, control), 2)}   (Welch vs control: "
              f"uncorrected verdict, Holm verdict in brackets)")
        for a, r_, a_ in zip(joint, raw, adj):
            print(f"    {a:12s} {fmt_iqr(aris(scores, a), 2)}  "
                  f"{'* ' if r_ < alpha else 'ns'} [{'* ' if a_ < alpha else 'ns'}]")


def main():
    cfg = load_experiments_yaml().get("seg_ablation", {})
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default=cfg.get("experiment", "segabl_oscillation"))
    ap.add_argument("--control", default=cfg.get("control", "none"),
                    help="tags.seg_ablation of the matched-control arm")
    ap.add_argument("--alpha", type=float, default=0.05)
    add_source_args(ap)
    args = apply_source_args(ap.parse_args())
    print(f"tracking uri: {setup_mlflow_uri()}")

    scores, diag = load_levels(args.experiment, PRIMARY)
    if args.control not in scores:
        raise SystemExit(f"no `{args.control}` arm in {args.experiment!r}; "
                         f"there is no matched control to test against")
    c_m, c_se, c_n = stat(scores, args.control, 0)
    print(f"matched control `{args.control}`: ARI median [IQR] "
          f"{fmt_iqr(aris(scores, args.control))}   (mean {c_m:.3f}, "
          f"sd {c_se * np.sqrt(c_n):.3f}, n={c_n})")
    clean_exp, clean = published_clean()
    if clean.size:
        print(f"published clean [{clean_exp}]: ARI median [IQR] {fmt_iqr(clean)}   "
              f"(mean {clean.mean():.3f}, sd {clean.std(ddof=1):.3f}, n={clean.size})"
              f"  — context only")

    single_table(scores, diag, args.control, args.alpha)
    joint_table(args.experiment, args.control, args.alpha)


if __name__ == "__main__":
    main()
