"""Table tab:timescales — oscillation PERIOD separation, plus its figure.

Companion to figure4.py, reusing its metric and idiom: k-means ARI at known k
(`figure4._recovery_ari`), the mean-ARI seed for the scatter (embeddings cannot be
averaged across seeds -- each sits in its own rotated/reflected frame), and per-seed
SD error bars.

    a  parameter embedding of the mean-ARI seed, coloured by oscillation period
    b  k-means ARI (k=4) over seeds, with the paper's amplitude arm for reference
    c  pairwise 2-means ARI for all six period pairs -- where the P=7/P=10 limit shows

The period experiment comes from `timescales` in experiments.yaml; the amplitude
reference row is figure 4's oscillation spec (`sweeps.oscillation`). Set
MLFLOW_URI to the tracking store.

Usage:  python paper_figures/figure_osc_period.py [--save-dir DIR]
"""
import argparse
import itertools
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

import figure4 as f4  # noqa: E402  (same directory)
from _combo_common import fmt_iqr
from _run_lookup import load_experiments_yaml

EXPERIMENT = load_experiments_yaml().get("timescales", {}).get(
    "experiment", "osc_period_clusters")
PRIMARY = "area_fluct_periods"
FULL_COLOR = "#785ef0"
_CB = sns.color_palette("colorblind", 10)
PALETTE = _CB[2:6]          # 4 period levels, same hue family as figure4 row 1


def collect():
    """-> (per-seed rows, mean-ARI seed's (emb, periods))."""
    runs = f4._fetch_full_runs(EXPERIMENT, {})["dino"]
    rows, seeds = [], []
    for r in runs:
        emb, meta = f4._load_emb_meta(r)
        if emb is None or PRIMARY not in meta:
            continue
        per = meta[PRIMARY]
        lev = np.unique(per)
        lab = np.searchsorted(lev, per)
        ari = f4._recovery_ari(emb, per)

        pair = {}
        for a, b in itertools.combinations(range(len(lev)), 2):
            m = (lab == a) | (lab == b)
            km = KMeans(2, n_init=10, random_state=0).fit_predict(emb[m])
            pair[(int(lev[a]), int(lev[b]))] = adjusted_rand_score(
                (lab[m] == b).astype(int), km)

        # HDBSCAN with the repo recipe: min_cluster_size = max(20, N // (2k)).
        # Unlike k-means it discovers k, so `kfound` is the diagnostic that
        # goes with its ARI.
        h = HDBSCAN(min_cluster_size=max(20, len(emb) // 8)).fit_predict(emb)
        rows.append(dict(seed=int(r.data.params.get("model.seed", -1)),
                         ARI=ari,
                         HDB=adjusted_rand_score(lab, h),
                         kfound=len(set(h)) - (1 if -1 in h else 0),
                         **{f"{a}v{b}": v for (a, b), v in pair.items()}))
        if np.isfinite(ari):
            seeds.append((ari, f4._normalize_emb(emb), per))

    aris = np.array([s[0] for s in seeds])
    chosen = seeds[int(np.argmin(np.abs(aris - aris.mean())))]
    rows.sort(key=lambda d: d["seed"])
    return rows, chosen


def main(save_dir):
    sns.set_theme(style="ticks", context="paper")
    mpl.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })

    rows, (chosen_ari, emb_n, per) = collect()
    aris = np.array([r["ARI"] for r in rows])
    lev = np.unique(per)
    pairs = [f"{a}v{b}" for a, b in itertools.combinations([int(v) for v in lev], 2)]

    fig = plt.figure(figsize=(9.0, 3.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.72, 1.15], wspace=0.42,
                          left=0.06, right=0.98, top=0.88, bottom=0.22)

    # --- a: embedding of the mean-ARI seed -------------------------------
    ax = fig.add_subplot(gs[0, 0])
    cat, cmap, ticks, tickvals = f4._categorical_cmap(per, PALETTE)
    sc = ax.scatter(emb_n[:, 0], emb_n[:, 1], c=cat, cmap=cmap,
                    vmin=-0.5, vmax=cmap.N - 0.5, s=3.0,
                    edgecolors="face", linewidths=0.2, alpha=0.9)
    ax.set_box_aspect(1)
    ax.set_xlabel(r"$\lambda_0$", labelpad=2)
    ax.set_ylabel(r"$\lambda_1$", labelpad=2)
    ax.text(0.04, 0.04, f"ARI={chosen_ari:.3f}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.75))
    ax.text(0.04, 0.96, "a", transform=ax.transAxes, fontsize=10,
            fontweight="bold", ha="left", va="top")
    ax.set_title("mean-ARI seed", pad=3)
    for s in ax.spines.values():
        s.set_edgecolor("black"); s.set_linewidth(0.8); s.set_visible(True)
    cax = ax.inset_axes([1.03, 0.0, 0.045, 1.0], transform=ax.transAxes)
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r"period $P$", labelpad=2)
    cb.ax.tick_params(pad=1, labelsize=6, length=2)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{v:.0f}" for v in tickvals])

    # --- b: overall ARI, ours vs the paper's amplitude arm ---------------
    ax = fig.add_subplot(gs[0, 1])
    hdbs = np.array([r["HDB"] for r in rows])
    amp_km, amp_hdb = paper_amplitude_aris()
    series = [("k-means (k known)", FULL_COLOR, aris, amp_km),
              ("HDBSCAN (k found)", "0.55", hdbs, amp_hdb)]
    w, rng = 0.38, np.random.default_rng(0)
    for i, (lbl, col, ours, theirs) in enumerate(series):
        xs = np.array([0, 1]) + (i - 0.5) * w
        ax.bar(xs, [ours.mean(), theirs.mean()], w * 0.9,
               yerr=[ours.std(), theirs.std()], color=col, edgecolor="none",
               error_kw={"linewidth": 1.4}, label=lbl)
        for x, v in zip(xs, (ours, theirs)):
            ax.scatter(np.full(len(v), x) + rng.normal(0, .045, len(v)),
                       v, s=4, color="k", alpha=.4, zorder=3)
    ax.axhline(0.0, ls="--", color="black", lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["period\n(this work)", "amplitude\n(fig. 4)"])
    ax.set_ylim(-0.05, 1.28)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("ARI", labelpad=1)
    ax.legend(fontsize=6, frameon=False, loc="upper center", ncol=1,
              handlelength=1.1, borderpad=0.1, labelspacing=0.25)
    ax.text(-0.30, 1.06, "b", transform=ax.transAxes, fontsize=10,
            fontweight="bold", ha="right", va="bottom")
    sns.despine(ax=ax)

    # --- c: pairwise separability ----------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    pv = [np.array([r[p] for r in rows]) for p in pairs]
    lbl = [p.replace("v", " vs ") for p in pairs]
    cols = [FULL_COLOR if not p.startswith("7v") else "#d55e00" for p in pairs]
    ax.bar(range(len(pairs)), [v.mean() for v in pv],
           yerr=[v.std() for v in pv], color=cols, edgecolor="none",
           error_kw={"linewidth": 1.4})
    for i, v in enumerate(pv):
        ax.scatter(np.full(len(v), i) + np.random.default_rng(1).normal(0, .07, len(v)),
                   v, s=5, color="k", alpha=.45, zorder=3)
    ax.axhline(0.0, ls="--", color="black", lw=0.8)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(lbl, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("pairwise 2-means ARI", labelpad=1)
    ax.text(-0.16, 1.06, "c", transform=ax.transAxes, fontsize=10,
            fontweight="bold", ha="right", va="bottom")
    sns.despine(ax=ax)

    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(save_dir / f"figure_osc_period.{ext}", dpi=200)
    print(f"saved {save_dir/'figure_osc_period.pdf'}")

    # --- numbers for the markdown table ----------------------------------
    print(f"\nn seeds = {len(rows)}")
    print(f"  ARI(k=4)        {aris.mean():.3f} +/- {aris.std():.3f}   "
          f"[{aris.min():.3f}, {aris.max():.3f}]   median {np.median(aris):.3f}")
    no0 = aris[[r["seed"] != 0 for r in rows]]
    print(f"  ARI excl.seed0  {no0.mean():.3f} +/- {no0.std():.3f}")
    kf = np.array([r["kfound"] for r in rows])
    print(f"  HDBSCAN ARI     {hdbs.mean():.3f} +/- {hdbs.std():.3f}   "
          f"[{hdbs.min():.3f}, {hdbs.max():.3f}]")
    print(f"  clusters found  mean {kf.mean():.2f}, "
          f"counts {dict(zip(*[a.tolist() for a in np.unique(kf, return_counts=True)]))}")
    for p, v in zip(lbl, pv):
        print(f"  {p:>10}      {v.mean():.3f} +/- {v.std():.3f}   [{v.min():.3f}, {v.max():.3f}]")

    print("\nTable tab:timescales — k-means ARI, median [IQR]   (mean ± sd, sd with ddof=0 / ddof=1)")
    for name, v in ((f"periods {{{', '.join(f'{x:.0f}' for x in lev)}}} [{EXPERIMENT}]", aris),
                    (f"amplitudes, period fixed [{f4.SPECS[0]['experiment']}]", amp_km)):
        print(f"  {name:<52} {fmt_iqr(v)}   ({v.mean():.3f} ± {v.std():.3f} / "
              f"{v.std(ddof=1):.3f}, n={v.size})")


def paper_amplitude_aris():
    """figure 4's oscillating row (amplitude) -> (k-means ARI, HDBSCAN ARI)."""
    km, hdb = [], []
    spec = f4.SPECS[0]
    for r in f4._fetch_full_runs(spec["experiment"], spec["filters"])["dino"]:
        emb, meta = f4._load_emb_meta(r)
        if emb is None or "area_fluct_amps" not in meta:
            continue
        a = f4._recovery_ari(emb, meta["area_fluct_amps"])
        if not np.isfinite(a):
            continue
        t = meta["area_fluct_amps"]
        h = HDBSCAN(min_cluster_size=max(20, len(emb) // 8)).fit_predict(emb)
        km.append(a)
        hdb.append(adjusted_rand_score(np.searchsorted(np.unique(t), t), h))
    return np.array(km), np.array(hdb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", default="figures")
    main(ap.parse_args().save_dir)
