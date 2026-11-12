"""Table tab:stochastic — the self-propulsion noise f_p is needed to recover
differential adhesion.

The ladder datasets are the paper's differential-adhesion config with only the
polarity force `fp` changed (0 / 25 / 100). The f_p = 50 row is the paper's
own adhesion experiment (DINO, full_int), whose cache dir carries no fp
marker, so it is labelled explicitly.

Metric of record is the paper's: k-means ARI at known k against `_meta_khs`
(`figure4._recovery_ari`'s labelling), FINAL checkpoint by default. HDBSCAN ARI
(on non-noise points only, so read it with the noise fraction), neighbor
consistency and its chance level are printed alongside.

The ladder experiment comes from `stochastic` in experiments.yaml, the
f_p = 50 experiment from `sweeps.adhesion`; set MLFLOW_URI to the tracking
store.

    python paper_figures/stochastic_table.py [--checkpoint final|best]
"""
import argparse
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from scipy.spatial import KDTree
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

from _ckpt_store import (CkptRun, add_source_args, apply_source_args, ckpt_path,
                         ckpt_root)
from _combo_common import fmt_iqr
from _run_lookup import load_experiments_yaml, search_runs, setup_mlflow_uri

FACTOR = "khs"
N_BINS = 20        # figure4's cap on label bins
K_NEIGHBORS = 10   # NC neighbourhood size
ORDER = ["fp0", "fp25", "fp50", "fp100"]


def run_dir(run):
    """Where the checkpoint landed — artifact dir, else the hydra dir."""
    art = Path(urlparse(run.info.artifact_uri).path)
    if (art / "graph_decoder_model.pth").exists():
        return art
    hdir = run.data.tags.get("hydra_output_dir")
    if hdir and (Path(hdir) / "graph_decoder_model.pth").exists():
        return Path(hdir)
    return None


def load_emb_meta(run, which):
    if isinstance(run, CkptRun):
        path = ckpt_path(run, which) or ckpt_path(run, "final")
        if path is None:
            return None, None
    else:
        rdir = run_dir(run)
        if rdir is None:
            return None, None
        path = rdir / ("graph_decoder_model.pth" if which == "final"
                       else "graph_decoder_model_best.pth")
        if not path.exists():          # fall back rather than drop the arm
            path = rdir / "graph_decoder_model.pth"
    sd = torch.load(path, map_location="cpu", weights_only=False)
    emb = sd["parameter_embedding"].numpy()
    meta = {k[len("_meta_"):]: sd[k].numpy() for k in sd if k.startswith("_meta_")}
    return emb, meta


def discrete_labels(target):
    """figure4._recovery_ari's labelling: exact levels if few, else binned."""
    unique = np.unique(target)
    if len(unique) <= N_BINS:
        return np.searchsorted(unique, target)
    bins = np.linspace(target.min() - 1e-9, target.max() + 1e-9, N_BINS + 1)
    return np.digitize(target, bins)


def scores(emb, target):
    mask = np.isfinite(target) & np.isfinite(emb).all(axis=1)
    emb, target = emb[mask], target[mask]
    if len(target) < 4 or np.std(target) < 1e-12:
        return None
    labels = discrete_labels(target)
    k = len(np.unique(labels))
    if k < 2:
        return None
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(emb)
    hdb = HDBSCAN(min_cluster_size=max(20, len(emb) // (2 * k))).fit_predict(emb)
    keep = hdb >= 0
    _, idx = KDTree(emb).query(emb, k=K_NEIGHBORS + 1)
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return dict(k=k,
                ari_km=float(adjusted_rand_score(labels, km)),
                ari_hdb=(float(adjusted_rand_score(labels[keep], hdb[keep]))
                         if keep.sum() >= 2 else np.nan),
                n_hdb=int(len(set(hdb[keep]))) if keep.sum() else 0,
                noise_frac=float((~keep).mean()),
                nc=float((labels[idx[:, 1:]] == labels[:, None]).mean()),
                nc_chance=float((p ** 2).sum()))


def arm_of(run):
    """Which fp arm a run belongs to, from its logged config."""
    fp = run.data.params.get("data.params.fp")
    if fp is not None:
        return f"fp{int(float(fp))}"
    cache = run.data.params.get("data.generator.cache_dir", "")
    for tag in ("fp100", "fp25", "fp0"):   # longest first
        if tag in cache:
            return tag
    return "unknown"


def collect(experiment, which, by_arm, *, arm=None, embedding=None, ablation=None):
    runs = search_runs(experiment)
    if runs is None:
        print(f"  [{experiment}] not found in MLflow")
        return
    for r in runs:
        if r.info.status != "FINISHED":
            continue
        if embedding and r.data.tags.get("embedding") != embedding:
            continue
        if ablation and r.data.tags.get("ablation") != ablation:
            continue
        emb, meta = load_emb_meta(r, which)
        if emb is None or FACTOR not in meta:
            continue
        s = scores(emb, meta[FACTOR].ravel())
        if s is None:
            continue
        s.update(seed=r.data.params.get("model.seed"), run=r.info.run_name)
        by_arm[arm or arm_of(r)].append(s)


def fmt(v, spec="6.3f"):
    return "   nan" if v is None or not np.isfinite(v) else format(v, spec)


def main():
    cfg = load_experiments_yaml()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment",
                    default=cfg.get("stochastic", {}).get("experiment", "diff_adhesion_fp_ladder"),
                    help="MLflow experiment holding the fp = 0 / 25 / 100 arms")
    ap.add_argument("--fp50-experiment",
                    default=cfg.get("sweeps", {}).get("adhesion", "diff_adhesion_ablations"),
                    help="the paper's adhesion experiment (the fp = 50 arm)")
    ap.add_argument("--checkpoint", default="final", choices=["final", "best"])
    add_source_args(ap)
    args = apply_source_args(ap.parse_args())
    if args.checkpoint == "best" and ckpt_root() is not None:
        raise SystemExit("--checkpoint best: the exported checkpoints are final only.")
    print(f"tracking uri: {setup_mlflow_uri()}")

    by_arm = defaultdict(list)
    collect(args.experiment, args.checkpoint, by_arm)
    collect(args.fp50_experiment, args.checkpoint, by_arm,
            arm="fp50", embedding="dino", ablation="full_int")
    arms = [a for a in ORDER if a in by_arm] + [a for a in by_arm if a not in ORDER]

    print(f"checkpoint: {args.checkpoint}   factor: {FACTOR}")
    hdr = (f"{'arm':>6} {'seed':>5} | {'ARI_km':>6} {'ARI_hdb':>7} {'hdb#/noise':>11} "
           f"| {'NC':>6} {'chance':>6} | run")
    print(hdr + "\n" + "-" * len(hdr))
    for arm in arms:
        for x in sorted(by_arm[arm], key=lambda x: int(x["seed"] or 0)):
            print(f"{arm:>6} {str(x['seed']):>5} | {fmt(x['ari_km'])} "
                  f"{fmt(x['ari_hdb'], '7.3f')} {x['n_hdb']:>3d}/{x['noise_frac'] * 100:>5.1f}%  "
                  f"| {fmt(x['nc'])} {fmt(x['nc_chance'])} | {x['run']}")
        print()

    print("Table tab:stochastic — k-means ARI for k_h, median [IQR]   (mean ± sd)")
    for arm in arms:
        v = np.array([x["ari_km"] for x in by_arm[arm]], float)
        v = v[np.isfinite(v)]
        if v.size:
            sd = v.std(ddof=1) if v.size > 1 else 0.0
            print(f"  f_p = {arm[2:]:>4}   {fmt_iqr(v)}   ({v.mean():.3f} ± {sd:.3f}, n={v.size})")


if __name__ == "__main__":
    main()
