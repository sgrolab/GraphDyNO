"""Table tab:complex_cases, live-cell CTC column (DIC-C2DH-HeLa x PhC-C2DL-PSC).

One row per run of the reported recipe, then the column value: the k-means
ARI against the cell's dataset, mean +/- sd over runs.

PINNED READOUT:
    KMeans(k, n_init=10) on the raw embedding (figure 4's recipe),
on the FINAL checkpoint. With only ~139 scorable slots, taking the best of
several (space x covariance) recipes per run reaches ARI ~0.2 on embeddings
whose linear-probe AUC is chance, so the recipe is fixed rather than searched.

Slots whose `parameter_embedding` row never received gradient keep their
random init and are dropped first (see `trainable_mask`).

Columns
    std       median per-dim embedding std (< 0.05 = collapsed; read first)
    linAUC    5-fold logistic AUC on the standardized embedding
    rfAUC     5-fold RandomForest(300) AUC
    ARI       pinned k-means fit vs the label; purity = best-match accuracy
    kmPCA     KMeans(k, n_init=10) in StandardScaler -> PCA(2) space
    seqAUC    rfAUC for PSC/01 vs PSC/02 alone — two sequences of ONE dataset.
              A batch control: if it rivals rfAUC, the "dataset" signal may be
              per-video structure.

The experiment, filters and label come from `ctc` in experiments.yaml; set
MLFLOW_URI to the tracking store.

    python paper_figures/ctc_table.py [--cache-dir DIR]
"""
import argparse
import glob
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from _combo_common import fmt_iqr, summarize
from _run_lookup import (find_runs_in_experiment, load_experiments_yaml,
                         setup_mlflow_uri)

warnings.filterwarnings("ignore")


def trainable_mask(cache_dir, val_ratio=0.0, loss_mask_mode="strict"):
    """(N,) bool mask — True where the slot fed at least one training loss.

    On real CTC data most slots are padding in most windows, and a slot only
    contributes when `prev_valid & valid & next_valid` all hold at some
    TRAINING timestep. Pure padding, validation-only timesteps, and short
    tracks at a window edge (valid, but never a prev/cur/next triple) all
    leave a row at its random init; clustering those rows adds pure noise.

    Recomputed from the trajectory cache, reproducing the dataloader's
    indexing: a temporal split keeps timesteps [0, T - int(T * val_ratio));
    timestep 0 yields no sample (no prev); sample t carries valid[t],
    next_valid[t] and prev_valid = valid[t-1]. train.py writes the same set to
    untrained_slots.json in the run dir.
    """
    files = sorted(f for f in glob.glob(os.path.join(cache_dir, "*.pt"))
                   if not os.path.basename(f).startswith(("sdf_", "dino_", "finetuned_")))
    if not files:
        raise FileNotFoundError(f"no trajectory .pt files in {cache_dir}")
    trainable = None
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        valid = d["valid"].numpy()              # (T, N)
        next_valid = d["next_valid"].numpy()    # (T, N)
        T = valid.shape[0]
        n_train = T - int(T * val_ratio)
        m = valid[1:n_train] & next_valid[1:n_train]
        if loss_mask_mode == "strict":
            m = m & valid[0:n_train - 1]
        m = m.any(axis=0)
        trainable = m if trainable is None else (trainable | m)
    return trainable


def auc(X, y, forest):
    clf = (RandomForestClassifier(300, random_state=0, n_jobs=-1) if forest
           else LogisticRegression(max_iter=5000))
    n_cls = len(np.unique(y))
    n_splits = int(min(5, np.bincount(y).min()))
    if n_splits < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=0)
    return cross_val_score(clf, X, y, cv=cv,
                           scoring="roc_auc" if n_cls == 2 else "roc_auc_ovr").mean()


def kmeans_ari(X, y, k):
    pred = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    cm = confusion_matrix(y, pred)
    r, c = linear_sum_assignment(-cm)
    return adjusted_rand_score(y, pred), cm[r, c].sum() / cm.sum()


def logged_cache_dir(run):
    cfg = OmegaConf.load(Path(run.data.tags["hydra_output_dir"]) / ".hydra" / "config.yaml")
    return str(cfg.data.generator.cache_dir)


def main():
    cfg = load_experiments_yaml().get("ctc", {})
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default=cfg.get("experiment"))
    ap.add_argument("--label", default=cfg.get("label", "dataset_id"),
                    help="_meta_* buffer to score against")
    ap.add_argument("--cache-dir", default=None,
                    help="trajectory cache to compute trainable slots from "
                         "(default: each run's logged data.generator.cache_dir)")
    args = ap.parse_args()
    print(f"tracking uri: {setup_mlflow_uri()}")

    runs = [r for r in find_runs_in_experiment(
                args.experiment, tag_filters=cfg.get("tag_filters"),
                param_filters=cfg.get("param_filters"))
            if (Path(r.data.tags.get("hydra_output_dir", "")) / "graph_decoder_model.pth").exists()]
    print(f"experiment {args.experiment}   label {args.label}   "
          f"filters {cfg.get('tag_filters')} {cfg.get('param_filters')}   "
          f"pinned: KMeans(k, n_init=10) raw   {len(runs)} runs")
    hdr = (f"{'seed':>5}{'dim':>5}{'std':>7}{'linAUC':>8}{'rfAUC':>7}{'ARI':>7}"
           f"{'purity':>8}{'kmPCA':>7}{'seqAUC':>8}  run")
    print(hdr + "\n" + "-" * len(hdr))

    masks, aris = {}, []
    for r in sorted(runs, key=lambda x: int(x.data.params.get("model.seed", 0))):
        sd = torch.load(Path(r.data.tags["hydra_output_dir"]) / "graph_decoder_model.pth",
                        map_location="cpu", weights_only=False)
        cd = args.cache_dir or logged_cache_dir(r)
        if cd not in masks:
            masks[cd] = trainable_mask(
                cd, val_ratio=float(r.data.params.get("dataloader.val_ratio", 0.0)))
        keep = masks[cd]
        emb = sd["parameter_embedding"].numpy()[keep]
        y = sd[f"_meta_{args.label}"].numpy().ravel()[keep].astype(int)
        vid = sd["_meta_video_id"].numpy().ravel()[keep].astype(int)
        ds = sd["_meta_dataset_id"].numpy().ravel()[keep].astype(int)
        Xs = StandardScaler().fit_transform(emb)
        k = len(np.unique(y))
        ari, pur = kmeans_ari(emb, y, k)
        km_pca, _ = kmeans_ari(PCA(2, random_state=0).fit_transform(Xs), y, k)
        # PSC is the larger dataset_id; its two sequences are two video ids.
        psc = ds == ds.max()
        seq = (auc(Xs[psc], (vid[psc] == vid[psc].max()).astype(int), True)
               if len(np.unique(vid[psc])) == 2 else float("nan"))
        aris.append(ari)
        print(f"{r.data.params.get('model.seed', '?'):>5}"
              f"{r.data.params.get('model.embedding_dim'):>5}"
              f"{np.median(emb.std(0)):>7.3f}{auc(Xs, y, False):>8.3f}{auc(Xs, y, True):>7.3f}"
              f"{ari:>7.3f}{pur:>8.3f}{km_pca:>7.3f}{seq:>8.3f}  "
              f"{r.info.run_name}")
    for cd, m in masks.items():
        print(f"\ntrainable slots: {int(m.sum())}/{m.size}  ({cd})")

    s = summarize(aris)
    print(f"\nTable tab:complex_cases — Live cell CTC, k-means ARI, median [IQR]: "
          f"{fmt_iqr(aris)}   ({s['mean']:.3f} ± {s['sd']:.3f}, n={s['n']})")


if __name__ == "__main__":
    main()
