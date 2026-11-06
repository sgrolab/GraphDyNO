"""Table tab:complex_cases, two- and three-parameter columns.

Each dataset varies several mechanisms at once (2 oscillation amplitudes x 2
adhesion strengths, k = 4; x 2 membrane tensions, k = 8), and one config is
trained over many `model.seed`s. Every cell gets a label from its full factor
tuple, read from the checkpoint's `_meta_*` buffers, and the parameter
embedding is clustered at known k. The table reports k-means on the raw
embedding at the FINAL checkpoint, mean +/- sd over seeds; the standardized
variant is printed alongside, and the best checkpoint as a diagnostic.

Caveat: `model.seed` varies initialization and the noise generator only. The
data split and batch order are fixed by `dataloader.seed`, so the spread is
initialization variance, not data-resampling variance.

Experiments and k come from `combined_params` in experiments.yaml; set
MLFLOW_URI to the tracking store.

    python paper_figures/combined_params_table.py
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from _combo_common import kmeans_ari, nc, summarize
from _run_lookup import (find_runs_in_experiment, load_experiments_yaml,
                         setup_mlflow_uri)

CKPTS = {"best": "graph_decoder_model_best.pth", "final": "graph_decoder_model.pth"}

# name -> (scorer, standardize)
RECIPES = {
    "raw": (kmeans_ari, False),
    "std": (kmeans_ari, True),
}
#: The recipe and checkpoint the table reports.
TABLE_RECIPE = ("final", "raw")

# A swept factor has a handful of levels (2 in both datasets). Anything with
# more per-cell values is a nuisance variable, not part of the design.
MAX_LEVELS = 12


def load_ckpt(run, which):
    """(embedding, {factor: per-cell values}) from a run's checkpoint, or None."""
    d = run.data.tags.get("hydra_output_dir")
    if not d:
        return None
    p = Path(d) / CKPTS[which]
    if not p.exists():
        return None
    sd = torch.load(p, map_location="cpu", weights_only=False)
    emb = sd["parameter_embedding"].numpy()
    n = emb.shape[0]
    meta = {k[len("_meta_"):]: np.round(sd[k].numpy().ravel(), 6)
            for k in sd if k.startswith("_meta_")}
    # A design factor varies AND has few levels. A constant buffer is a
    # parameter the sim emitted but did not sweep; `area_fluct_phases` is a
    # per-cell random phase that would make every cell its own group.
    meta = {k: v for k, v in meta.items()
            if v.size == n and 1 < len(np.unique(v)) <= MAX_LEVELS}
    return emb, meta


def combo_labels(meta, k):
    """Integer label per cell from the full factor tuple (order-independent)."""
    names = sorted(meta)
    design = np.stack([meta[n] for n in names], axis=1)
    _, labels = np.unique(design, axis=0, return_inverse=True)
    n_groups = len(np.unique(labels))
    if n_groups != k:
        levels = {n_: len(np.unique(meta[n_])) for n_ in names}
        raise ValueError(f"design resolves to {n_groups} groups, expected k={k}. "
                         f"Factors found: {levels}")
    return labels, names


def eval_arm(label, experiment, k):
    runs = find_runs_in_experiment(experiment)
    print(f"\n{'=' * 78}\n### {label}: {experiment}   (k={k}, {len(runs)} FINISHED runs)")
    rows, fingerprints = [], []
    for r in runs:
        seed = r.data.params.get("model.seed")
        row = dict(seed=int(seed) if seed is not None else None,
                   val=float(r.data.metrics.get("val_loss", np.nan)))
        for which in CKPTS:
            got = load_ckpt(r, which)
            if got is None:
                continue
            emb, meta = got
            labels, names = combo_labels(meta, k)
            for rec, (fn, std) in RECIPES.items():
                row[f"{which}_{rec}"] = fn(emb, labels, k, std)
            if which == "final":
                for n_ in names:
                    row[f"NC_{n_}"] = nc(emb, meta[n_])
                fingerprints.append(float(np.abs(emb).sum()))
        rows.append(row)
    rows.sort(key=lambda x: (x["seed"] is None, x["seed"]))
    if not rows:
        print("  no evaluable runs")
        return None

    nc_cols = sorted({c for x in rows for c in x if c.startswith("NC_")})
    hdr = (f"{'seed':>4} | " + " ".join(f"{r:>8}" for r in RECIPES) + " | "
           + " ".join(f"{c[3:][:7]:>7}" for c in nc_cols) + f" | {'val_loss':>9}")
    print("\nfinal checkpoint, ARI by recipe")
    print(hdr + "\n" + "-" * len(hdr))
    for x in rows:
        def f(key, w=8):
            v = x.get(key, np.nan)
            return f"{v:>{w}.4f}" if np.isfinite(v) else f"{'--':>{w}}"
        print(f"{str(x['seed']):>4} | "
              + " ".join(f(f"final_{r}") for r in RECIPES) + " | "
              + " ".join(f(c, 7) for c in nc_cols) + f" | {f('val', 9)}")

    print(f"\n{'recipe':<24} {'n':>3} {'mean':>7} {'sd':>7} {'median':>7} "
          f"{'min':>7} {'max':>7} {'95% CI':>17}")
    print("-" * 82)
    for which in CKPTS:
        for rec in RECIPES:
            s = summarize([x.get(f"{which}_{rec}", np.nan) for x in rows])
            star = " *" if (which, rec) == TABLE_RECIPE else "  "
            name = f"{which} k-means {rec}{star}"
            print(f"{name:<24} {s['n']:>3} {s['mean']:>7.4f} {s['sd']:>7.4f} "
                  f"{s['median']:>7.4f} {s['lo']:>7.4f} {s['hi']:>7.4f} "
                  f"  [{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]")
    print("* = the recipe the table reports.")

    # Seeds that produced identical weights would make the sd meaningless.
    if len(fingerprints) > 1:
        n_uniq = len(np.unique(np.round(fingerprints, 6)))
        print(f"\ndistinct parameter_embedding fingerprints: {n_uniq}/{len(fingerprints)}"
              + ("" if n_uniq == len(fingerprints) else "   !! seeds are NOT independent"))
    return summarize([x.get("_".join(TABLE_RECIPE), np.nan) for x in rows])


def main():
    cfg = load_experiments_yaml().get("combined_params", {})
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--two-params", default=cfg.get("two_params", {}).get("experiment"),
                    help="MLflow experiment of the k=4 dataset")
    ap.add_argument("--three-params", default=cfg.get("three_params", {}).get("experiment"),
                    help="MLflow experiment of the k=8 dataset")
    args = ap.parse_args()
    print(f"tracking uri: {setup_mlflow_uri()}")

    arms = [("Two parameters", args.two_params, cfg.get("two_params", {}).get("k", 4)),
            ("Three parameters", args.three_params,
             cfg.get("three_params", {}).get("k", 8))]
    table = {label: eval_arm(label, exp, k) for label, exp, k in arms}

    print(f"\n{'=' * 78}\nTable tab:complex_cases — k-means ARI "
          f"({TABLE_RECIPE[1]}, {TABLE_RECIPE[0]} checkpoint), median [IQR]   (mean ± sd)")
    for label, s in table.items():
        if s is not None:
            print(f"  {label:<18} {s['median']:.3f} [{s['q1']:.3f}, {s['q3']:.3f}]   "
                  f"({s['mean']:.3f} ± {s['sd']:.3f}, n={s['n']})")


if __name__ == "__main__":
    main()
