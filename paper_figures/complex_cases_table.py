"""Table tab:complex_cases — all three columns, written as the paper's LaTeX table.

k-means ARI at known k, median [IQR] over runs, in one DINOv3 row:

  Two parameters     2 oscillation amplitudes x 2 adhesion strengths (k = 4),
                     labelled by the full factor tuple; BEST checkpoint
  Three parameters   x 2 membrane stiffnesses (k = 8); BEST checkpoint
  Real cells (CTC)   DIC-C2DH-HeLa vs PhC-C2DL-PSC, labelled by dataset, with
                     the untrained slots dropped first; FINAL checkpoint

Checkpoints: each column uses the checkpoint the export holds for its
experiment (`COLUMNS`), best for the two synthetic columns and final for CTC.

Every column is scored by the same pinned readout,
KMeans(k, n_init=10, random_state=0) on the raw embedding, so the spread is
training-seed spread only. The readout is fixed rather than searched: with
only ~139 scorable CTC slots, taking the best of several recipes per run
reaches ARI ~0.2 on embeddings whose linear-probe AUC is chance.

Synthetic labels: a cell's label is its full factor tuple, read from the
checkpoint's `_meta_*` buffers. Only design factors count, i.e. buffers that
vary and have at most MAX_LEVELS levels. `model.seed` varies initialization
and the noise generator only; the data split and batch order are fixed by
`dataloader.seed`, so the spread is not data-resampling variance.

CTC slots: a `parameter_embedding` row that never fed a training loss keeps
its random init and would add pure noise, so it is dropped first. The set
comes from `untrained_slots.json` beside an exported checkpoint, else it is
recomputed from the trajectory cache (`trainable_mask`).

Experiments, k, the CTC filters and the CTC label come from the
`combined_params` and `ctc` blocks of experiments.yaml; set MLFLOW_URI to the
tracking store, or pass --ckpt-dir to read the exported checkpoints. Writes,
into --save-dir:

  complex_cases_table.tex       the tab:complex_cases table environment
  complex_cases_per_seed.csv    case, checkpoint, seed, ARI, run_id per run scored

    python paper_figures/complex_cases_table.py [--save-dir figures] [--ckpt-dir]
"""
import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from _ckpt_store import (CkptRun, add_source_args, apply_source_args, ckpt_path,
                         ckpt_root)
from _combo_common import fmt_iqr, kmeans_ari, summarize, tex_iqr
from _run_lookup import (find_runs_in_experiment, load_experiments_yaml,
                         setup_mlflow_uri)

#: (key, column header, checkpoint).
COLUMNS = [
    ("two_params", "Two parameters", "best"),
    ("three_params", "Three parameters", "best"),
    ("ctc", "Real cells (CTC)", "final"),
]
ROW_LABEL = "DINOv3"
PREC = 2
CAPTION = ("GraphDyNO accurately clusters cells by combinations of parameters and "
           "real segmented cells by cell type. The values are the $k$-means ARI "
           "reported as median [IQR] over {n} runs. ")
LABEL = "tab:complex_cases"

#: Checkpoint file names in an MLflow run's hydra output dir.
CKPT_FILES = {"best": "graph_decoder_model_best.pth", "final": "graph_decoder_model.pth"}

# A swept factor has a handful of levels (2 in both datasets). Anything with
# more per-cell values is a nuisance variable, not part of the design.
MAX_LEVELS = 12


# --- loading ---------------------------------------------------------------

def checkpoint_path(run, which):
    """Path of a run's `which` checkpoint ("best" / "final"), or None."""
    if isinstance(run, CkptRun):
        return ckpt_path(run, which)
    d = run.data.tags.get("hydra_output_dir")
    p = Path(d) / CKPT_FILES[which] if d else None
    return p if p is not None and p.exists() else None


def _seed(run):
    s = run.data.params.get("model.seed")
    return int(s) if s is not None else None


def _by_seed(runs):
    return sorted(runs, key=lambda r: (_seed(r) is None, _seed(r) or 0))


# --- synthetic labels ------------------------------------------------------

def design_factors(sd):
    """{factor: per-cell values} for the swept factors among a state dict's `_meta_*`."""
    n = sd["parameter_embedding"].shape[0]
    meta = {k[len("_meta_"):]: np.round(sd[k].numpy().ravel(), 6)
            for k in sd if k.startswith("_meta_")}
    # A design factor varies AND has few levels. A constant buffer is a
    # parameter the sim emitted but did not sweep; `area_fluct_phases` is a
    # per-cell random phase that would make every cell its own group.
    return {k: v for k, v in meta.items()
            if v.size == n and 1 < len(np.unique(v)) <= MAX_LEVELS}


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
    return labels


# --- CTC trainable slots ---------------------------------------------------

def trainable_mask(cache_dir, val_ratio=0.0, loss_mask_mode="strict"):
    """(N,) bool mask — True where the slot fed at least one training loss.

    On real CTC data most slots are padding in most windows, and a slot only
    contributes when `prev_valid & valid & next_valid` all hold at some
    TRAINING timestep. Pure padding, validation-only timesteps, and short
    tracks at a window edge (valid, but never a prev/cur/next triple) all
    leave a row at its random init.

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


def logged_cache_dir(run):
    """The run's trajectory cache: logged in its hydra config, or DATA_DIR's for an export."""
    if isinstance(run, CkptRun):
        return os.path.join(os.environ.get("DATA_DIR", "./data"), "ctc_dic_psc_late_cache/")
    cfg = OmegaConf.load(Path(run.data.tags["hydra_output_dir"]) / ".hydra" / "config.yaml")
    return str(cfg.data.generator.cache_dir)


def untrained_json(run, path):
    """untrained_slots.json beside an exported checkpoint, or None."""
    if isinstance(run, CkptRun):
        p = Path(path).parent / "untrained_slots.json"
        if p.exists():
            return p
    return None


def json_mask(path):
    d = json.loads(Path(path).read_text())
    keep = np.ones(d["n_slots"], dtype=bool)
    keep[d["untrained_slots"]] = False
    return keep


# --- scoring ---------------------------------------------------------------

def score_combo(experiment, k, which):
    """[{seed, ari, run_id}] for every run of a combined-parameter experiment."""
    runs = find_runs_in_experiment(experiment)
    if not runs:
        print(f"  WARNING: no runs found for '{experiment}'")
    rows = []
    for r in _by_seed(runs):
        path = checkpoint_path(r, which)
        if path is None:
            print(f"  WARNING: {r.info.run_name} ({experiment}) has no {which} "
                  f"checkpoint; skipped")
            continue
        sd = torch.load(path, map_location="cpu", weights_only=False)
        labels = combo_labels(design_factors(sd), k)
        emb = sd["parameter_embedding"].numpy()
        rows.append(dict(seed=_seed(r), ari=kmeans_ari(emb, labels, k),
                         run_id=r.info.run_id))
    return rows


def score_ctc(experiment, cfg, label, which, cache_dir=None):
    """[{seed, ari, run_id}] for the CTC runs, trainable slots only."""
    runs = find_runs_in_experiment(experiment, tag_filters=cfg.get("tag_filters"),
                                   param_filters=cfg.get("param_filters"))
    if not runs:
        print(f"  WARNING: no runs found for '{experiment}' with filters "
              f"{cfg.get('tag_filters')} {cfg.get('param_filters')}")
    masks, rows = {}, []
    for r in _by_seed(runs):
        path = checkpoint_path(r, which)
        if path is None:
            print(f"  WARNING: {r.info.run_name} ({experiment}) has no {which} "
                  f"checkpoint; skipped")
            continue
        sd = torch.load(path, map_location="cpu", weights_only=False)
        js = untrained_json(r, path)
        cd = str(js) if js else (cache_dir or logged_cache_dir(r))
        if cd not in masks:
            masks[cd] = json_mask(js) if js else trainable_mask(
                cd, val_ratio=float(r.data.params.get("dataloader.val_ratio", 0.0)))
        keep = masks[cd]
        emb = sd["parameter_embedding"].numpy()[keep]
        y = sd[f"_meta_{label}"].numpy().ravel()[keep].astype(int)
        rows.append(dict(seed=_seed(r), ari=kmeans_ari(emb, y, len(np.unique(y))),
                         run_id=r.info.run_id))
    for cd, m in masks.items():
        print(f"  trainable slots: {int(m.sum())}/{m.size}  ({cd})")
    return rows


# --- output ----------------------------------------------------------------

def emit_table(results):
    """The tab:complex_cases table environment, with n filled into the caption."""
    ns = sorted({len(results[key]) for key, *_ in COLUMNS})
    if len(ns) > 1:
        print(f"  WARNING: n differs between columns "
              + ", ".join(f"{key}={len(results[key])}" for key, *_ in COLUMNS)
              + "; the caption gives the range")
    n = str(ns[0]) if len(ns) == 1 else f"{ns[0]}--{ns[-1]}"
    cells = [tex_iqr([x["ari"] for x in results[key]], prec=PREC, macro="iqrv")
             for key, *_ in COLUMNS]
    lines = [
        r"\begin{table}",
        r"  \centering",
        r"  \caption{" + CAPTION.format(n=n) + "}",
        r"  \label{" + LABEL + "}",
        r"  \begin{tabular}{l" + "c" * len(COLUMNS) + "}",
        r"    \toprule",
        "    & " + " & ".join(hdr for _, hdr, _ in COLUMNS) + r" \\",
        r"    \midrule",
        f"    {ROW_LABEL} & " + " & ".join(cells) + r" \\",
        "",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    yml = load_experiments_yaml()
    combo, ctc = yml.get("combined_params", {}), yml.get("ctc", {})
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--two-params", default=combo.get("two_params", {}).get("experiment"),
                    help="MLflow experiment of the k=4 dataset")
    ap.add_argument("--three-params",
                    default=combo.get("three_params", {}).get("experiment"),
                    help="MLflow experiment of the k=8 dataset")
    ap.add_argument("--ctc-experiment", default=ctc.get("experiment"),
                    help="MLflow experiment of the CTC runs")
    ap.add_argument("--cache-dir", default=None,
                    help="CTC trajectory cache to compute trainable slots from "
                         "(default: untrained_slots.json beside the export, else "
                         "each run's logged data.generator.cache_dir)")
    ap.add_argument("--save-dir", default="figures")
    add_source_args(ap)
    args = apply_source_args(ap.parse_args())
    root = ckpt_root()
    print(f"checkpoints: {root}" if root else f"tracking uri: {setup_mlflow_uri()}")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    k2 = combo.get("two_params", {}).get("k", 4)
    k3 = combo.get("three_params", {}).get("k", 8)
    label = ctc.get("label", "dataset_id")
    scorers = {
        "two_params": (f"{args.two_params}, k={k2}",
                       lambda w: score_combo(args.two_params, k2, w)),
        "three_params": (f"{args.three_params}, k={k3}",
                         lambda w: score_combo(args.three_params, k3, w)),
        "ctc": (f"{args.ctc_experiment}, label {label}",
                lambda w: score_ctc(args.ctc_experiment, ctc, label, w, args.cache_dir)),
    }

    results = {}
    for key, hdr, which in COLUMNS:
        desc, score = scorers[key]
        print(f"\n{'=' * 78}\n### {hdr}: {desc}   ({which} checkpoint)")
        rows = score(which)
        results[key] = rows
        print(f"{'seed':>6} {'ARI':>8}  run")
        for x in rows:
            print(f"{str(x['seed']):>6} {x['ari']:>8.4f}  {x['run_id']}")

    print(f"\n{'=' * 78}\nTable {LABEL} — k-means ARI (raw), median [IQR]   "
          f"(mean ± sd, n)")
    for key, hdr, which in COLUMNS:
        aris = [x["ari"] for x in results[key]]
        s = summarize(aris)
        print(f"  {hdr:<18} {which:<5}  {fmt_iqr(aris):<22}  "
              f"({s['mean']:.3f} ± {s['sd']:.3f}, n={s['n']})")

    table = emit_table(results)
    print("\n" + table)
    tex_path = save_dir / "complex_cases_table.tex"
    tex_path.write_text(table + "\n")
    print(f"\nSaved {tex_path}")

    csv_path = save_dir / "complex_cases_per_seed.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "checkpoint", "seed", "ARI", "run_id"])
        for key, _, which in COLUMNS:
            for x in results[key]:
                w.writerow([key, which, x["seed"], x["ari"], x["run_id"]])
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
