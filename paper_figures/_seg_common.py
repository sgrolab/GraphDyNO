"""Loading and scoring shared by the segmentation-ablation tables.

Scoring is the paper's own: k-means ARI at known k (`figure4._recovery_ari`)
and neighbor consistency, both on the FINAL checkpoint. Under an ablation
`argmin(val_loss)` is driven by unpredictable corrupted transitions, so the
best-val checkpoint is not a model selector.

Two zero points exist, and they differ. The *published* clean runs
(`sweeps.oscillation` in experiments.yaml) are the number the paper reports
for the uncorrupted model. The *matched control* is the `bank_none` level
(tagged `none`): no perturbation, but the same `GraphModelSegAblation` with
Huber position/velocity and masked losses that every ablated arm uses, where
the published runs are plain `GraphModelDINO` with MSE. Deltas are measured
from the matched control, the only zero point that isolates the ablation.
"""
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch

import figure4
from _combo_common import nc
from _run_lookup import find_runs_in_experiment

#: The error kinds with a rate ladder, in table order.
RATE_KINDS = ["partial", "scale", "miss", "merge", "id_swap"]
KIND_LABELS = {"partial": "Partial mask", "scale": "Boundary offset",
               "miss": "Missed detection", "merge": "Merged neighbor",
               "id_swap": "ID swap"}
DOSES = [1, 2, 3, 4]
#: The balanced, non-episodic joint family, in dose order.
JOINT_PCT = ["joint_1pct", "joint_2pct", "joint_5pct", "joint_10pct"]

FINAL = "graph_decoder_model.pth"
BEST = "graph_decoder_model_best.pth"


def checkpoint(run, name=FINAL):
    """Path of one of a run's checkpoints (artifact dir, else hydra dir)."""
    for base in (Path(urlparse(run.info.artifact_uri).path),
                 Path(run.data.tags.get("hydra_output_dir", "/nonexistent"))):
        p = base / name
        if p.exists():
            return p
    return None


def load_emb_meta(run, name=FINAL):
    p = checkpoint(run, name)
    if p is None:
        return None, None
    sd = torch.load(p, map_location="cpu", weights_only=False)
    emb = sd["parameter_embedding"].numpy()
    meta = {k[len("_meta_"):]: sd[k].numpy().ravel()
            for k in sd if k.startswith("_meta_")}
    return emb, meta


def score(run, primary, name=FINAL):
    """(ARI, NC) for one run, or None if it has no such checkpoint."""
    emb, meta = load_emb_meta(run, name)
    if emb is None or primary not in meta:
        return None
    tgt = meta[primary]
    return figure4._recovery_ari(emb, tgt), nc(emb, tgt)


def realized(run):
    """Per-kind realized rate, averaged over trajectories.

    Realized, not requested: train.py logs what each run actually got
    (`segabl.t<i>.<kind>.realized_rate`). The planner drops an episode wherever
    the contact graph offers no free adjacent window, so on a sparse cache the
    pair kinds can land short of the requested rate.
    """
    def mean_of(pattern):
        vals = [float(v) for k, v in run.data.params.items()
                if re.fullmatch(pattern, k) and v not in (None, "None")]
        return float(np.mean(vals)) if vals else 0.0

    out = {kind: mean_of(rf"segabl\.t\d+\.{kind}\.realized_rate")
           for kind in RATE_KINDS}
    # Segmentation cell-frames: `merge` counts twice (the absorber gets the
    # blob, the partner goes invalid) and `id_swap` is a tracking error.
    out["seg_rate"] = float(out["partial"] + out["scale"] + out["miss"]
                            + 2.0 * out["merge"])
    return out


@lru_cache(maxsize=None)
def level_runs(experiment):
    """{level tag: [FINISHED runs]} for a seg-ablation experiment.

    Cached: on a file store, listing ~400 runs with ~270 params each is most
    of the runtime, and the joint tables read the same runs again per checkpoint.
    """
    out = {}
    for r in find_runs_in_experiment(experiment):
        lvl = r.data.tags.get("seg_ablation")
        if lvl is not None:
            out.setdefault(lvl, []).append(r)
    return out


def load_levels(experiment, primary, name=FINAL, levels=None):
    """{level: [(ARI, NC), ...]} and {level: [realized-rate dict, ...]}.

    `levels` restricts loading to those tags (default: every level).
    """
    scores, diag, n_skipped = {}, {}, 0
    by_level = level_runs(experiment)
    for lvl, runs in by_level.items():
        if levels is not None and lvl not in levels:
            continue
        for r in runs:
            s = score(r, primary, name)
            if s is None:
                n_skipped += 1
                continue
            scores.setdefault(lvl, []).append(s)
            diag.setdefault(lvl, []).append(
                dict(realized(r), seed=r.data.params.get("model.seed")))
    print(f"[{experiment}] {sum(len(v) for v in scores.values())} FINISHED runs "
          f"over {len(scores)} levels ({name})"
          + (f", {n_skipped} skipped: no checkpoint" if n_skipped else ""))
    return scores, diag


def mean_se(values):
    """(mean, standard error, n). SE is 0 for a single seed, not NaN."""
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return np.nan, np.nan, 0
    se = float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return float(a.mean()), se, int(a.size)


def stat(scores, level, idx):
    return mean_se([s[idx] for s in scores.get(level, [])])


def delta(scores, level, ref, idx):
    """level - ref, with the SE of the difference (independent seed sets).

    `ref` is either a level name in `scores` or a precomputed (mean, se, n).
    """
    m_a, se_a, n_a = stat(scores, level, idx)
    m_b, se_b, n_b = ref if isinstance(ref, tuple) else stat(scores, ref, idx)
    if not (np.isfinite(m_a) and np.isfinite(m_b)):
        return np.nan, np.nan, 0
    return m_a - m_b, float(np.hypot(se_a, se_b)), min(n_a, n_b)


def dose_x(diag, level, key):
    """The realized x-coordinate of one level, averaged over its seeds."""
    vals = [d[key] for d in diag.get(level, []) if np.isfinite(d.get(key, np.nan))]
    return float(np.mean(vals)) if vals else np.nan
