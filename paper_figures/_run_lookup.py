"""Helpers for paper-figure scripts to find MLflow runs by experiment.

The figure scripts in this directory used to hardcode randomly-generated
MLflow `run_name` strings (e.g. "bald-skink-299") that don't exist in a
fresh run set. This module replaces that pattern with experiment-name +
parameter-filter lookups so figures regenerate from any reviewer's runs.

Usage:
    from _run_lookup import setup_mlflow_uri, find_run_in_experiment

    setup_mlflow_uri()
    run = find_run_in_experiment(
        "membrane_stiffness_clusters",
        tag_filters={"embedding": "dino", "ablation": "full_int"},
        metric="best_val_loss",
    )
    model = run["model"]; cfg = run["cfg"]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import mlflow
import torch
import yaml

from _ckpt_store import CkptRun, StateDictView, ckpt_path, ckpt_root, list_runs

REPO_ROOT = Path(__file__).resolve().parents[1]

from graph_dyno.plotting.plot import load_run  # noqa: E402  (after sys.path tweak)


def setup_mlflow_uri(uri: Optional[str] = None) -> str:
    """Point MLflow at MLFLOW_URI (env or arg) or the repo's ./mlruns directory.

    Returns the URI that was set.
    """
    if uri is None:
        uri = os.environ.get("MLFLOW_URI")
    if uri is None:
        uri = f"file://{(REPO_ROOT / 'mlruns').resolve()}"
    elif not uri.startswith(("file://", "sqlite:", "http://", "https://")):
        # Treat bare paths as filesystem URIs.
        uri = f"file://{Path(uri).resolve()}"
    mlflow.set_tracking_uri(uri)
    return uri


def load_experiments_yaml() -> dict:
    """Read paper_figures/experiments.yaml — figure → experiment-name mapping."""
    p = Path(__file__).resolve().parent / "experiments.yaml"
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def search_runs(experiment_name: str):
    """Every run of one experiment, or None if there is no such experiment."""
    if ckpt_root() is not None:
        return list_runs(experiment_name)
    client = mlflow.tracking.MlflowClient()
    exps = [e.experiment_id for e in client.search_experiments()
            if e.name == experiment_name]
    if not exps:
        return None
    return mlflow.search_runs(experiment_ids=exps, output_format="list")


def load_emb_meta(run):
    """(parameter_embedding, {meta key: array}) from a run's final checkpoint."""
    if isinstance(run, CkptRun):
        path = ckpt_path(run, "final")
    else:
        art = Path(urlparse(run.info.artifact_uri).path)
        hdir = run.data.tags.get("hydra_output_dir")
        if (art / "graph_decoder_model.pth").exists():
            path = art / "graph_decoder_model.pth"
        elif hdir and (Path(hdir) / "graph_decoder_model.pth").exists():
            path = Path(hdir) / "graph_decoder_model.pth"
        else:
            path = None
    if path is None:
        return None, None
    sd = torch.load(path, map_location="cpu", weights_only=False)
    emb = sd["parameter_embedding"].numpy()
    meta = {k[len("_meta_"):]: sd[k].numpy()
            for k in sd if k.startswith("_meta_")}
    if "area_fluct_periods" in meta and "area_fluct_amps" in meta:
        meta["area_fluct_amp_x_period"] = (
            meta["area_fluct_periods"] * meta["area_fluct_amps"])
    return emb, meta


def _ckpt_matches(run, tag_filters, param_filters):
    """Exact match, skipping keys the export layout doesn't define."""
    for filters, source in ((tag_filters, run.data.tags),
                            (param_filters, run.data.params)):
        for key, expected in (filters or {}).items():
            if key in source and str(source[key]) != str(expected):
                return False
    return True


def _ckpt_find_run(experiment_name, *, metric, tag_filters, param_filters,
                   use_last, run_name):
    runs = find_runs_in_experiment(experiment_name, tag_filters=tag_filters,
                                   param_filters=param_filters)
    if run_name:
        runs = [r for r in runs if r.info.run_name == run_name]
    if not runs:
        raise ValueError(
            f"No checkpoints of '{experiment_name}' under {ckpt_root()} match "
            f"tags={tag_filters} params={param_filters} run={run_name}.")
    if metric is None or run_name:
        chosen = runs[0]
    else:
        chosen = min(runs, key=lambda r: r.info.start_time)
        if len(runs) > 1:
            print(f"  WARNING: checkpoints carry no `{metric}`; using the lowest "
                  f"seed of {len(runs)} in '{experiment_name}' "
                  f"({chosen.info.run_name}). Pass a seed_<N> run name to choose.")
    which = "final" if use_last else "best"
    path = ckpt_path(chosen, which)
    if path is None:
        raise ValueError(f"'{experiment_name}' {chosen.info.run_name}: the export "
                         f"holds no {which} checkpoint.")
    return {"model": StateDictView(torch.load(path, map_location="cpu",
                                              weights_only=False))}


def find_run_in_experiment(
    experiment_name: str,
    *,
    metric: Optional[str] = None,
    direction: str = "min",
    tag_filters: Optional[dict] = None,
    param_filters: Optional[dict] = None,
    use_last: bool = False,
    status: str = "FINISHED",
    run_name: Optional[str] = None,
) -> dict:
    """Return the loaded `run` dict (per `plot.load_run`) for one MLflow run.

    Selection rules, applied in order:
    1. Filter by experiment_name and `status` (default FINISHED).
    2. Apply tag_filters (exact match on `run.data.tags`).
    3. Apply param_filters (exact match on `run.data.params`).
    4. If `metric` is given, pick the run optimizing it (ascending if
       direction="min", descending if "max"). Otherwise, return the most
       recent run.

    Raises ValueError if no run matches.

    From a checkpoint tree, the lowest seed stands in for a metric and the
    model is a `StateDictView`; `run_name` (seed_<N>) picks a run.
    """
    if ckpt_root() is not None:
        return _ckpt_find_run(experiment_name, metric=metric,
                              tag_filters=tag_filters, param_filters=param_filters,
                              use_last=use_last, run_name=run_name)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(
            f"MLflow experiment '{experiment_name}' not found at "
            f"{mlflow.get_tracking_uri()}."
        )

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id], output_format="list"
    )
    runs = [r for r in runs if r.info.status == status]

    def _matches(run, filters, source):
        for key, expected in (filters or {}).items():
            if str(source(run).get(key, "")) != str(expected):
                return False
        return True

    runs = [
        r for r in runs
        if _matches(r, tag_filters, lambda r: r.data.tags)
        and _matches(r, param_filters, lambda r: r.data.params)
    ]
    if not runs:
        raise ValueError(
            f"No runs in experiment '{experiment_name}' match "
            f"tags={tag_filters} params={param_filters}."
        )

    if metric is not None:
        sign = 1 if direction == "min" else -1
        runs = sorted(
            runs,
            key=lambda r: sign * (r.data.metrics.get(metric, float("inf")) or float("inf")),
        )
    else:
        runs = sorted(runs, key=lambda r: r.info.start_time, reverse=True)

    chosen = runs[0]
    hydra_dir = (
        chosen.data.tags.get("hydra_output_dir")
        or chosen.data.params.get("hydra_output_dir")
    )
    if hydra_dir is None:
        raise ValueError(
            f"Run {chosen.info.run_id} has no `hydra_output_dir` tag/param; "
            "cannot resolve checkpoint path."
        )
    return load_run(hydra_dir, use_last=use_last)


def find_runs_in_experiment(
    experiment_name: str,
    *,
    tag_filters: Optional[dict] = None,
    param_filters: Optional[dict] = None,
    status: str = "FINISHED",
) -> list:
    """Return raw mlflow Run objects matching the filters (no load_run)."""
    if ckpt_root() is not None:
        return [r for r in list_runs(experiment_name) or []
                if _ckpt_matches(r, tag_filters, param_filters)]
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return []
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id], output_format="list"
    )
    runs = [r for r in runs if r.info.status == status]

    def _matches(run, filters, source):
        for key, expected in (filters or {}).items():
            if str(source(run).get(key, "")) != str(expected):
                return False
        return True

    return [
        r for r in runs
        if _matches(r, tag_filters, lambda r: r.data.tags)
        and _matches(r, param_filters, lambda r: r.data.params)
    ]
