"""Helpers for paper-figure scripts to find MLflow runs by experiment.

The figure scripts in this directory used to hardcode randomly-generated
MLflow `run_name` strings (e.g. "bald-skink-299") that don't exist in a
fresh run set. This module replaces that pattern with experiment-name +
parameter-filter lookups so figures regenerate from any reviewer's runs.

Usage:
    from _run_lookup import setup_mlflow_uri, find_run_in_experiment

    setup_mlflow_uri()
    run = find_run_in_experiment(
        "tension_clusters",
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

import mlflow
import yaml

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


def find_run_in_experiment(
    experiment_name: str,
    *,
    metric: Optional[str] = None,
    direction: str = "min",
    tag_filters: Optional[dict] = None,
    param_filters: Optional[dict] = None,
    use_last: bool = False,
    status: str = "FINISHED",
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
    """
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
