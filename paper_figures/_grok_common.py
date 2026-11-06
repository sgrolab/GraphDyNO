"""Per-epoch embedding snapshots of the H = 64 oscillation runs, shared by
figure3.py (row 2), figure3_supp.py and grok_recovery_table.py.

The runs come from `run_training.sh grok-h64`, which saves the parameter
embedding every 2 epochs. Each figure and table reads two snapshots of the same
run, the early one (epoch 40, where the embedding is organised along
alpha*omega) and the late one (epoch 200, where alpha and omega have split),
instead of a best or last checkpoint. The experiment, filters and epochs come
from `grok_h64` in experiments.yaml.

Snapshots are read from `embedding_snapshots.pt` (written when
`training.embedding_snapshot_every` is set), falling back to the
`param_embedding_epoch_<N>.pt` files train.py saves every 20 epochs.
"""
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch

from _run_lookup import find_runs_in_experiment, load_experiments_yaml

GROK_CFG = load_experiments_yaml().get("grok_h64", {})
EARLY = GROK_CFG.get("early_epoch", 40)
LATE = GROK_CFG.get("late_epoch", 200)
TARGETS = ("alpha", "omega", "alpha_omega")


def _dirs(run):
    """Where a run's files can live: its artifact dir and its hydra dir."""
    out = [Path(urlparse(run.info.artifact_uri).path)]
    hdir = run.data.tags.get("hydra_output_dir") or run.data.params.get("hydra_output_dir")
    if hdir:
        out.append(Path(hdir))
    return out


class Snapshots:
    """epoch -> parameter_embedding for one run, plus its alpha/omega targets."""

    def __init__(self, run):
        self.run = run
        self.dirs = _dirs(run)
        self._dict = None
        for d in self.dirs:
            if (d / "embedding_snapshots.pt").exists():
                self._dict = torch.load(d / "embedding_snapshots.pt", map_location="cpu")
                break
        state = None
        for d in self.dirs:
            for name in ("graph_decoder_model.pth", "graph_decoder_model_best.pth"):
                if (d / name).exists():
                    state = torch.load(d / name, map_location="cpu", weights_only=False)
                    break
            if state is not None:
                break
        if state is None:
            raise FileNotFoundError(f"{run.info.run_name}: no checkpoint in {self.dirs}")
        alpha = state["_meta_area_fluct_amps"].numpy().ravel()
        omega = 2 * np.pi / state["_meta_area_fluct_periods"].numpy().ravel()
        self.targets = {"alpha": alpha, "omega": omega, "alpha_omega": alpha * omega}

    def epochs(self):
        if self._dict is not None:
            return sorted(self._dict)
        found = set()
        for d in self.dirs:
            for p in d.glob("param_embedding_epoch_*.pt"):
                found.add(int(p.stem.rsplit("_", 1)[1]))
        return sorted(found)

    def __getitem__(self, epoch):
        if self._dict is not None:
            return self._dict[epoch].numpy().astype(np.float64)
        for d in self.dirs:
            p = d / f"param_embedding_epoch_{epoch}.pt"
            if p.exists():
                sd = torch.load(p, map_location="cpu")
                return sd["parameter_embedding"].numpy().astype(np.float64)
        raise KeyError(f"{self.run.info.run_name}: no snapshot at epoch {epoch}")


def find_runs(experiment=None, param_filters=None):
    """The matching FINISHED runs, lowest model.seed first."""
    experiment = experiment or GROK_CFG.get("experiment")
    if param_filters is None:
        param_filters = GROK_CFG.get("param_filters")
    runs = find_runs_in_experiment(experiment, param_filters=param_filters)
    runs.sort(key=lambda r: int(r.data.params.get("model.seed", 0)))
    print(f"[{experiment}] {len(runs)} runs matching {param_filters}: "
          + ", ".join(f"{r.info.run_name} (seed {r.data.params.get('model.seed')})" for r in runs))
    if not runs:
        raise SystemExit("no runs to score")
    return runs


def pick_snapshots(experiment=None, run_name=None):
    """Snapshots of the run a single-run figure draws: `run_name` if given,
    else the lowest model.seed. (These runs log no best_val_loss to select by.)
    """
    runs = find_runs(experiment)
    if run_name is None:
        return Snapshots(runs[0])
    for r in runs:
        if r.info.run_name == run_name:
            return Snapshots(r)
    raise SystemExit(f"--grok-run {run_name!r} is not among the matched runs")
