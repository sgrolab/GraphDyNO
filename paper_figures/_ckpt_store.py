"""Read runs from the exported `model_checkpoints/` tree instead of MLflow.

Opt in with `--ckpt-dir [DIR]` (bare flag: <repo>/model_checkpoints) or
GRAPHDYNO_CKPT_DIR. Each export file is a bare state dict, `seed_<N>.pth`
(or `.pt`), holding `parameter_embedding` and the `_meta_*` buffers. Its
directory levels stand in for the MLflow tags and params the scripts filter
on (`LAYOUTS`). There are no metrics.
"""
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "model_checkpoints"
ENV_VAR = "GRAPHDYNO_CKPT_DIR"

_cli_root = None


def ckpt_root():
    """The checkpoint tree in use, or None for the default MLflow path."""
    root = _cli_root or os.environ.get(ENV_VAR)
    return Path(root) if root else None


def add_source_args(parser):
    parser.add_argument(
        "--ckpt-dir", nargs="?", const=str(DEFAULT_ROOT), default=None, metavar="DIR",
        help=f"read the exported checkpoints instead of MLflow "
             f"(bare flag: {DEFAULT_ROOT}; also ${ENV_VAR})")
    return parser


def apply_source_args(args):
    global _cli_root
    if getattr(args, "ckpt_dir", None):
        _cli_root = args.ckpt_dir
    return args


class Layout(NamedTuple):
    subpath: str
    #: One parser per directory level below `subpath`: name -> (tags, params).
    levels: tuple = ()
    tags: dict = {}
    params: dict = {}
    #: Which checkpoint the exported files are.
    file: str = "final"


def _tag(key, fmt: "str | Callable" = "{}"):
    return lambda name: ({key: fmt(name) if callable(fmt) else fmt.format(name)}, {})


def _param(key, fmt: "str | Callable" = "{}"):
    return lambda name: ({}, {key: fmt(name) if callable(fmt) else fmt.format(name)})


def _arch_axis(name):
    axis, n = name.rsplit("_", 1)
    return {"arch_axis": axis}, {f"model.{axis}": n}


_SCALE_CACHE = {"more_osc_clusters": "more_oscillating_clusters_new/",
                "membrane_stiffness_two_clusters": "membrane_stiffness_two_clusters_cache/",
                "differential_adhesion_clusters": "differential_adhesion_clusters_cache/"}
_DINO_FULL = {"embedding": "dino", "ablation": "full_int"}

LAYOUTS = {
    **{exp: Layout(f"fig4_fig5_sweeps/{exp}",
                   (_tag("embedding"), _tag("ablation", "{}_int")))
       for exp in ("oscillation_paper", "membrane_stiffness_clusters", "diff_adhesion_ablations")},
    "scale": Layout("tab_scale_only",
                    (_param("data.generator.cache_dir", _SCALE_CACHE.__getitem__),),
                    tags={"ablation": "scale_only"}),
    "osc_arch_sweep": Layout("tab_arch_sweep/osc_arch_sweep", (_arch_axis,), tags=_DINO_FULL),
    **{exp: Layout(f"tab_encoder_cell_dino/{exp}",
                   tags={"embedding": "cell_dino", "ablation": "full_int"})
       for exp in ("oscillation_cell_dino", "membrane_stiffness_cell_dino", "diff_adhesion_cell_dino")},
    "osc_amp_x_adhesion_seeds20":
        Layout("tab_complex_cases/two_params/osc_amp_x_adhesion_seeds20", file="best"),
    "osc_amp_x_adhesion_x_membrane_stiffness_kh1e5_seeds20":
        Layout("tab_complex_cases/three_params/osc_amp_x_adhesion_x_membrane_stiffness_kh1e5_seeds20",
               file="best"),
    "ctc_dic_psc_late": Layout("tab_complex_cases/ctc",
                               tags={"ablation": "b_shape_scale_int"},
                               params={"model.embedding_dim": "16",
                                       "dataloader.val_ratio": "0.0"}),
    "segabl_oscillation": Layout("tab_seg_ablation/segabl_oscillation",
                                 (_tag("seg_ablation"),)),
    "diff_adhesion_fp_ladder": Layout("tab_stochastic/diff_adhesion_fp_ladder",
                                      (_param("data.params.fp", lambda s: s[len("fp_"):]),)),
    "osc_period_clusters": Layout("tab_timescales/osc_period_clusters", tags=_DINO_FULL),
    "growing_cells_dino": Layout("fig3/growing_cells_dino", file="best"),
    "osc_grok_param_count": Layout("fig3/osc_grok_param_count"),
}

_SEED_FILE = re.compile(r"^seed_(\d+)\.(pth|pt)$")


class CkptRun:
    """The fields of an mlflow `Run` the figure scripts read, for one export file."""

    def __init__(self, path, seed, tags, params, which):
        self.info = SimpleNamespace(
            run_name=f"seed_{seed}", run_id=str(path.with_suffix("")),
            status="FINISHED", start_time=seed, artifact_uri=path.parent.as_uri())
        self.data = SimpleNamespace(tags=tags, params={**params, "model.seed": str(seed)},
                                    metrics={})
        snap = path.with_name(f"seed_{seed}_snapshots.pt")
        fig = path.with_name("fig_seed.pt")
        self.files = {"final": None, "best": None,
                      "snapshots": snap if snap.exists() else None,
                      "fig_seed": fig if fig.exists() else None}
        self.files[which] = path


def list_runs(experiment):
    """Every export run of `experiment`, highest seed first (MLflow's
    start_time-DESC order), or None if the tree doesn't hold it."""
    layout, root = LAYOUTS.get(experiment), ckpt_root()
    if layout is None or not (root / layout.subpath).is_dir():
        return None
    base = root / layout.subpath
    dirs = sorted(base.glob("/".join(["*"] * len(layout.levels)))) if layout.levels else [base]
    runs = []
    for d in dirs:
        if not d.is_dir():
            continue
        tags, params = dict(layout.tags), dict(layout.params)
        for parse, name in zip(layout.levels, d.relative_to(base).parts):
            t, p = parse(name)
            tags.update(t)
            params.update(p)
        for f in sorted(d.iterdir()):
            m = _SEED_FILE.match(f.name)
            if m:
                runs.append(CkptRun(f, int(m[1]), dict(tags), dict(params), layout.file))
    runs.sort(key=lambda r: r.info.start_time, reverse=True)
    return runs


def ckpt_path(run, which):
    """Path of a CkptRun's `which` checkpoint, or None if the export holds the other one."""
    return run.files[which]


class StateDictView:
    """The model attributes figure3 reads, over a bare state dict."""

    def __init__(self, sd):
        self._sd = sd
        self.parameter_embedding = sd["parameter_embedding"]

    def get_metadata(self, key):
        return self._sd.get(f"_meta_{key}")
