# Discovering interaction laws via graph neural networks

This repository accompanies the NeurIPS 2026 submission "Discovering biophysical parameters of simulated multicellular systems from image sequences of cell shapes" by Corey Herr, Cedric Allier, Stephan Saalfeld, and Allyson Sgro. It contains the training pipeline, the
PolyHoop-based synthetic-trajectory generator, and the figure-generation
scripts used to produce the paper figures.

## Contents

```
configs/                Hydra configs (data, model, training, embedding, ablation)
src/graph_dyno/         Library package: models, data loaders, plotting, training utilities
src/dino/               DINOv3 projection-head finetuning (LiveCell), runnable via `python -m dino.train_dino`
paper_figures/          Scripts that produce every figure in the paper
scripts/                Auxiliary utilities (data, plots, monitoring)
tests/                  Pipeline-verification scripts
train.py                GNN training entry point
train_autoencoder.py    Conv shape-autoencoder pretraining
run_training.sh         Bash entry point that wraps every paper training run, with per-section and per-figure targets
```

## Setup

With [uv](https://docs.astral.sh/uv/), one command builds the environment the
paper's runs used: Python 3.12 (`.python-version`), every package pinned by
`uv.lock`, and the `graph_dyno` + `dino` packages installed editable.

```bash
uv sync
source .venv/bin/activate
```

On Linux, PyTorch 2.9.1 comes from the CUDA 13 wheel index
(`torch==2.9.1+cu130`); on macOS uv installs the CPU wheel. Without uv:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
pip install -e .                      # exposes the graph_dyno + dino packages
```

Our modified PolyHoop bindings are in `_polyhoop/` and are imported as
`polyhoop` in `graph_dyno.data.polyhoop_simulation`. Build them once inside the
environment, and again after editing `polyhoop_bindings.cpp` (needs a C++17
compiler; OpenMP is used when available):

```bash
python _polyhoop/build.py      # Intel icpx if available, else c++
```

### External datasets and pretrained checkpoints

The DINOv3 projection-head and the conv autoencoder are trained on
[LiveCell](https://sartorius-research.github.io/LIVECell/). The mask extractor
downloads the dataset (~14 GB) on first run; pick a local directory and point
`LIVECELL_DIR` at it:

```bash
export LIVECELL_DIR=/path/to/livecell           # writable, ~30 GB after extraction
python -m dino.cell_mask_extractor              # downloads + builds ${LIVECELL_DIR}/cell_masks/
```

After extraction the directory layout is:

```
${LIVECELL_DIR}/
  raw/         # LiveCell images and COCO annotations (downloaded)
  cell_masks/  # sharded per-cell SDF crops (.pt files), used by both encoders
```

Both `train_autoencoder.py` (conv autoencoder) and `dino.train_dino`
(DINOv3 projection-head decoder) read from `${LIVECELL_DIR}/cell_masks/`.

### DINOv3 access

The DINOv3 backbones are gated on Hugging Face. Request access to the
[facebook/dinov3 collection](https://huggingface.co/collections/facebook/dinov3),
accept the license terms, and export a token with read access before running
any DINO encoder code:

```bash
export HF_TOKEN=hf_...                  # token from https://huggingface.co/settings/tokens
```

`dino.feature_extractor` calls `huggingface_hub.login(HF_TOKEN)` on first use
and downloads the requested backbone (`S`, `SP`, `B`, `L`, `H`, or `F`).

### Pretraining the encoders

```bash
# DINOv3 projection-head decoder. Defaults to dino_model=SP, bottleneck_dim=256
# — the variant used in the paper figures. precompute_embeddings runs
# automatically the first time a new (dino_model, split) combination is seen
# and caches CLS+patch tokens in mlruns artifacts.
python -m dino.train_dino dino.model.dino_model=SP dino.model.bottleneck_dim=256

# To reproduce the encoder-comparison sweep, repeat with each
# (dino_model, bottleneck_dim) pair listed in configs/pth/default.yaml:
#   dino_model in {S, SP, B, L, H, F}, bottleneck_dim in {256, 1024, 2048}.

# Conv autoencoder sweep (one run per bottleneck dim).
python train_autoencoder.py -m shape.bottleneck_dim=16,32,64,128,256
```

Each pretraining run writes a Hydra output directory containing `best.pt`.
Copy or symlink each `best.pt` into the layout the GNN configs expect:

```
mkdir -p ./checkpoints
ln -s <hydra-out-dir>/best.pt ./checkpoints/SP_256.pt    # for the DINOv3 SP/256 run
ln -s <hydra-out-dir>/best.pt ./checkpoints/Conv_256.pt  # for the conv autoencoder (+embedding=conv_256)
# ... one entry per (model, bottleneck) combination you want to load.
```

`configs/pth/default.yaml` resolves these to
`${oc.env:CHECKPOINTS_DIR,./checkpoints}/<KEY>.pt`, so set `CHECKPOINTS_DIR`
to whichever directory you put the checkpoints in (default: `./checkpoints`).
Fourier descriptors (`+embedding=fourier_*`) are parametric and need no
checkpoint.

### Cell-DINO (optional; Appendix "GraphDyNO functions well with CellDINO")

Cell-DINO's code and weights are **CC-BY-NC — research use only, non-commercial**,
so neither is vendored here and any result produced with them carries that
restriction.

1. Accept the licence at
   <https://ai.meta.com/resources/models-and-libraries/cell-dino-downloads/>;
   the download URLs arrive by e-mail. Fetch the channel-adaptive ViT-L/16
   backbone with `wget` into `${CHECKPOINTS_DIR:-./checkpoints}/`
   (`channel_adaptive_dino_vitl16_pretrain_cells-ef7c17ff.pth`).
2. Clone the model code out of tree, at the commit we used, and point
   `DINOV2_REPO` at it:
   ```bash
   git clone https://github.com/facebookresearch/dinov2 /path/to/dinov2
   git -C /path/to/dinov2 checkout 7764ea0f
   export DINOV2_REPO=/path/to/dinov2
   ```
3. Train the projection-head decoder on LiveCell and link it as
   `CellDino_CA_256.pt`:
   ```bash
   python -m dino.train_dino dino/model=cell
   ln -s <hydra-out-dir>/best.pt ./checkpoints/CellDino_CA_256.pt
   ```

`+embedding=cell_dino` then selects it in place of DINOv3.

### Cell Tracking Challenge data (live-cell experiment)

The live-cell experiment (`real_dic_psc_late`) pools two datasets from the
[Cell Tracking Challenge](https://celltrackingchallenge.net/2d-datasets/):
**DIC-C2DH-HeLa** and **PhC-C2DL-PSC**. The CTC data's terms do not allow
redistribution, so we do not provide those datasets here . Download each dataset's *training* zip 
and unzip it into `real_cell_data/` using the following bash command:

```bash
cd real_cell_data
for d in DIC-C2DH-HeLa PhC-C2DL-PSC; do                  # ~42 MB and ~145 MB
    wget https://data.celltrackingchallenge.net/training-datasets/$d.zip
    unzip -q $d.zip && rm $d.zip
done
cd ..
```

If you keep the data elsewhere, pass `data.params.root=/path/to/ctc` to
`train.py`; relative paths resolve against the repo root.
`real_cell_data/CTC_DATA_FORMAT.md` describes the full layout.

### Simulated datasets and trained checkpoints (Git LFS)

The simulated trajectory datasets (`data/<dataset>/trajectories_<hash>.tar.zst`,
~6 GB in total) and the trained checkpoints behind the figures
(`model_checkpoints.tar.zst`, ~3.4 GB) are stored with
[Git LFS](https://git-lfs.com) as zstd-compressed tarballs. Install `git-lfs`
and `zstd` first (e.g. `conda install -c conda-forge git-lfs zstd`,
`brew install git-lfs zstd`, or `apt install git-lfs zstd`).

Unpacked, they are large: each dataset folder expands to roughly 50–250 GB,
and the checkpoints to ~20 GB. To download only what you need, clone without
the LFS files and pull selectively:

```bash
git lfs install                                        # once per machine
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/sgrolab/GraphDyNO.git
cd GraphDyNO
git lfs pull --include="model_checkpoints.tar.zst"     # checkpoints only
git lfs pull --include="data/growing_cell_cache/*"     # a single dataset
git lfs pull                                           # or everything (~9.4 GB)
```

A plain `git clone` with git-lfs installed downloads every archive. If you
cloned before installing git-lfs, the `.tar.zst` files are small pointer
stubs; run `git lfs install && git lfs pull` to fetch them.

Unpack the checkpoints at the repo root, which creates `model_checkpoints/`:

```bash
zstd -dc model_checkpoints.tar.zst | tar -xf -
```

Each dataset archive unpacks into the folder that contains it:

```bash
d=data/growing_cell_cache
zstd -dc $d/trajectories_*.tar.zst | tar -xf - -C $d

# or every dataset at once
for a in data/*/trajectories_*.tar.zst; do
    zstd -dc "$a" | tar -xf - -C "$(dirname "$a")"
done
```

The `cache_dir` in each `configs/experiment/*.yaml` names the folder that
experiment reads. To keep the unpacked data on another disk, unpack into
`/path/to/data/<dataset>/` (same folder names) and
`export DATA_DIR=/path/to/data`.

## Reproducing the paper

### Train all models

`run_training.sh` wraps every training command used in the paper, grouped by
paper section. Pass one or more targets to reproduce just that work, or `all`
to run everything end-to-end:

```bash
./run_training.sh --help                         # list all targets
./run_training.sh encoders                       # pretrain encoders only (do this first)
./run_training.sh fig4                           # all training needed for Figure 4
./run_training.sh membrane-stiffness adhesion    # multiple sections in one invocation
./run_training.sh all                            # every section, in order
```

The commands are Hydra multiruns (`-m`), which run their sweeps one after
another on the local machine; drop `-m` inside the script if you only want a
single run. Encoder pretraining is a prerequisite for every other
section — run `./run_training.sh encoders` (or follow the manual steps under
"Pretraining the encoders" above) first if `${CHECKPOINTS_DIR:-./checkpoints}`
is empty.

| Paper section                           | `run_training.sh` target     | Experiment name               |
|-----------------------------------------|------------------------------|-------------------------------|
| Encoder pretraining (App. A.1)          | `encoders`                   | encoder checkpoints           |
| Linear growth (Fig 2)                   | `linear-cluster`             | `fig2_best`                   |
| Linear growth recovery (Fig 3a–b)       | `linear-continuous`          | `growing_cells_dino`          |
| Oscillation sweep (Figs 4–5)            | `oscillation-clusters`       | `oscillation_paper`           |
| Membrane stiffness sweep (Figs 3c, 4–5) | `membrane-stiffness`         | `membrane_stiffness_clusters` |
| Adhesion sweep (Figs 4–5)               | `adhesion`                   | `diff_adhesion_ablations`     |
| Scale-only supplement                   | `scale-only`                 | `scale`                       |
| Combined parameters (Table 1)           | `combined-params`            | `osc_amp_x_adhesion_seeds20`, `osc_amp_x_adhesion_x_membrane_stiffness_kh1e5_seeds20` |
| Live-cell CTC (Table 1)                 | `ctc`                        | `ctc_dic_psc_late`            |
| Oscillatory recovery (Fig 3d–f)         | `grok-h64`                   | `osc_grok_param_count`        |
| Architecture sweep (App.)               | `arch-sweep`                 | `osc_arch_sweep`              |
| Cell-DINO encoder (App.)                | `cell-dino`                  | `oscillation_cell_dino`, `membrane_stiffness_cell_dino`, `diff_adhesion_cell_dino` |
| Segmentation ablation (App.)            | `seg-ablation`               | `segabl_oscillation`          |
| Stochasticity ladder (App.)             | `stochastic`                 | `diff_adhesion_fp_ladder`     |
| Oscillation timescales (App.)           | `timescales`                 | `osc_period_clusters`         |

The figure aliases bundle every section that contributes to a given paper
figure: e.g. `fig3` runs `linear-continuous`, `membrane-stiffness`, and `grok-h64`. Figures 4 and 5 share trained models, so running either alias
trains the same set of models once.

The total GPU budget is ~1200 GNN runs and ~210 GPU-hours of training. A
150-epoch run takes ~9–12 min. The first seed of each configuration takes
20–30 min, because it also fine-tunes the shape decoder and builds the
cached dataset. Precomputing the frozen-encoder crop embeddings adds
~20 GPU-hours. Encoder pretraining adds ~2 GPU-hours each for the DINOv3
projection head and the conv autoencoder, and ~4 GPU-hours for the Cell-DINO
head. Because the multiruns run one job at a time, a single GPU needs
roughly 10 days for `all`.

| `run_training.sh` target | Runs | Median min/run | GPU-hours |
|--------------------------|-----:|---------------:|----------:|
| `linear-cluster`         |    1 |             21 |       0.4 |
| `linear-continuous`      |    1 |             19 |       0.3 |
| `oscillation-clusters`   |  150 |              9 |        23 |
| `membrane-stiffness`     |  150 |             10 |        27 |
| `adhesion`               |  150 |             10 |        27 |
| `scale-only`             |   60 |              9 |         9 |
| `combined-params`        |   40 |             19 |        12 |
| `ctc` (800 epochs)       |   20 |             16 |       5.5 |
| `grok-h64` (200 epochs)  |    5 |             13 |       1.2 |
| `arch-sweep`             |  240 |              9 |        35 |
| `cell-dino`              |   60 |             11 |        11 |
| `seg-ablation`           |  250 |             10 |        42 |
| `stochastic`             |   60 |             12 |        13 |
| `timescales`             |   20 |              9 |         3 |
| **Total**                | **1207** |            |  **~210** |

These are wall times from a full rerun on RTX PRO 6000 Blackwell and H100
GPUs; the `combined-params` times come from the original runs.

### Regenerate figures

All figure scripts read from MLflow. Set `MLFLOW_URI` if your tracking
directory is not the repo's `./mlruns/`:

```bash
export MLFLOW_URI=/path/to/mlruns           # or file:///path/...
```

Then run:

```bash
python paper_figures/figure2.py             # Figure 1b — autoregressive rollout
python paper_figures/figure3.py             # Figure 3 — embedding panels
python paper_figures/figure3_supp.py        # Supp — alpha/omega projections
python paper_figures/figure4.py             # Figure 4 — encoder×dataset grid + ARI bars
python paper_figures/figure5.py             # Figure 5 — DINO ablation NC
python paper_figures/all_embeddings_supp.py # Supp — encoder×ablation grids
python paper_figures/scale_only_supp.py     # Supp — scale-only ablation
python paper_figures/metrics_table.py       # Supp — extended metrics table
python paper_figures/results_table.py       # Supp — NC tables
python paper_figures/loss_table.py          # Supp — val-loss table
python paper_figures/grok_recovery_table.py # Supp — epoch 40/200 recovery R² (H = 64)
python paper_figures/polyhoop_snapshots_combined.py  # Supp — trajectory snapshots
python paper_figures/complex_cases_table.py    # Table 1 — all three columns (+ .tex)
python paper_figures/arch_sweep_table.py       # Supp — architecture sweep table
python paper_figures/cell_dino_table.py        # Supp — Cell-DINO vs DINOv3 table
python paper_figures/seg_ablation_tables.py    # Supp — segmentation-error tables
python paper_figures/stochastic_table.py       # Supp — f_p noise ladder table
```

The mapping from figure script → MLflow experiment name lives in
`paper_figures/experiments.yaml`. Each script also accepts CLI overrides
(e.g. `--experiment <name>`, `--oscillation-experiment <name>`); see
`<script>.py --help`.

The single-run figures (`figure2.py`, `figure3.py`) pick the best run
from each experiment by `val_loss`. The oscillatory panels (figure3 d–f,
`figure3_supp.py`) instead draw the epoch-40 and epoch-200 embedding
snapshots of one H = 64 run, seed 40 by default (`grok_h64.figure_seed` in
`experiments.yaml`; `--grok-run` picks another). To force a specific seed elsewhere, pass `--run-name
<mlflow-run-name>` to `figure2.py` or `--growth-run` / `--kl-run` to
`figure3.py`.

To regenerate from the exported `model_checkpoints/` tree instead of MLflow,
pass `--ckpt-dir` (a bare flag means `./model_checkpoints`) or set
`GRAPHDYNO_CKPT_DIR`. Seg-ablation doses are the
requested rates rather than the realized ones, and the figure3 kl panel and
figure4's DINO membrane-stiffness panel use the seeds pinned in `experiments.yaml`
(`figure3.kl_ckpt_seed`, `figure4.ckpt_median_seeds`), which match the MLflow
picks. 

## Hardware

Each job ran on a single NVIDIA RTX PRO 6000 Blackwell (96 GB) or H100
(80 GB), with no observable runtime difference. The pipeline
runs end-to-end on a single GPU; multi-GPU is not required.

## License

This code is released under the Janelia Open‑Source License (see `LICENSE`).
