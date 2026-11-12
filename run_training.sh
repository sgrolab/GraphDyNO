#!/usr/bin/env bash
# Conventions
# -----------
# - Run from the repo root.
# - Caches live under `${DATA_DIR:-./data}/<subdir>/`. Set `DATA_DIR` to point
#   at an existing simulation cache; otherwise data is generated on first run.
# - MLflow tracking writes to `./mlruns/` by default. Override with `MLFLOW_URI`
#   or `+mlflow_uri=...` on the command line if you keep runs elsewhere.
# - Pretrained shape-encoder weights are loaded from
#   `${CHECKPOINTS_DIR:-./checkpoints}/<name>.pt` (see configs/pth/default.yaml).
#
# Seed budget
# -----------
# Per encoder×ablation sweep (the exact `model.seed=range(...)` values are in
# the commands below):
#   oscillation_paper           dino_sp: 20 seeds, full ablation set (full+spatial+static+temporal)
#                               fourier_32 / conv_256: 5 seeds, encoder-comparison ablations only
#                                                     (spatial+static+temporal)
#   membrane_stiffness_clusters dino_sp: 20 seeds, full ablation set
#                               fourier_32 / conv_256: 20 seeds, `full` ablation only
#   diff_adhesion_ablations     dino_sp: 20 seeds, full ablation set
#                               fourier_32 / conv_256: 5 seeds, full ablation set
#   combined parameters         20 seeds per dataset
#   CTC DIC-HeLa x PSC          20 seeds
#   grok (H = 64)               5 seeds
#   architecture sweep          20 seeds per arm, 12 arms
#   Cell-DINO                   20 seeds per dataset
#   segmentation ablation       10 seeds per level, 25 levels
#   f_p ladder                  20 seeds per f_p
#   oscillation timescales      20 seeds

set -euo pipefail

usage() {
    cat <<'EOF'
Commands to reproduce every paper figure.

Usage: ./run_training.sh [--dry-run] [target ...]

Options:
  -n, --dry-run           Print every training command without executing it.

Sections (pass one or more):
  encoders                Encoder pretraining (LiveCell + DINO + Conv autoencoder)
  linear-cluster          Linear growth — cluster (Figure 1b / 2)
  linear-continuous       Linear growth — continuous (Figure 3 growth panel)
  oscillation-clusters    Oscillation — clusters (Figure 4 row 1, Figure 5 left)
  membrane-stiffness      Membrane stiffness (Figure 3 kl, Figure 4 row 2, Figure 5 middle)
  adhesion                Differential adhesion (Figure 4 row 3, Figure 5 right)
  scale-only              Scale-only ablation (Supplementary: scale_only_supp.py)
  combined-params         Combined parameters (Table: complex cases, 2- and 3-parameter columns)
  ctc                     Live-cell CTC DIC-HeLa x PSC (Table: complex cases, CTC column)
  grok-h64                Oscillation, grokking at H = 64 (Figure 3 d-f, Supp: figure3_supp, Table: grok recovery)
  arch-sweep              Architecture robustness sweep (Table: arch sweep)
  cell-dino               Cell-DINO encoder (Table: encoder)
  seg-ablation            Segmentation/tracking ablation (Tables: seg single, seg joint)
  stochastic              f_p noise ladder (Table: stochastic)
  timescales              Oscillation periods beyond the horizon (Table: timescales)

Figure aliases (run every section needed for the figure):
  fig2                    Figure 1b / 2     -> linear-cluster
  fig3                    Figure 3          -> linear-continuous, membrane-stiffness, grok-h64
  fig4                    Figure 4          -> oscillation-clusters, membrane-stiffness, adhesion
  fig5                    Figure 5          -> oscillation-clusters, membrane-stiffness, adhesion
                          (Fig 4 and 5 share trained models — run once for both.)
  scale-only-supp         Supp scale-only   -> scale-only

  all                     Run every section above, in order

Encoder pretraining is a prerequisite for every other section; run `encoders`
first if you do not already have checkpoints under ${CHECKPOINTS_DIR:-./checkpoints}.
EOF
}

##############################################################################
# Encoder pretraining (Appendix: app:encoders)
##############################################################################
# Trained on the LiveCell dataset; produces the shape-decoder checkpoints
# referenced by configs/pth/default.yaml. Fourier descriptors are parametric
# and require no training.
#
# Set LIVECELL_DIR to a local directory; the extractor will download LiveCell
# (~14 GB) into ${LIVECELL_DIR}/raw/ on first run and produce per-cell SDF
# crops in ${LIVECELL_DIR}/cell_masks/.
#
# Place finished checkpoints into ${CHECKPOINTS_DIR:-./checkpoints}/ with the
# names defined in configs/pth/default.yaml (S_256.pt, SP_256.pt, ...,
# Conv_256.pt, ...). The training scripts write to a Hydra output dir; copy or
# symlink the resulting best.pt into ./checkpoints/<KEY>.pt.
run_encoders() {
    # 1. Download LiveCell and extract per-cell SDF crops (run once).
    python -m dino.cell_mask_extractor

    # 2a. DINOv3 projection-head decoder — train one variant per (model, bottleneck).
    #    The paper uses SP_256; rerun with dino.model.dino_model in {S,SP,B,L,H,F}
    #    and dino.model.bottleneck_dim in {256,1024,2048} for the encoder
    #    comparison. precompute_embeddings is invoked automatically on first run
    #    per (model, split).
    python -m dino.train_dino dino.model.dino_model=SP dino.model.bottleneck_dim=256

    # 2b. Conv autoencoder — sweep the bottleneck dims used in the paper.
    python train_autoencoder.py -m shape.bottleneck_dim=16,32,64,128,256

    # 2c. Cell-DINO projection-head decoder (Appendix: Cell-DINO). Needs the
    #     gated backbone weights and the dinov2 clone (see README); link the
    #     resulting best.pt as ./checkpoints/CellDino_CA_256.pt.
    python -m dino.train_dino dino/model=cell
}

##############################################################################
# Linear growth — cluster (Figure 1b / 2)
##############################################################################
# Single best-seed run used as the autoregressive-rollout demo (figure2.py).
run_linear_cluster() {
    python train.py -m \
        +experiment=more_grow \
        +experiment_name=fig2_best \
        model=history \
        +embedding=dino_sp \
        +ablation=full \
        model.history_window=0 \
        model.hidden_dim=64 \
        model.L_dropout_prob=0 \
        model.scale_dropout_prob=0.5 \
        model.embedding_noise_std=0.1 \
        model.L_skip_connection=true \
        model.interaction_dim=2 \
        model.skip_connection=False \
        training.loss_terms.parameter_embedding_l2=0 \
        training.rollout.enabled=True \
        model.input_noise_std=0.1 \
        +training.plot_every=0 \
        model.seed=10
}

##############################################################################
# Linear growth — continuous (Figure 3, growth panel)
##############################################################################
run_linear_continuous() {
    python train.py \
        +experiment=deformable_cells_no_movement_dino_repro \
        +experiment_name=growing_cells_dino \
        +embedding=dino_sp \
        model.L_dropout_prob=0 \
        training.n_epochs=80
}

##############################################################################
# Oscillation — clusters (Figure 4 row 1, Figure 5 left panel)
##############################################################################
run_oscillation_clusters() {
    # dino_sp — 20 seeds across the full ablation set (full, spatial, static, temporal).
    python train.py -m \
        +experiment=more_osc_clusters \
        +experiment_name=oscillation_paper \
        +embedding=dino_sp \
        '+ablation/int=glob(*)' \
        model.seed=range\(0,200,10\)

    # fourier_32 / conv_256 — 20 seeds, full ablation only (full).
    python train.py -m \
        +experiment=more_osc_clusters \
        +experiment_name=oscillation_paper \
        +embedding=fourier_32,conv_256 \
        '+ablation/int=full' \
        model.seed=range\(0,200,10\)

    # fourier_32 / conv_256 — 5 seeds, encoder-comparison ablations only.
    python train.py -m \
        +experiment=more_osc_clusters \
        +experiment_name=oscillation_paper \
        +embedding=fourier_32,conv_256 \
        '+ablation/int=spatial,static,temporal' \
        model.seed=range\(0,50,10\)
}

##############################################################################
# Membrane stiffness (Figure 3 kl panel, Figure 4 row 2, Figure 5 middle panel)
##############################################################################
run_membrane_stiffness() {
    # dino_sp — 20 seeds across the full ablation set.
    python train.py -m \
        model=history \
        +experiment=membrane_stiffness_two_clusters \
        +experiment_name=membrane_stiffness_clusters \
        +embedding=dino_sp \
        '+ablation/int=glob(*)' \
        training.rollout.enabled=false \
        model.embedding_noise_std=0.2 \
        model.seed=range\(0,200,10\)

    # fourier_32 / conv_256 — 20 seeds, full ablation only.
    python train.py -m \
        model=history \
        +experiment=membrane_stiffness_two_clusters \
        +experiment_name=membrane_stiffness_clusters \
        +embedding=fourier_32,conv_256 \
        +ablation/int=full \
        training.rollout.enabled=false \
        model.embedding_noise_std=0.2 \
        model.seed=range\(0,200,10\)

    # fourier_32 / conv_256 — 5 seeds, encoder-comparison ablations only.
    python train.py -m \
        model=history \
        +experiment=membrane_stiffness_two_clusters \
        +experiment_name=membrane_stiffness_clusters \
        +embedding=fourier_32,conv_256 \
        +ablation/int=static,spatial,temporal \
        training.rollout.enabled=false \
        model.embedding_noise_std=0.2 \
        model.seed=range\(0,50,10\)
}

##############################################################################
# Differential adhesion (Figure 4 row 3, Figure 5 right panel)
##############################################################################
run_adhesion() {
    # dino_sp — 20 seeds across the full ablation set.
    python train.py -m \
        model=history \
        +experiment=differential_adhesion_clusters \
        +experiment_name=diff_adhesion_ablations \
        +embedding=dino_sp \
        '+ablation/int=glob(*)' \
        model.seed=range\(0,200,10\)

    # fourier_32 / conv_256 — 20 seeds for the full ablation only
    python train.py -m \
        model=history \
        +experiment=differential_adhesion_clusters \
        +experiment_name=diff_adhesion_ablations \
        +embedding=fourier_32,conv_256 \
        '+ablation/int=full' \
        model.seed=range\(0,200,10\)

    # fourier_32 / conv_256 — 5 seeds for the encoder-comparison ablations only
    python train.py -m \
        model=history \
        +experiment=differential_adhesion_clusters \
        +experiment_name=diff_adhesion_ablations \
        +embedding=fourier_32,conv_256 \
        '+ablation/int=static,spatial,temporal' \
        model.seed=range\(0,50,10\)
}

##############################################################################
# Scale-only ablation (Supplementary figure: scale_only_supp.py)
##############################################################################
# Scale-only ablation across all three sorting/oscillation regimes; consumed by
# paper_figures/scale_only_supp.py.
run_scale_only() {
    python train.py -m \
        model=history \
        +experiment=differential_adhesion_clusters,more_osc_clusters,membrane_stiffness_two_clusters \
        +experiment_name=scale \
        +embedding=dino_sp \
        +ablation=scale_only \
        training.loss_terms.position_mse=0 \
        training.loss_terms.shape_mse=0 \
        training.loss_terms.velocity_mse=0 \
        training.loss_terms.parameter_embedding_l2=0 \
        training.n_epochs=150 \
        model.seed=range\(0,200,10\)
}

##############################################################################
# Combined parameters (Table: complex cases, two- and three-parameter columns)
##############################################################################
# Scored by paper_figures/complex_cases_table.py.
run_combined_params() {
    # 2 oscillation amplitudes x 2 adhesion strengths (k = 4).
    python train.py -m \
        +experiment=osc_amp_x_adhesion \
        +experiment_name=osc_amp_x_adhesion_seeds20 \
        +embedding=dino_sp \
        +ablation=full_no_history \
        model.embedding_dim=16 \
        model.interaction_dim=32 \
        dataloader.gpu_cache_memory=60GB \
        model.seed=range\(0,20\)

    # 2 amplitudes x 2 adhesion strengths x 2 membrane stiffnesses (k = 8).
    python train.py -m \
        +experiment=osc_amp_x_adhesion_x_membrane_stiffness_kh1e5 \
        +experiment_name=osc_amp_x_adhesion_x_membrane_stiffness_kh1e5_seeds20 \
        +embedding=dino_sp \
        +ablation=full_no_history \
        model.embedding_dim=32 \
        model.interaction_dim=32 \
        dataloader.gpu_cache_memory=60GB \
        model.seed=range\(0,20\)
}

##############################################################################
# Live-cell CTC: DIC-C2DH-HeLa x PhC-C2DL-PSC (Table: complex cases, CTC column)
##############################################################################
# Needs the two CTC training datasets under real_cell_data/ (see README). The
# recipe (embedding_dim 16, 800 epochs, batch size 2) lives in the experiment
# config. Scored by paper_figures/complex_cases_table.py.
run_ctc() {
    python train.py -m \
        +experiment=real_dic_psc_late \
        +experiment_name=ctc_dic_psc_late \
        +embedding=dino_sp \
        +ablation=b_sweep/b_shape_scale_int \
        model.seed=range\(0,200,10\)
}

##############################################################################
# Oscillation, grokking at hidden width H = 64 (Figure 3 d-f, figure3_supp,
# Table grok_recovery)
##############################################################################
# Saves the per-cell embedding every 2 epochs; paper_figures/figure3.py,
# figure3_supp.py and grok_recovery_table.py read the epoch-40 and epoch-200
# snapshots.
# plot_every=0 is part of the recipe: the diagnostic plots draw from the
# global RNG.
run_grok_h64() {
    python train.py -m \
        +experiment=oscillating_cell_vary_amp_dino_repro \
        +experiment_name=osc_grok_param_count \
        +embedding=dino_sp \
        +ablation=full_velocity \
        model.L_dropout_prob=0.3 \
        model.scale_dropout_prob=0.3 \
        model.embedding_dim=3 \
        model.hidden_dim=64 \
        training.n_epochs=200 \
        +training.embedding_snapshot_every=2 \
        +training.plot_every=0 \
        model.seed=0,10,20,30,40
}

##############################################################################
# Architecture robustness sweep (Table: arch sweep)
##############################################################################
# One width moved at a time around the oscillation-clusters configuration,
# whose dino_sp/full runs are the baseline row. Scored by
# paper_figures/arch_sweep_table.py, which groups arms by the arch_axis tag.
run_arch_sweep() {
    local axis_levels
    for axis_levels in hidden_dim=64,128,256,1024 \
                       interaction_dim=4,8,16,32 \
                       embedding_dim=3,4,8,16; do
        python train.py -m \
            +experiment=more_osc_clusters \
            +experiment_name=osc_arch_sweep \
            +embedding=dino_sp \
            +ablation/int=full \
            +tags.arch_axis="${axis_levels%%=*}" \
            model."${axis_levels}" \
            model.seed=range\(0,200,10\)
    done
}

##############################################################################
# Cell-DINO encoder (Table: encoder)
##############################################################################
# Needs ./checkpoints/CellDino_CA_256.pt from `encoders` step 2c. Same GNN
# inputs as the DINOv3 arm (+ablation=full_int matches +ablation/int=full).
run_cell_dino() {
    python train.py -m \
        +experiment=more_osc_clusters \
        +experiment_name=oscillation_cell_dino \
        +embedding=cell_dino \
        +ablation=full_int \
        dataloader.gpu_cache_memory=60GB \
        model.seed=range\(0,200,10\)

    python train.py -m \
        model=history \
        +experiment=membrane_stiffness_two_clusters \
        +experiment_name=membrane_stiffness_cell_dino \
        +embedding=cell_dino \
        +ablation/int=full \
        training.rollout.enabled=false \
        model.embedding_noise_std=0.2 \
        dataloader.gpu_cache_memory=60GB \
        model.seed=range\(0,200,10\)

    python train.py -m \
        model=history \
        +experiment=differential_adhesion_clusters \
        +experiment_name=diff_adhesion_cell_dino \
        +embedding=cell_dino \
        +ablation/int=full \
        dataloader.gpu_cache_memory=60GB \
        model.seed=range\(0,200,10\)
}

##############################################################################
# Segmentation / tracking ablation (Tables: seg single, seg joint)
##############################################################################
# Needs the more_osc_clusters cache (run `oscillation-clusters` first). Step 1
# encodes every corrupted crop once into an embedding bank; step 2 trains each
# level, which train.py materializes from the bank in process. Scored by
# paper_figures/seg_ablation_tables.py.
run_seg_ablation() {
    local bank="+experiment=more_osc_clusters +embedding=dino_sp +seg_bank=ladder_v1"
    local t k
    # 1. Plan (CPU), encode (GPU) and merge the bank.
    for t in 0 1 2; do
        python scripts/plan_seg_bank.py $bank +traj_idx=$t
    done
    for t in 0 1 2; do
        for k in partial scale merge; do
            python scripts/build_seg_bank_worker.py $bank \
                +traj_idx=$t +kind=$k +n_chunks=1 +chunk_idx=0
        done
    done
    python scripts/merge_seg_bank.py $bank

    # 2a. Matched control + the per-kind dose ladders.
    python train.py -m \
        +experiment=more_osc_clusters \
        +experiment_name=segabl_oscillation \
        +embedding=dino_sp \
        +ablation/int=full \
        +seg_ablation=bank_none,partial_d1,partial_d2,partial_d3,partial_d4,scale_d1,scale_d2,scale_d3,scale_d4,miss_d1,miss_d2,miss_d3,miss_d4,merge_d1,merge_d2,merge_d3,merge_d4,id_swap_d1,id_swap_d2,id_swap_d3,id_swap_d4 \
        model.seed=range\(0,100,10\)

    # 2b. Balanced joint corruption.
    python train.py -m \
        +experiment=more_osc_clusters \
        +experiment_name=segabl_oscillation \
        +embedding=dino_sp \
        +ablation/int=full \
        +seg_ablation=joint_1pct,joint_2pct,joint_5pct,joint_10pct \
        model.seed=range\(10,110,10\)
}

##############################################################################
# f_p noise ladder (Table: stochastic)
##############################################################################
# The differential-adhesion experiment at f_p = 0, 25, 100; the f_p = 50 row
# is the `adhesion` section's dino_sp/full runs. Scored by
# paper_figures/stochastic_table.py.
run_stochastic() {
    python train.py -m \
        model=history \
        +experiment=diff_adhesion_fp0,diff_adhesion_fp25,diff_adhesion_fp100 \
        +experiment_name=diff_adhesion_fp_ladder \
        +embedding=dino_sp \
        +ablation/int=full \
        model.seed=range\(0,200,10\)
}

##############################################################################
# Oscillation periods beyond the forecasting horizon (Table: timescales)
##############################################################################
# The amplitude row is the `oscillation-clusters` section. Scored by
# paper_figures/figure_osc_period.py.
run_timescales() {
    python train.py -m \
        +experiment=osc_period_clusters \
        +experiment_name=osc_period_clusters \
        +embedding=dino_sp \
        +ablation=int/full \
        model.seed=range\(0,200,10\)
}

# Figure aliases — bundle every section needed for a given paper figure.
run_fig2() {
    run_linear_cluster
}

run_fig3() {
    run_linear_continuous
    run_membrane_stiffness
    run_grok_h64
}

run_fig4() {
    run_oscillation_clusters
    run_membrane_stiffness
    run_adhesion
}

# Figure 5 reuses the same trained models as Figure 4.
run_fig5() {
    run_fig4
}

run_scale_only_supp() {
    run_scale_only
}

run_all() {
    run_encoders
    run_linear_cluster
    run_linear_continuous
    run_oscillation_clusters
    run_membrane_stiffness
    run_adhesion
    run_scale_only
    run_combined_params
    run_ctc
    run_grok_h64
    run_arch_sweep
    run_cell_dino
    run_seg_ablation
    run_stochastic
    run_timescales
}

DRY_RUN=0
TARGETS=()
for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY_RUN=1 ;;
        -h|--help)    usage; exit 0 ;;
        *)            TARGETS+=("$arg") ;;
    esac
done

# Override `python` to echo when --dry-run is set; otherwise defer to the real
# interpreter. %q emits shell-safe quoting so the printed line can be copied
# and run verbatim.
python() {
    if [[ $DRY_RUN -eq 1 ]]; then
        printf 'python'
        printf ' %q' "$@"
        printf '\n\n'
    else
        command python "$@"
    fi
}

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    usage
    exit 1
fi

for target in "${TARGETS[@]}"; do
    case "$target" in
        encoders)               run_encoders ;;
        linear-cluster)         run_linear_cluster ;;
        linear-continuous)      run_linear_continuous ;;
        oscillation-clusters)   run_oscillation_clusters ;;
        membrane-stiffness)     run_membrane_stiffness ;;
        adhesion)               run_adhesion ;;
        scale-only)             run_scale_only ;;
        combined-params)        run_combined_params ;;
        ctc)                    run_ctc ;;
        grok-h64)               run_grok_h64 ;;
        arch-sweep)             run_arch_sweep ;;
        cell-dino)              run_cell_dino ;;
        seg-ablation)           run_seg_ablation ;;
        stochastic)             run_stochastic ;;
        timescales)             run_timescales ;;
        fig2)                   run_fig2 ;;
        fig3)                   run_fig3 ;;
        fig4)                   run_fig4 ;;
        fig5)                   run_fig5 ;;
        scale-only-supp)        run_scale_only_supp ;;
        all)                    run_all ;;
        *)
            echo "Unknown target: $target" >&2
            usage
            exit 1
            ;;
    esac
done
