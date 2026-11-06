"""Encode ONE time-chunk of ONE kind's seg-ablation bank — the GPU unit of work.

    # dry run: assert the job count, no GPU touched
    python scripts/build_seg_bank_worker.py +experiment=<name> \
        +embedding=<emb> +seg_bank=ladder_v1 +n_chunks=1 +list_parts=true

    for t in 0 1 2; do for k in partial scale merge; do
        python scripts/build_seg_bank_worker.py +experiment=<name> \
            +embedding=<emb> +seg_bank=ladder_v1 \
            +traj_idx=$t +kind=$k +n_chunks=1 +chunk_idx=0
    done; done

Work is `traj_idx x kind x chunk_idx`; `+n_chunks=K` splits each kind into K
independent jobs (`+chunk_idx=0..K-1`) for running them in parallel. Rows are per (episode, frame), so a
chunk boundary inside an episode costs nothing — the rows on either side are
independent and the merge reassembles them by the stored `chunk_idx`, never by
filename.

Only `partial`, `merge` and `scale` appear here: `miss` and `id_swap` are index
edits and `centroid_noise` is a scalar, so they are planned into the bank but
need no re-encode.

`shape.sdf_workers` sets the SDF pool. Under a scheduler that enforces CPU
limits, match it to the job's allocation — a wider pool only time-slices.
"""

import os
import sys

import hydra
import torch
from omegaconf import DictConfig

from graph_dyno.seg_ablation.bank import (
    BANK_KINDS,
    bank_id,
    load_bank_spec,
    normalize_bank_spec,
    rows_filename,
)
from graph_dyno.seg_ablation.bank_build import encode_chunk


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from graph_dyno.encoders.factory import get_companion_suffix, create_shape_encoder
    from graph_dyno.models.model_factory import resolve_shape_config
    from graph_dyno.data.trajectory_generator import generate_config_hash

    resolve_shape_config(cfg)
    if cfg.get("seg_bank", None) is None:
        raise SystemExit("pass +seg_bank=<name> (see configs/seg_bank/)")
    bspec = normalize_bank_spec(cfg.seg_bank)
    if bank_id(bspec) != bank_id(load_bank_spec(bspec["name"])):
        raise SystemExit("the composed seg_bank differs from its config file; "
                         "see scripts/plan_seg_bank.py::_resolve_bank")
    bid = bank_id(bspec)
    cache_id = generate_config_hash(cfg.data)
    cache_dir = cfg.data.generator.cache_dir
    companion_suffix = get_companion_suffix(cfg)
    if companion_suffix.endswith("None"):
        raise SystemExit(
            f"companion suffix is {companion_suffix!r} — the embedding overlay "
            f"does not pin shape.dino_model. The precompute path is weight-free "
            f"by design and never auto-resolves it.")
    n_traj = int(cfg.data.generator.n_trajectories)
    n_chunks = int(cfg.get("n_chunks", 1))

    if cfg.get("list_parts", False):
        print(f"{n_traj} traj x {len(BANK_KINDS)} kinds x {n_chunks} chunks = "
              f"{n_traj * len(BANK_KINDS) * n_chunks} parts "
              f"(bank {bspec['name']} / {bid}, suffix {companion_suffix})")
        missing = 0
        for i in range(n_traj):
            for kind in BANK_KINDS:
                for c in range(n_chunks):
                    p = os.path.join(cache_dir, rows_filename(
                        bid, kind, companion_suffix, cache_id, i, c))
                    done = os.path.join(cache_dir, rows_filename(
                        bid, kind, companion_suffix, cache_id, i))
                    ok = os.path.exists(p) or os.path.exists(done)
                    missing += not ok
                    print(f"  traj={i} kind={kind:8s} chunk={c:02d}  "
                          f"{'EXISTS' if ok else 'missing'}")
        print(f"{missing} parts still to encode.")
        return

    traj_idx = cfg.get("traj_idx", None)
    kind = cfg.get("kind", None)
    chunk_idx = cfg.get("chunk_idx", None)
    if traj_idx is None or kind is None or chunk_idx is None:
        raise ValueError(
            "Specify the unit of work: +traj_idx=<int> +kind=<partial|scale|"
            "merge> +chunk_idx=<int> (+n_chunks=<int>). See +list_parts=true.")
    traj_idx, chunk_idx = int(traj_idx), int(chunk_idx)
    kind = str(kind)
    if kind not in BANK_KINDS:
        raise ValueError(f"+kind={kind} needs no re-encode; bankable kinds are "
                         f"{list(BANK_KINDS)}")

    if os.path.exists(os.path.join(cache_dir, rows_filename(
            bid, kind, companion_suffix, cache_id, traj_idx))):
        print("already merged, nothing to do")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name} ({round(p.total_memory / 1024 ** 3)} GB)")

    encode_chunk(
        cache_dir, bspec, kind, cache_id=cache_id, traj_idx=traj_idx,
        companion_suffix=companion_suffix, encoder=create_shape_encoder(
            cfg, device=device),
        n_chunks=n_chunks, chunk_idx=chunk_idx,
        workers=int(cfg.shape.get("sdf_workers", 32)),
        batch_size=int(cfg.get("batch_size", 64)),
        force=bool(cfg.get("force", False)))


if __name__ == "__main__":
    sys.argv.extend([
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
    ])
    main()
