"""Write one bank-derived level's sidecars to disk — an ON-DEMAND tool.

    python scripts/materialize_seg_level.py +experiment=<name> \
        +embedding=<emb> +seg_ablation=joint_d2 +traj_idx=0

**This is not part of the scan.** `train.py` materializes its own level in
process (`seg_ablation/bank_apply.py`), so a 29-level x 10-seed grid needs no
materialize step at all. Run this to inspect a level's geometry sidecar
offline without starting a training job.

It is GPU-free like the read path it wraps: every corrupted row already lives
in the bank.
"""

import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from graph_dyno.seg_ablation.bank import bank_id, load_bank_spec
from graph_dyno.seg_ablation.bank_apply import materialize_level
from graph_dyno.seg_ablation.build import sidecar_paths
from graph_dyno.seg_ablation.plan import abl_id, normalize_spec


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from graph_dyno.encoders.factory import get_companion_suffix
    from graph_dyno.models.model_factory import resolve_shape_config
    from graph_dyno.data.trajectory_generator import generate_config_hash

    resolve_shape_config(cfg)
    if cfg.get("seg_ablation", None) is None:
        raise SystemExit("pass +seg_ablation=<level> (see configs/seg_ablation/)")
    spec = normalize_spec(cfg.seg_ablation)
    if not spec["bank"]:
        raise SystemExit(
            "this level is not bank-derived; build it with "
            "scripts/build_seg_ablation.py")
    abl = abl_id(spec)
    bspec = load_bank_spec(spec["bank"])
    cache_id = generate_config_hash(cfg.data)
    cache_dir = cfg.data.generator.cache_dir
    companion_suffix = get_companion_suffix(cfg)
    n_traj = int(cfg.data.generator.n_trajectories)

    print(f"cache_dir       {cache_dir}")
    print(f"cache_id        {cache_id}")
    print(f"abl_id          {abl}")
    print(f"bank            {bspec['name']} ({bank_id(bspec)})")
    print(f"companion       {companion_suffix}")
    print(OmegaConf.to_yaml(OmegaConf.create(spec)))

    if cfg.get("list_parts", False):
        for i in range(n_traj):
            g, e = sidecar_paths(cache_dir, abl, companion_suffix, cache_id, i)
            print(f"  traj_idx={i:3d}  geo {'EXISTS' if os.path.exists(g) else 'missing'}"
                  f"  emb {'EXISTS' if os.path.exists(e) else 'missing'}")
        return

    traj_idx = cfg.get("traj_idx", None)
    idxs = range(n_traj) if traj_idx is None else [int(traj_idx)]
    for i in idxs:
        geo, emb, stats = materialize_level(
            cache_dir, spec, cache_id=cache_id, traj_idx=i,
            companion_suffix=companion_suffix, dt=cfg.data.dt,
            save_geo=True, save_emb=bool(cfg.get("save_emb", False)),
            force=bool(cfg.get("force", False)))
        g, e = sidecar_paths(cache_dir, abl, companion_suffix, cache_id, i)
        print(f"\ntrajectory {i}: {stats['n_events']} episodes, "
              f"{stats['n_reencoded']} banked rows, "
              f"{stats['n_invalid_cell_frames']} invalid cell-frames "
              f"({stats['invalid_frac'] * 100:.2f}%)")
        for kind, d in stats["per_kind"].items():
            if not d.get("n_episodes"):
                continue
            print(f"  {kind:9s} realized rate {d['realized_rate']:.4f}  "
                  f"episodes {d['n_episodes']}  "
                  f"collision drops {d.get('n_collision_drops', 0)}")
        print(f"  edges dropped {(1 - stats['edges_kept_frac']) * 100:.2f}%   "
              f"centroid sigma {stats['centroid_sigma_px']:.3f} px   "
              f"velocity SNR {stats['vel_snr']:.2f}")
        w = stats.get("worst_label_correlation")
        if w is not None:
            print(f"  worst per-slot/label correlation: "
                  f"r({w[0]}, {w[1]}) = {w[2]:+.3f}  "
                  f"(limit {spec['max_label_correlation']})")
        print(f"  {g}  ({os.path.getsize(g) / 1e6:.0f} MB)")
        if os.path.exists(e):
            print(f"  {e}  ({os.path.getsize(e) / 1e6:.0f} MB)")
        else:
            # Redundant with the bank, and ~180 MB/trajectory. The movie reads
            # only the geometry half; pass `+save_emb=true` if you want the
            # pair on disk anyway.
            print("  (no embedding sidecar written — pass +save_emb=true)")


if __name__ == "__main__":
    sys.argv.extend([
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
    ])
    main()
