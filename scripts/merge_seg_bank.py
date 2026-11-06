"""Concatenate the seg-ablation bank chunks, then report the decorrelation.

    python scripts/merge_seg_bank.py +experiment=<name> +embedding=<emb> \
        +seg_bank=ladder_v1

One job: the merge is a concatenation, and the whole point of the bank is that
it happens once. Unlike `precompute_embeddings.merge_chunks` — which
concatenates whatever it globs, deletes it, and leaves an unrecoverable short
file when a part is missing — `bank.merge_chunks` asserts the stored chunk
indices are exactly `0..n_chunks-1` **before** deleting anything, and orders by
that field rather than by filename (so `chunk10 < chunk2` cannot bite).

Also writes `segbankdecorr_{bank_id}_{cache_id}.json`: how differently the same
cell is corrupted in each trajectory. The requirement it checks is that a given
cell id must NOT receive the same corruption everywhere, while its episode
COUNT stays equal (the latter is `strict_balance`, and is the label-confound
control). Read, not gated on.
"""

import os
import sys

import json

import hydra
import torch
from omegaconf import DictConfig

from graph_dyno.seg_ablation.bank import (
    BANK_KINDS,
    PLAN_KINDS,
    bank_id,
    decorr_filename,
    decorrelation,
    load_bank_spec,
    load_plan,
    merge_chunks,
    normalize_bank_spec,
    rows_filename,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from graph_dyno.encoders.factory import get_companion_suffix
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
    n_traj = int(cfg.data.generator.n_trajectories)
    n_chunks = cfg.get("n_chunks", None)
    n_chunks = None if n_chunks is None else int(n_chunks)

    print(f"bank {bspec['name']} ({bid})  cache {cache_id}  "
          f"suffix {companion_suffix}\n")

    for traj_idx in range(n_traj):
        for kind in BANK_KINDS:
            merge_chunks(cache_dir, bid, kind, companion_suffix, cache_id,
                         traj_idx, expect_n_chunks=n_chunks)

    print("\nverifying merged banks:")
    for traj_idx in range(n_traj):
        for kind in BANK_KINDS:
            _verify(cache_dir, bid, kind, companion_suffix, cache_id, traj_idx)

    _write_decorrelation(cache_dir, bid, cache_id, n_traj)


def _verify(cache_dir, bid, kind, companion_suffix, cache_id, traj_idx):
    """Every planned episode must be represented, with the right row count."""
    path = os.path.join(cache_dir, rows_filename(
        bid, kind, companion_suffix, cache_id, traj_idx))
    r = torch.load(path, map_location="cpu", weights_only=False)
    plan = load_plan(cache_dir, bid, kind, cache_id, traj_idx)
    e = r["embeddings"]
    if e.ndim != 2 or e.shape[0] != int(r["K"]):
        raise SystemExit(f"{os.path.basename(path)}: embeddings {tuple(e.shape)}"
                         f" against K={r['K']}")
    if not torch.isfinite(e).all():
        raise SystemExit(f"{os.path.basename(path)}: non-finite embeddings")
    if e.numel() and float(e.std()) < 1e-3:
        raise SystemExit(f"{os.path.basename(path)}: embedding std "
                         f"{float(e.std()):.2e} — the encode collapsed")
    cs = r["crop_sides"]
    if cs.dtype != torch.int32:
        raise SystemExit(f"{os.path.basename(path)}: crop_sides is {cs.dtype}, "
                         f"must be int32")
    if cs.numel() and int(cs.min()) <= 0:
        raise SystemExit(f"{os.path.basename(path)}: crop_side <= 0 on a pixel "
                         f"row — the perturbation emptied a mask")
    ep = r["episode_id"].numpy()
    n_ep = int(plan["rank"].size)
    import numpy as np
    counts = np.bincount(ep, minlength=n_ep)
    want = (np.asarray(plan["t1"], dtype=np.int64)
            - np.asarray(plan["t0"], dtype=np.int64))
    if not np.array_equal(counts, want):
        bad = int((counts != want).sum())
        raise SystemExit(
            f"{os.path.basename(path)}: {bad} of {n_ep} episodes have the "
            f"wrong row count — the chunk set is incomplete.")
    print(f"  traj {traj_idx} {kind:8s} OK: {r['K']} rows x {r['D']}, "
          f"{n_ep} episodes, {os.path.getsize(path) / 1e6:.0f} MB")


def _write_decorrelation(cache_dir, bid, cache_id, n_traj):
    out = {"bank_id": bid, "cache_id": cache_id, "kinds": {}}
    for kind in PLAN_KINDS:
        plans = {}
        for i in range(n_traj):
            try:
                plans[i] = load_plan(cache_dir, bid, kind, cache_id, i)
            except FileNotFoundError:
                continue
        if len(plans) < 2:
            continue
        N = int(next(iter(plans.values()))["N"])
        out["kinds"][kind] = decorrelation(plans, N)

    path = os.path.join(cache_dir, decorr_filename(bid, cache_id))
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

    print(f"\nper-trajectory decorrelation -> {path}")
    for kind, d in out["kinds"].items():
        spread = d.get("per_slot_count_spread", {})
        print(f"  {kind:9s} per-slot count spread per traj "
              f"{ {k: v for k, v in spread.items()} }")
        for pair, e in d["pairs"].items():
            bits = [f"episode overlap {e['episode_overlap_frac'] * 100:.2f}%",
                    f"count r {e['per_slot_count_r']:+.3f}"]
            if "mean_abs_axis_delta" in e:
                bits.append(f"mean |d axis| {e['mean_abs_axis_delta']:.3f} "
                            f"(pi/3 = 1.047 if independent)")
            if "side_agreement" in e:
                bits.append(f"side agree {e['side_agreement'] * 100:.0f}%")
            if "partner_agreement" in e:
                bits.append(f"partner agree {e['partner_agreement'] * 100:.0f}%")
            print(f"    traj {pair}: " + ", ".join(bits))


if __name__ == "__main__":
    sys.argv.extend([
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
    ])
    main()
