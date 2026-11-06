"""Plan ONE trajectory's seg-ablation embedding bank — the CPU half of a bank.

No GPU, no encoding: this writes the episode plans every level sub-selects
from, plus a machine-readable realized-rate report.

    # dry run: assert the job count and show what is already on disk
    python scripts/plan_seg_bank.py +experiment=<name> +embedding=<emb> \
        +seg_bank=ladder_v1 +list_parts=true

    for t in 0 1 2; do
        python scripts/plan_seg_bank.py +experiment=<name> +embedding=<emb> \
            +seg_bank=ladder_v1 +traj_idx=$t
    done

Banks are planned **per trajectory** so a given cell is corrupted differently
in each one (different cut axis, area fraction, frames, merge partner) while
`strict_balance` still gives it the same episode count everywhere. That is why
this runs once per trajectory and not once per cache.

Gate on completion and file count only. The realized rates and the
decorrelation statistics go into the JSON reports and are read at ANALYSIS
time — `more_osc_clusters` is contact-starved, so merge and id_swap are quite
unlikely to be fully realizable at the ladder's top rate, and waiting to find
out would just be a probe-read-retune-relaunch round trip. Request the ladder,
encode what lands, describe it honestly at the end.
"""

import os
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from graph_dyno.seg_ablation.bank import (
    BANK_KINDS,
    PLAN_KINDS,
    bank_id,
    build_bank_plan,
    load_bank_spec,
    normalize_bank_spec,
    plan_filename,
    realized_stats,
    report_filename,
    rows_filename,
    select,
    write_report,
)
from graph_dyno.seg_ablation.build import _atomic_save

#: Ratios of the bank rate at which the realized rate is tabulated. A level's
#: own realized numbers are logged per RUN by `train._log_seg_stats`; this
#: curve is the dataset-level answer to "what does asking for rho actually
#: get you here", available before a single training job has run.
RATE_CURVE_RATIOS = [round(0.05 * i, 3) for i in range(1, 21)]


def _resolve_bank(cfg):
    """The composed `seg_bank` node and its config FILE must agree.

    A level names its bank with a string, and `bank_apply` resolves that name
    by reading `configs/seg_bank/<name>.yaml`. A CLI override applied here but
    not visible there would build a bank no level could ever address.
    """
    if cfg.get("seg_bank", None) is None:
        raise SystemExit("pass +seg_bank=<name> (see configs/seg_bank/)")
    composed = normalize_bank_spec(cfg.seg_bank)
    on_disk = load_bank_spec(composed["name"])
    if bank_id(composed) != bank_id(on_disk):
        raise SystemExit(
            f"the composed seg_bank ({bank_id(composed)}) differs from "
            f"configs/seg_bank/{composed['name']}.yaml ({bank_id(on_disk)}). "
            f"`seg_ablation.bank` resolves by NAME through the file, so a "
            f"CLI-overridden bank would be unaddressable. Edit the file.")
    return composed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from graph_dyno.encoders.factory import get_companion_suffix
    from graph_dyno.models.model_factory import resolve_shape_config
    from graph_dyno.data.trajectory_generator import generate_config_hash

    resolve_shape_config(cfg)
    bspec = _resolve_bank(cfg)
    bid = bank_id(bspec)
    cache_id = generate_config_hash(cfg.data)
    cache_dir = cfg.data.generator.cache_dir
    companion_suffix = get_companion_suffix(cfg)
    n_traj = int(cfg.data.generator.n_trajectories)

    print(f"cache_dir       {cache_dir}")
    print(f"cache_id        {cache_id}   (unchanged by seg_ablation, by design)")
    print(f"bank            {bspec['name']}  ({bid})")
    print(f"companion       {companion_suffix}")
    print(OmegaConf.to_yaml(OmegaConf.create(bspec)))

    if cfg.get("list_parts", False):
        print(f"{n_traj} parts (one per trajectory):")
        missing = 0
        for i in range(n_traj):
            have = [k for k in PLAN_KINDS
                    if os.path.exists(os.path.join(
                        cache_dir, plan_filename(bid, k, cache_id, i)))]
            missing += len(have) != len(PLAN_KINDS)
            print(f"  traj_idx={i:3d}  {len(have)}/{len(PLAN_KINDS)} plans  "
                  f"report={'yes' if os.path.exists(os.path.join(cache_dir, report_filename(bid, cache_id, i))) else 'no'}")
        print(f"{missing} of {n_traj} still to plan.")
        print(f"\nGPU wave after this: {n_traj} traj x {len(BANK_KINDS)} kinds "
              f"x n_chunks jobs ({', '.join(BANK_KINDS)}); "
              f"{', '.join(k for k in PLAN_KINDS if k not in BANK_KINDS)} "
              f"need no re-encode.")
        return

    traj_idx = cfg.get("traj_idx", None)
    if traj_idx is None:
        raise ValueError(
            "Specify a trajectory to plan: +traj_idx=<int> (see "
            "+list_parts=true). Banks are planned per trajectory on purpose — "
            "that is what decorrelates which corruption a given cell receives.")
    traj_idx = int(traj_idx)
    force = bool(cfg.get("force", False))

    traj_path = os.path.join(cache_dir, f"{cache_id}_{traj_idx:04d}.pt")
    if not os.path.exists(traj_path):
        raise SystemExit(f"no trajectory at {traj_path}")
    traj = torch.load(traj_path, map_location="cpu", weights_only=False,
                      mmap=True)
    T, N, _ = traj["states"].shape
    states = traj["states"].numpy()
    W = traj["W_interactions"]
    print(f"trajectory {traj_idx}: T={T}, N={N}\n")

    report = {
        "bank_id": bid,
        "bank_name": bspec["name"],
        "bank_spec": bspec,
        "cache_id": cache_id,
        "traj_idx": traj_idx,
        "T": int(T), "N": int(N),
        "companion_suffix": companion_suffix,
        "kinds": {},
    }

    for kind in PLAN_KINDS:
        path = os.path.join(cache_dir, plan_filename(bid, kind, cache_id,
                                                     traj_idx))
        if os.path.exists(path) and not force:
            plan = torch.load(path, map_location="cpu", weights_only=False)
            print(f"  {kind:9s} plan present, reusing "
                  f"({plan['rank'].size} episodes)")
        else:
            plan = build_bank_plan(
                bspec, kind, cache_id=cache_id, traj_idx=traj_idx, T=T, N=N,
                states=states, W_interactions=W)
            _atomic_save(plan, path)
        rep = dict(plan["report"])
        rep["requested_rate"] = float(bspec["kinds"][kind]["rate"])
        rep["n_infeasible_slots"] = len(plan["infeasible_slots"])
        rep["needs_gpu"] = kind in BANK_KINDS
        rep["rate_curve"] = _rate_curve(plan)
        report["kinds"][kind] = rep
        _print_kind(kind, rep)

    p = write_report(cache_dir, bid, cache_id, traj_idx, report)
    print(f"\nwrote {p}")
    for kind in PLAN_KINDS:
        f = os.path.join(cache_dir, plan_filename(bid, kind, cache_id, traj_idx))
        print(f"  {f}  ({os.path.getsize(f) / 1e6:.1f} MB)"
              + ("" if kind in BANK_KINDS else "   [no GPU rows needed]"))
    if BANK_KINDS:
        print("\nnext: scripts/build_seg_bank_worker.py, then "
              "scripts/merge_seg_bank.py; the rows land in\n  "
              + os.path.join(cache_dir, rows_filename(
                  bid, BANK_KINDS[0], companion_suffix, cache_id, traj_idx)))


def _rate_curve(plan):
    """Realized rate as a function of the requested fraction of the bank."""
    out = []
    bank_rate = float(plan["rate"])
    for r in RATE_CURVE_RATIOS:
        keep = select(plan, r * bank_rate)
        st = realized_stats(plan, keep)
        out.append({"ratio": r, "requested_rate": r * bank_rate,
                    "realized_rate": st["realized_rate"],
                    "n_episodes": st["n_episodes"],
                    "slot_count_spread": st["slot_count_spread"]})
    return out


def _print_kind(kind, rep):
    line = (f"  {kind:9s} requested {rep['requested_rate']:.4f}  realized "
            f"{rep['realized_rate']:.4f}  episodes {rep['n_episodes']}"
            f"/{rep.get('n_requested', 0)}  len {rep.get('mean_len', 0):.1f}  "
            f"spread init {rep.get('slot_count_spread', 0)}")
    if "partner_count_spread" in rep:
        line += (f" partner {rep['partner_count_spread']}  slots touched "
                 f"{rep.get('n_slots_touched', 0)}/"
                 f"{rep.get('n_feasible_slots', 0)} "
                 f"({rep['n_infeasible_slots']} infeasible"
                 + (f", {rep['n_stranded_slots']} STRANDED"
                    if rep.get("n_stranded_slots") else "") + ")")
    print(line)
    sf = rep.get("shortfall", 0)
    if sf:
        print(f"            shortfall {sf} episodes — the contact graph, not "
              f"a bug. Recorded, not gated on.")


if __name__ == "__main__":
    # Only these two: `hydra/job_logging=none` would remove the per-job
    # `.hydra` dir a --multirun sweep leaves behind, and pinning
    # hydra.sweep.dir here would stop --multirun honoring it from the CLI.
    sys.argv.extend([
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
    ])
    main()
