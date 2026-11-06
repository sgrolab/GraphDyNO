"""Materialize one ablation level straight out of a bank — no GPU, no wave.

`train.py` used to need `ensure_sidecars` (a GPU re-encode) followed by
`apply_all`. With a bank, the re-encode is already done: turning a bank into a
level is index selection plus the CPU tail that `build._finalize_sidecar`
shares with the direct builder. So the training job does it itself, and a
29-level x 10-seed scan needs no materialize wave at all — 87 fewer jobs, ~16 GB
less disk, and one fewer barrier between the bank merge and the first GPU
training run.

Two things make that affordable at 290 concurrent jobs:

  * **It is GPU-free.** Every corrupted row this needs is in the bank; the only
    encode a level could still want is the single all-zero-mask row, and that
    is cached in the bank file (`bank.find_empty_emb`) for exactly this reason.
  * **The geometry half is cached opportunistically.** The first job for a
    level pays the states/edges rebuild and writes
    `segabl_{abl_id}_{cache_id}_{traj}.pt` through the existing skip-if-exists
    `_atomic_save`; its nine sibling seeds read it. Concurrent duplicates are
    harmless (pid-unique temps, atomic replace) and bounded to whatever races
    on one level.

The corrupted `(T, N, D)` is assembled by `apply.apply_sidecar` verbatim — the
id-swap gather, the delta scatter, the int32 `crop_sides` check and the
`W_nnz` / `W_indices_sha1` edge-mask guard are all the existing code. The bank
changes only where the rows came from.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from graph_dyno.seg_ablation.apply import apply_sidecar
from graph_dyno.seg_ablation.bank import (
    BANK_KINDS,
    PLAN_KINDS,
    SEVERITY_FIELDS,
    bank_id,
    find_empty_emb,
    load_bank_spec,
    load_plan,
    frame_plan,
    load_rows,
    plan_events_of,
    realized_stats,
    resolve_collisions,
    select,
    select_frames,
)
from graph_dyno.seg_ablation.build import (
    _emb_dict,
    _finalize_sidecar,
    _sort_rowmajor,
    assert_state_identities,
    sidecar_paths,
)
from graph_dyno.seg_ablation.plan import (
    INVALID_OPS,
    PAIR_KINDS,
    OP_MERGE_ABSORB,
    OP_NONE,
    OP_PARTIAL,
    OP_SCALE,
    SEG_ABLATION_CODE_VERSION,
    PlanReport,
    abl_id,
    build_index_maps,
    normalize_spec,
)

#: ops whose row comes out of the bank rather than being the empty-mask row.
PIXEL_OPS = (OP_PARTIAL, OP_SCALE, OP_MERGE_ABSORB)

#: The op each banked kind writes, i.e. how to find a kind's cell-frames in the
#: op map. One-to-one: a merge writes OP_MERGE_ABSORB at the absorber and
#: OP_MERGE_GONE (an INVALID_OP, no banked row) at the partner.
KIND_OP = {"partial": OP_PARTIAL, "scale": OP_SCALE,
           "merge": OP_MERGE_ABSORB}

#: Fraction of a single-slot kind's requested frames that may go unfilled
#: before `episodic: false` refuses the level outright.
#:
#: Not zero, because the ceiling is not a clean number: frame capacity is
#: shared across kinds through the MIN_EPISODE_GAP band, so at high total load
#: a handful of unlucky cells run out while the other 498 are fine. On the 10%
#: joint arm that is 2 of 500 slots taking 42 frames instead of 50 — 0.06% of
#: the requested frames, far below anything the ARI can resolve. Not unbounded
#: either, because the failure it guards against is a level quietly training on
#: materially less error than its name claims.
MAX_FRAME_SHORTFALL = 0.02


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def apply_bank(embeddings, cache_dir, spec, *, cache_id, companion_suffix, dt,
               bank_configs_dir=None, verbose=True):
    """Bank-backed replacement for `ensure_sidecars` + `apply_all`.

    Returns the same list of `stats` dicts `apply_all` does, so
    `train._log_seg_stats`, the `seg_ablation_meta` attribute and the
    `_segabl_id` checkpoint buffer are all unchanged. The per-level REALIZED
    numbers are therefore still recorded — per *run* now, which is strictly
    more than the build-time print gave.
    """
    s = normalize_spec(spec)
    stats = []
    for traj_idx in sorted(embeddings.keys()):
        geo, emb, st = materialize_level(
            cache_dir, s, cache_id=cache_id, traj_idx=traj_idx,
            companion_suffix=companion_suffix, dt=dt,
            bank_configs_dir=bank_configs_dir, verbose=verbose)
        traj_path = os.path.join(cache_dir, f"{cache_id}_{traj_idx:04d}.pt")
        apply_sidecar(embeddings[traj_idx], geo, emb, traj_path, spec=s, dt=dt)
        stats.append(st)
        if verbose:
            print(f"  [seg_ablation] traj {traj_idx} ({geo['abl_id']}, bank "
                  f"{geo['bank']['bank_id']}): {st['n_events']} events, "
                  f"{st['invalid_frac'] * 100:.2f}% cell-frames invalid, "
                  f"{(1 - st['edges_kept_frac']) * 100:.2f}% edges dropped")
    return stats


def materialize_level(cache_dir, spec, *, cache_id, traj_idx,
                      companion_suffix, dt, save_geo=True, save_emb=False,
                      force=False, bank_configs_dir=None, verbose=True):
    """One level, one trajectory: (geo, emb, stats).

    `save_emb` is False by default — the embedding sidecar is redundant when a
    bank exists, and 29 levels x 3 trajectories of it is ~16 GB nobody reads.
    `scripts/materialize_seg_level.py` sets it when you actually want the pair
    on disk (to film a level, or to inspect it offline).
    """
    s = normalize_spec(spec)
    if not s["bank"]:
        raise ValueError(
            "materialize_level needs seg_ablation.bank set; use "
            "seg_ablation.build.build_sidecar for a bank-free spec")
    abl = abl_id(s)
    # `bank_configs_dir` is for tests and for a caller that keeps its bank
    # configs outside the repo; production resolves `configs/seg_bank/`.
    bspec = load_bank_spec(s["bank"], bank_configs_dir)
    bid = bank_id(bspec)

    geo_path, _ = sidecar_paths(cache_dir, abl, companion_suffix, cache_id,
                                traj_idx)
    traj_path = os.path.join(cache_dir, f"{cache_id}_{traj_idx:04d}.pt")
    traj = torch.load(traj_path, map_location="cpu", weights_only=False,
                      mmap=True)
    T, N, _ = traj["states"].shape

    geo = None
    if not force and os.path.exists(geo_path):
        geo = _load_cached_geometry(geo_path, abl, bid, T, N)
        if geo is not None and verbose:
            print(f"  [seg_ablation] traj {traj_idx}: geometry cached ({abl}), "
                  f"gathering banked rows")

    if geo is None:
        selected, events, src_t, src_n, op, report = _plan_selection(
            cache_dir, s, bspec, bid, cache_id=cache_id, traj_idx=traj_idx,
            T=T, N=N, verbose=verbose)
    else:
        # The cached op map is all the row gather needs — it addresses rows by
        # cell-frame, so `geo["bank"]["selected"]` is provenance only.
        op = geo["op"].numpy()
        selected = events = src_t = src_n = report = None

    pixel = _gather_pixel_table(
        cache_dir, bid, companion_suffix, cache_id, traj_idx, op=op, T=T, N=N)

    if geo is None:
        seed = int(s["seed"]) * 1_000_003 + traj_idx
        geo, emb, stats = _finalize_sidecar(
            cache_dir, s, abl, cache_id=cache_id, traj_idx=traj_idx,
            companion_suffix=companion_suffix, dt=dt, traj=traj,
            events=events, report=report, src_t=src_t, src_n=src_n, op=op,
            pixel=pixel, seed=seed,
            n_clipped=pixel["n_clipped"],
            n_crop_disagree=pixel["n_crop_disagree"],
            extra_geo={"bank": {
                "bank_id": bid,
                "bank_name": bspec["name"],
                "bank_spec": bspec,
                "selected": {k: v.astype(np.int32)
                             for k, v in selected.items()},
            }},
            save_geo=save_geo, save_emb=save_emb, verbose=verbose)
        return geo, emb, stats

    emb = _emb_dict(abl, companion_suffix, cache_id, traj_idx,
                    pixel["idx_t"], pixel["idx_n"], pixel["delta_emb"],
                    pixel["delta_crop"])
    if save_emb:
        from graph_dyno.seg_ablation.build import _atomic_save
        _, emb_path = sidecar_paths(cache_dir, abl, companion_suffix,
                                    cache_id, traj_idx)
        _atomic_save(emb, emb_path)
    assert_state_identities(geo["states"], geo["next_states"], dt,
                            where=f"cached bank geometry {abl}")
    return geo, emb, geo["stats"]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _plan_selection(cache_dir, s, bspec, bid, *, cache_id, traj_idx, T, N,
                    verbose=True):
    """Select from every banked kind, resolve collisions, build the maps."""
    episodic = bool(s["episodic"])
    plans, keeps, n_banked = {}, {}, {}
    raw_plans, rates = {}, {}
    for kind in PLAN_KINDS:
        rate = float(s["kinds"][kind]["rate"])
        if rate <= 0:
            continue
        _check_severity(kind, s["kinds"][kind], bspec["kinds"][kind],
                        bspec["name"], episodic=episodic)
        plan = load_plan(cache_dir, bid, kind, cache_id, traj_idx,
                         expect_T=T, expect_N=N)
        n_banked[kind] = int(np.asarray(plan["rank"]).size)
        raw_plans[kind] = plan
        rates[kind] = rate
        if episodic:
            plans[kind] = plan
            keeps[kind] = select(plan, rate)

    if not episodic and raw_plans:
        # One shared occupancy across all kinds, so a collision costs a retry
        # rather than a frame — see `bank.select_frames`. Each selection is then
        # wrapped in a plan-shaped dict of single-frame "episodes" so the rest
        # of the pipeline runs unchanged.
        selected_frames, shortfalls = select_frames(
            raw_plans, rates, level_seed=int(s["seed"]))
        for kind, short in sorted(shortfalls.items()):
            worst = min(short.values())
            want = int(round(rates[kind] * T))
            lost = sum(want - n for n in short.values())
            frac = lost / float(max(want * N, 1))
            # Saturate the capacity-limited slots and REPORT them rather than
            # silently skewing — the position `_waterfill` already takes on the
            # episodic path, and the reason the criterion actually enforced
            # downstream is `max_label_correlation` rather than exact balance.
            # Redistributing the shortfall onto the well-connected cells would
            # buy back the nominal rate at the price of making per-slot counts
            # a function of contact degree, which is the confound the
            # stratification exists to remove.
            if verbose:
                print(f"  [seg_ablation] {kind}: {len(short)} of {N} slots "
                      f"capacity-limited below the {want}-frame quota (worst "
                      f"{worst}), {frac * 100:.2f}% of requested frames lost; "
                      f"realized lands below nominal — quote realized.")
            if kind in PAIR_KINDS:
                # Structural and expected, not a fault of the request: exact
                # per-slot balance for merge/id_swap is a Hall condition on the
                # contact graph, and `more_osc_clusters` does not satisfy it
                # (1.8 union-distinct partners per cell, min 0). Roughly a
                # quarter of cells can never merge at all, so this shortfall is
                # large by construction and is equally present in the episodic
                # merge arms.
                continue
            if frac > MAX_FRAME_SHORTFALL:
                # A single-slot kind competes with the others only through the
                # shared gap band, so a LARGE shortfall means the level is
                # over-subscribed and would quietly train on materially less
                # error than its name claims. Refuse, as `select` refuses an
                # over-rate request.
                raise ValueError(
                    f"level requests {kind} at rate {rates[kind]} ({want} "
                    f"frames per cell) but {len(short)} slot(s) of bank {bid} "
                    f"could only take {worst} — {frac * 100:.2f}% of the "
                    f"requested frames, over the "
                    f"{MAX_FRAME_SHORTFALL * 100:g}% tolerance. Frame capacity "
                    f"is roughly `bank_rate * T / (MIN_EPISODE_GAP + 1)` per "
                    f"kind and is shared across kinds. Lower {kind} to <= "
                    f"{worst / float(T):.5f}, lower the other kinds, or raise "
                    f"the bank rate and re-encode.")
        for kind, (ep_idx, t_sel) in selected_frames.items():
            plans[kind] = frame_plan(raw_plans[kind], ep_idx, t_sel)
            keeps[kind] = np.ones(ep_idx.size, dtype=bool)

    # In frame mode this is a self-check, not a filter: `select_frames` already
    # enforced the identical gap rule against a shared occupancy, so it must
    # drop nothing. A nonzero `n_collision_drops` here is a selection bug.
    accepted, _, dropped = resolve_collisions(plans, keeps, T=T, N=N)
    if not episodic and any(dropped.values()):
        raise RuntimeError(
            f"frame selection produced overlaps that resolve_collisions had to "
            f"drop ({dropped}) — select_frames and resolve_collisions disagree "
            f"about the gap rule")

    selected, events = {}, []
    for kind, mask in accepted.items():
        ids = np.flatnonzero(mask).astype(np.int64)
        selected[kind] = ids
        events.extend(plan_events_of(plans[kind], ids.tolist()))
    events.sort(key=lambda e: (e.t0, e.slot, e.kind))

    src_t, src_n, op = build_index_maps(events, T=T, N=N)
    _assert_swap_disjoint(src_n, op, T, N)

    report = PlanReport()
    for kind in PLAN_KINDS:
        if kind not in plans:
            report.per_kind[kind] = {"n_episodes": 0, "realized_rate": 0.0}
            continue
        rep = dict(plans[kind]["report"])
        rep.update(realized_stats(plans[kind], accepted[kind]))
        # In frame mode the pseudo-plan's `rank` counts selected frames, not
        # what the bank holds; report the bank's own total.
        rep["n_banked"] = n_banked[kind]
        # An "episode" is one frame in frame mode, so the request is
        # `rate * T` frames per slot rather than `rate * T * N / mean_len`
        # episodes. Dividing by the bank's mean_len here would understate the
        # request by 8x and make every arm look like it overshot.
        unit_len = 1.0 if not episodic else float(s["kinds"][kind]["mean_len"])
        rep["n_requested"] = int(round(
            float(s["kinds"][kind]["rate"]) * T * N / unit_len))
        rep["shortfall"] = max(rep["n_requested"] - rep["n_episodes"], 0)
        rep["n_collision_drops"] = int(dropped[kind])
        rep["bank_rate"] = float(plans[kind]["rate"])
        rep["episodic"] = episodic
        report.per_kind[kind] = rep
        report.infeasible_slots[kind] = list(plans[kind]["infeasible_slots"])
    return selected, events, src_t, src_n, op, report


def _check_severity(kind, level_cfg, bank_cfg, bank_name, *, episodic=True):
    """A level may vary `rate`; varying severity would be a silent lie.

    The banked rows ARE the bank's severity. If a level asked for a 0.35-0.65
    area fraction while the bank encoded 0.55-0.85, nothing would fail — the
    run would simply report the level's own config while training on the
    bank's corruption. That is why the committed mild->extreme ladder (which
    co-varies rate and severity) is unbankable and why this scan pins severity
    at the `moderate` calibration and sweeps frequency alone.

    `mean_len` is exempt under `episodic: false`, and only that field: a frame
    selection deliberately cuts the bank's episodes to length 1, so requiring
    the level to echo the bank's 8 would be demanding a number that describes
    neither the level nor anything the level can control. Every field that
    describes what a corrupted crop LOOKS like is still pinned, which is the
    property that matters — those are what the banked pixels encode.
    """
    fields = SEVERITY_FIELDS.get(kind, ())
    if not episodic:
        fields = tuple(f for f in fields if f != "mean_len")
    for f in fields:
        a, b = level_cfg[f], bank_cfg[f]
        same = (abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
                if isinstance(a, float) or isinstance(b, float) else a == b)
        if not same:
            raise ValueError(
                f"seg_ablation.kinds.{kind}.{f} = {a!r} but bank "
                f"{bank_name!r} encoded {b!r}. A bank pins severity; only "
                f"`rate` may vary between levels. Either match the bank or "
                f"build a new one.")


def _assert_swap_disjoint(src_n, op, T, N):
    """The two invariants the late permutation depends on. Same as build.py."""
    identity_n = np.broadcast_to(np.arange(N, dtype=np.int32)[None, :], (T, N))
    swapped = src_n != identity_n
    if np.any(swapped & (op != OP_NONE)):
        raise RuntimeError("id_swap collides with a pixel op — selection bug")
    if swapped.any():
        rows_ = np.arange(T)[:, None]
        if not np.array_equal(src_n[rows_, src_n], identity_n):
            raise RuntimeError("src_n is not an involution — selection bug")


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def _gather_pixel_table(cache_dir, bid, companion_suffix, cache_id, traj_idx,
                        *, op, T, N):
    """Pull the selected rows out of the bank and add the empty-mask rows.

    Row order mirrors `build_sidecar`: re-encoded crops first (row-major by
    `(t, n)`), then the invalid cell-frames. Order is not load-bearing for the
    scatter (there are no duplicate `(t, n)`), but keeping it identical makes
    a bank-derived and a directly-built sidecar comparable field by field.
    """
    idx_t_parts, idx_n_parts, emb_parts = [], [], []
    crop_parts, cd_parts = [], []
    n_clipped = n_crop_disagree = 0
    empty_emb = None
    D = None

    for kind in BANK_KINDS:
        # Rows are addressed by CELL-FRAME, not by episode: `episode_id` indexes
        # the plan, and under `episodic: false` the level owns individual frames
        # of an episode rather than all of it. The op map is the authority on
        # which cell-frames this level actually claims, and it is built from the
        # accepted events in both modes — so keying on `(t, n)` is exact for
        # frame selection and identical to the old `episode_id` match for
        # episode selection (the equality is re-asserted against `pixel_map`
        # below).
        want_tn = np.argwhere(op == KIND_OP[kind])
        if want_tn.size == 0:
            continue
        rows = load_rows(cache_dir, bid, kind, companion_suffix, cache_id,
                         traj_idx)
        empty_emb = rows["empty_emb"].clone().float()
        D = int(rows["D"])
        row_key = (rows["idx_t"].numpy().astype(np.int64) * N
                   + rows["idx_n"].numpy().astype(np.int64))
        want_key = want_tn[:, 0].astype(np.int64) * N + want_tn[:, 1]
        sel = np.flatnonzero(np.isin(row_key, want_key))
        if sel.size != want_key.size:
            missing = sorted(set(want_key.tolist())
                             - set(row_key[sel].tolist()))[:8]
            raise ValueError(
                f"bank {bid}/{kind} traj {traj_idx} is missing rows for "
                f"{want_key.size - sel.size} selected cell-frames (e.g. "
                f"{[(int(k) // N, int(k) % N) for k in missing]} as (t, n)). "
                f"The chunk merge is incomplete or the plan changed under it.")
        idx_t_parts.append(rows["idx_t"].numpy()[sel])
        idx_n_parts.append(rows["idx_n"].numpy()[sel])
        emb_parts.append(rows["embeddings"][torch.from_numpy(sel)].float())
        crop_parts.append(rows["crop_sides"].numpy()[sel])
        cd_parts.append(rows["centroid_delta"].numpy()[sel])
        n_clipped += int(rows["clipped"].numpy()[sel].sum())
        n_crop_disagree += int(rows["crop_disagree"].numpy()[sel].sum())

    pixel_map = np.isin(op, PIXEL_OPS)
    n_pix = int(pixel_map.sum())
    if idx_t_parts:
        it = np.concatenate(idx_t_parts).astype(np.int64)
        inn = np.concatenate(idx_n_parts).astype(np.int64)
        order = np.lexsort((inn, it))
        it, inn = it[order], inn[order]
        pix_emb = torch.cat(emb_parts, 0)[torch.from_numpy(order)]
        pix_crop = np.concatenate(crop_parts).astype(np.int32)[order]
        pix_cd = np.concatenate(cd_parts).astype(np.float64)[order]
        # The rows and the op map must agree exactly: a row the map does not
        # claim would be scattered onto a cell-frame the states rebuild treats
        # as clean, which is the one inconsistency `apply_sidecar` cannot see.
        got_map = np.zeros((T, N), dtype=bool)
        got_map[it, inn] = True
        if not np.array_equal(got_map, pixel_map) or it.size != n_pix:
            raise ValueError(
                f"banked rows cover {it.size} cell-frames but the op map has "
                f"{n_pix} pixel ops — selection and geometry disagree")
    else:
        if n_pix:
            raise ValueError(f"op map has {n_pix} pixel ops but no banked "
                             f"rows were selected")
        it = np.zeros(0, dtype=np.int64)
        inn = np.zeros(0, dtype=np.int64)
        pix_emb = None
        pix_crop = np.zeros(0, dtype=np.int32)
        pix_cd = np.zeros((0, 2), dtype=np.float64)

    invalid_idx = _sort_rowmajor(np.argwhere(np.isin(op, INVALID_OPS)))
    n_invalid = int(invalid_idx.shape[0])
    if n_invalid and empty_emb is None:
        empty_emb = find_empty_emb(cache_dir, bid, companion_suffix, cache_id,
                                   traj_idx)
        D = int(empty_emb.shape[0])

    if pix_emb is None and empty_emb is None:
        # A level with no pixel and no validity events (`none`, or centroid
        # noise alone) touches no embedding row at all, so it needs no bank
        # file to exist. D is then irrelevant; keep the table empty but typed.
        delta_emb = torch.zeros((0, 0), dtype=torch.float32)
    else:
        D = int(D)
        delta_emb = torch.empty((it.size + n_invalid, D), dtype=torch.float32)
        if it.size:
            delta_emb[:it.size] = pix_emb
        if n_invalid:
            delta_emb[it.size:] = empty_emb.unsqueeze(0).expand(n_invalid, -1)

    centroid_delta = np.zeros((T, N, 2), dtype=np.float64)
    if it.size:
        centroid_delta[it, inn] = pix_cd

    return {
        "idx_t": np.concatenate([it, invalid_idx[:, 0]]).astype(np.int32),
        "idx_n": np.concatenate([inn, invalid_idx[:, 1]]).astype(np.int32),
        "delta_emb": delta_emb,
        "delta_crop": np.concatenate([
            pix_crop, np.zeros(n_invalid, dtype=np.int32)]).astype(np.int32),
        "centroid_delta": centroid_delta,
        "crop_override": np.full((T, N), -1, dtype=np.int32),
        "n_pix": n_pix,
        "n_clipped": n_clipped,
        "n_crop_disagree": n_crop_disagree,
    }


# ---------------------------------------------------------------------------
# Geometry cache
# ---------------------------------------------------------------------------

def _load_cached_geometry(geo_path, abl, bid, T, N):
    """Reuse a geometry sidecar written by a sibling run, or return None.

    Returns None (rather than raising) only when the file is unusable for a
    benign reason. Anything that indicates the artifact is *wrong* for this
    spec — a stale code version, a different `abl_id`, a different bank — is
    fatal: silently rebuilding over it would hide exactly the collision the
    `abl_id` hash exists to prevent.
    """
    geo = torch.load(geo_path, map_location="cpu", weights_only=False)
    if geo.get("code_version") != SEG_ABLATION_CODE_VERSION:
        raise ValueError(
            f"{os.path.basename(geo_path)} was built with "
            f"{geo.get('code_version')!r} but this code is "
            f"{SEG_ABLATION_CODE_VERSION!r} — delete it and rebuild.")
    if geo.get("abl_id") != abl:
        raise ValueError(f"{os.path.basename(geo_path)} carries abl_id "
                         f"{geo.get('abl_id')} but this spec is {abl}")
    bank = geo.get("bank")
    if bank is None:
        raise ValueError(
            f"{os.path.basename(geo_path)} was built WITHOUT a bank but this "
            f"spec names one. `bank` is part of the hashed payload, so this "
            f"should be impossible — the file predates that change; delete it.")
    if bank.get("bank_id") != bid:
        raise ValueError(f"{os.path.basename(geo_path)} came from bank "
                         f"{bank.get('bank_id')}, this spec uses {bid}")
    if tuple(geo["states"].shape[:2]) != (T, N):
        raise ValueError(f"{os.path.basename(geo_path)} is "
                         f"{tuple(geo['states'].shape[:2])}, cache is {(T, N)}")
    return geo
