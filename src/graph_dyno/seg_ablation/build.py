"""Build the two seg-ablation sidecars for one trajectory.

Nothing here mutates the trajectory cache. Every trajectory `.pt` stays
byte-identical; all the ablation lives in

    segabl_{abl_id}_{cache_id}_{traj:04d}.pt                     backbone-INDEPENDENT
    segabl_{abl_id}_{companion_suffix}_{cache_id}_{traj:04d}.pt  backbone-DEPENDENT

The split matters: embedding width D depends on `shape.dino_model` /
`dino_use_patches` (768 / 1024 / 2048), so the backbone-dependent half must
carry `companion_suffix` or two `+embedding=` runs collide. It also lets an
`+embedding=` sweep reuse the geometry half unchanged — `more_osc_clusters`
already holds `dino_SP`, `dino_CELL_CA_L16`, `conv_16`, `conv_256` and
`fourier_32` companions, so that reuse is real.

Writes are atomic (`<name>.tmp` then `os.replace`): the skip-if-exists resume
semantics inherited from `precompute_all` (`precompute_embeddings.py:143-144`)
would otherwise accept a truncated file from a killed job as complete.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from graph_dyno.seg_ablation import ops
from graph_dyno.seg_ablation.plan import (
    INVALID_OPS,
    OP_MERGE_ABSORB,
    OP_NONE,
    OP_PARTIAL,
    OP_SCALE,
    SEG_ABLATION_CODE_VERSION,
    abl_id,
    build_index_maps,
    normalize_spec,
    per_slot_counts,
    plan_events,
)

#: masks are pickled one-per-item into the SDF pool and buffered as an
#: (M, 224, 224) float16 array, so a whole trajectory's worth of 256^2 crops
#: would pickle ~16 GB in a single call. Chunk the encode instead.
ENCODE_CHUNK = 4096


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def geometry_filename(abl, cache_id, traj_idx):
    return f"segabl_{abl}_{cache_id}_{traj_idx:04d}.pt"


def embedding_filename(abl, companion_suffix, cache_id, traj_idx):
    return f"segabl_{abl}_{companion_suffix}_{cache_id}_{traj_idx:04d}.pt"


def sidecar_paths(cache_dir, abl, companion_suffix, cache_id, traj_idx):
    return (
        os.path.join(cache_dir, geometry_filename(abl, cache_id, traj_idx)),
        os.path.join(cache_dir,
                     embedding_filename(abl, companion_suffix, cache_id, traj_idx)),
    )


def _atomic_save(obj, path):
    """Write via a PROCESS-UNIQUE temp file, then rename.

    The temp name carries the pid. A shared `<name>.tmp` is not merely
    redundant under concurrency, it is corrupting: two cluster jobs that both
    find the sidecar missing would write the same temp path, and the first
    `os.replace` would publish whatever the second had written so far. Every
    ladder arm calls `ensure_sidecars` on the same cache dir, so that race is
    the default on a cluster, not an edge case.

    `os.replace` is atomic within a filesystem, so the loser of the race
    publishes a complete (identical) file and the winner's is overwritten
    whole. Building the sidecars once up front is still the right move; this
    just makes the race harmless.
    """
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Perturbation worker (top-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------

def _perturb_worker(item):
    """Apply one pixel op. Returns (mask_or_None, crop_side, delta_units, clipped)."""
    op_code, mask_a, mask_b, params, px_per_unit, pad_factor, close_px = item

    if op_code == OP_SCALE:
        # v2 re-encodes this: the SDF barely moves (MAE ~0.01 at 2 px) but the
        # DINO embedding moves a third of the between-cell distance, so a
        # crop_side-only treatment would discard most of the perturbation.
        base = ops.observed_centroid_units(mask_a, px_per_unit)
        m = ops.morph_mask(mask_a, params["delta_px"])
        got = ops.observed_centroid_units(m, px_per_unit)
        delta = np.zeros(2) if (base is None or got is None) else (got - base)
        return ops.pad_to(m, pad_factor), ops.crop_side_of(m), delta, False

    if op_code == OP_PARTIAL:
        base = ops.observed_centroid_units(mask_a, px_per_unit)
        cut = ops.half_plane_cut(mask_a, params["axis"], params["area_frac"],
                                 params["side"])
        got = ops.observed_centroid_units(cut, px_per_unit)
        delta = np.zeros(2) if (base is None or got is None) else (got - base)
        # Padded to the merge canvas so one uniform batch reaches the encoder.
        # `_mask_to_sdf` crops to the tight bbox first, so padding is a no-op
        # on the SDF for any mask that does not touch the 128 border.
        return ops.pad_to(cut, pad_factor), ops.crop_side_of(cut), delta, False

    if op_code == OP_MERGE_ABSORB:
        base = ops.observed_centroid_units(ops.pad_to(mask_a, pad_factor),
                                           px_per_unit)
        blob, clipped = ops.merge_pair(mask_a, mask_b, params["offset_px"],
                                       pad_factor, close_px)
        got = ops.observed_centroid_units(blob, px_per_unit)
        delta = np.zeros(2) if (base is None or got is None) else (got - base)
        return blob, ops.crop_side_of(blob), delta, bool(clipped)

    raise ValueError(f"_perturb_worker got op {op_code}")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_sidecar(cache_dir, spec, *, cache_id, traj_idx, companion_suffix,
                  dt, encoder_factory=None, workers=32, batch_size=64,
                  force=False, verbose=True):
    """Build (or reuse) the sidecars for one trajectory.

    Args:
        cache_dir: the trajectory cache directory.
        spec: raw or normalized seg_ablation spec.
        cache_id: `generate_config_hash(cfg.data)`.
        traj_idx: trajectory index.
        companion_suffix: `get_companion_suffix(cfg)`.
        dt: `cfg.data.dt` — the finite-difference step the velocity identity uses.
        encoder_factory: zero-arg callable returning a ShapeEncoder. Called
            **lazily**, and only when a pixel or validity event actually exists
            (mirroring `precompute_all:117-132`, which defers weight loading
            past its skip-if-exists check).
        workers: CPU workers for the perturbation pool.
        force: rebuild even when both sidecars exist.

    Returns:
        (geo_path, emb_path, stats) — stats is None when both files were reused.
    """
    s = normalize_spec(spec)
    abl = abl_id(s)
    geo_path, emb_path = sidecar_paths(cache_dir, abl, companion_suffix,
                                       cache_id, traj_idx)
    if not force and os.path.exists(geo_path) and os.path.exists(emb_path):
        if verbose:
            print(f"  [seg_ablation] traj {traj_idx}: sidecars present ({abl}), skipping")
        return geo_path, emb_path, None

    traj_path = os.path.join(cache_dir, f"{cache_id}_{traj_idx:04d}.pt")
    traj = torch.load(traj_path, map_location="cpu", mmap=True)

    clean_states = traj["states"]
    clean_next = traj["next_states"]
    images = traj["channeled_images"]
    W = traj["W_interactions"]
    r_max = float(traj["sim_metadata"]["r_max"])
    T, N, _ = clean_states.shape
    image_size = int(images.shape[-1])
    px_per_unit = ops.scale_px(image_size, r_max)
    dt = float(dt)

    # Verified on the INPUT before we rebuild anything: both identities hold
    # with exactly zero error on all three paper caches.
    assert_state_identities(clean_states, clean_next, dt,
                            where=f"clean cache {os.path.basename(traj_path)}")

    # ---- plan --------------------------------------------------------------
    # `* 1_000_003 + traj_idx` decorrelates trajectories (plan.py:485). Under
    # `shared_across_trajectories` the traj term is dropped on purpose, so the
    # corruption is a property of the cell id rather than of (cell, traj) —
    # see the spec-default note in plan.py for why that changes the result.
    seed = int(s["seed"]) * 1_000_003
    if not s["shared_across_trajectories"]:
        seed += traj_idx
    events, report = plan_events(
        s, T=T, N=N, states=clean_states.numpy(), W_interactions=W, seed=seed)
    src_t, src_n, op = build_index_maps(events, T=T, N=N)

    identity_n = np.broadcast_to(np.arange(N, dtype=np.int32)[None, :], (T, N))
    swapped = src_n != identity_n
    # Every op below is computed in TRUE cell space and permuted only at the
    # very end, which is only valid because a swapped slot never also carries
    # a pixel op. plan_events marks occupancy for both roles; assert it.
    if np.any(swapped & (op != OP_NONE)):
        raise RuntimeError("id_swap collides with a pixel op — planner bug")
    if swapped.any():
        rows_ = np.arange(T)[:, None]
        if not np.array_equal(src_n[rows_, src_n], identity_n):
            raise RuntimeError("src_n is not an involution — planner bug")

    # ---- pixel work --------------------------------------------------------
    pad_factor = int(s["kinds"]["merge"]["pad_factor"])
    close_px = float(s["kinds"]["merge"]["close_px"])
    ev_at = _event_lookup(events)

    # Sorted by (t, n): measured 1.41 ms/read sorted vs 1.81 unsorted on
    # NFS (nfs4, rsize=1M) — a free 25%.
    pixel_idx = _sort_rowmajor(
        np.argwhere(np.isin(op, (OP_PARTIAL, OP_MERGE_ABSORB, OP_SCALE))))

    centroid_delta = np.zeros((T, N, 2), dtype=np.float64)
    # Retained in the schema for an op that changes reported scale without
    # changing pixels. No such op exists today: erode/dilate looked like one
    # (its SDF MAE really is ~0.01) but moves the DINO embedding by a third of
    # the between-cell distance, so v2 re-encodes it. See
    # SEG_ABLATION_CODE_VERSION.
    crop_override = np.full((T, N), -1, dtype=np.int32)
    n_clipped = 0
    canvas = pad_factor * image_size

    invalid_map = np.isin(op, INVALID_OPS)
    invalid_idx = _sort_rowmajor(np.argwhere(invalid_map))
    n_invalid = int(invalid_idx.shape[0])
    n_pix = int(pixel_idx.shape[0])

    idx_t = np.zeros(0, dtype=np.int32)
    idx_n = np.zeros(0, dtype=np.int32)
    delta_emb = torch.zeros((0, 0), dtype=torch.float32)
    delta_crop = np.zeros(0, dtype=np.int32)

    if n_pix or n_invalid:
        if encoder_factory is None:
            raise ValueError("encoder_factory is required: this spec has pixel "
                             "or validity events that need embeddings")
        encoder = encoder_factory()
        emb_parts, crop_parts = [], []
        n_crop_disagree = 0

        for lo in range(0, n_pix, ENCODE_CHUNK):
            sub = pixel_idx[lo:lo + ENCODE_CHUNK]
            items = []
            for t, n in sub:
                t, n = int(t), int(n)
                ev = ev_at[(t, n)]
                if op[t, n] in (OP_PARTIAL, OP_SCALE):
                    items.append((int(op[t, n]), _mask(images, t, n), None,
                                  dict(ev.params), px_per_unit, pad_factor,
                                  close_px))
                else:
                    j = ev.partner
                    # states[t] is the geometry the raster at index t depicts
                    off = (clean_states[t, j, :2] - clean_states[t, n, :2]).numpy()
                    items.append((OP_MERGE_ABSORB, _mask(images, t, n),
                                  _mask(images, t, j),
                                  {"offset_px": off.astype(np.float64) * px_per_unit},
                                  px_per_unit, pad_factor, close_px))
            results = _run_pool(items, workers)

            masks = np.zeros((len(results), canvas, canvas), dtype=np.uint8)
            local_crop = np.zeros(len(results), dtype=np.int32)
            for k, ((t, n), res) in enumerate(zip(sub, results)):
                masks[k] = res[0]
                local_crop[k] = res[1]
                centroid_delta[t, n] = res[2]
                n_clipped += int(res[3])

            e, c = encoder.encode_masks(torch.from_numpy(masks),
                                        batch_size=batch_size)
            emb_parts.append(e.float().cpu())
            # `_mask_to_sdf`'s own crop_side is the single source of truth (it
            # is what the clean companions store, and `scale_mse` trains
            # against it). ops.crop_side_of is the same bbox formula and only
            # disagrees for masks that touched the 128 border before padding.
            crop_parts.append(c.cpu().to(torch.int32))
            n_crop_disagree += int((c.cpu().numpy() != local_crop).sum())

        # One extra all-zero mask: `_mask_to_sdf_worker`
        # (precompute_dino_embeddings.py:106-110) returns the deterministic
        # all-(-1) SDF and crop_side 0 for it, so a single encode covers every
        # missed and every absorbed cell in the whole dataset.
        e0, c0 = encoder.encode_masks(
            torch.zeros((1, canvas, canvas), dtype=torch.uint8),
            batch_size=batch_size)
        if int(c0[0]) != 0:
            raise RuntimeError(
                f"empty-mask encode returned crop_side {int(c0[0])}, expected 0")
        empty_emb = e0.float().cpu()[0]

        idx_t = np.concatenate([pixel_idx[:, 0], invalid_idx[:, 0]]).astype(np.int32)
        idx_n = np.concatenate([pixel_idx[:, 1], invalid_idx[:, 1]]).astype(np.int32)
        delta_emb = torch.empty((idx_t.size, empty_emb.shape[0]), dtype=torch.float32)
        if n_pix:
            delta_emb[:n_pix] = torch.cat(emb_parts, 0)
        if n_invalid:
            delta_emb[n_pix:] = empty_emb.unsqueeze(0).expand(n_invalid, -1)
        delta_crop = np.concatenate([
            (torch.cat(crop_parts, 0).numpy() if crop_parts
             else np.zeros(0, dtype=np.int32)),
            np.zeros(n_invalid, dtype=np.int32),
        ]).astype(np.int32)
    else:
        n_crop_disagree = 0

    geo, emb, stats = _finalize_sidecar(
        cache_dir, s, abl, cache_id=cache_id, traj_idx=traj_idx,
        companion_suffix=companion_suffix, dt=dt, traj=traj,
        events=events, report=report, src_t=src_t, src_n=src_n, op=op,
        pixel={"idx_t": idx_t, "idx_n": idx_n, "delta_emb": delta_emb,
               "delta_crop": delta_crop, "centroid_delta": centroid_delta,
               "crop_override": crop_override, "n_pix": n_pix},
        seed=seed, n_clipped=n_clipped, n_crop_disagree=n_crop_disagree,
        verbose=verbose)
    return geo_path, emb_path, stats


def _finalize_sidecar(cache_dir, s, abl, *, cache_id, traj_idx,
                      companion_suffix, dt, traj, events, report,
                      src_t, src_n, op, pixel, seed,
                      n_clipped=0, n_crop_disagree=0, extra_geo=None,
                      save_geo=True, save_emb=True, verbose=True):
    """The CPU tail every builder shares: states, edges, audit, stats, write.

    Extracted so `build_sidecar` (which re-encodes its own crops) and the bank
    materializer (`seg_ablation/bank_apply.py`, which gathers pre-encoded rows)
    cannot drift apart on a single invariant. Everything load-bearing lives
    here: the two `assert_state_identities` checks, the "no edge touches an
    invalid slot" rule, the `W_nnz` + `W_indices_sha1` provenance stamp, the
    label-correlation gate, and the `clean_disp_std` that must NOT vary with
    the ablation level.

    `save_geo` / `save_emb` exist for the bank path, which writes the geometry
    half opportunistically (so the first run of a level pays the rebuild and
    its sibling seeds read it) and never writes an embedding half at all —
    those rows already live in the bank.

    Args:
        pixel: the delta table — `idx_t`, `idx_n`, `delta_emb`, `delta_crop`,
            the full `(T, N, 2)` `centroid_delta`, the `(T, N)` int32
            `crop_override`, and `n_pix` (rows that are re-encoded crops rather
            than the empty-mask row).

    Returns:
        (geo, emb, stats) — the two sidecar dicts and the stats block, whether
        or not they were written.
    """
    idx_t = pixel["idx_t"]
    idx_n = pixel["idx_n"]
    delta_emb = pixel["delta_emb"]
    delta_crop = pixel["delta_crop"]
    centroid_delta = pixel["centroid_delta"]
    crop_override = pixel["crop_override"]
    n_pix = int(pixel["n_pix"])

    geo_path, emb_path = sidecar_paths(cache_dir, abl, companion_suffix,
                                       cache_id, traj_idx)
    clean_states = traj["states"]
    clean_next = traj["next_states"]
    W = traj["W_interactions"]
    r_max = float(traj["sim_metadata"]["r_max"])
    T, N, _ = clean_states.shape
    image_size = int(traj["channeled_images"].shape[-1])
    px_per_unit = ops.scale_px(image_size, r_max)
    pad_factor = int(s["kinds"]["merge"]["pad_factor"])
    dt = float(dt)

    invalid_map = np.isin(op, INVALID_OPS)
    n_invalid = int(invalid_map.sum())

    # crop_sides_override (erode/dilate) and the delta table must be disjoint:
    # two sources of truth for one (t, n) is exactly the class of bug the
    # sidecar's asserts exist to prevent.
    if np.any((crop_override >= 0) & (invalid_map | (op != OP_NONE))):
        raise RuntimeError("crop_sides_override overlaps a delta-table entry")

    # ---- states ------------------------------------------------------------
    rng = np.random.default_rng(seed + 977)
    pos = clean_states[..., :2].numpy().astype(np.float64)
    median_disp_px = float(
        np.median(np.linalg.norm(pos[1:] - pos[:-1], axis=-1)) * px_per_unit)

    # Parametrised against per-frame displacement, not pixels or cell radius:
    # velocity is recomputed as (p~(t) - p~(t-1))/dt, which amplifies sigma by
    # sqrt(2)/dt = 141x, so a "realistic" 1 px centroid error gives 7.5
    # units/time of velocity noise against a true mean speed of 2.79.
    sigma_units = float(s["centroid_noise_sigma_rel"]) * median_disp_px / px_per_unit

    obs_pos = pos + centroid_delta
    if sigma_units > 0:
        obs_pos = obs_pos + rng.normal(0.0, sigma_units, size=obs_pos.shape)

    valid_true = ~invalid_map
    pad_pos = -4.0 * r_max          # real_data/generator.py:64
    obs_pos[~valid_true] = pad_pos

    # Late permutation, exactly as the composition order says: everything above
    # is in true-cell space; the observer's slot relabelling is last.
    rows = np.arange(T)[:, None]
    obs_pos = obs_pos[rows, src_n]
    valid = valid_true[rows, src_n]

    # Rebuild in float32 with the *same* arithmetic the identity check uses, so
    # the invariant holds bit-exactly rather than to a tolerance.
    pos32 = torch.from_numpy(obs_pos.astype(np.float32))
    states = clean_states.clone()
    states[..., :2] = pos32
    states[1:, :, 2:4] = (pos32[1:] - pos32[:-1]) / dt
    # t = 0 is never served as a sample (dataloader.py:660 returns None), so it
    # has no defined finite difference; keep the cache's own value.
    states[0, :, 2:4] = clean_states[0, :, 2:4]

    next_pos_last = clean_next[-1, :, :2].numpy().astype(np.float64)
    if sigma_units > 0:
        next_pos_last = next_pos_last + rng.normal(0.0, sigma_units,
                                                   size=next_pos_last.shape)
    next_pos_last = next_pos_last[src_n[-1]]
    next_pos_last[~valid[-1]] = pad_pos
    next_pos_last32 = torch.from_numpy(next_pos_last.astype(np.float32))

    next_states = clean_next.clone()
    next_states[:-1] = states[1:]
    next_states[-1, :, :2] = next_pos_last32
    next_states[-1, :, 2:4] = (next_pos_last32 - pos32[-1]) / dt

    next_valid = np.empty_like(valid)
    next_valid[:-1] = valid[1:]
    # No `valid` exists for the frame past the window; carry the slot's own
    # status forward (1 frame in 2000).
    next_valid[-1] = valid[-1]

    assert_state_identities(states, next_states, dt,
                            where="rebuilt seg-ablation states")

    # ---- edges -------------------------------------------------------------
    W_idx = W._indices()
    nnz = int(W_idx.shape[1])
    W_sha1 = hashlib.sha1(np.ascontiguousarray(W_idx.numpy()).tobytes()).hexdigest()
    et, ei, ej = W_idx[0].numpy(), W_idx[1].numpy(), W_idx[2].numpy()
    # Edges are stored in TRUE cell space, and an id swap only relabels the
    # endpoints (apply.py does that), so survival depends on true-cell validity.
    # Dropping every edge touching an invalid slot is what guarantees "padded
    # slots have no incident edges", which is what makes them harmless in
    # gnn.py (never a src/tgt, so scatter_add_ leaves the message at zero).
    keep = valid_true[et, ei] & valid_true[et, ej]

    # ---- clean, arm-independent displacement scales -----------------------
    clean_pos = clean_states[..., :2].numpy().astype(np.float64)
    clean_vel = clean_states[..., 2:4].numpy().astype(np.float64)
    dx = np.linalg.norm(clean_next[..., :2].numpy() - clean_pos, axis=-1)
    dv = np.linalg.norm(clean_next[..., 2:4].numpy() - clean_vel, axis=-1)
    mean_speed = float(np.mean(np.linalg.norm(clean_vel, axis=-1)))
    vel_sigma = sigma_units * np.sqrt(2.0) / dt

    # ---- the stratification audit ------------------------------------------
    # Equal event counts do not imply equal supervision — a miss costs 3
    # transitions under `strict` (graph_model_real.py:159-173) while a blind
    # partial costs none — hence per-kind counts AND the realized supervised
    # count are both audited.
    counts = per_slot_counts(events, N)
    sup = (valid[:-1] & valid[1:] & next_valid[1:]).sum(axis=0).astype(np.int64)
    counts["supervised_transitions"] = sup
    corr, worst = _label_correlations(counts, traj["sim_metadata"], N)
    if worst is not None and abs(worst[2]) > float(s["max_label_correlation"]):
        msg = (f"seg_ablation: per-slot {worst[0]!r} correlates with the varied "
               f"parameter {worst[1]!r} at r = {worst[2]:+.3f} (limit "
               f"{s['max_label_correlation']}). Contact degree is a function of "
               f"adhesion, so an unbalanced role makes per-slot supervision a "
               f"monotone function of the factor being clustered — the "
               f"separation result would then be partly an artifact of the "
               f"sampler.")
        if s["strict_balance"]:
            raise RuntimeError(
                msg + " Lower the merge/id_swap rate, or set "
                      "seg_ablation.strict_balance=false to accept it knowingly "
                      "(and report the correlation alongside the result).")
        print(f"[seg_ablation] WARNING: {msg}")

    stats = {
        "abl_id": abl,
        "traj_idx": traj_idx,
        "T": T, "N": N,
        "n_events": len(events),
        "per_slot_counts": {k: v for k, v in counts.items()},
        "label_correlations": corr,
        "worst_label_correlation": worst,
        "per_kind": report.per_kind,
        "n_infeasible_slots": {k: len(v) for k, v in report.infeasible_slots.items()},
        "n_invalid_cell_frames": n_invalid,
        "invalid_frac": n_invalid / float(T * N),
        "n_reencoded": n_pix,
        "n_merge_clipped": n_clipped,
        "n_crop_side_disagree": n_crop_disagree,
        "median_disp_px": median_disp_px,
        "centroid_sigma_px": sigma_units * px_per_unit,
        "centroid_sigma_units": sigma_units,
        "mean_speed_units_per_time": mean_speed,
        "vel_noise_sigma": float(vel_sigma),
        "vel_snr": float(mean_speed / vel_sigma) if vel_sigma > 0 else float("inf"),
        "edges_kept_frac": float(keep.mean()) if nnz else 1.0,
        "W_nnz": nnz,
        "pad_pos": pad_pos,
        "px_per_unit": px_per_unit,
        "sigma_x": float(dx.std()),
        "sigma_v": float(dv.std()),
    }

    geo = {
        "spec": s,
        "abl_id": abl,
        "code_version": SEG_ABLATION_CODE_VERSION,
        "seed": seed,
        "cache_id": cache_id,
        "traj_idx": traj_idx,
        "dt": dt,
        "r_max": r_max,
        "image_size": image_size,
        "pad_factor": pad_factor,
        "events": [(e.kind, e.t0, e.t1, e.slot, e.partner, dict(e.params))
                   for e in events],
        "src_t": torch.from_numpy(src_t),
        "src_n": torch.from_numpy(src_n),
        "op": torch.from_numpy(op),
        "states": states,
        "next_states": next_states,
        "valid": torch.from_numpy(valid),
        "next_valid": torch.from_numpy(next_valid),
        "W_edge_keep": torch.from_numpy(keep),
        "W_nnz": nnz,
        "W_indices_sha1": W_sha1,
        "crop_sides_override": torch.from_numpy(crop_override),
        # `input_noise_std` multiplies these, so letting them drift with the
        # ablation level would give every arm a different regularisation
        # strength for reasons unrelated to segmentation. Computed on the CLEAN
        # states, so every arm — including `none` — gets the same number.
        "clean_disp_std": {"sigma_x": stats["sigma_x"], "sigma_v": stats["sigma_v"]},
        "stats": stats,
    }
    if extra_geo:
        geo.update(extra_geo)
    emb = _emb_dict(abl, companion_suffix, cache_id, traj_idx,
                    idx_t, idx_n, delta_emb, delta_crop)

    if save_geo:
        _atomic_save(geo, geo_path)
    if save_emb:
        _atomic_save(emb, emb_path)
    if verbose:
        print(f"  [seg_ablation] traj {traj_idx}: {len(events)} events, "
              f"{n_pix} re-encoded, {n_invalid} invalid cell-frames, "
              f"{100 * (1 - stats['edges_kept_frac']):.2f}% edges dropped")
    return geo, emb, stats


def _emb_dict(abl, companion_suffix, cache_id, traj_idx,
              idx_t, idx_n, delta_emb, delta_crop):
    """The backbone-dependent sidecar half.

    Split out because the bank path builds one in memory and never writes it —
    its rows already live in the bank — but `apply.apply_sidecar` still expects
    exactly this shape, including the `D` it checks against the companion
    width and the int32 `crop_sides` that `sparse_collate_fn` requires.
    """
    return {
        "abl_id": abl,
        "companion_suffix": companion_suffix,
        "cache_id": cache_id,
        "traj_idx": traj_idx,
        "idx_t": torch.from_numpy(np.asarray(idx_t, dtype=np.int32)),
        "idx_n": torch.from_numpy(np.asarray(idx_n, dtype=np.int32)),
        "embeddings": delta_emb,
        "crop_sides": torch.from_numpy(np.asarray(delta_crop, dtype=np.int32)),
        "K": int(np.asarray(idx_t).size),
        "D": int(delta_emb.shape[1]),
    }


def ensure_sidecars(cache_dir, cfg, *, cache_id, companion_suffix, traj_indices,
                    device=None, force=False, verbose=True):
    """Build any missing sidecars for `cfg.seg_ablation`, then return the stats.

    The encoder is created lazily and at most once across all trajectories:
    loading DINOv3 weights costs ~20 s and a fully-cached rerun should not pay
    it (the same reason `precompute_all:117-132` defers it past its own
    skip-if-exists check).
    """
    from graph_dyno.encoders.factory import create_shape_encoder

    spec = normalize_spec(cfg.seg_ablation)
    holder = {}

    def _factory():
        if "enc" not in holder:
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            print("  [seg_ablation] loading shape encoder for re-encoding...")
            holder["enc"] = create_shape_encoder(cfg, device=str(dev))
        return holder["enc"]

    out = []
    for traj_idx in traj_indices:
        _, _, stats = build_sidecar(
            cache_dir, spec, cache_id=cache_id, traj_idx=traj_idx,
            companion_suffix=companion_suffix, dt=cfg.data.dt,
            encoder_factory=_factory,
            workers=int(cfg.shape.get("sdf_workers", 32)),
            force=force, verbose=verbose)
        out.append(stats)
    holder.clear()
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_correlations(counts, sim_metadata, N):
    """Pearson r between each per-slot count and each *varied* sim parameter.

    Only varied parameters can label anything, so constants are skipped.
    Oscillation phases are excluded: they are randomised per trajectory and
    are not a clustering target.
    """
    skip = {"r_max", "area_fluct_phases", "gam_fluct_phases", "kl_fluct_phases"}
    factors = {}
    for k, v in sim_metadata.items():
        if k in skip or not isinstance(v, torch.Tensor):
            continue
        a = v.detach().cpu().numpy().ravel().astype(np.float64)
        if a.size != N or a.std() < 1e-12:
            continue
        factors[k] = a
    out = {}
    worst = None
    for cname, c in counts.items():
        c = c.astype(np.float64)
        if c.std() < 1e-12:
            continue
        for fname, f in factors.items():
            r = float(np.corrcoef(c, f)[0, 1])
            out[f"{cname}|{fname}"] = r
            if worst is None or abs(r) > abs(worst[2]):
                worst = (cname, fname, r)
    return out, worst


def _sort_rowmajor(idx):
    if idx.size == 0:
        return idx.reshape(0, 2)
    return idx[np.lexsort((idx[:, 1], idx[:, 0]))]


def _mask(images, t, n):
    """One 128x128 crop out of the mmap'd (T, N, H, W) float32 array."""
    return (images[int(t), int(n)].numpy() > 0.5).astype(np.uint8)


def _event_lookup(events):
    """(t, slot) -> Event for the slot that carries the pixel op."""
    out = {}
    for e in events:
        if e.kind == "id_swap":
            continue
        for t in range(e.t0, e.t1):
            out[(t, e.slot)] = e
    return out


def _run_pool(items, workers):
    if not items:
        return []
    if workers <= 1 or len(items) < 64:
        return [_perturb_worker(i) for i in items]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_perturb_worker, items, chunksize=32))


def assert_state_identities(states, next_states, dt, where=""):
    """The two invariants every downstream consumer silently depends on.

    `dataloader.py:643` reads `next_states[timestep_idx]` directly rather than
    `states[t+1]`, and `indices_of_mlp_features: [2, 3]` makes `x_features`
    *be* the velocity — so perturbing position while leaving velocity clean
    would hide the error from every MLP input. Both hold with exactly zero
    error on the clean caches and are rebuilt to hold exactly here.

    Index 0 carries no finite difference (it is never served as a sample) and
    is excluded from the velocity identity by the `[1:]` slice.
    """
    a = (next_states[:-1] - states[1:]).abs().max().item()
    if a != 0:
        raise AssertionError(
            f"next_states[t] != states[t+1] ({where}): max |diff| = {a}")
    pos = states[..., :2]
    pred = (pos[1:] - pos[:-1]) / float(dt)
    b = (states[1:, :, 2:4] - pred).abs().max().item()
    if b != 0:
        raise AssertionError(
            f"states[t, 2:4] != (pos[t] - pos[t-1])/dt ({where}): "
            f"max |diff| = {b}")
