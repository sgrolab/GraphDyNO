"""Encode one time-chunk of one bank kind — the only GPU step in the bank flow.

Kept separate from `bank.py` (pure identity / planning / selection) and
`bank_apply.py` (the GPU-free read path) so that the one place a shape encoder
is touched is obvious, and so the tests can drive exactly the code the cluster
worker runs rather than a re-implementation of it.

The perturbation itself is `build._perturb_worker` verbatim — same half-plane
cut, same signed morphology, same merge canvas and closing radius — so a banked
row and a directly-built row are the same corrupted crop.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from graph_dyno.seg_ablation import ops
from graph_dyno.seg_ablation.bank import (
    BANK_KINDS,
    bank_id,
    chunk_bounds,
    load_plan,
    normalize_bank_spec,
    rows_filename,
)
from graph_dyno.seg_ablation.build import ENCODE_CHUNK, _atomic_save, _mask, _run_pool
from graph_dyno.seg_ablation.plan import (
    OP_MERGE_ABSORB,
    OP_PARTIAL,
    OP_SCALE,
    SEG_ABLATION_CODE_VERSION,
)

_OP_OF_KIND = {"partial": OP_PARTIAL, "scale": OP_SCALE,
               "merge": OP_MERGE_ABSORB}


def episode_rows(plan, lo, hi):
    """(row_t, row_episode) for every frame of every episode inside [lo, hi).

    An episode spanning a chunk boundary contributes rows to both chunks and
    costs nothing: rows are per (episode, frame) and independent, and the merge
    reassembles by the stored `chunk_idx`.

    Sorted by (t, slot) so the encode reads the mmap'd raster row-major —
    measured 1.41 ms/read sorted against 1.81 unsorted on NFS.
    """
    t0 = np.asarray(plan["t0"], dtype=np.int64)
    t1 = np.asarray(plan["t1"], dtype=np.int64)
    slot = np.asarray(plan["slot"], dtype=np.int64)
    rt, re = [], []
    for i in range(t0.size):
        a, b = max(int(t0[i]), int(lo)), min(int(t1[i]), int(hi))
        if b <= a:
            continue
        rt.append(np.arange(a, b, dtype=np.int64))
        re.append(np.full(b - a, i, dtype=np.int64))
    if not rt:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    rt = np.concatenate(rt)
    re = np.concatenate(re)
    order = np.lexsort((slot[re], rt))
    return rt[order], re[order]


def encode_chunk(cache_dir, bank_spec, kind, *, cache_id, traj_idx,
                 companion_suffix, encoder, n_chunks=1, chunk_idx=0,
                 workers=32, batch_size=64, force=False, verbose=True):
    """Encode one `(traj, kind, chunk)` of a bank. Returns (path, payload).

    Skip-if-exists at both levels: an already-merged rows file, or an existing
    chunk, short-circuits — the same resume semantics as `precompute_all`, and
    written through `_atomic_save` so a killed job never leaves a truncated
    file that looks complete.
    """
    bspec = normalize_bank_spec(bank_spec)
    bid = bank_id(bspec)
    if kind not in BANK_KINDS:
        raise ValueError(f"{kind!r} needs no re-encode; bankable kinds are "
                         f"{list(BANK_KINDS)}")

    out_path = os.path.join(cache_dir, rows_filename(
        bid, kind, companion_suffix, cache_id, traj_idx, chunk_idx))
    merged_path = os.path.join(cache_dir, rows_filename(
        bid, kind, companion_suffix, cache_id, traj_idx))
    if os.path.exists(merged_path):
        if verbose:
            print(f"{os.path.basename(merged_path)} already merged")
        return merged_path, None
    if os.path.exists(out_path) and not force:
        if verbose:
            print(f"{os.path.basename(out_path)} exists, skipping")
        return out_path, None

    traj_path = os.path.join(cache_dir, f"{cache_id}_{traj_idx:04d}.pt")
    traj = torch.load(traj_path, map_location="cpu", weights_only=False,
                      mmap=True)
    images = traj["channeled_images"]
    clean_states = traj["states"]
    T = int(clean_states.shape[0])
    image_size = int(images.shape[-1])
    px_per_unit = ops.scale_px(image_size, float(traj["sim_metadata"]["r_max"]))

    plan = load_plan(cache_dir, bid, kind, cache_id, traj_idx, expect_T=T)
    pad_factor = int(bspec["kinds"]["merge"]["pad_factor"])
    close_px = float(bspec["kinds"]["merge"]["close_px"])
    canvas = pad_factor * image_size
    lo, hi = chunk_bounds(T, n_chunks, chunk_idx)

    row_t, row_ep = episode_rows(plan, lo, hi)
    slot = np.asarray(plan["slot"], dtype=np.int64)
    partner = np.asarray(plan["partner"], dtype=np.int64)
    row_n = slot[row_ep]
    K = int(row_t.size)
    if verbose:
        print(f"bank {bid} / {kind} / traj {traj_idx} / chunk {chunk_idx} "
              f"[{lo}, {hi}): {K} rows from {len(set(row_ep.tolist()))} episodes")

    op_code = _OP_OF_KIND[kind]
    emb_parts, crop_parts = [], []
    centroid = np.zeros((K, 2), dtype=np.float64)
    clipped = np.zeros(K, dtype=np.uint8)
    disagree = np.zeros(K, dtype=np.uint8)

    for a in range(0, K, ENCODE_CHUNK):
        sub = slice(a, min(a + ENCODE_CHUNK, K))
        items = []
        for t, e in zip(row_t[sub], row_ep[sub]):
            t, n, e = int(t), int(slot[e]), int(e)
            if kind == "merge":
                j = int(partner[e])
                # states[t] is the geometry the raster at index t depicts
                off = (clean_states[t, j, :2] - clean_states[t, n, :2]).numpy()
                params = {"offset_px": off.astype(np.float64) * px_per_unit}
                items.append((op_code, _mask(images, t, n), _mask(images, t, j),
                              params, px_per_unit, pad_factor, close_px))
            else:
                items.append((op_code, _mask(images, t, n), None,
                              dict(plan["params"][e]), px_per_unit, pad_factor,
                              close_px))
        results = _run_pool(items, workers)

        masks = np.zeros((len(results), canvas, canvas), dtype=np.uint8)
        local_crop = np.zeros(len(results), dtype=np.int32)
        for k, res in enumerate(results):
            masks[k] = res[0]
            local_crop[k] = res[1]
            centroid[a + k] = res[2]
            clipped[a + k] = int(res[3])
        e, c = encoder.encode_masks(torch.from_numpy(masks),
                                    batch_size=batch_size)
        emb_parts.append(e.float().cpu())
        # `_mask_to_sdf`'s own crop_side is the single source of truth (it is
        # what the clean companions store and what `scale_mse` trains against).
        crop_parts.append(c.cpu().to(torch.int32))
        disagree[a:a + len(results)] = (c.cpu().numpy() != local_crop)
        if verbose:
            print(f"  encoded {min(a + ENCODE_CHUNK, K)}/{K}", flush=True)

    # One all-zero mask: `_mask_to_sdf_worker` returns the deterministic
    # all-(-1) SDF and crop_side 0 for it, so this single row covers every
    # missed and every absorbed cell in the dataset. Cached in EVERY chunk so
    # `bank_apply` never needs an encoder — 290 concurrent training jobs each
    # loading DINOv3 to encode one zero mask is the failure this avoids.
    e0, c0 = encoder.encode_masks(
        torch.zeros((1, canvas, canvas), dtype=torch.uint8),
        batch_size=batch_size)
    if int(c0[0]) != 0:
        raise RuntimeError(
            f"empty-mask encode returned crop_side {int(c0[0])}, expected 0")
    empty_emb = e0.float().cpu()[0]

    D = int(empty_emb.shape[0])
    embeddings = (torch.cat(emb_parts, 0) if emb_parts
                  else torch.zeros((0, D), dtype=torch.float32))
    crop_sides = (torch.cat(crop_parts, 0) if crop_parts
                  else torch.zeros(0, dtype=torch.int32))
    if not torch.isfinite(embeddings).all():
        raise RuntimeError("non-finite embedding in the bank chunk")
    if embeddings.numel() and int(embeddings.shape[1]) != D:
        raise RuntimeError(f"row width {embeddings.shape[1]} != empty-mask "
                           f"width {D}")

    payload = {
        "bank_id": bid,
        "bank_name": bspec["name"],
        "code_version": SEG_ABLATION_CODE_VERSION,
        "kind": kind,
        "companion_suffix": companion_suffix,
        "cache_id": cache_id,
        "traj_idx": int(traj_idx),
        "chunk_idx": int(chunk_idx),
        "n_chunks": int(n_chunks),
        "t_lo": lo, "t_hi": hi,
        "idx_t": torch.from_numpy(row_t.astype(np.int32)),
        "idx_n": torch.from_numpy(row_n.astype(np.int32)),
        "episode_id": torch.from_numpy(row_ep.astype(np.int32)),
        "embeddings": embeddings,
        "crop_sides": crop_sides,
        # float64, NOT float32. `_finalize_sidecar` adds this to the float64
        # true position and only then rounds to float32, so a float32 round
        # trip here perturbs the last bit of `states` for whichever cells sit
        # on a rounding boundary — invisible on a 60x10 fixture, ~150k chances
        # to bite per real trajectory. It is 6 MB/trajectory.
        "centroid_delta": torch.from_numpy(centroid.astype(np.float64)),
        "clipped": torch.from_numpy(clipped),
        "crop_disagree": torch.from_numpy(disagree),
        "empty_emb": empty_emb,
        "K": K,
        "D": D,
    }
    _atomic_save(payload, out_path)
    if verbose:
        print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.0f} MB, "
              f"{K} rows, {int(clipped.sum())} clipped)")
    return out_path, payload
