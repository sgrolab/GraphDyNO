"""Graft seg-ablation sidecars onto the in-memory companion dicts.

`train.py` loads the clean companion embeddings into
`{traj_idx: {"embeddings": (T,N,D), "crop_sides": (T,N)}}`. This module turns
each of those into the *observed* dataset by scattering the delta rows and
adding the geometry keys (`states`, `next_states`, `valid`, `next_valid`,
`W_interactions`), which `dataloader.py` then grafts over the on-disk values.

The trajectory `.pt` is opened here **explicitly** (mmap) to read
`W_interactions`: the edge mask is positional over that exact `(3, nnz)` index
tensor, so `W_nnz` + `W_indices_sha1` are asserted before it is applied. A mask
silently applied to a different serialization would drop the wrong edges and
look like a plausible result.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch

from graph_dyno.seg_ablation.build import sidecar_paths
from graph_dyno.seg_ablation.plan import SEG_ABLATION_CODE_VERSION, abl_id, normalize_spec

#: keys the sidecar adds to a companion entry, i.e. the keys the dataloader
#: graft will override on the on-disk trajectory dict.
GEOMETRY_KEYS = ("states", "next_states", "valid", "next_valid", "W_interactions")


def load_sidecars(cache_dir, spec, *, cache_id, traj_idx, companion_suffix):
    """Load the two sidecars for one trajectory, or raise with a build hint."""
    abl = abl_id(spec)
    geo_path, emb_path = sidecar_paths(cache_dir, abl, companion_suffix,
                                       cache_id, traj_idx)
    for p in (geo_path, emb_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing seg-ablation sidecar {p}. Build it with\n"
                f"  python scripts/build_seg_ablation.py +experiment=<name> "
                f"+embedding=<name> +seg_ablation=<level>"
            )
    return (torch.load(geo_path, map_location="cpu", weights_only=False),
            torch.load(emb_path, map_location="cpu", weights_only=False))


def apply_sidecar(entry, geo, emb, traj_path, *, spec=None, dt=None):
    """Turn one clean companion entry into the observed one, in place.

    Args:
        entry: `{"embeddings": (T,N,D) float32, "crop_sides": (T,N) int32}`.
        geo: the backbone-independent sidecar dict.
        emb: the backbone-dependent sidecar dict.
        traj_path: path to `{cache_id}_{traj:04d}.pt` — opened mmap'd purely to
            read `W_interactions`.
        spec: optional; when given, its `abl_id` must match the sidecar's.
        dt: optional; when given, the state identities are re-asserted.

    Returns:
        The same `entry`, with the delta rows scattered in and the geometry
        keys added.
    """
    _check_identity(geo, emb, spec)

    e = entry["embeddings"]
    cs = entry["crop_sides"]
    T, N = cs.shape
    src_n = geo["src_n"]
    src_t = geo["src_t"]

    if tuple(geo["states"].shape[:2]) != (T, N):
        raise ValueError(
            f"sidecar is (T, N) = {tuple(geo['states'].shape[:2])} but the "
            f"companion is {(T, N)} — mismatched cache or trajectory")
    # No temporal-misalignment op exists yet; the identity is the artifact's
    # written record of the frame-grid convention (see plan.py).
    if not torch.equal(src_t, torch.arange(T, dtype=torch.int32)[:, None]
                       .expand(T, N).contiguous()):
        raise ValueError("src_t is not the identity — unsupported sidecar")

    # ---- 1. ID swaps are a pure gather over the *clean* rows ---------------
    ident = torch.arange(N, dtype=src_n.dtype).unsqueeze(0)
    swap_rc = torch.nonzero(src_n != ident, as_tuple=False)
    if swap_rc.numel():
        rt, rn = swap_rc[:, 0], swap_rc[:, 1]
        src = src_n[rt, rn].long()
        # Gather into a buffer first: a swap moves i -> j and j -> i in the
        # same frame, so an in-place write would read a value it had already
        # overwritten.
        e_buf = e[rt, src].clone()
        cs_buf = cs[rt, src].clone()
        e[rt, rn] = e_buf
        cs[rt, rn] = cs_buf

    # ---- 2. erode / dilate: crop_side only, no re-encode -------------------
    ov = geo["crop_sides_override"]
    has_ov = ov >= 0
    if bool(has_ov.any()):
        cs[has_ov] = ov[has_ov].to(cs.dtype)

    # ---- 3. delta rows (partial / merge / miss / absorbed) -----------------
    K = int(emb["K"])
    if K:
        it = emb["idx_t"].long()
        inn = emb["idx_n"].long()
        de = emb["embeddings"]
        if de.shape[1] != e.shape[-1]:
            raise ValueError(
                f"sidecar embedding width {de.shape[1]} != companion width "
                f"{e.shape[-1]}. The backbone-dependent sidecar carries "
                f"companion_suffix={emb.get('companion_suffix')!r}; this run "
                f"is using a different encoder.")
        e[it, inn] = de.to(e.dtype)
        cs[it, inn] = emb["crop_sides"].to(cs.dtype)

    # `precompute_dino_embeddings.py:168` stores crop_sides as int32 and
    # `torch.stack` in sparse_collate_fn fails on a dtype-mixed batch.
    if cs.dtype != torch.int32:
        raise TypeError(f"crop_sides must stay int32, got {cs.dtype}")

    # ---- 4. geometry -------------------------------------------------------
    entry["states"] = geo["states"]
    entry["next_states"] = geo["next_states"]
    entry["valid"] = geo["valid"]
    entry["next_valid"] = geo["next_valid"]
    entry["W_interactions"] = _rebuild_edges(traj_path, geo, N)

    if dt is not None:
        from graph_dyno.seg_ablation.build import assert_state_identities
        assert_state_identities(entry["states"], entry["next_states"], dt,
                                where=f"applied sidecar {geo['abl_id']}")
    return entry


def apply_all(embeddings, cache_dir, spec, *, cache_id, companion_suffix, dt,
              verbose=True):
    """Apply sidecars to every trajectory in a `_load_companion_embeddings` dict."""
    stats = []
    for traj_idx in sorted(embeddings.keys()):
        geo, emb = load_sidecars(cache_dir, spec, cache_id=cache_id,
                                 traj_idx=traj_idx,
                                 companion_suffix=companion_suffix)
        traj_path = os.path.join(cache_dir, f"{cache_id}_{traj_idx:04d}.pt")
        apply_sidecar(embeddings[traj_idx], geo, emb, traj_path,
                      spec=spec, dt=dt)
        stats.append(geo["stats"])
        if verbose:
            s = geo["stats"]
            print(f"  [seg_ablation] traj {traj_idx} ({geo['abl_id']}): "
                  f"{s['n_events']} events, {s['invalid_frac'] * 100:.2f}% "
                  f"cell-frames invalid, "
                  f"{(1 - s['edges_kept_frac']) * 100:.2f}% edges dropped")
    return stats


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _check_identity(geo, emb, spec):
    if geo["code_version"] != SEG_ABLATION_CODE_VERSION:
        raise ValueError(
            f"sidecar was built with {geo['code_version']!r} but this code is "
            f"{SEG_ABLATION_CODE_VERSION!r}. The perturbation math changed — "
            f"rebuild, do not reuse.")
    if geo["abl_id"] != emb["abl_id"]:
        raise ValueError(
            f"sidecar halves disagree: geometry {geo['abl_id']} vs embedding "
            f"{emb['abl_id']}")
    if spec is not None:
        want = abl_id(normalize_spec(spec))
        if want != geo["abl_id"]:
            raise ValueError(
                f"sidecar {geo['abl_id']} does not match the active spec "
                f"({want}); the spec changed since it was built")


def _rebuild_edges(traj_path, geo, N):
    """Drop edges on invalid slots, relabel through the ID-swap permutation.

    Edges are removed by **masking** the existing `(3, nnz)` index tensor,
    never by concatenation: `torch.stack` of sparse tensors leaves the result
    uncoalesced through `sparse_collate_fn:114` -> `gnn.py:57`, and
    `gnn.py:90`'s `scatter_add_` *sums* duplicates, so a duplicated edge would
    double its message. An ID swap only relabels endpoints, which is a
    bijection and therefore also introduces no duplicates.
    """
    traj = torch.load(traj_path, map_location="cpu", mmap=True)
    W = traj["W_interactions"]
    idx = W._indices()
    nnz = int(idx.shape[1])
    if nnz != int(geo["W_nnz"]):
        raise ValueError(
            f"{os.path.basename(traj_path)} has nnz={nnz} but the sidecar's "
            f"edge mask was built against nnz={int(geo['W_nnz'])}")
    sha1 = hashlib.sha1(np.ascontiguousarray(idx.numpy()).tobytes()).hexdigest()
    if sha1 != geo["W_indices_sha1"]:
        raise ValueError(
            f"W_interactions indices of {os.path.basename(traj_path)} hash to "
            f"{sha1[:12]} but the sidecar's positional edge mask was built "
            f"against {geo['W_indices_sha1'][:12]}. Masking it would drop the "
            f"wrong edges.")

    keep = geo["W_edge_keep"]
    kept = idx[:, keep]
    vals = W._values()[keep]

    src_n = geo["src_n"]
    ident = torch.arange(N, dtype=src_n.dtype).unsqueeze(0)
    if bool((src_n != ident).any()):
        # src_n maps observed -> true and is an involution (asserted at build
        # time), so it is also its own inverse: true -> observed.
        t = kept[0]
        kept = torch.stack([t,
                            src_n[t, kept[1]].long(),
                            src_n[t, kept[2]].long()])
        order = np.lexsort((kept[2].numpy(), kept[1].numpy(), kept[0].numpy()))
        order = torch.from_numpy(order)
        kept = kept[:, order]
        vals = vals[order]

    return torch.sparse_coo_tensor(kept, vals, W.shape, is_coalesced=True)
