"""Event planning for segmentation / tracking ablations.

Pure module: no image I/O, no filesystem, no torch autograd. Given a spec, the
trajectory length, the cell count and the contact graph, produce the list of
observation events plus the `(T, N)` index/op maps the builder consumes.

Frame-grid convention (stated once, applied everywhere)
-------------------------------------------------------
`polyhoop_simulation.py` records `states[k]`, `channeled_images[k]` and
`W_interactions[k]` at the same instant, so the `(t, n)` grid used here is both
**the dataloader's pairing** — index `t` means "the observation `__getitem__`
serves at `timestep_idx = t`" — and the physical frame. Merge offsets and
adjacency therefore read `states[t]` and `W_interactions[t]`, the geometry the
raster at index `t` depicts.

Stratification
--------------
Events are stratified over **two roles**, per kind, because two distinct
confounds live here:

1. *Initiator imbalance.* Under i.i.d. sampling per-slot event counts are
   Binomial(T, rate), a ~10% spread in effective training signal per slot,
   injected straight into the per-slot `parameter_embedding` the metric
   clusters.
2. *Partner imbalance, which is label-correlated.* Merge/swap partners come
   from `W_interactions`, and contact degree is a function of adhesion
   (measured on the kh1e5 cache: mean per-slot degree 3.008 at kh=1e4 vs 3.805
   at kh=1e5, pearson r(degree, log10 kh) = 0.438). A high-kh cell would be
   absorbed ~27% more often, making per-slot supervised-transition count a
   monotone function of the factor being clustered.

So per-slot counts are held to a uniform cap in the initiator role *and* the
partner role, per kind, and filled round-robin; only *which* frames and *which*
pairing vary. Feasibility is respected: a slot that never touches another cell
cannot host a merge or a swap in either role, and is excluded (and reported).

Exact "spread <= 1" balance is nevertheless not always attainable, and not
because of the sampler: the role assignment is a degree-constrained bipartite
b-matching whose feasibility is a Hall condition on the contact graph, and that
condition does not depend on the rate. `more_osc_clusters` has 122 degree-1
slots among 379 with any contact at all (union-distinct partners per cell: 1.8,
min 0), so several share a single possible partner and NO positive rate admits
an exactly balanced assignment there. membrane_stiffness/adhesion, at 22-25 partners per
cell, are fine. What is therefore *enforced* is the harm rather than the proxy:
`build.py` correlates the realized per-slot counts against every varied
simulation parameter and refuses the build when a per-slot event or supervision
count turns out to be a monotone function of the factor being clustered.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Bump whenever the perturbation math changes. The sidecar filename carries
# only `abl_id`, so an unbumped change would silently serve the wrong artifact
# — the lesson of `_assert_sdf_target_size` (precompute_embeddings.py:191).
#
#   v1 -> v2: erode/dilate is re-encoded instead of being applied as a
#             crop_side override. Measured on the live SP embedding (192
#             probe cells, between-cell reference ||E_i - E_j|| = 6.82):
#
#                 delta_px   ||dE||/ref   SDF MAE   d crop_side
#                      0.5        0.000   0.00000             0
#                      1.0        0.000   0.00000             0
#                      2.0        0.328   0.01048             2
#                      3.0        0.429   0.02032             4
#                      4.0        0.408   0.01885             6
#
#             The SDF MAE really is tiny — `_mask_to_sdf`'s tight-bbox crop and
#             resize to 224 normalise a uniform morphological offset almost
#             away, reproducing the 0.012 / 0.026 figures the design was based
#             on. But DINO is sensitive to what survives: a third to nearly
#             half of the between-cell distance is NOT a no-op, and skipping
#             the re-encode would have made the `scale_only` arm report
#             "boundary error barely matters" when in fact most of the
#             perturbation had been discarded. (delta <= 1 px is a literal
#             no-op at 128 px resolution — `sdf > -delta` excludes the
#             distance-1 ring — which the pre-flight gate now flags.)
#
#             v2 also made `morph_mask` symmetric in sign and added the
#             MIN_EPISODE_GAP guard band. No v2 artifact predates either, so
#             they are folded into the same version rather than given their own.
SEG_ABLATION_CODE_VERSION = "segabl-v2"

KINDS = ("merge", "id_swap", "miss", "partial", "scale")

#: Kinds needing a partner slot drawn from the contact graph.
PAIR_KINDS = ("merge", "id_swap")

#: Kinds whose pixels change and therefore need a GPU re-encode. `scale` joined
#: in v2 — see the SEG_ABLATION_CODE_VERSION note.
PIXEL_KINDS = ("partial", "merge", "scale")

# op codes written into the (T, N) uint8 map. Ordered by precedence: a higher
# code wins if two events ever landed on the same (t, n). `plan_events`
# guarantees they never do; `build_index_maps` enforces it anyway.
OP_NONE = 0
OP_SCALE = 1          # erode/dilate — re-encode (v2; see the version note)
OP_PARTIAL = 2        # half-plane cut — re-encode
OP_MERGE_ABSORB = 3   # absorber gets the blended blob — re-encode
OP_MERGE_GONE = 4     # absorbed cell has no mask this frame — invalid
OP_MISS = 5           # dropped detection — invalid

OP_NAMES = {
    OP_NONE: "none",
    OP_SCALE: "scale",
    OP_PARTIAL: "partial",
    OP_MERGE_ABSORB: "merge_absorb",
    OP_MERGE_GONE: "merge_gone",
    OP_MISS: "miss",
}

#: ops that zero the mask and mark the slot invalid.
INVALID_OPS = (OP_MERGE_GONE, OP_MISS)

_KIND_DEFAULTS: dict[str, dict[str, Any]] = {
    # Shape errors are temporally persistent — a 1-frame corruption is a
    # high-frequency impulse a temporal model can largely reject, so mean_len
    # is exposed per kind and defaults > 1 for shape.
    "partial": {"rate": 0.0, "mean_len": 8.0,
                "area_frac_min": 0.55, "area_frac_max": 0.85},
    # Merges default to iid (mean_len 1): a merge is a per-frame association
    # failure, not a persistent mis-segmentation.
    "merge": {"rate": 0.0, "mean_len": 1.0, "pad_factor": 2,
              # A merge IS a lost membrane, so the union is closed into one
              # connected region. Measured: the raw union is already connected
              # 99.7% of the time; 1 px takes it to 100% for +0.1% area.
              "close_px": 1.0},
    "miss": {"rate": 0.0, "mean_len": 3.0},
    "id_swap": {"rate": 0.0, "mean_len": 40.0},
    "scale": {"rate": 0.0, "mean_len": 8.0, "delta_px": 2.0},
}

_SPEC_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "seed": 0,
    # sigma_px = sigma_rel * median true per-frame displacement in px. See the
    # centroid-noise note in configs/seg_ablation/README-style comments: the
    # sim frame interval is far shorter than a microscopy frame interval, which
    # is this study's main domain-mismatch caveat.
    "centroid_noise_sigma_rel": 0.0,
    # Hard cap on a sampled episode length (frames).
    "max_episode_len": 200,
    # Refuse to build when a realized per-slot count correlates with a varied
    # simulation parameter above `max_label_correlation` (see _check_balance
    # and build._label_correlations). This is a REGRESSION GUARD against losing
    # the stratification, not a claim of zero confound. Measured |r| between
    # per-slot merge counts and the clustered factor, at the `moderate` rate:
    #     dataset      i.i.d. edge sampling   stratified (this planner)
    #     more_osc            0.13                    0.15
    #     membrane_stiffness             0.04                    0.08
    #     adhesion            0.07                    0.13
    # i.e. on these three paper caches contact degree barely tracks the label
    # either way, and the residual is dominated by which slots the contact
    # graph *strands* rather than by degree bias. On the kh1e5 cache the plan
    # cites, r(degree, log10 kh) = 0.438 — that is the case this guard exists
    # to catch.
    "strict_balance": True,
    "max_label_correlation": 0.25,
    # When True the corruption is a property of the CELL, not of the
    # (cell, trajectory) pair: slot 20 takes the same partial cut, at the same
    # frames, in every trajectory.
    #
    # Why it matters: `parameter_embedding` is per-SLOT and shared across
    # trajectories (one vector per cell id, `model.num_nodes`). Under the
    # default per-trajectory draw, a cell corrupted in trajectory 0 is clean in
    # 1 and 2, so its embedding still sees mostly-clean supervision and the
    # corruption averages out — which is why the membrane_stiffness ladder came back flat
    # at ARI 1.000 all the way to 4x. Sharing removes that escape hatch.
    #
    # EXACT for the single-slot kinds (partial, miss, scale) — identical t0,
    # duration and params in every trajectory. BEST-EFFORT for merge/id_swap:
    # they need a partner adjacent for every frame of the episode, and the
    # contact graph genuinely differs between trajectories, so the partner (and
    # sometimes the placement) cannot be shared. `build_sidecar` reports the
    # achieved cross-trajectory identity per kind.
    "shared_across_trajectories": False,
    # Name of a `configs/seg_bank/<name>.yaml` embedding bank, or None.
    #
    # When set, the pixel kinds are NOT planned from scratch: they are a nested
    # sub-selection of a pre-encoded superset (see seg_ablation/bank.py), so a
    # whole dose ladder costs one GPU wave instead of one per level. The field
    # is part of the hashed payload, and deliberately so — a bank-derived
    # sidecar and a directly-built one of otherwise identical spec are NOT the
    # same artifact (different episode placements, different `abl_id`), and
    # letting them share a filename is exactly the silent-collision class the
    # `abl_id` + code-version pair exists to prevent.
    "bank": None,
    # Are errors temporally PERSISTENT (a "bad cell" that stays bad for a
    # geometric run of `mean_len` frames), or independent per cell-frame?
    #
    # True (the default, and what every published arm used) is the realistic
    # model of a mis-segmentation: a cell cut on one axis stays cut while the
    # cause persists, and `_sample_durations` draws the run length. That
    # persistence is the point of the single-kind ladders — a 1-frame impulse
    # is something a temporal model can largely reject, so an episodic error
    # is the harder and more honest test.
    #
    # False makes the selection unit the CELL-FRAME instead of the episode,
    # which is what a BALANCED JOINT arm needs. Under `strict_balance` an
    # episodic level can only add whole episodes to every cell at once, so its
    # realized rate is quantized in steps of `mean_len / T` — 0.004 for
    # partial/scale (mean_len 8, T 2000) and 0.020 for id_swap (mean_len 40).
    # A four-kind arm balanced to 1% of cell-frames wants 0.25% per kind and is
    # therefore unreachable: the floor is ~1.4%. Frame selection drops the
    # quantum to `1 / T` = 0.0005 for every kind, so the rate becomes
    # effectively continuous and `rate` means what it says (frames per cell
    # = rate * T) instead of needing numerical inversion.
    #
    # Bank-backed levels only — the banked rows are already per cell-frame
    # (`idx_t`/`idx_n`), so this costs no re-encode. The direct builder in
    # `build.py` plans episodes from scratch and rejects it.
    "episodic": True,
}

#: `_SPEC_DEFAULTS` entries dropped from the `abl_id` payload when they hold
#: this exact value. Added AFTER 340 runs were on disk; including a new key
#: unconditionally would re-key every existing level and orphan every sidecar
#: and every logged `abl_id`. A spec that leaves the key at its legacy default
#: therefore hashes byte-identically to the same spec written before the key
#: existed. `tests/test_seg_ablation_bank.py::test_committed_abl_ids_are_frozen`
#: pins all 54 of them.
_ABL_ID_LEGACY_DEFAULTS: dict[str, Any] = {"episodic": True}


@dataclass(frozen=True)
class Event:
    """One observation-error episode.

    Attributes:
        kind: one of KINDS.
        t0, t1: half-open frame range on the dataloader's `(t, n)` grid.
        slot: the initiator slot (absorber for merge, the swapped-into slot
            for id_swap, the affected cell otherwise).
        partner: the partner slot for PAIR_KINDS, else -1.
        params: episode-constant scalars (axis angle, area fraction, signed
            morphological offset). Frozen for the whole episode so shape
            errors are temporally persistent rather than flickering.
    """

    kind: str
    t0: int
    t1: int
    slot: int
    partner: int = -1
    params: dict[str, float] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.t1 - self.t0


# ---------------------------------------------------------------------------
# Spec normalisation + identity
# ---------------------------------------------------------------------------

def normalize_spec(spec) -> dict:
    """Fill defaults and coerce to a plain, JSON-ish dict.

    Accepts an OmegaConf node or a plain mapping. Unknown keys are a hard
    error — a typo'd `rate` silently meaning "no events" is exactly the kind of
    thing that produces a whole ablation ladder of identical runs.
    """
    from omegaconf import DictConfig, OmegaConf

    if spec is None:
        raise ValueError("seg_ablation spec is None")
    if isinstance(spec, DictConfig):
        spec = OmegaConf.to_container(spec, resolve=True)
    if not isinstance(spec, dict):
        raise TypeError(f"seg_ablation spec must be a mapping, got {type(spec)}")

    out = dict(_SPEC_DEFAULTS)
    kinds_in = spec.get("kinds", {}) or {}

    unknown = set(spec) - set(_SPEC_DEFAULTS) - {"kinds"}
    if unknown:
        raise ValueError(
            f"Unknown seg_ablation keys {sorted(unknown)}. "
            f"Known: {sorted(set(_SPEC_DEFAULTS) | {'kinds'})}"
        )
    for k, v in spec.items():
        if k != "kinds":
            out[k] = v

    unknown_kinds = set(kinds_in) - set(KINDS)
    if unknown_kinds:
        raise ValueError(
            f"Unknown seg_ablation kinds {sorted(unknown_kinds)}. Known: {list(KINDS)}"
        )

    kinds = {}
    for kind in KINDS:
        entry = dict(_KIND_DEFAULTS[kind])
        given = kinds_in.get(kind, {}) or {}
        unknown_fields = set(given) - set(entry)
        if unknown_fields:
            raise ValueError(
                f"Unknown fields {sorted(unknown_fields)} for seg_ablation kind "
                f"{kind!r}. Known: {sorted(entry)}"
            )
        entry.update(given)
        entry["rate"] = float(entry["rate"])
        entry["mean_len"] = float(entry["mean_len"])
        if entry["rate"] < 0 or entry["rate"] > 1:
            raise ValueError(f"seg_ablation.kinds.{kind}.rate must be in [0, 1]")
        if entry["mean_len"] < 1:
            raise ValueError(f"seg_ablation.kinds.{kind}.mean_len must be >= 1")
        kinds[kind] = entry
    out["kinds"] = kinds

    out["seed"] = int(out["seed"])
    out["centroid_noise_sigma_rel"] = float(out["centroid_noise_sigma_rel"])
    out["max_episode_len"] = int(out["max_episode_len"])
    out["strict_balance"] = bool(out["strict_balance"])
    out["max_label_correlation"] = float(out["max_label_correlation"])
    out["enabled"] = bool(out["enabled"])
    out["shared_across_trajectories"] = bool(out["shared_across_trajectories"])
    out["episodic"] = bool(out["episodic"])
    out["bank"] = None if out["bank"] is None else str(out["bank"])
    if not out["episodic"] and out["bank"] is None:
        # Frame selection sub-selects the frames of a banked episode. Without a
        # bank there is nothing to sub-select: `plan_events` places episodes
        # directly, and honouring the flag would mean a second planner.
        raise ValueError(
            "seg_ablation.episodic: false requires seg_ablation.bank — frame "
            "selection sub-selects banked cell-frames (see seg_ablation/"
            "bank.py:select_frames). The direct builder plans episodes.")
    if out["bank"] is not None and out["shared_across_trajectories"]:
        # A bank is planned per trajectory precisely so a given cell does not
        # receive the same corruption everywhere; sharing is the exact opposite
        # request and would silently be ignored by the bank path.
        raise ValueError(
            "seg_ablation.shared_across_trajectories is not supported with "
            "seg_ablation.bank: banks are planned per trajectory by design "
            "(see seg_ablation/bank.py)")
    return out


def per_slot_counts(events, N):
    """Realized per-slot counts, per (kind, role) — the stratification audit.

    Returns {f"{kind}_{role}": (N,) int array} with role in
    {"initiator", "partner"} ("partner" present only for PAIR_KINDS).
    """
    out = {}
    for e in events:
        key = f"{e.kind}_initiator"
        out.setdefault(key, np.zeros(N, dtype=np.int64))[e.slot] += 1
        if e.partner >= 0:
            key = f"{e.kind}_partner"
            out.setdefault(key, np.zeros(N, dtype=np.int64))[e.partner] += 1
    return out


def spec_is_active(spec) -> bool:
    """True when the normalized spec would actually perturb anything."""
    s = normalize_spec(spec)
    if not s["enabled"]:
        return False
    if s["centroid_noise_sigma_rel"] > 0:
        return True
    return any(k["rate"] > 0 for k in s["kinds"].values())


def abl_id(spec) -> str:
    """8-char identity of a spec *and* the perturbation code that realises it.

    The sidecar filename carries nothing else, so both halves matter: bumping
    SEG_ABLATION_CODE_VERSION invalidates every artifact whose math changed.
    """
    from omegaconf import OmegaConf

    s = normalize_spec(spec)
    for k, legacy in _ABL_ID_LEGACY_DEFAULTS.items():
        if s.get(k) == legacy:
            s.pop(k)
    payload = OmegaConf.to_yaml(OmegaConf.create(s), sort_keys=True)
    h = hashlib.md5()
    h.update(payload.encode())
    h.update(SEG_ABLATION_CODE_VERSION.encode())
    return h.hexdigest()[:8]


# ---------------------------------------------------------------------------
# Contact-graph preprocessing
# ---------------------------------------------------------------------------

def _edge_arrays(W_interactions) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical (t, i, j) arrays with i < j, deduplicated.

    The cache stores W symmetrically (verified: every (i,j) has its (j,i)), so
    half the entries are redundant for partner selection.
    """
    import torch

    if isinstance(W_interactions, torch.Tensor):
        if not W_interactions.is_sparse:
            raise TypeError("W_interactions must be a sparse COO tensor")
        idx = W_interactions._indices().cpu().numpy()
    else:
        idx = np.asarray(W_interactions)
    t, i, j = idx[0], idx[1], idx[2]
    keep = i < j
    return t[keep].astype(np.int64), i[keep].astype(np.int64), j[keep].astype(np.int64)


def _contact_runs(t, i, j, T, N):
    """Maximal runs of consecutive frames during which a pair stays in contact.

    Returns:
        runs_by_slot: list of length N; entry n is (partner, start, length),
            each a numpy array, sorted ascending by `length` so a duration
            filter is one `searchsorted`.
        has_contact: (T, N) bool — slot has at least one neighbour this frame.
    """
    has_contact = np.zeros((T, N), dtype=bool)
    if t.size:
        has_contact[t, i] = True
        has_contact[t, j] = True

    if t.size == 0:
        empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64))
        return [empty for _ in range(N)], has_contact

    key = i * N + j
    order = np.lexsort((t, key))
    key_s, t_s = key[order], t[order]

    new = np.empty(key_s.size, dtype=bool)
    new[0] = True
    new[1:] = (key_s[1:] != key_s[:-1]) | (t_s[1:] != t_s[:-1] + 1)
    starts = np.flatnonzero(new)
    lengths = np.diff(np.append(starts, key_s.size))
    run_t0 = t_s[starts]
    run_key = key_s[starts]
    run_i = run_key // N
    run_j = run_key % N

    # Each run is usable from both endpoints.
    src = np.concatenate([run_i, run_j])
    dst = np.concatenate([run_j, run_i])
    st = np.concatenate([run_t0, run_t0])
    ln = np.concatenate([lengths, lengths])

    order2 = np.lexsort((ln, src))          # group by slot, ascending length
    src, dst, st, ln = src[order2], dst[order2], st[order2], ln[order2]
    bounds = np.searchsorted(src, np.arange(N + 1))
    runs_by_slot = [
        (dst[bounds[n]:bounds[n + 1]], st[bounds[n]:bounds[n + 1]],
         ln[bounds[n]:bounds[n + 1]])
        for n in range(N)
    ]
    return runs_by_slot, has_contact


# ---------------------------------------------------------------------------
# Quota helpers
# ---------------------------------------------------------------------------

def _largest_remainder(total: int, feasible: np.ndarray, rng=None) -> np.ndarray:
    """Spread `total` over the feasible slots so counts differ by at most 1.

    Which slots get the extra one is chosen at *random* (seeded), never lowest
    index first. Slot index is not label-neutral: the paper configs lay factors
    out with `dist: {steps, repeats}`, and a `block: repeats` layout would put
    whole groups in contiguous index ranges. Taking the lowest `rem` indices
    would then hand the extra event to one group systematically. (The three
    caches here happen to be interleaved — measured r(index, label) = 0.008 /
    0.003 — but that is a property of those configs, not of this function.)
    """
    n_feasible = int(feasible.sum())
    counts = np.zeros(feasible.size, dtype=np.int64)
    if n_feasible == 0 or total <= 0:
        return counts
    base, rem = divmod(total, n_feasible)
    idx = np.flatnonzero(feasible)
    counts[idx] = base
    if rem:
        pick = (rng.permutation(idx)[:rem] if rng is not None else idx[:rem])
        counts[pick] += 1
    return counts


def _waterfill(total: int, caps: np.ndarray, feasible: np.ndarray,
               rng=None) -> np.ndarray:
    """Equal quotas, except where a slot's own capacity forbids it.

    Perfect balance is not always available: on `more_osc_clusters` a cell
    touches 1.8 distinct neighbours across the *whole* 2000-frame trajectory
    (min 0), against 22-25 on membrane_stiffness/adhesion. Rather than silently skewing,
    saturate the capacity-limited slots at their cap and split the remainder
    equally over the rest — then report which slots were capped, so the
    residual imbalance is visible instead of hidden.
    """
    quota = np.zeros(caps.size, dtype=np.int64)
    free = feasible & (caps > 0)
    remaining = int(total)
    while remaining > 0 and free.any():
        level = remaining // int(free.sum())
        if level == 0:
            # Fewer episodes than slots: hand out one each by largest
            # remainder. Returning all-zero here would silently disable the
            # kind whenever `rate * T * N / mean_len < N` — which is exactly
            # the id_swap regime (rate 0.002, mean_len 40).
            quota += _largest_remainder(remaining, free, rng)
            break
        sat = free & (caps <= level)
        if not sat.any():
            add = _largest_remainder(remaining, free, rng)
            over = free & (quota + add > caps)
            if not over.any():
                quota += add
                remaining = 0
                break
            sat = over
            # fall through: saturate the over-cap slots and re-level the rest
        quota[sat] = caps[sat]
        remaining -= int(caps[sat].sum())
        free = free & ~sat
    return np.minimum(quota, caps)


def _sample_durations(rng, n, mean_len, max_len):
    """Geometric(1/mean_len) on {1, 2, ...}, clipped to `max_len`."""
    if mean_len <= 1.0:
        return np.ones(n, dtype=np.int64)
    d = rng.geometric(1.0 / mean_len, size=n)
    return np.clip(d, 1, max_len).astype(np.int64)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@dataclass
class PlanReport:
    """What the planner actually achieved, for the build-time print + MLflow."""

    per_kind: dict[str, dict[str, Any]] = field(default_factory=dict)
    infeasible_slots: dict[str, list[int]] = field(default_factory=dict)

    def as_flat(self) -> dict[str, float]:
        out = {}
        for kind, d in self.per_kind.items():
            for k, v in d.items():
                if isinstance(v, (int, float)):
                    out[f"segabl_{kind}_{k}"] = float(v)
        return out


def plan_events(spec, *, T, N, states=None, W_interactions=None, seed=None):
    """Plan every observation-error episode for one trajectory.

    Args:
        spec: raw or normalized seg_ablation spec.
        T, N: trajectory length and slot count on the dataloader's grid.
        states: (T, N, 4) — the geometry the raster at index t depicts.
            Used only for reporting merge-offset statistics; the plan itself is
            a pure function of the contact graph.
        W_interactions: sparse (T, N, N) contact graph, or a (3, nnz) index
            array. Required whenever a PAIR_KIND has a nonzero rate.
        seed: overrides `spec.seed` (used to decorrelate trajectories).

    Returns:
        (events, report)
    """
    s = normalize_spec(spec)
    rng = np.random.default_rng(s["seed"] if seed is None else seed)
    max_len = int(s["max_episode_len"])
    report = PlanReport()

    need_pairs = any(s["kinds"][k]["rate"] > 0 for k in PAIR_KINDS)
    if need_pairs and W_interactions is None:
        raise ValueError("W_interactions is required when merge/id_swap rate > 0")

    if W_interactions is not None:
        te, ie, je = _edge_arrays(W_interactions)
        runs_by_slot, has_contact = _contact_runs(te, ie, je, T, N)
    else:
        runs_by_slot, has_contact = None, None

    occ = np.zeros((T, N), dtype=bool)
    events: list[Event] = []

    shared = s["shared_across_trajectories"]

    # Ordering. Default: pair kinds first, because they are the most
    # constrained (a partner must be adjacent for *every* frame of the episode)
    # and so deserve first pick of frames.
    #
    # Under `shared_across_trajectories` that priority is inverted, and it has
    # to be. The single-slot kinds are the ones that CAN be shared exactly, but
    # only if they see the same free-frame landscape everywhere — and pair
    # placement is trajectory-specific, so letting it mark `occ` first would
    # perturb every later `_find_free_window` and destroy the exactness. Shape
    # kinds therefore go first, on empty occupancy, and the pair kinds take
    # what is left. The cost is real: merge/id_swap lose first pick and may
    # achieve fewer episodes than under the default ordering.
    order = (("miss", "partial", "scale", "merge", "id_swap") if shared
             else ("merge", "id_swap", "miss", "partial", "scale"))

    def _kind_rng(kind):
        """Per-kind stream, so one kind's draw count cannot shift another's.

        With a single shared stream the pair kinds consume a
        trajectory-dependent number of draws, which would desynchronise every
        kind planned after them even at an identical seed.
        """
        if not shared:
            return rng
        base = s["seed"] if seed is None else seed
        return np.random.default_rng(
            [int(base), 1 + KINDS.index(kind)])

    for kind in order:
        cfg = s["kinds"][kind]
        if cfg["rate"] <= 0:
            report.per_kind[kind] = {"n_episodes": 0, "realized_rate": 0.0}
            continue
        k_rng = _kind_rng(kind)
        if kind in PAIR_KINDS:
            ev, rep, infeasible = _plan_pair_kind(
                kind, cfg, k_rng, T=T, N=N, occ=occ,
                runs_by_slot=runs_by_slot, has_contact=has_contact,
                max_len=max_len, strict=s["strict_balance"],
            )
            report.infeasible_slots[kind] = infeasible
        else:
            ev, rep = _plan_single_kind(
                kind, cfg, k_rng, T=T, N=N, occ=occ, max_len=max_len,
                strict=s["strict_balance"],
            )
        events.extend(ev)
        report.per_kind[kind] = rep

    if states is not None and report.per_kind.get("merge", {}).get("n_episodes", 0):
        report.per_kind["merge"]["mean_offset_units"] = _merge_offset_stat(
            events, states)

    events.sort(key=lambda e: (e.t0, e.slot, e.kind))
    return events, report


#: Salt for the per-slot rank stream. Any fixed odd constant would do; it only
#: has to be distinct from the `+ 977` the centroid-noise stream uses.
_RANK_SALT = 7919


def plan_bank(kind, kind_cfg, *, T, N, states=None, W_interactions=None,
              seed, max_episode_len=200, strict_balance=True):
    """Plan ONE kind's dense superset, ranked so every lower rate nests inside.

    This is `plan_events` for a single kind on empty occupancy, plus the two
    fields that make a *sub-selection* well-behaved:

        rank        0..m-1 within the episode's own initiator slot, shuffled by
                    a per-slot RNG.
        slot_total  m, that slot's episode count.

    Selecting `rank < round(rho / rho_bank * slot_total)` then gives

      * **strict nesting** — the selection at a lower rate is a subset of the
        selection at a higher one, so the four doses of a ladder are literally
        d1 subset d2 subset d3 subset d4 rather than four independent draws;
      * **per-slot balance for free** — `_largest_remainder` already holds
        `slot_total` to a spread of 1, and rounding a common ratio against two
        integers differing by 1 also differs by at most 1;
      * **MIN_EPISODE_GAP inherited** — the superset is placed with the guard
        band, and any subset of a gap-respecting placement still respects it.

    Ranking is by INITIATOR slot even for pair kinds. Partner-role balance is
    therefore only approximately preserved by a sub-selection; the criterion
    that is actually enforced downstream is the label correlation
    (`build._label_correlations`), which is checked per materialized level.

    Args:
        kind: one of KINDS.
        kind_cfg: that kind's entry of a normalized spec (rate, mean_len, ...).
        seed: RNG seed. Callers pass a per-(bank, trajectory, kind) value so
            that a cell is corrupted *differently* in every trajectory.

    Returns:
        (events, report, rank, slot_total, infeasible) — `events` sorted by
        (t0, slot), `rank`/`slot_total` aligned to it, `infeasible` the
        contact-graph-excluded slots (empty for single-slot kinds).
    """
    if kind not in KINDS:
        raise ValueError(f"unknown seg_ablation kind {kind!r}")
    rng = np.random.default_rng(seed)
    occ = np.zeros((T, N), dtype=bool)

    if kind in PAIR_KINDS:
        if W_interactions is None:
            raise ValueError(f"W_interactions is required to plan a {kind} bank")
        te, ie, je = _edge_arrays(W_interactions)
        runs_by_slot, has_contact = _contact_runs(te, ie, je, T, N)
        events, rep, infeasible = _plan_pair_kind(
            kind, kind_cfg, rng, T=T, N=N, occ=occ, runs_by_slot=runs_by_slot,
            has_contact=has_contact, max_len=int(max_episode_len),
            strict=bool(strict_balance))
    else:
        events, rep = _plan_single_kind(
            kind, kind_cfg, rng, T=T, N=N, occ=occ,
            max_len=int(max_episode_len), strict=bool(strict_balance))
        infeasible = []

    if kind == "merge" and states is not None and rep.get("n_episodes"):
        rep["mean_offset_units"] = _merge_offset_stat(events, states)

    # Episode identity is the index into THIS list, so the order has to be a
    # pure function of the plan, not of placement order (which depends on the
    # round-robin pass structure).
    events.sort(key=lambda e: (e.t0, e.slot, e.partner))
    rank, slot_total = rank_episodes(events, N, seed)
    return events, rep, rank, slot_total, infeasible


def rank_episodes(events, N, seed):
    """Per-initiator-slot rank (shuffled) and per-slot episode total."""
    by_slot: dict[int, list[int]] = {}
    for i, e in enumerate(events):
        by_slot.setdefault(int(e.slot), []).append(i)
    rank = np.zeros(len(events), dtype=np.int32)
    slot_total = np.zeros(len(events), dtype=np.int32)
    for slot, idxs in by_slot.items():
        # Per-SLOT stream: an episode's rank must not depend on how many
        # episodes other slots happened to receive, or the nesting of one slot
        # would shift when an unrelated slot's count changed.
        r = np.random.default_rng([int(seed), _RANK_SALT, int(slot)])
        order = r.permutation(len(idxs))
        for k, i in enumerate(idxs):
            rank[i] = int(order[k])
            slot_total[i] = len(idxs)
    return rank, slot_total


def _plan_single_kind(kind, cfg, rng, *, T, N, occ, max_len, strict):
    """Per-slot episodes with exactly-balanced counts (spread <= 1)."""
    mean_len = cfg["mean_len"]
    n_ep = int(round(cfg["rate"] * T * N / mean_len))
    feasible = np.ones(N, dtype=bool)
    quota = _largest_remainder(n_ep, feasible, rng)

    events = []
    placed = np.zeros(N, dtype=np.int64)
    for _ in range(int(quota.max()) + EXTRA_PLACEMENT_PASSES if quota.any() else 0):
        progressed = False
        for n in np.flatnonzero(quota - placed > 0).tolist():
            d = int(_sample_durations(rng, 1, mean_len, max_len)[0])
            t0 = _find_free_window(rng, occ, [n], d, T)
            if t0 is None:
                continue
            occ[t0:t0 + d, n] = True
            events.append(Event(kind=kind, t0=t0, t1=t0 + d, slot=n,
                                params=_episode_params(kind, cfg, rng)))
            placed[n] += 1
            progressed = True
        if not progressed:
            break

    covered = int(sum(e.length for e in events))
    spread = _spread(placed, quota > 0)
    rep = {
        "n_episodes": len(events),
        "n_requested": n_ep,
        "n_quota": int(quota.sum()),
        "realized_rate": covered / float(T * N),
        "mean_len": covered / max(len(events), 1),
        "slot_count_spread": spread,
        "n_feasible_slots": int(feasible.sum()),
        "shortfall": int((quota - placed).clip(min=0).sum()),
    }
    _check_balance(kind, "initiator", spread, strict)
    return events, rep


#: extra placement passes beyond the largest per-slot quota. Placement is
#: round-robin — one episode per slot per pass — because filling each slot to
#: quota before moving on drains its neighbours' partner budgets and strands
#: the slots that come last, even on a 2-regular ring where a perfectly
#: balanced assignment exists.
EXTRA_PLACEMENT_PASSES = 8


def _plan_pair_kind(kind, cfg, rng, *, T, N, occ, runs_by_slot, has_contact,
                    max_len, strict):
    """Episodes over adjacent pairs, balanced in BOTH roles.

    A partner must be adjacent for **every** frame of the episode, so the
    candidate set for an initiator is its contact *runs* of length >= duration.
    Placement is deficit-driven and most-constrained-first: slots with the
    fewest distinct possible partners are served before slots with many, and
    the chosen partner is always the one with the most role budget left. That
    ordering is what makes exact balance reachable on a sparse contact graph,
    where a naive shuffle strands the low-degree slots.
    """
    mean_len = cfg["mean_len"]
    feasible = has_contact.any(axis=0)
    infeasible = np.flatnonzero(~feasible).tolist()
    n_ep = int(round(cfg["rate"] * T * N / mean_len))
    n_free = int(feasible.sum())
    if n_free == 0 or n_ep == 0:
        return [], {"n_episodes": 0, "n_requested": n_ep, "n_quota": 0,
                    "realized_rate": 0.0, "mean_len": 0.0,
                    "slot_count_spread": 0, "partner_count_spread": 0,
                    "n_feasible_slots": n_free,
                    "n_infeasible_slots": len(infeasible),
                    "n_capacity_capped_slots": 0, "shortfall": n_ep,
                    "n_zero_event_slots": n_free}, infeasible

    # A slot spends `mean_len` frames per episode, in either role, and can only
    # do so on a frame where it has a neighbour.
    contact_frames = has_contact.sum(axis=0)
    cap_role = np.floor(contact_frames / (2.0 * mean_len)).astype(np.int64)
    # A uniform per-role CAP rather than a pre-assigned per-slot quota. With a
    # quota, `n_ep < N` (the id_swap regime: rate 0.002, mean_len 40 -> 50
    # episodes over 379 feasible slots) hands the initiator and partner budgets
    # to two *independently chosen* small subsets that are almost never
    # adjacent, and the kind silently plans ZERO episodes. A cap plus a
    # round-robin fill that stops at `n_ep` keeps counts within 1 of each other
    # in both regimes and lets the contact graph choose the pairs.
    level = int(np.ceil(n_ep / n_free))
    cap = np.minimum(level, np.maximum(cap_role, 0)) * feasible
    capped = feasible & (cap < level)

    used_init = np.zeros(N, dtype=np.int64)
    used_part = np.zeros(N, dtype=np.int64)
    degree = np.array([np.unique(runs_by_slot[n][0]).size for n in range(N)])

    events = []
    n_passes = level + EXTRA_PLACEMENT_PASSES
    for _ in range(n_passes):
        progressed = False
        for role in ("init", "part"):
            if len(events) >= n_ep:
                break
            used = used_init if role == "init" else used_part
            order = np.flatnonzero(cap - used > 0)
            if order.size == 0:
                continue
            # most constrained (fewest distinct partners) first, random ties
            order = order[np.lexsort((rng.random(order.size), degree[order]))]
            for n in order.tolist():
                if len(events) >= n_ep:
                    break
                # ONE episode per slot per pass — see EXTRA_PLACEMENT_PASSES.
                d = int(_sample_durations(rng, 1, mean_len, max_len)[0])
                other_used = used_part if role == "init" else used_init
                got = _try_place_pair(rng, occ, runs_by_slot[n], n, d, T,
                                      cap - other_used)
                if got is None:
                    continue
                other, t0 = got
                occ[t0:t0 + d, n] = True
                occ[t0:t0 + d, other] = True
                slot, partner = (n, other) if role == "init" else (other, n)
                used_init[slot] += 1
                used_part[partner] += 1
                events.append(Event(kind=kind, t0=t0, t1=t0 + d, slot=slot,
                                    partner=partner,
                                    params=_episode_params(kind, cfg, rng)))
                progressed = True
        if not progressed or len(events) >= n_ep:
            break

    covered = int(sum(e.length for e in events))
    balanced = feasible & ~capped
    init_spread = _spread(used_init, balanced)
    part_spread = _spread(used_part, balanced)
    rep = {
        "n_episodes": len(events),
        "n_requested": n_ep,
        "n_quota": int(cap.sum()),
        "realized_rate": covered / float(T * N),
        "mean_len": covered / max(len(events), 1),
        "slot_count_spread": init_spread,
        "partner_count_spread": part_spread,
        "n_feasible_slots": int(feasible.sum()),
        "n_infeasible_slots": len(infeasible),
        "n_capacity_capped_slots": int(capped.sum()),
        "shortfall": max(n_ep - len(events), 0),
        "n_slots_touched": int((used_init + used_part > 0).sum()),
        # Feasible slots that got nothing *despite there being enough episodes
        # to go round*. Only meaningful when `n_ep >= n_free`: below that,
        # slots with zero events are simple arithmetic, not a pathology. The
        # real pathology is a Hall violation — several degree-1 slots sharing
        # one possible partner — where some slot gets nothing however the
        # sampler is ordered.
        "n_stranded_slots": (int((feasible & (used_init + used_part == 0)).sum())
                             if n_ep >= n_free else 0),
    }
    _check_balance(kind, "initiator", init_spread, strict)
    _check_balance(kind, "partner", part_spread, strict)
    return events, rep, infeasible


def _spread(used, active):
    if not active.any():
        return 0
    return int(used[active].max() - used[active].min())


def _try_place_pair(rng, occ, runs, slot, d, T, other_budget, tries=32):
    """Find (other, t0) for one episode of length `d` starting at `slot`."""
    partners, starts, lengths = runs
    if partners.size == 0:
        return None
    lo = int(np.searchsorted(lengths, d, side="left"))
    if lo >= partners.size:
        return None
    p, st, ln = partners[lo:], starts[lo:], lengths[lo:]
    b = other_budget[p]
    ok = (b > 0) & (p != slot)
    if not ok.any():
        return None
    p, st, ln, b = p[ok], st[ok], ln[ok], b[ok]
    # Most remaining budget first; random tie-break so the same pair is not
    # always chosen for a slot whose neighbours are all equally free.
    order = np.lexsort((rng.random(p.size), -b))
    for k in order[:tries]:
        t0 = _find_free_window(rng, occ, [slot, int(p[k])], d, T,
                               lo_t=int(st[k]), hi_t=int(st[k]) + int(ln[k]) - d)
        if t0 is not None:
            return int(p[k]), t0
    return None


def _check_balance(kind, role, spread, strict):
    """Report the realized per-slot spread. Informative, never fatal.

    Exact balance is not always attainable, and not because of the sampler:
    the initiator/partner assignment is a degree-constrained bipartite
    b-matching on the contact graph, and its feasibility is a Hall condition
    that does not depend on the rate. `more_osc_clusters` has 122 degree-1
    slots against 379 slots with any contact at all, so several of them share
    a single possible partner and NO positive rate admits an exactly balanced
    assignment there. (membrane_stiffness/adhesion, at 22-25 union-distinct partners per
    cell, are fine.)

    The criterion that is actually enforced is therefore the *harm* rather than
    this proxy: `build.py` correlates the realized per-slot counts against every
    varied simulation parameter and refuses the build when a per-slot event or
    supervision count turns out to be a monotone function of the factor being
    clustered. See `seg_ablation.max_label_correlation`.
    """
    if spread <= 1:
        return
    print(f"[seg_ablation] {kind}: per-slot {role} counts differ by {spread} "
          f"(exact balance unattainable on this contact graph; the enforced "
          f"criterion is the label correlation, checked at build time)")


#: Frames of clearance required between two episodes on the same slot.
#:
#: Equal event counts do NOT imply equal supervision. Under `loss_mask_mode:
#: strict` an invalid frame kills three transitions (t-1, t, t+1), so two
#: invalid frames within 2 of each other cost less than two isolated ones. That
#: turned out to be label-correlated in exactly the direction the stratification
#: exists to prevent: on the membrane_stiffness cache at the merge_only rate, per-slot
#: absorbed counts were perfectly balanced (spread 1) yet
#: r(near-adjacent absorbed pairs, log10 kl) = +0.265 and
#: r(lost transitions, log10 kl) = -0.264 — WORSE than i.i.d. edge sampling,
#: which sits at 0.005. High-stiffness cells have fewer/shorter contact runs, so
#: the planner was forced to pack their episodes together.
#:
#: A 2-frame guard band makes the +-1 mask windows of two episodes on one slot
#: disjoint by construction, so lost transitions become exactly
#: `3 * events` for every slot and the confound is removed rather than bounded.
MIN_EPISODE_GAP = 2


def _find_free_window(rng, occ, slots, d, T, lo_t=0, hi_t=None, tries=24,
                      gap=MIN_EPISODE_GAP):
    """Uniformly sample a start with all `slots` free over [t0-gap, t0+d+gap).

    The guard band is only *checked*, never marked: the occupancy a placed
    episode claims stays exactly its own frames, so the check is symmetric
    between the episode placed first and the one placed second.
    """
    hi = (T - d) if hi_t is None else min(hi_t, T - d)
    if hi < lo_t:
        return None
    for _ in range(tries):
        t0 = int(rng.integers(lo_t, hi + 1))
        lo_g = max(t0 - gap, 0)
        hi_g = min(t0 + d + gap, T)
        if not any(occ[lo_g:hi_g, n].any() for n in slots):
            return t0
    return None


def _episode_params(kind, cfg, rng):
    """Episode-constant scalars, frozen for the whole run of frames."""
    if kind == "partial":
        return {
            "axis": float(rng.uniform(0.0, np.pi)),
            "area_frac": float(rng.uniform(cfg["area_frac_min"],
                                           cfg["area_frac_max"])),
            "side": float(rng.integers(0, 2)),
        }
    if kind == "scale":
        sign = 1.0 if rng.integers(0, 2) else -1.0
        return {"delta_px": float(sign * cfg["delta_px"])}
    return {}


def _merge_offset_stat(events, states):
    st = np.asarray(states)
    vals = []
    for e in events:
        if e.kind != "merge":
            continue
        t = min(e.t0, st.shape[0] - 1)
        vals.append(float(np.linalg.norm(st[t, e.partner, :2] - st[t, e.slot, :2])))
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Index maps
# ---------------------------------------------------------------------------

def build_index_maps(events, *, T, N):
    """Turn events into the three `(T, N)` maps the builder and sidecar carry.

    Returns:
        src_t: (T, N) int32 — which frame slot n reads at index t. Always the
            identity today (no temporal misalignment op exists); carried so the
            grid convention is explicit in the artifact and so a future
            frame-drop op needs no schema change. Asserted identity on load.
        src_n: (T, N) int32 — which *true* cell slot n reads. Deviates from the
            identity exactly on id_swap episodes.
        op:    (T, N) uint8 — the pixel/validity op, see OP_*.
    """
    src_t = np.broadcast_to(np.arange(T, dtype=np.int32)[:, None], (T, N)).copy()
    src_n = np.broadcast_to(np.arange(N, dtype=np.int32)[None, :], (T, N)).copy()
    op = np.zeros((T, N), dtype=np.uint8)

    def _set(sl, n, code):
        cur = op[sl, n]
        if np.any(cur != OP_NONE) and np.any(cur != code):
            raise RuntimeError(
                f"two seg_ablation events collide on slot {n} in frames "
                f"[{sl.start}, {sl.stop}) (ops {sorted(set(cur.tolist()))} vs "
                f"{code}). plan_events marks occupancy for both roles, so this "
                f"is a planner bug, not a config problem."
            )
        # Precedence, defensive: the higher code wins.
        op[sl, n] = np.maximum(cur, code)

    for e in events:
        sl = slice(e.t0, e.t1)
        if e.kind == "id_swap":
            src_n[sl, e.slot] = e.partner
            src_n[sl, e.partner] = e.slot
        elif e.kind == "merge":
            _set(sl, e.slot, OP_MERGE_ABSORB)
            _set(sl, e.partner, OP_MERGE_GONE)
        elif e.kind == "miss":
            _set(sl, e.slot, OP_MISS)
        elif e.kind == "partial":
            _set(sl, e.slot, OP_PARTIAL)
        elif e.kind == "scale":
            _set(sl, e.slot, OP_SCALE)
        else:
            raise ValueError(f"unknown event kind {e.kind!r}")

    return src_t, src_n, op
