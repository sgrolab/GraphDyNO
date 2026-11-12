"""Reusable embedding banks — one GPU wave for a whole ablation ladder.

The builder in `build.py` re-encodes every corrupted crop a level needs, so an
N-level scan pays N independent GPU builds. That is affordable for six arms and
absurd for twenty-nine.

A **bank** breaks the coupling. Plan one *dense superset* of corrupted episodes
per kind, at the ladder's top rate, and pre-encode it once:

    segbankplan_{bank_id}_{kind}_{cache_id}_{traj:04d}.pt        CPU: episodes
    segbankreport_{bank_id}_{cache_id}_{traj:04d}.json           realized rates
    segbank_{bank_id}_{kind}_{suffix}_{cache_id}_{traj:04d}.pt   GPU: the rows
    segbank_..._chunk{c:02d}.pt                                  pre-merge

Every level below the bank rate is then a **sub-selection of banked rows** —
pure index math, no GPU — which is what lets `train.py` materialize its own
level in-process (`bank_apply.apply_bank`) instead of a materialize wave.

Why the selection is exact rather than approximate
--------------------------------------------------
Each banked episode carries a `rank` in `0..m-1` within its own *initiator
slot*, shuffled by a per-slot RNG, and its slot's total `m`. Selecting rate rho
keeps `rank < round(rho/rho_bank * m)`, which gives:

  * **strict nesting** — d1 subset d2 subset d3 subset d4 = the whole bank, so
    a dose ladder is a nested sequence rather than four independent draws and
    the dose-response curve is not confounded by resampling;
  * **per-slot balance at every dose** — `_largest_remainder` already holds `m`
    to a spread of 1, and rounding a common ratio against two integers that
    differ by 1 also differs by at most 1;
  * **MIN_EPISODE_GAP inherited** — the superset is placed with the guard band,
    and any subset of a gap-respecting placement still respects it.

What is *not* banked, and why it does not matter
------------------------------------------------
`miss` and `id_swap` are index edits — no pixels change, so they need no
re-encode. They are nevertheless **planned into the bank** (a plan file, no
rows file) so that they get the same nesting, the same balance, and a place in
the realized-rate report. Centroid noise is a per-level scalar and lives
entirely in `build._finalize_sidecar`.

Per-trajectory decorrelation
----------------------------
Banks are planned **per trajectory**, with `traj_idx` in the RNG seed, and each
kind gets its own stream. So cell 217 is cut on a different axis, at a
different area fraction, at different frames, and merged with a different
neighbour in every trajectory — while `strict_balance` still equalises its
*count* everywhere, which is the label-confound control. That is the reason the
plan wave is three jobs and not one.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from graph_dyno.seg_ablation.plan import (
    KINDS,
    MIN_EPISODE_GAP,
    PAIR_KINDS,
    PIXEL_KINDS,
    SEG_ABLATION_CODE_VERSION,
    Event,
    _KIND_DEFAULTS,
    plan_bank,
)

#: Kinds that get a GPU rows file. The others are planned into the bank but
#: realised by index arithmetic at apply time.
BANK_KINDS = PIXEL_KINDS

#: Order the plan wave and the report use. Deterministic, not significant.
PLAN_KINDS = ("partial", "scale", "merge", "miss", "id_swap")

#: Per-kind fields that describe *severity* rather than frequency. A level may
#: freely vary `rate`; if it varies any of these it is asking for a corruption
#: the bank never encoded, and `bank_apply` refuses rather than silently
#: serving the bank's own severity under the level's name. Pinning severity and
#: sweeping only frequency is what makes the four doses one nested population.
SEVERITY_FIELDS = {
    "partial": ("mean_len", "area_frac_min", "area_frac_max"),
    "merge": ("mean_len", "pad_factor", "close_px"),
    "scale": ("mean_len", "delta_px"),
    "miss": ("mean_len",),
    "id_swap": ("mean_len",),
}

_BANK_DEFAULTS: dict[str, Any] = {
    "name": None,
    "seed": 0,
    "max_episode_len": 200,
    "strict_balance": True,
}


# ---------------------------------------------------------------------------
# Spec + identity
# ---------------------------------------------------------------------------

def normalize_bank_spec(spec) -> dict:
    """Fill defaults and coerce a bank spec to a plain dict.

    Unknown keys are fatal, exactly as in `plan.normalize_spec`: a typo'd
    `rate` here would quietly under-build the bank and cap the whole ladder.
    """
    from omegaconf import DictConfig, OmegaConf

    if spec is None:
        raise ValueError("seg_bank spec is None")
    if isinstance(spec, DictConfig):
        spec = OmegaConf.to_container(spec, resolve=True)
    if not isinstance(spec, dict):
        raise TypeError(f"seg_bank spec must be a mapping, got {type(spec)}")

    unknown = set(spec) - set(_BANK_DEFAULTS) - {"kinds"}
    if unknown:
        raise ValueError(
            f"Unknown seg_bank keys {sorted(unknown)}. "
            f"Known: {sorted(set(_BANK_DEFAULTS) | {'kinds'})}")

    out = dict(_BANK_DEFAULTS)
    for k, v in spec.items():
        if k != "kinds":
            out[k] = v
    if not out["name"]:
        raise ValueError("seg_bank.name is required — it is half of the "
                         "bank's identity and the key levels refer to")

    kinds_in = spec.get("kinds", {}) or {}
    unknown_kinds = set(kinds_in) - set(PLAN_KINDS)
    if unknown_kinds:
        raise ValueError(
            f"Unknown seg_bank kinds {sorted(unknown_kinds)}. "
            f"Known: {list(PLAN_KINDS)}")

    kinds = {}
    for kind in PLAN_KINDS:
        entry = dict(_KIND_DEFAULTS[kind])
        given = kinds_in.get(kind, {}) or {}
        bad = set(given) - set(entry)
        if bad:
            raise ValueError(
                f"Unknown fields {sorted(bad)} for seg_bank kind {kind!r}. "
                f"Known: {sorted(entry)}")
        entry.update(given)
        entry["rate"] = float(entry["rate"])
        entry["mean_len"] = float(entry["mean_len"])
        if not 0 < entry["rate"] <= 1:
            raise ValueError(
                f"seg_bank.kinds.{kind}.rate must be in (0, 1] — a bank with a "
                f"zero rate has nothing to sub-select from")
        if entry["mean_len"] < 1:
            raise ValueError(f"seg_bank.kinds.{kind}.mean_len must be >= 1")
        kinds[kind] = entry
    out["kinds"] = kinds

    out["name"] = str(out["name"])
    out["seed"] = int(out["seed"])
    out["max_episode_len"] = int(out["max_episode_len"])
    out["strict_balance"] = bool(out["strict_balance"])
    return out


def bank_id(spec) -> str:
    """8-char identity of a bank spec *and* the perturbation code behind it.

    Mirrors `plan.abl_id`: the filenames carry nothing else, so bumping
    SEG_ABLATION_CODE_VERSION must invalidate every banked row whose math
    changed. `traj_idx` is deliberately NOT part of the identity — it enters
    the RNG seed and the filename, exactly as it does for the sidecars.
    """
    from omegaconf import OmegaConf

    s = normalize_bank_spec(spec)
    payload = OmegaConf.to_yaml(OmegaConf.create(s), sort_keys=True)
    h = hashlib.md5()
    h.update(payload.encode())
    h.update(SEG_ABLATION_CODE_VERSION.encode())
    return h.hexdigest()[:8]


def bank_config_path(name, configs_dir=None) -> str:
    if configs_dir is None:
        configs_dir = os.path.join(
            Path(__file__).resolve().parents[3], "configs", "seg_bank")
    return os.path.join(configs_dir, f"{name}.yaml")


def load_bank_spec(name, configs_dir=None) -> dict:
    """Read `configs/seg_bank/<name>.yaml` and normalize it.

    A level names its bank with a string (`seg_ablation.bank: ladder_v1`), so
    the read path has to resolve that name to a spec without Hydra having
    composed the `seg_bank` group. Reading the same file the group composes
    from keeps one source of truth; `scripts/plan_seg_bank.py` asserts that the
    composed node and this file agree, so a CLI override cannot orphan a bank.
    """
    from omegaconf import OmegaConf

    path = bank_config_path(name, configs_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No bank config {path}. `seg_ablation.bank: {name}` refers to "
            f"configs/seg_bank/{name}.yaml.")
    node = OmegaConf.load(path)
    node = node.get("seg_bank", node)
    s = normalize_bank_spec(node)
    if s["name"] != name:
        raise ValueError(
            f"{path} declares name={s['name']!r} but was loaded as {name!r}; "
            f"the name is part of the bank identity")
    return s


def kind_seed(bank_spec, traj_idx, kind) -> int:
    """The per-(bank, trajectory, kind) RNG stream.

    Three independent axes, and all three matter. `traj_idx` is what makes a
    given cell suffer *differently* in each trajectory (different cut axis,
    area fraction, frames, merge partner) while `strict_balance` still gives it
    the same episode COUNT everywhere. `kind` keeps one kind's episode count
    from shifting another kind's draws, which a single shared stream would do
    as soon as the contact graph changed.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    ss = np.random.SeedSequence(
        [int(bank_spec["seed"]), int(traj_idx), KINDS.index(kind)])
    return int(ss.generate_state(1, dtype=np.uint64)[0])


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def plan_filename(bid, kind, cache_id, traj_idx):
    return f"segbankplan_{bid}_{kind}_{cache_id}_{traj_idx:04d}.pt"


def report_filename(bid, cache_id, traj_idx):
    return f"segbankreport_{bid}_{cache_id}_{traj_idx:04d}.json"


def decorr_filename(bid, cache_id):
    """Cross-trajectory decorrelation. A DIFFERENT prefix on purpose —
    `read_reports` globs the per-trajectory reports and reads `traj_idx` off
    each, so a whole-bank file must not land in that glob."""
    return f"segbankdecorr_{bid}_{cache_id}.json"


def rows_filename(bid, kind, companion_suffix, cache_id, traj_idx,
                  chunk_idx=None):
    """The banked rows for one kind.

    `chunk_idx` is **zero-padded**. `precompute_embeddings.merge_chunks` sorts
    its glob lexicographically, which is why that path has to cap `n_chunks` at
    10 (`chunk10` sorts before `chunk2`, giving a time-scrambled trajectory).
    Padding removes the trap instead of working around it — and this merge
    orders by the stored `chunk_idx` anyway, never by filename.
    """
    base = f"segbank_{bid}_{kind}_{companion_suffix}_{cache_id}_{traj_idx:04d}"
    if chunk_idx is None:
        return base + ".pt"
    return f"{base}_chunk{int(chunk_idx):02d}.pt"


def chunk_glob(bid, kind, companion_suffix, cache_id, traj_idx):
    return (f"segbank_{bid}_{kind}_{companion_suffix}_{cache_id}_"
            f"{traj_idx:04d}_chunk*.pt")


def chunk_bounds(T, n_chunks, chunk_idx):
    """Half-open [lo, hi) frame range of one chunk.

    Exact division is not required (the bounds are computed, not assumed), but
    an uneven split just makes one job longer than the rest.
    """
    n_chunks = int(n_chunks)
    if not 0 <= int(chunk_idx) < n_chunks:
        raise ValueError(f"chunk_idx {chunk_idx} out of range for "
                         f"n_chunks={n_chunks}")
    edges = np.linspace(0, int(T), n_chunks + 1).astype(np.int64)
    return int(edges[int(chunk_idx)]), int(edges[int(chunk_idx) + 1])


# ---------------------------------------------------------------------------
# Planning one bank
# ---------------------------------------------------------------------------

def build_bank_plan(bank_spec, kind, *, cache_id, traj_idx, T, N,
                    states=None, W_interactions=None):
    """Plan one kind's superset for one trajectory. Pure; writes nothing."""
    s = normalize_bank_spec(bank_spec)
    seed = kind_seed(s, traj_idx, kind)
    events, rep, rank, slot_total, infeasible = plan_bank(
        kind, s["kinds"][kind], T=T, N=N, states=states,
        W_interactions=W_interactions, seed=seed,
        max_episode_len=s["max_episode_len"],
        strict_balance=s["strict_balance"])
    return {
        "bank_id": bank_id(s),
        "bank_name": s["name"],
        "bank_spec": s,
        "code_version": SEG_ABLATION_CODE_VERSION,
        "kind": kind,
        "cache_id": cache_id,
        "traj_idx": int(traj_idx),
        "T": int(T),
        "N": int(N),
        "seed": int(seed),
        "rate": float(s["kinds"][kind]["rate"]),
        "kind_cfg": dict(s["kinds"][kind]),
        "t0": np.array([e.t0 for e in events], dtype=np.int32),
        "t1": np.array([e.t1 for e in events], dtype=np.int32),
        "slot": np.array([e.slot for e in events], dtype=np.int32),
        "partner": np.array([e.partner for e in events], dtype=np.int32),
        "params": [dict(e.params) for e in events],
        "rank": rank,
        "slot_total": slot_total,
        "report": rep,
        "infeasible_slots": infeasible,
    }


def plan_events_of(plan, ids):
    """Rebuild `Event`s for the selected episode ids, in plan order."""
    kind = plan["kind"]
    return [
        Event(kind=kind, t0=int(plan["t0"][i]), t1=int(plan["t1"][i]),
              slot=int(plan["slot"][i]), partner=int(plan["partner"][i]),
              params=dict(plan["params"][i]))
        for i in ids
    ]


def load_plan(cache_dir, bid, kind, cache_id, traj_idx, *, expect_T=None,
              expect_N=None):
    import torch

    path = os.path.join(cache_dir, plan_filename(bid, kind, cache_id, traj_idx))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing bank plan {path}. Build it with\n"
            f"  python scripts/plan_seg_bank.py +experiment=<name> "
            f"+embedding=<name> +seg_bank=<bank> +traj_idx={traj_idx}")
    p = torch.load(path, map_location="cpu", weights_only=False)
    if p["code_version"] != SEG_ABLATION_CODE_VERSION:
        raise ValueError(
            f"{os.path.basename(path)} was planned with "
            f"{p['code_version']!r} but this code is "
            f"{SEG_ABLATION_CODE_VERSION!r} — replan, do not reuse.")
    if p["bank_id"] != bid or p["kind"] != kind:
        raise ValueError(f"{os.path.basename(path)} is {p['bank_id']}/"
                         f"{p['kind']}, expected {bid}/{kind}")
    if p["cache_id"] != cache_id or int(p["traj_idx"]) != int(traj_idx):
        raise ValueError(f"{os.path.basename(path)} is for "
                         f"{p['cache_id']}/{p['traj_idx']}")
    if expect_T is not None and int(p["T"]) != int(expect_T):
        raise ValueError(f"{os.path.basename(path)} has T={p['T']}, "
                         f"trajectory has {expect_T}")
    if expect_N is not None and int(p["N"]) != int(expect_N):
        raise ValueError(f"{os.path.basename(path)} has N={p['N']}, "
                         f"trajectory has {expect_N}")
    return p


# ---------------------------------------------------------------------------
# Sub-selection
# ---------------------------------------------------------------------------

def select(plan, rate):
    """Nested per-slot sub-selection of a bank at `rate`. Returns a bool mask.

    `round(rho/rho_bank * slot_total)` rather than a global count: the whole
    point of the per-slot rank is that balance survives the selection, and a
    global top-k would reintroduce exactly the per-slot spread the
    stratification exists to remove.
    """
    bank_rate = float(plan["rate"])
    m = np.asarray(plan["slot_total"], dtype=np.int64)
    rank = np.asarray(plan["rank"], dtype=np.int64)
    if rate <= 0:
        return np.zeros(rank.size, dtype=bool)
    if rate > bank_rate * (1 + 1e-9):
        raise ValueError(
            f"level requests {plan['kind']} at rate {rate} but the bank "
            f"{plan['bank_id']} only holds {bank_rate}. A level can never "
            f"exceed its bank — raise the bank rate and re-encode.")
    ratio = min(rate / bank_rate, 1.0)
    return rank < np.rint(ratio * m).astype(np.int64)


#: Salt for the frame-selection RNG. Distinct from the episode `rank` stream so
#: that a non-episodic level draws its own order rather than inheriting one
#: correlated with the episodic ladder's.
_FRAME_SALT = 0x5EAF12


def select_frames(plans, rates, *, gap=MIN_EPISODE_GAP, level_seed=0):
    """Fill every kind's per-cell frame quota against ONE shared occupancy.

    The frame-granular counterpart of `select`, for `episodic: false`. Where
    `select` hands each cell a whole number of episodes — quantizing the
    realized rate at `mean_len / T`, which is 0.004 for partial/scale and 0.020
    for id_swap, and so puts a balanced 1% joint arm out of reach — this hands
    each cell a whole number of *frames*, so the quantum is `1 / T` and `rate`
    means exactly "frames per cell / T".

    It costs no GPU: the bank stores one re-encoded row per corrupted cell-frame
    (`idx_t`/`idx_n`), so the frames of a banked episode are already
    independently addressable. Only the *selection* unit changes.

    Why all kinds are selected TOGETHER rather than per kind then reconciled
    ---------------------------------------------------------------------
    `resolve_collisions` drops the loser of an overlap, and a dropped frame is
    gone: the kind ends up below its quota. That loss is not uniform — a pair
    kind occupies two slots per event and so collides about twice as often —
    which reintroduces exactly the composition skew frame selection exists to
    remove. Measured on the first cut of the 5% arm, per-kind requests of
    0.0125/0.0125/0.0125/0.006 realized 0.0115/0.0115/0.0115/**0.0038**: merge
    lost 37% of its request while the single-slot kinds lost 8%.

    Walking every candidate frame against a shared occupancy fixes it at the
    source. A collision costs a *retry*, not a frame — the kind simply takes its
    next candidate — so each kind reaches its quota whenever the bank holds the
    capacity, and realized equals requested to the frame.

    The three properties `select` has are kept:

      * **exact per-slot balance** — every slot gets exactly `round(rate * T)`
        frames of each kind, tighter than the spread-of-1 `select` gives;
      * **MIN_EPISODE_GAP preserved** — enforced here rather than inherited,
        since consecutive frames of one banked episode are adjacent by
        construction; the rule matches `resolve_collisions` exactly (unpadded
        marks, padded test => accepted frames on a slot are `gap + 1` apart), so
        running that afterwards is a free self-check that drops nothing;
      * **no kind systematically favoured** — one global random priority over
        all candidates, the same argument `resolve_collisions` makes for its
        hash tiebreak.

    Nesting across rates is NOT offered. Quota filling is a function of the
    whole request, so a lower dose is not a subset of a higher one; the episodic
    ladder keeps that property and this family does not need it (its arms are a
    dose axis, not a nested population).

    Args:
        plans: {kind: bank plan}, as `load_plan` returns.
        rates: {kind: rate}. Kinds absent or <= 0 are skipped.
        level_seed: the LEVEL's `seg_ablation.seed`, mixed into the candidate
            ordering so that two levels differing only in seed draw a different
            REALIZATION at identical per-slot quotas. Without it a level is one
            draw of the corruption, and a seed wave varies only model init — so
            "which frames were hit" is confounded with dose across the arms of
            a family. Note the episodic path (`select`) does NOT take a level
            seed: its placement is the bank's `rank`, and re-seeding it would
            change every already-published episodic level.

            `level_seed=0` contributes nothing to the mix, so it reproduces the
            pre-existing artifacts bit-for-bit.

    Returns:
        (selected, shortfall) — `selected[kind]` is `(ep_idx, t)`, parallel
        int64 arrays sorted by `(t, ep_idx)`; `shortfall[kind]` is
        `{slot: n_taken}` for slots that could not reach the quota, either
        because the bank strands them or because the other kinds crowded them
        out. Slots with no banked frame at all are omitted, matching how
        `infeasible_slots` already treats a cell the contact graph strands.
    """
    kinds = [k for k in sorted(plans) if float(rates.get(k, 0.0)) > 0]
    if not kinds:
        return {}, {}
    T = int(plans[kinds[0]]["T"])
    N = int(plans[kinds[0]]["N"])

    ck, cep, ct, cs, cp = [], [], [], [], []
    for ki, kind in enumerate(kinds):
        p = plans[kind]
        t0 = np.asarray(p["t0"], dtype=np.int64)
        t1 = np.asarray(p["t1"], dtype=np.int64)
        lens = t1 - t0
        n = int(lens.sum())
        if not n:
            continue
        ep = np.repeat(np.arange(t0.size, dtype=np.int64), lens)
        offs = np.arange(n, dtype=np.int64) - np.repeat(np.cumsum(lens) - lens,
                                                        lens)
        ck.append(np.full(n, ki, dtype=np.int64))
        cep.append(ep)
        ct.append(np.repeat(t0, lens) + offs)
        cs.append(np.asarray(p["slot"], dtype=np.int64)[ep])
        cp.append(np.asarray(p["partner"], dtype=np.int64)[ep])
    if not ck:
        return {k: (np.zeros(0, np.int64), np.zeros(0, np.int64))
                for k in kinds}, {}
    ck = np.concatenate(ck)
    cep = np.concatenate(cep)
    ct = np.concatenate(ct)
    cs = np.concatenate(cs)
    cp = np.concatenate(cp)

    seed = _FRAME_SALT
    for kind in kinds:
        seed ^= int(plans[kind]["seed"])
    # Multiplied by an odd 32-bit constant so consecutive seeds decorrelate;
    # zero maps to zero, which is what keeps seed 0 byte-identical to before.
    seed ^= (int(level_seed) * 0x9E3779B1) & 0xFFFFFFFF
    order = np.argsort(np.random.default_rng(seed).random(ct.size),
                       kind="stable")

    quota = np.array([int(np.rint(float(rates[k]) * T)) for k in kinds],
                     dtype=np.int64)
    taken = np.zeros((len(kinds), N), dtype=np.int64)
    # Slot-major and offset by `gap`: accepting t marks [t, t + 2*gap + 1),
    # which covers t-gap..t+gap, so testing a candidate is the single lookup
    # `occ[slot, t + gap]`.
    occ = np.zeros((N, T + 2 * gap), dtype=bool)
    keep = np.zeros(ct.size, dtype=bool)

    ck_l, ct_l, cs_l, cp_l = (ck.tolist(), ct.tolist(), cs.tolist(),
                              cp.tolist())
    span = 2 * gap + 1
    for idx in order.tolist():
        ki = ck_l[idx]
        s = cs_l[idx]
        if taken[ki, s] >= quota[ki]:
            continue
        t = ct_l[idx]
        pr = cp_l[idx]
        if occ[s, t + gap] or (pr >= 0 and occ[pr, t + gap]):
            continue
        occ[s, t:t + span] = True
        if pr >= 0:
            occ[pr, t:t + span] = True
        taken[ki, s] += 1
        keep[idx] = True

    selected, shortfall = {}, {}
    for ki, kind in enumerate(kinds):
        m = keep & (ck == ki)
        ep_out, t_out = cep[m], ct[m]
        o = np.lexsort((ep_out, t_out))
        selected[kind] = (ep_out[o], t_out[o])
        # A slot with no candidate at all is stranded, not short — the same
        # distinction `plan_bank` already draws with `infeasible_slots`.
        has_cand = np.bincount(cs[ck == ki], minlength=N) > 0
        short = np.flatnonzero(has_cand & (taken[ki] < quota[ki]))
        if short.size:
            shortfall[kind] = {int(s): int(taken[ki, s]) for s in short}
    return selected, shortfall


def frame_plan(plan, ep_idx, t):
    """A plan-shaped dict whose "episodes" are the selected single frames.

    Lets `resolve_collisions`, `plan_events_of` and `realized_stats` run on a
    frame selection unchanged — they only ever read `t0`/`t1`/`slot`/
    `partner`/`params`. `src_episode` keeps the provenance for the report.
    """
    out = dict(plan)
    out["t0"] = t.astype(np.int32)
    out["t1"] = (t + 1).astype(np.int32)
    out["slot"] = np.asarray(plan["slot"], dtype=np.int64)[ep_idx].astype(np.int32)
    out["partner"] = np.asarray(plan["partner"], dtype=np.int64)[ep_idx].astype(np.int32)
    out["params"] = [dict(plan["params"][i]) for i in ep_idx.tolist()]
    out["rank"] = np.arange(ep_idx.size, dtype=np.int64)
    out["src_episode"] = ep_idx
    return out


def resolve_collisions(plans, keeps, *, T, N, gap=MIN_EPISODE_GAP):
    """Drop cross-kind overlaps with a deterministic RANDOM tiebreak.

    Banks are planned per kind on empty occupancy, so a joint arm can select
    two episodes on the same `(t, slot)`. Resolution is by `sha1(kind:id)`
    order, **not** kind priority: priority is what made the saturated membrane_stiffness
    levels realise wildly different compositions under the two sharing modes,
    because the first kind in the order took everything it wanted and the last
    took what was left. A hash order keeps the realized composition roughly
    proportional to the requested one at any saturation.

    The guard band is enforced across kinds too, so `MIN_EPISODE_GAP`'s
    guarantee — that two episodes on one slot have disjoint +-1 mask windows,
    hence exactly `3 * events` lost transitions under `strict` — survives
    mixing.

    Returns:
        (accepted, occ, n_dropped) — `accepted[kind]` a bool mask over that
        kind's episodes, `occ` the (T, N) occupancy, `n_dropped` per kind.
    """
    items = []
    for kind in sorted(plans):
        for i in np.flatnonzero(keeps[kind]).tolist():
            key = hashlib.sha1(f"{kind}:{i}".encode()).digest()
            items.append((key, kind, int(i)))
    items.sort(key=lambda x: x[0])

    occ = np.zeros((int(T), int(N)), dtype=bool)
    accepted = {k: np.zeros(np.asarray(keeps[k]).size, dtype=bool)
                for k in plans}
    n_dropped = {k: 0 for k in plans}
    for _, kind, i in items:
        p = plans[kind]
        t0, t1 = int(p["t0"][i]), int(p["t1"][i])
        slots = [int(p["slot"][i])]
        partner = int(p["partner"][i])
        if partner >= 0:
            slots.append(partner)
        lo, hi = max(t0 - gap, 0), min(t1 + gap, int(T))
        if any(occ[lo:hi, n].any() for n in slots):
            n_dropped[kind] += 1
            continue
        for n in slots:
            occ[t0:t1, n] = True
        accepted[kind][i] = True
    return accepted, occ, n_dropped


def realized_stats(plan, accepted):
    """Per-kind realized numbers for one selection — what the analysis plots."""
    ids = np.flatnonzero(accepted)
    t0 = np.asarray(plan["t0"], dtype=np.int64)[ids]
    t1 = np.asarray(plan["t1"], dtype=np.int64)[ids]
    covered = int((t1 - t0).sum())
    T, N = int(plan["T"]), int(plan["N"])
    slot = np.asarray(plan["slot"], dtype=np.int64)[ids]
    counts = np.bincount(slot, minlength=N)
    active = counts > 0
    return {
        "n_episodes": int(ids.size),
        "n_banked": int(np.asarray(plan["rank"]).size),
        "realized_rate": covered / float(T * N),
        "mean_len": covered / max(int(ids.size), 1),
        "slot_count_spread": (int(counts[active].max() - counts[active].min())
                              if active.any() else 0),
        "n_slots_touched": int(active.sum()),
    }


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

def load_rows(cache_dir, bid, kind, companion_suffix, cache_id, traj_idx,
              *, mmap=True):
    import torch

    path = os.path.join(cache_dir, rows_filename(
        bid, kind, companion_suffix, cache_id, traj_idx))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing banked rows {path}. Encode them with\n"
            f"  python scripts/build_seg_bank_worker.py "
            f"+experiment=<name> +embedding=<name> "
            f"+seg_bank=<bank> +traj_idx={traj_idx} +kind={kind} ...\n"
            f"  python scripts/merge_seg_bank.py ...")
    r = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    if r["bank_id"] != bid or r["kind"] != kind:
        raise ValueError(f"{os.path.basename(path)} is {r['bank_id']}/"
                         f"{r['kind']}, expected {bid}/{kind}")
    if r["companion_suffix"] != companion_suffix:
        raise ValueError(
            f"{os.path.basename(path)} carries companion_suffix "
            f"{r['companion_suffix']!r}; this run uses {companion_suffix!r}")
    if r["cache_id"] != cache_id or int(r["traj_idx"]) != int(traj_idx):
        raise ValueError(f"{os.path.basename(path)} is for "
                         f"{r['cache_id']}/{r['traj_idx']}")
    return r


def find_empty_emb(cache_dir, bid, companion_suffix, cache_id, traj_idx):
    """The all-zero-mask embedding row, from whichever bank file exists.

    Every missed and every absorbed cell in the dataset carries this one
    vector (`_mask_to_sdf_worker` returns the deterministic all-(-1) SDF and
    crop_side 0 for an empty mask), so caching it in the bank is what keeps
    `apply_bank` **GPU-free**. 290 concurrent training jobs each loading
    DINOv3 to encode a single zero mask is precisely the N-times-GPU failure
    the runbook warns about for train-time builds.
    """
    import torch

    for kind in BANK_KINDS:
        path = os.path.join(cache_dir, rows_filename(
            bid, kind, companion_suffix, cache_id, traj_idx))
        if os.path.exists(path):
            r = torch.load(path, map_location="cpu", weights_only=False,
                           mmap=True)
            return r["empty_emb"].clone().float()
    raise FileNotFoundError(
        f"This level needs the empty-mask row (it has missed or absorbed "
        f"cells) but no bank {bid} rows file exists for {companion_suffix} / "
        f"{cache_id} / traj {traj_idx}. Encode the bank first.")


def merge_chunks(cache_dir, bid, kind, companion_suffix, cache_id, traj_idx,
                 *, expect_n_chunks=None, delete=True, verbose=True):
    """Concatenate a kind's chunks into one rows file.

    Unlike `precompute_embeddings.merge_chunks` (which concatenates whatever it
    globs, deletes it, and leaves an unrecoverable short file when a part is
    missing) this **asserts completeness before deleting anything**: the stored
    `chunk_idx` values must be exactly `0..n_chunks-1`, and ordering is by that
    field, never by filename.
    """
    import torch

    from graph_dyno.seg_ablation.build import _atomic_save

    final = os.path.join(cache_dir, rows_filename(
        bid, kind, companion_suffix, cache_id, traj_idx))
    if os.path.exists(final):
        if verbose:
            print(f"  [seg_bank] {os.path.basename(final)} present, skipping")
        return final, None

    paths = sorted(glob.glob(os.path.join(
        cache_dir, chunk_glob(bid, kind, companion_suffix, cache_id, traj_idx))))
    if not paths:
        raise FileNotFoundError(
            f"No chunks to merge for {bid}/{kind}/traj {traj_idx}")

    parts = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    n_chunks = {int(p["n_chunks"]) for p in parts}
    if len(n_chunks) != 1:
        raise ValueError(f"chunks disagree on n_chunks: {sorted(n_chunks)}")
    n_chunks = n_chunks.pop()
    if expect_n_chunks is not None and n_chunks != int(expect_n_chunks):
        raise ValueError(f"chunks say n_chunks={n_chunks}, expected "
                         f"{expect_n_chunks}")
    got = sorted(int(p["chunk_idx"]) for p in parts)
    if got != list(range(n_chunks)):
        raise ValueError(
            f"incomplete chunk set for {bid}/{kind}/traj {traj_idx}: have "
            f"{got}, need {list(range(n_chunks))}. Refusing to merge — a "
            f"missing part becomes an unrecoverable short file.")
    parts.sort(key=lambda p: int(p["chunk_idx"]))

    D = {int(p["D"]) for p in parts}
    if len(D) != 1:
        raise ValueError(f"chunks disagree on embedding width: {sorted(D)}")
    empties = torch.stack([p["empty_emb"] for p in parts])
    if not bool((empties == empties[0]).all()):
        raise ValueError("chunks disagree on the empty-mask embedding row")

    merged = {
        "bank_id": bid,
        "bank_name": parts[0].get("bank_name"),
        "code_version": SEG_ABLATION_CODE_VERSION,
        "kind": kind,
        "companion_suffix": companion_suffix,
        "cache_id": cache_id,
        "traj_idx": int(traj_idx),
        "n_chunks": n_chunks,
        "D": D.pop(),
        "empty_emb": parts[0]["empty_emb"].clone(),
    }
    for key in ("idx_t", "idx_n", "episode_id", "crop_sides", "clipped",
                "crop_disagree"):
        merged[key] = torch.cat([p[key] for p in parts], 0)
    merged["embeddings"] = torch.cat([p["embeddings"] for p in parts], 0)
    merged["centroid_delta"] = torch.cat([p["centroid_delta"] for p in parts], 0)
    merged["K"] = int(merged["idx_t"].numel())

    _atomic_save(merged, final)
    if delete:
        for p in paths:
            os.remove(p)
    if verbose:
        print(f"  [seg_bank] {os.path.basename(final)}: {merged['K']} rows "
              f"x {merged['D']} from {n_chunks} chunks "
              f"({os.path.getsize(final) / 1e6:.0f} MB)")
    return final, merged


# ---------------------------------------------------------------------------
# Feasibility / decorrelation report
# ---------------------------------------------------------------------------

def write_report(cache_dir, bid, cache_id, traj_idx, payload):
    """Machine-readable realized rates — read at ANALYSIS time, never gated on.

    `more_osc_clusters` is contact-starved (1.8 union-distinct partners per
    cell, 74% of cell-frames with any neighbour at all), so merge and id_swap
    at the ladder's top rate are quite unlikely to be fully realizable and
    `joint_d4` certainly is not. That does not hold up the GPU: request the
    ladder, encode whatever lands, and plot every curve against the REALIZED
    rate recorded here.
    """
    path = os.path.join(cache_dir, report_filename(bid, cache_id, traj_idx))
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=_jsonable)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def read_reports(cache_dir, bid, cache_id, traj_indices=None):
    """Every per-trajectory report for one bank, keyed by traj_idx."""
    out = {}
    pattern = os.path.join(cache_dir, f"segbankreport_{bid}_{cache_id}_*.json")
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            d = json.load(f)
        i = int(d["traj_idx"])
        if traj_indices is None or i in traj_indices:
            out[i] = d
    return out


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serialisable: {type(o)}")


def decorrelation(plans_by_traj, N):
    """How differently the same cell is corrupted in each trajectory.

    The requirement this measures: a given cell id must NOT receive the same
    corruption in all three trajectories, while its episode COUNT stays equal
    everywhere (that equality is the label-confound control, and is what
    `strict_balance` buys).

    Args:
        plans_by_traj: {traj_idx: plan} for ONE kind.

    Returns per-pair overlap of the `(t0, slot)` episode set, the correlation
    between per-slot counts, and — for the kinds that have them — the mean
    absolute cut-axis difference, the `side` agreement rate and the merge
    partner agreement rate.
    """
    trajs = sorted(plans_by_traj)
    out = {"traj_indices": trajs, "pairs": {}}
    counts = {}
    for i in trajs:
        p = plans_by_traj[i]
        counts[i] = np.bincount(np.asarray(p["slot"], dtype=np.int64),
                                minlength=N).astype(float)
        out.setdefault("per_slot_count_spread", {})[str(i)] = int(
            counts[i].max() - counts[i].min()) if counts[i].size else 0

    for a in range(len(trajs)):
        for b in range(a + 1, len(trajs)):
            i, j = trajs[a], trajs[b]
            pi, pj = plans_by_traj[i], plans_by_traj[j]
            si = {(int(t), int(s)) for t, s in zip(pi["t0"], pi["slot"])}
            sj = {(int(t), int(s)) for t, s in zip(pj["t0"], pj["slot"])}
            union = len(si | sj)
            entry = {
                "episode_overlap_frac": (len(si & sj) / union) if union else 0.0,
                "per_slot_count_r": _safe_corr(counts[i], counts[j]),
            }
            # Params are per-episode, so compare them on the slots both
            # trajectories touched — that is where "same cell, same corruption"
            # would show up.
            ax_i = _first_param_by_slot(pi, "axis")
            ax_j = _first_param_by_slot(pj, "axis")
            shared = sorted(set(ax_i) & set(ax_j))
            if shared:
                d = np.abs(np.array([ax_i[s] for s in shared])
                           - np.array([ax_j[s] for s in shared]))
                # `axis` is an angle on [0, pi); the mean |difference| of two
                # independent U(0, pi) draws is pi/3 = 1.047.
                entry["mean_abs_axis_delta"] = float(d.mean())
                entry["n_shared_slots"] = len(shared)
            sd_i = _first_param_by_slot(pi, "side")
            sd_j = _first_param_by_slot(pj, "side")
            shared = sorted(set(sd_i) & set(sd_j))
            if shared:
                entry["side_agreement"] = float(np.mean(
                    [sd_i[s] == sd_j[s] for s in shared]))
            if pi["kind"] in PAIR_KINDS:
                pi_map = _first_partner_by_slot(pi)
                pj_map = _first_partner_by_slot(pj)
                shared = sorted(set(pi_map) & set(pj_map))
                if shared:
                    entry["partner_agreement"] = float(np.mean(
                        [pi_map[s] == pj_map[s] for s in shared]))
            out["pairs"][f"{i}-{j}"] = entry
    return out


def _safe_corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _first_param_by_slot(plan, key):
    out = {}
    for s, params in zip(plan["slot"], plan["params"]):
        if key in params:
            out.setdefault(int(s), float(params[key]))
    return out


def _first_partner_by_slot(plan):
    out = {}
    for s, p in zip(plan["slot"], plan["partner"]):
        out.setdefault(int(s), int(p))
    return out
