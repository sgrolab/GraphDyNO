"""Tests for the reusable seg-ablation embedding bank.

The bank replaces one GPU build per level with one GPU wave per *dataset*, so
the things that can silently go wrong are all about whether a sub-selection of
banked rows really is the same artifact a direct build would have produced:

  - Selection    nesting, per-slot balance, the MIN_EPISODE_GAP guard band,
                 and cross-kind collision resolution
  - Identity     `abl_id` separates bank-derived from directly-built specs;
                 severity may not drift from the bank's
  - Merge        an incomplete chunk set must NOT be merged (the failure mode
                 `precompute_embeddings.merge_chunks` has)
  - Equivalence  a bank-materialized level is bit-identical to `build_sidecar`
                 on the same episodes — the load-bearing check that the
                 `_finalize_sidecar` extraction kept every invariant
  - Determinism  two independent materializations agree, including under the
                 lazy geometry cache and across processes
  - Decorrelation  the same cell is corrupted DIFFERENTLY in each trajectory
                 while its episode COUNT stays equal

Run with: pytest tests/test_seg_ablation_bank.py -v
"""

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from graph_dyno.seg_ablation import bank as bank_mod
from graph_dyno.seg_ablation.bank import (
    BANK_KINDS,
    PLAN_KINDS,
    bank_id,
    build_bank_plan,
    decorrelation,
    kind_seed,
    load_bank_spec,
    merge_chunks,
    normalize_bank_spec,
    plan_filename,
    resolve_collisions,
    rows_filename,
    select,
    select_frames,
)
from graph_dyno.seg_ablation.bank_apply import apply_bank, materialize_level
from graph_dyno.seg_ablation.bank_build import encode_chunk
from graph_dyno.seg_ablation.build import _atomic_save, sidecar_paths
from graph_dyno.seg_ablation.plan import MIN_EPISODE_GAP, abl_id, normalize_spec

# The mini cache, the deterministic encoder and the from-scratch re-encoder all
# already exist next door; reusing them is what makes "same as a direct build"
# an honest comparison rather than a re-implementation.
from test_seg_ablation import (  # noqa: F401  (mini_cache is a fixture)
    FakeEncoder,
    _clean_entry,
    _from_scratch,
    chain_graph,
    mini_cache,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: A bank sized for the 60-frame / 10-cell mini cache. `strict_balance` is off
#: because exact two-role balance on a 10-cell contact graph is a Hall
#: condition, not a property of the sampler — see plan.py's module docstring.
TEST_BANK = {
    "name": "testbank",
    "seed": 3,
    "max_episode_len": 20,
    "strict_balance": False,
    "kinds": {
        "partial": {"rate": 0.12, "mean_len": 4,
                    "area_frac_min": 0.55, "area_frac_max": 0.85},
        "scale": {"rate": 0.12, "mean_len": 4, "delta_px": 2.0},
        "merge": {"rate": 0.06, "mean_len": 1, "pad_factor": 2,
                  "close_px": 1.0},
        "miss": {"rate": 0.12, "mean_len": 3},
        "id_swap": {"rate": 0.06, "mean_len": 5},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _level(rates, *, sigma=0.0, bank="testbank", seed=0):
    """A level spec pinned to TEST_BANK's severities, varying only rates."""
    kinds = {}
    for k in PLAN_KINDS:
        entry = {kk: vv for kk, vv in TEST_BANK["kinds"][k].items()}
        entry["rate"] = float(rates.get(k, 0.0))
        kinds[k] = entry
    return normalize_spec({
        "enabled": True, "seed": seed, "bank": bank,
        "strict_balance": False, "max_episode_len": 20,
        "centroid_noise_sigma_rel": sigma, "kinds": kinds,
    })


def _write_bank_config(tmp_path, spec=None):
    """A `configs/seg_bank/`-shaped directory the read path can resolve."""
    spec = spec or TEST_BANK
    d = os.path.join(str(tmp_path), "seg_bank_configs")
    os.makedirs(d, exist_ok=True)
    OmegaConf.save(OmegaConf.create({"seg_bank": spec}),
                   os.path.join(d, f"{spec['name']}.yaml"))
    return d


def _plan_all(mini, bank_spec=None, traj_idx=0):
    """Write every kind's plan file for one trajectory."""
    bank_spec = bank_spec or TEST_BANK
    bid = bank_id(bank_spec)
    traj = torch.load(
        os.path.join(mini["dir"], f"{mini['cache_id']}_{traj_idx:04d}.pt"),
        map_location="cpu", weights_only=False)
    T, N, _ = traj["states"].shape
    plans = {}
    for kind in PLAN_KINDS:
        p = build_bank_plan(bank_spec, kind, cache_id=mini["cache_id"],
                            traj_idx=traj_idx, T=T, N=N,
                            states=traj["states"].numpy(),
                            W_interactions=traj["W_interactions"])
        _atomic_save(p, os.path.join(
            mini["dir"], plan_filename(bid, kind, mini["cache_id"], traj_idx)))
        plans[kind] = p
    return plans


def _encode_all(mini, bank_spec=None, traj_idx=0, n_chunks=2):
    """Encode + merge every bankable kind with the deterministic FakeEncoder."""
    bank_spec = bank_spec or TEST_BANK
    bid = bank_id(bank_spec)
    for kind in BANK_KINDS:
        for c in range(n_chunks):
            encode_chunk(mini["dir"], bank_spec, kind,
                         cache_id=mini["cache_id"], traj_idx=traj_idx,
                         companion_suffix=FakeEncoder.companion_suffix,
                         encoder=FakeEncoder(), n_chunks=n_chunks, chunk_idx=c,
                         workers=1, verbose=False)
        merge_chunks(mini["dir"], bid, kind, FakeEncoder.companion_suffix,
                     mini["cache_id"], traj_idx, expect_n_chunks=n_chunks,
                     verbose=False)


@pytest.fixture
def banked(mini_cache, tmp_path):
    """A mini cache with a fully planned and encoded `testbank`."""
    _plan_all(mini_cache)
    _encode_all(mini_cache)
    mini_cache["bank_configs"] = _write_bank_config(tmp_path)
    return mini_cache


def _materialize(mini, spec, **kw):
    return materialize_level(
        mini["dir"], spec, cache_id=mini["cache_id"], traj_idx=0,
        companion_suffix=FakeEncoder.companion_suffix, dt=mini["dt"],
        bank_configs_dir=mini["bank_configs"], verbose=False, **kw)


def _resolvable(cfg):
    """Supply the keys a level's interpolations reference.

    `joint_*pct` set `seed: ${model.seed}` so that a seed wave varies the
    corruption realization and not only model init. Hydra provides `model` at
    compose time; a test loading the level file on its own does not, and the
    interpolation would raise. Seed 0 is the right stand-in — it is the value
    `abl_id` was frozen against.
    """
    return OmegaConf.merge(OmegaConf.create({"model": {"seed": 0}}), cfg)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

class TestSelection:

    def test_select_is_strictly_nested(self, chain_graph):
        """d1 subset d2 subset d3 subset d4 — the whole point of the rank."""
        T, N, W = chain_graph
        plan = build_bank_plan(TEST_BANK, "partial", cache_id="x", traj_idx=0,
                               T=T, N=N, W_interactions=W)
        rates = [0.02, 0.05, 0.10, TEST_BANK["kinds"]["partial"]["rate"]]
        prev = None
        for r in rates:
            ids = set(np.flatnonzero(select(plan, r)).tolist())
            assert ids, f"rate {r} selected nothing"
            if prev is not None:
                assert prev <= ids, f"rate {r} is not a superset of the one below"
            prev = ids
        assert prev == set(range(plan["rank"].size)), \
            "selecting at the bank rate must take the whole bank"

    def test_per_slot_balance_survives_every_dose(self, chain_graph):
        T, N, W = chain_graph
        plan = build_bank_plan(TEST_BANK, "partial", cache_id="x", traj_idx=0,
                               T=T, N=N, W_interactions=W)
        slot = np.asarray(plan["slot"])
        for r in (0.02, 0.05, 0.10, 0.12):
            keep = select(plan, r)
            counts = np.bincount(slot[keep], minlength=N)
            assert counts.max() - counts.min() <= 1, \
                f"per-slot spread {counts.max() - counts.min()} at rate {r}"

    def test_min_episode_gap_holds_on_every_subset(self, chain_graph):
        """A subset of a gap-respecting placement still respects the gap."""
        T, N, W = chain_graph
        plan = build_bank_plan(TEST_BANK, "partial", cache_id="x", traj_idx=0,
                               T=T, N=N, W_interactions=W)
        for r in (0.02, 0.05, 0.10, 0.12):
            ids = np.flatnonzero(select(plan, r))
            by_slot = {}
            for i in ids:
                by_slot.setdefault(int(plan["slot"][i]), []).append(
                    (int(plan["t0"][i]), int(plan["t1"][i])))
            for slot, spans in by_slot.items():
                spans.sort()
                for a, b in zip(spans, spans[1:]):
                    assert b[0] - a[1] >= MIN_EPISODE_GAP, \
                        f"slot {slot} episodes {a} and {b} are too close"

    def test_selecting_above_the_bank_rate_is_fatal(self, chain_graph):
        T, N, W = chain_graph
        plan = build_bank_plan(TEST_BANK, "partial", cache_id="x", traj_idx=0,
                               T=T, N=N, W_interactions=W)
        with pytest.raises(ValueError, match="only holds"):
            select(plan, 0.5)


class TestCollisions:

    def _two_kind_plans(self, chain_graph):
        T, N, W = chain_graph
        plans, keeps = {}, {}
        for kind in ("partial", "scale"):
            p = build_bank_plan(TEST_BANK, kind, cache_id="x", traj_idx=0,
                                T=T, N=N, W_interactions=W)
            plans[kind] = p
            keeps[kind] = select(p, TEST_BANK["kinds"][kind]["rate"])
        return T, N, plans, keeps

    def test_resolution_is_deterministic(self, chain_graph):
        T, N, plans, keeps = self._two_kind_plans(chain_graph)
        a, _, da = resolve_collisions(plans, keeps, T=T, N=N)
        b, _, db = resolve_collisions(plans, keeps, T=T, N=N)
        for k in plans:
            assert np.array_equal(a[k], b[k])
        assert da == db

    def test_resolution_leaves_no_overlap_and_keeps_the_gap(self, chain_graph):
        T, N, plans, keeps = self._two_kind_plans(chain_graph)
        accepted, occ, _ = resolve_collisions(plans, keeps, T=T, N=N)
        spans = {}
        for kind, mask in accepted.items():
            for i in np.flatnonzero(mask):
                spans.setdefault(int(plans[kind]["slot"][i]), []).append(
                    (int(plans[kind]["t0"][i]), int(plans[kind]["t1"][i])))
        for slot, sp in spans.items():
            sp.sort()
            for a, b in zip(sp, sp[1:]):
                assert b[0] - a[1] >= MIN_EPISODE_GAP
        assert int(occ.sum()) == sum(
            b - a for sp in spans.values() for a, b in sp)

    def test_no_kind_is_systematically_favoured(self, chain_graph):
        """A random tiebreak, not a priority order.

        Kind priority is what made the saturated membrane_stiffness levels realise wildly
        different compositions in the two sharing modes: the first kind took
        everything it wanted and the last took what was left. Under a hash
        tiebreak both kinds keep a similar FRACTION of what they asked for.
        """
        T, N, plans, keeps = self._two_kind_plans(chain_graph)
        accepted, _, dropped = resolve_collisions(plans, keeps, T=T, N=N)
        frac = {k: accepted[k].sum() / max(keeps[k].sum(), 1) for k in plans}
        assert min(frac.values()) > 0.5, frac
        assert abs(frac["partial"] - frac["scale"]) < 0.15, frac
        assert sum(dropped.values()) > 0, \
            "these two kinds should collide at all — the test is vacuous"


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

class TestIdentity:

    def test_abl_id_separates_bank_derived_from_direct(self):
        direct = _level({"partial": 0.05}, bank=None)
        banked = _level({"partial": 0.05}, bank="testbank")
        assert direct["bank"] is None and banked["bank"] == "testbank"
        assert abl_id(direct) != abl_id(banked)

    def test_bank_id_changes_with_rate_and_with_code_version(self, monkeypatch):
        a = bank_id(TEST_BANK)
        other = copy.deepcopy(TEST_BANK)
        other["kinds"]["partial"]["rate"] = 0.2
        assert bank_id(other) != a
        monkeypatch.setattr(bank_mod, "SEG_ABLATION_CODE_VERSION", "segabl-v9")
        assert bank_id(TEST_BANK) != a

    def test_traj_idx_is_not_part_of_the_identity(self):
        assert bank_id(TEST_BANK) == bank_id(TEST_BANK)
        assert kind_seed(normalize_bank_spec(TEST_BANK), 0, "partial") != \
            kind_seed(normalize_bank_spec(TEST_BANK), 1, "partial")

    def test_shared_across_trajectories_with_a_bank_is_fatal(self):
        with pytest.raises(ValueError, match="not supported with"):
            normalize_spec({"bank": "testbank",
                            "shared_across_trajectories": True})

    def test_zero_rate_bank_kind_is_fatal(self):
        bad = copy.deepcopy(TEST_BANK)
        bad["kinds"]["miss"]["rate"] = 0.0
        with pytest.raises(ValueError, match="nothing to sub-select"):
            normalize_bank_spec(bad)

    def test_shipped_ladder_config_resolves(self):
        s = load_bank_spec("ladder_v1")
        assert s["name"] == "ladder_v1"
        assert set(s["kinds"]) == set(PLAN_KINDS)
        assert len(bank_id(s)) == 8

    def test_every_shipped_bank_level_is_selectable_from_its_own_bank(self):
        """Sweep ALL committed levels, each against the bank it declares.

        Not a parametrized sample and not pinned to one bank: a level that
        asked for more than its bank holds, or that drifted a severity field,
        would only fail at train time — after the data pipeline had already
        run. This is the cheap version of that check, and it also catches a
        level pointed at a bank that does not exist.
        """
        import glob

        checked = 0
        for path in sorted(glob.glob(os.path.join(
                ROOT, "configs", "seg_ablation", "*.yaml"))):
            cfg = _resolvable(OmegaConf.load(path))
            s = normalize_spec(cfg.seg_ablation)
            if not s["bank"]:
                continue                     # a directly-built level
            level = os.path.basename(path)[:-5]
            b = load_bank_spec(s["bank"])    # raises if the bank is missing
            for kind in PLAN_KINDS:
                assert s["kinds"][kind]["rate"] <= b["kinds"][kind]["rate"] + 1e-12, \
                    (f"{level}: {kind} rate {s['kinds'][kind]['rate']} exceeds "
                     f"bank {s['bank']}'s {b['kinds'][kind]['rate']}")
                for f in bank_mod.SEVERITY_FIELDS[kind]:
                    assert s["kinds"][kind][f] == b["kinds"][kind][f], \
                        f"{level}: {kind}.{f} drifts from bank {s['bank']}"
            checked += 1
        assert checked >= 25, f"only {checked} bank levels found"


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

class TestMerge:

    def test_refuses_to_delete_on_an_incomplete_part_set(self, mini_cache):
        """The failure `precompute_embeddings.merge_chunks` actually has.

        That function concatenates whatever it globs, deletes it, and leaves an
        unrecoverable short file. This one must refuse — and must leave the
        surviving chunks on disk so the wave can be re-run.
        """
        _plan_all(mini_cache)
        bid = bank_id(TEST_BANK)
        for c in range(3):
            encode_chunk(mini_cache["dir"], TEST_BANK, "partial",
                         cache_id=mini_cache["cache_id"], traj_idx=0,
                         companion_suffix=FakeEncoder.companion_suffix,
                         encoder=FakeEncoder(), n_chunks=3, chunk_idx=c,
                         workers=1, verbose=False)
        gone = os.path.join(mini_cache["dir"], rows_filename(
            bid, "partial", FakeEncoder.companion_suffix,
            mini_cache["cache_id"], 0, 1))
        os.remove(gone)

        with pytest.raises(ValueError, match="incomplete chunk set"):
            merge_chunks(mini_cache["dir"], bid, "partial",
                         FakeEncoder.companion_suffix,
                         mini_cache["cache_id"], 0, verbose=False)
        survivors = [c for c in (0, 2) if os.path.exists(os.path.join(
            mini_cache["dir"], rows_filename(
                bid, "partial", FakeEncoder.companion_suffix,
                mini_cache["cache_id"], 0, c)))]
        assert survivors == [0, 2], "a refused merge must delete nothing"
        assert not os.path.exists(os.path.join(
            mini_cache["dir"], rows_filename(
                bid, "partial", FakeEncoder.companion_suffix,
                mini_cache["cache_id"], 0)))

    def test_chunk_index_ordering_is_not_lexicographic(self, mini_cache):
        """`chunk10 < chunk2` must not be able to scramble time.

        Zero-padded names plus ordering by the stored `chunk_idx` — belt and
        braces, because the lexicographic sort is exactly the trap
        `precompute_embeddings.merge_chunks` has to cap `n_chunks <= 10` for.
        """
        _plan_all(mini_cache)
        bid = bank_id(TEST_BANK)
        n = 12
        for c in range(n):
            encode_chunk(mini_cache["dir"], TEST_BANK, "partial",
                         cache_id=mini_cache["cache_id"], traj_idx=0,
                         companion_suffix=FakeEncoder.companion_suffix,
                         encoder=FakeEncoder(), n_chunks=n, chunk_idx=c,
                         workers=1, verbose=False)
        _, merged = merge_chunks(
            mini_cache["dir"], bid, "partial", FakeEncoder.companion_suffix,
            mini_cache["cache_id"], 0, expect_n_chunks=n, verbose=False)
        t = merged["idx_t"].numpy()
        assert np.all(np.diff(t) >= 0), "merged rows are not time-ordered"

    def test_every_planned_episode_is_covered(self, banked):
        bid = bank_id(TEST_BANK)
        for kind in BANK_KINDS:
            rows = torch.load(os.path.join(banked["dir"], rows_filename(
                bid, kind, FakeEncoder.companion_suffix,
                banked["cache_id"], 0)), map_location="cpu",
                weights_only=False)
            plan = torch.load(os.path.join(banked["dir"], plan_filename(
                bid, kind, banked["cache_id"], 0)), map_location="cpu",
                weights_only=False)
            counts = np.bincount(rows["episode_id"].numpy(),
                                 minlength=plan["rank"].size)
            want = (np.asarray(plan["t1"], dtype=np.int64)
                    - np.asarray(plan["t0"], dtype=np.int64))
            assert np.array_equal(counts, want)
            assert torch.isfinite(rows["embeddings"]).all()
            assert rows["crop_sides"].dtype == torch.int32
            assert int(rows["crop_sides"].min()) > 0
            # float64, not float32. `_finalize_sidecar` adds this to the
            # float64 true position and only then rounds to float32, so a
            # float32 round trip perturbs the last bit of `states` for cells
            # on a rounding boundary. That is invisible on a 60x10 fixture and
            # showed up immediately on 150k real cell-frames, which is why the
            # dtype is asserted here rather than left to an equivalence test.
            assert rows["centroid_delta"].dtype == torch.float64


# ---------------------------------------------------------------------------
# Equivalence with a direct build
# ---------------------------------------------------------------------------

class TestEquivalence:

    @pytest.mark.parametrize("kind,rate", [("partial", 0.12), ("scale", 0.12),
                                           ("miss", 0.12)])
    def test_bank_level_matches_build_sidecar_bit_for_bit(
            self, mini_cache, tmp_path, monkeypatch, kind, rate):
        """The load-bearing check on the `_finalize_sidecar` extraction.

        Both paths are pinned to the SAME episodes by forcing the bank's
        per-(traj, kind) stream to the seed `build_sidecar` derives from the
        spec — `plan_events` with a single active kind and `plan_bank` then
        call `_plan_single_kind` with an identically seeded generator, so the
        placements coincide and everything downstream must agree exactly.
        """
        from graph_dyno.seg_ablation.build import build_sidecar

        direct_spec = _level({kind: rate}, sigma=1.5, bank=None)
        seed = int(direct_spec["seed"]) * 1_000_003 + 0
        monkeypatch.setattr(bank_mod, "kind_seed",
                            lambda spec, traj, k: seed)
        # Plan and encode AFTER the monkeypatch, on a fresh cache — the
        # `banked` fixture's rows belong to the unpinned seed.
        banked = mini_cache
        banked["bank_configs"] = _write_bank_config(tmp_path)
        _plan_all(banked)
        _encode_all(banked)

        _, _, direct_stats = build_sidecar(
            banked["dir"], direct_spec, cache_id=banked["cache_id"],
            traj_idx=0, companion_suffix=FakeEncoder.companion_suffix,
            dt=banked["dt"], encoder_factory=FakeEncoder, workers=1,
            verbose=False)
        d_geo_path, d_emb_path = sidecar_paths(
            banked["dir"], abl_id(direct_spec), FakeEncoder.companion_suffix,
            banked["cache_id"], 0)
        d_geo = torch.load(d_geo_path, map_location="cpu", weights_only=False)
        d_emb = torch.load(d_emb_path, map_location="cpu", weights_only=False)

        b_geo, b_emb, b_stats = _materialize(
            banked, _level({kind: rate}, sigma=1.5))

        assert b_geo["abl_id"] != d_geo["abl_id"], \
            "a bank-derived spec must not collide with a direct one"
        assert len(b_geo["events"]) == len(d_geo["events"])
        for key in ("op", "src_n", "src_t", "states", "next_states", "valid",
                    "next_valid", "W_edge_keep", "crop_sides_override"):
            assert torch.equal(b_geo[key], d_geo[key]), f"{key} differs"
        assert b_geo["W_indices_sha1"] == d_geo["W_indices_sha1"]
        assert b_stats["n_reencoded"] == direct_stats["n_reencoded"]
        assert b_stats["n_invalid_cell_frames"] == \
            direct_stats["n_invalid_cell_frames"]
        assert b_stats["edges_kept_frac"] == direct_stats["edges_kept_frac"]

        # The delta rows themselves: same (t, n) set, same values.
        def _table(e):
            return {(int(t), int(n)): (v.clone(), int(c))
                    for t, n, v, c in zip(e["idx_t"], e["idx_n"],
                                          e["embeddings"], e["crop_sides"])}
        bt, dt_ = _table(b_emb), _table(d_emb)
        assert set(bt) == set(dt_)
        for k in bt:
            assert torch.equal(bt[k][0], dt_[k][0]), f"embedding differs at {k}"
            assert bt[k][1] == dt_[k][1], f"crop_side differs at {k}"

    def test_applied_companion_matches_a_from_scratch_re_encode(self, banked):
        """End to end: corrupt the companion from the bank, then re-encode the
        whole observed trajectory independently and compare."""
        spec = _level({"partial": 0.12, "miss": 0.06, "id_swap": 0.06})
        entry = _clean_entry(banked)
        stats = apply_bank({0: entry}, banked["dir"], spec,
                           cache_id=banked["cache_id"],
                           companion_suffix=FakeEncoder.companion_suffix,
                           dt=banked["dt"],
                           bank_configs_dir=banked["bank_configs"],
                           verbose=False)
        assert len(stats) == 1 and stats[0]["n_events"] > 0

        geo_path, _ = sidecar_paths(
            banked["dir"], abl_id(spec), FakeEncoder.companion_suffix,
            banked["cache_id"], 0)
        geo = torch.load(geo_path, map_location="cpu", weights_only=False)
        e_ref, c_ref = _from_scratch(banked, geo)

        invalid = ~geo["valid"]
        got_e, got_c = entry["embeddings"], entry["crop_sides"]
        assert torch.allclose(got_e[~invalid], e_ref[~invalid], atol=1e-6)
        assert torch.equal(got_c[~invalid], c_ref[~invalid])
        assert got_c.dtype == torch.int32
        # Invalid slots carry the single empty-mask row and crop_side 0.
        assert int(got_c[invalid].abs().sum()) == 0

    def test_geometry_keys_reach_the_entry(self, banked):
        from graph_dyno.seg_ablation.apply import GEOMETRY_KEYS
        entry = _clean_entry(banked)
        apply_bank({0: entry}, banked["dir"], _level({"miss": 0.06}),
                   cache_id=banked["cache_id"],
                   companion_suffix=FakeEncoder.companion_suffix,
                   dt=banked["dt"], bank_configs_dir=banked["bank_configs"],
                   verbose=False)
        for k in GEOMETRY_KEYS:
            assert k in entry


# ---------------------------------------------------------------------------
# Determinism and the lazy geometry cache
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_two_materializations_agree(self, banked):
        """10 seeds of a level each recompute it independently."""
        spec = _level({"partial": 0.05, "merge": 0.03, "miss": 0.05,
                       "id_swap": 0.03, "scale": 0.05}, sigma=2.0)
        a_geo, a_emb, _ = _materialize(banked, spec, save_geo=False)
        b_geo, b_emb, _ = _materialize(banked, spec, save_geo=False)
        for key in ("op", "src_n", "states", "next_states", "valid",
                    "next_valid", "W_edge_keep"):
            assert torch.equal(a_geo[key], b_geo[key]), f"{key} differs"
        assert torch.equal(a_emb["embeddings"], b_emb["embeddings"])
        assert {k: v.tolist() for k, v in a_geo["bank"]["selected"].items()} \
            == {k: v.tolist() for k, v in b_geo["bank"]["selected"].items()}

    def test_cached_geometry_reproduces_a_cold_rebuild(self, banked):
        spec = _level({"partial": 0.05, "miss": 0.05, "id_swap": 0.03},
                      sigma=1.0)
        cold_geo, cold_emb, cold_stats = _materialize(banked, spec)
        geo_path, _ = sidecar_paths(
            banked["dir"], abl_id(spec), FakeEncoder.companion_suffix,
            banked["cache_id"], 0)
        assert os.path.exists(geo_path), "the first run must write the cache"

        warm_geo, warm_emb, warm_stats = _materialize(banked, spec)
        for key in ("op", "src_n", "states", "next_states", "valid",
                    "next_valid", "W_edge_keep"):
            assert torch.equal(cold_geo[key], warm_geo[key]), f"{key} differs"
        assert torch.equal(cold_emb["embeddings"], warm_emb["embeddings"])
        assert torch.equal(cold_emb["idx_t"], warm_emb["idx_t"])
        assert cold_stats["n_events"] == warm_stats["n_events"]

    def test_a_foreign_geometry_sidecar_is_fatal_not_silently_reused(
            self, banked):
        spec = _level({"partial": 0.05})
        geo_path, _ = sidecar_paths(
            banked["dir"], abl_id(spec), FakeEncoder.companion_suffix,
            banked["cache_id"], 0)
        _materialize(banked, spec)
        geo = torch.load(geo_path, map_location="cpu", weights_only=False)
        geo["bank"]["bank_id"] = "deadbeef"
        _atomic_save(geo, geo_path)
        with pytest.raises(ValueError, match="came from bank"):
            _materialize(banked, spec)

    def test_concurrent_materialization_is_safe(self, banked):
        """Pid-unique temp + os.replace: the loser publishes an identical file.

        Two processes, not two threads — `_atomic_save` guards against exactly
        the cross-process race the sibling seeds of one level create.
        """
        spec = _level({"partial": 0.05, "miss": 0.05})
        script = f"""
import os, sys, torch
sys.path.insert(0, {os.path.join(ROOT, 'tests')!r})
from test_seg_ablation import FakeEncoder
from graph_dyno.seg_ablation.bank_apply import materialize_level
import json
spec = json.loads(sys.argv[1])
geo, emb, st = materialize_level(
    {banked['dir']!r}, spec, cache_id={banked['cache_id']!r}, traj_idx=0,
    companion_suffix=FakeEncoder.companion_suffix, dt={banked['dt']!r},
    bank_configs_dir={banked['bank_configs']!r}, verbose=False)
torch.save({{"states": geo["states"], "valid": geo["valid"],
            "emb": emb["embeddings"], "idx_t": emb["idx_t"]}}, sys.argv[2])
"""
        outs = []
        procs = []
        for i in range(2):
            out = os.path.join(banked["dir"], f"race_{i}.pt")
            outs.append(out)
            procs.append(subprocess.Popen(
                [sys.executable, "-c", script, json.dumps(spec), out],
                cwd=ROOT))
        for p in procs:
            assert p.wait() == 0, "a concurrent materialization crashed"
        a = torch.load(outs[0], map_location="cpu", weights_only=False)
        b = torch.load(outs[1], map_location="cpu", weights_only=False)
        for k in a:
            assert torch.equal(a[k], b[k]), f"{k} differs between processes"


# ---------------------------------------------------------------------------
# Severity and portability
# ---------------------------------------------------------------------------

class TestSeverityAndPortability:

    def test_a_level_that_drifts_from_the_bank_severity_is_fatal(self, banked):
        spec = _level({"partial": 0.05})
        spec["kinds"]["partial"]["area_frac_min"] = 0.30
        with pytest.raises(ValueError, match="A bank pins severity"):
            _materialize(banked, spec)

    def test_a_level_above_the_bank_rate_is_fatal(self, banked):
        with pytest.raises(ValueError, match="only holds"):
            _materialize(banked, _level({"partial": 0.5}))

    def test_missing_rows_give_a_build_hint(self, banked):
        bid = bank_id(TEST_BANK)
        for kind in BANK_KINDS:
            os.remove(os.path.join(banked["dir"], rows_filename(
                bid, kind, FakeEncoder.companion_suffix,
                banked["cache_id"], 0)))
        with pytest.raises(FileNotFoundError, match="build_seg_bank_worker"):
            _materialize(banked, _level({"partial": 0.05}))

    def test_a_no_op_level_needs_no_bank_files_at_all(self, mini_cache,
                                                      tmp_path):
        """`bank_none`: every rate zero, so there is nothing to fetch."""
        mini_cache["bank_configs"] = _write_bank_config(tmp_path)
        geo, emb, stats = _materialize(mini_cache, _level({}))
        assert stats["n_events"] == 0
        assert int(emb["K"]) == 0
        assert bool(geo["valid"].all())
        assert torch.equal(geo["states"], mini_cache["states"])

    def test_the_same_bank_config_serves_a_second_cache(self, banked,
                                                        tmp_path):
        """Portability: nothing in the bank names a dataset.

        A second cache_id gets its own plan/rows/sidecar file set from the
        SAME `configs/seg_bank/` entry, with no config edits.
        """
        import shutil

        other_id = "cafef00d"
        shutil.copy(
            os.path.join(banked["dir"], f"{banked['cache_id']}_0000.pt"),
            os.path.join(banked["dir"], f"{other_id}_0000.pt"))
        other = dict(banked, cache_id=other_id)
        _plan_all(other)
        _encode_all(other)

        spec = _level({"partial": 0.05, "miss": 0.05})
        _materialize(banked, spec)
        _materialize(other, spec)
        bid = bank_id(TEST_BANK)
        for cid in (banked["cache_id"], other_id):
            assert os.path.exists(os.path.join(banked["dir"], plan_filename(
                bid, "partial", cid, 0)))
            g, _ = sidecar_paths(banked["dir"], abl_id(spec),
                                 FakeEncoder.companion_suffix, cid, 0)
            assert os.path.exists(g)


# ---------------------------------------------------------------------------
# Per-trajectory decorrelation
# ---------------------------------------------------------------------------

class TestDecorrelation:
    """The scan's explicit requirement: a given cell id must NOT receive the
    same corruption in all three trajectories, while `strict_balance` keeps its
    episode COUNT equal everywhere (that equality is the label-confound
    control, and is deliberately NOT decorrelated)."""

    @pytest.fixture
    def three_traj(self, mini_cache):
        import shutil
        for i in (1, 2):
            shutil.copy(
                os.path.join(mini_cache["dir"],
                             f"{mini_cache['cache_id']}_0000.pt"),
                os.path.join(mini_cache["dir"],
                             f"{mini_cache['cache_id']}_{i:04d}.pt"))
        return {kind: {i: _plan_all(mini_cache, traj_idx=i)[kind]
                       for i in (0, 1, 2)}
                for kind in PLAN_KINDS}

    def test_frames_and_params_differ_but_counts_do_not(self, three_traj,
                                                        mini_cache):
        N = mini_cache["N"]
        d = decorrelation(three_traj["partial"], N)
        for pair, e in d["pairs"].items():
            assert e["episode_overlap_frac"] < 0.05, \
                f"{pair}: episodes coincide across trajectories"
            assert abs(e["per_slot_count_r"]) < 0.9, pair
            # Two independent U(0, pi) draws differ by pi/3 = 1.047 on average.
            assert 0.6 < e["mean_abs_axis_delta"] < 1.5, pair
            # `side` is a coin flip per slot and this cache has ten of them, so
            # only the upper bound is a requirement: agreement near 1.0 would
            # mean the cut is a property of the cell rather than of the
            # (cell, trajectory) pair. A low value is just the coin.
            assert e["side_agreement"] < 0.9, pair
        # ... while the per-slot COUNTS stay equal everywhere.
        for i in (0, 1, 2):
            slot = np.asarray(three_traj["partial"][i]["slot"])
            c = np.bincount(slot, minlength=N)
            assert c.max() - c.min() <= 1

    def test_merge_partners_differ_across_trajectories(self, three_traj,
                                                       mini_cache):
        d = decorrelation(three_traj["merge"], mini_cache["N"])
        agreements = [e.get("partner_agreement") for e in d["pairs"].values()]
        agreements = [a for a in agreements if a is not None]
        assert agreements, "no shared slots to compare partners on"
        assert max(agreements) < 0.9

    def test_kind_streams_are_independent(self, mini_cache):
        """One kind's episode count must not shift another kind's draws."""
        s = normalize_bank_spec(TEST_BANK)
        seeds = {k: kind_seed(s, 0, k) for k in PLAN_KINDS}
        assert len(set(seeds.values())) == len(PLAN_KINDS)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

class TestReport:

    def test_realized_rates_are_recorded_per_run(self, banked):
        spec = _level({"partial": 0.06, "miss": 0.06})
        _, _, stats = _materialize(banked, spec)
        pk = stats["per_kind"]
        assert pk["partial"]["n_episodes"] > 0
        assert 0 < pk["partial"]["realized_rate"] <= 0.12
        assert pk["partial"]["bank_rate"] == 0.12
        assert "n_collision_drops" in pk["partial"]
        assert pk["scale"]["n_episodes"] == 0

    def test_doses_are_ordered_in_the_realized_rate(self, banked):
        """Monotone, not strictly increasing.

        `round(rho/rho_bank * slot_total)` can give two adjacent doses the same
        per-slot count when `slot_total` is small — on this 10-cell cache it is
        ~2, and it is ~4 for id_swap on the real one. That is exactly why the
        analysis plots against the REALIZED rate rather than the requested one.
        """
        got = []
        for r in (0.03, 0.06, 0.09, 0.12):
            _, _, stats = _materialize(banked, _level({"partial": r}),
                                       save_geo=False)
            got.append(stats["per_kind"]["partial"]["realized_rate"])
        assert all(b >= a for a, b in zip(got, got[1:])), got
        assert got[-1] > got[0], got


# ---------------------------------------------------------------------------
# Frame selection (`episodic: false`)
# ---------------------------------------------------------------------------

#: `abl_id` of every committed level, frozen on 2026-07-26 with 340 FINISHED
#: runs on disk in MLflow experiment `segabl_oscillation`.
#:
#: The sidecar filename carries nothing but this hash, so re-keying a level
#: orphans its geometry sidecar and silently divorces the analysis from the
#: runs that produced it. `episodic` was added to `_SPEC_DEFAULTS` after those
#: runs existed; `plan._ABL_ID_LEGACY_DEFAULTS` drops it from the payload when
#: it holds the legacy default so that specs which predate the key hash exactly
#: as before. This table is what proves that shim works — and it will catch the
#: next new spec key too, which is the point.
GOLDEN_ABL_IDS = {
    "bank_none": "ef956d6e",
    "id_swap_d1": "ebe3d6ad", "id_swap_d2": "ae1384a0",
    "id_swap_d3": "5b6b6c67", "id_swap_d4": "1c62c9d8",
    "merge_d1": "d3323146", "merge_d2": "e10a6e61",
    "merge_d3": "a62b5eee", "merge_d4": "5e3d8c5e",
    "miss_d1": "1c6150aa", "miss_d2": "c11d232e",
    "miss_d3": "8f191aee", "miss_d4": "8ce49736",
    "moderate": "534e3527",
    "partial_d1": "9918548a", "partial_d2": "8c7abd58",
    "partial_d3": "dc0b9e82", "partial_d4": "2a3073e8",
    "scale_d1": "068eccf9", "scale_d2": "ea23b633",
    "scale_d3": "6874aa21", "scale_d4": "62a15a96",
}


def test_committed_abl_ids_are_frozen():
    """No spec-schema change may re-key a level that already has runs."""
    import glob

    seen = {}
    for path in sorted(glob.glob(os.path.join(
            ROOT, "configs", "seg_ablation", "*.yaml"))):
        cfg = _resolvable(OmegaConf.load(path))
        if "seg_ablation" not in cfg:
            continue
        seen[os.path.basename(path)[:-len(".yaml")]] = abl_id(cfg.seg_ablation)

    drifted = {k: (v, seen[k]) for k, v in GOLDEN_ABL_IDS.items()
               if k in seen and seen[k] != v}
    assert not drifted, (
        f"abl_id changed for {sorted(drifted)} (was -> now: {drifted}). "
        f"Every sidecar and every logged abl_id for these levels is now "
        f"orphaned. If the change is intended, the runs must be re-done.")
    assert not (set(GOLDEN_ABL_IDS) - set(seen)), \
        f"committed level(s) disappeared: {sorted(set(GOLDEN_ABL_IDS) - set(seen))}"


class TestFrameSelection:
    """`episodic: false` — the cell-frame is the selection unit.

    Motivation, measured on `ladder_v1` / `more_osc_clusters`: under episodic
    selection a kind's realized rate moves in steps of `mean_len / T`, which is
    0.004 for partial and scale and 0.020 for id_swap. A joint arm balanced
    across four kinds at 1% of cell-frames needs 0.25% each and is therefore
    unbuildable — the attainable floor is ~1.4%. Frame selection drops the step
    to `1 / T` for every kind.
    """

    def _plans(self, chain_graph, kinds=("partial",)):
        T, N, W = chain_graph
        return {k: build_bank_plan(TEST_BANK, k, cache_id="x", traj_idx=0,
                                   T=T, N=N, W_interactions=W)
                for k in kinds}, T, N

    def test_rate_means_frames_per_cell(self, chain_graph):
        """`round(rate * T)` frames on every slot — the quantum is 1 / T."""
        plans, T, N = self._plans(chain_graph)
        for rate in (0.005, 0.01, 0.02, 0.03):
            sel, short = select_frames(plans, {"partial": rate})
            assert not short, f"rate {rate} ran out of capacity: {short}"
            ep, t = sel["partial"]
            counts = np.bincount(np.asarray(plans["partial"]["slot"])[ep],
                                 minlength=N)
            assert set(counts.tolist()) == {int(round(rate * T))}, \
                f"rate {rate}: per-slot counts {sorted(set(counts.tolist()))}"
            assert t.size == int(round(rate * T)) * N

    def test_it_reaches_rates_episode_selection_cannot(self, chain_graph):
        """The failure that motivated the flag, reproduced in miniature."""
        plans, T, N = self._plans(chain_graph)
        plan = plans["partial"]
        quantum = TEST_BANK["kinds"]["partial"]["mean_len"] / T   # 0.02

        def episodic_rate(r):
            keep = select(plan, r)
            t0 = np.asarray(plan["t0"])[keep]
            t1 = np.asarray(plan["t1"])[keep]
            return float((t1 - t0).sum()) / (T * N)

        def frame_rate(r):
            return float(select_frames(plans, {"partial": r})[0]["partial"][1]
                         .size) / (T * N)

        # A request of half the episodic quantum rounds to nothing at all...
        assert episodic_rate(quantum / 2) == 0.0
        # ...while frame selection lands on it to within one frame per cell.
        got = frame_rate(quantum / 2)
        assert abs(got - quantum / 2) <= 1.0 / T, got

        probes = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]
        assert len({round(frame_rate(r), 6) for r in probes}) > \
            len({round(episodic_rate(r), 6) for r in probes})

    def test_min_episode_gap_holds_between_selected_frames(self, chain_graph):
        """Without this an event costs fewer than 3 transitions under `strict`.

        Consecutive frames of one banked episode are adjacent by construction,
        so unlike `select` this cannot be inherited — it is enforced in the
        greedy accept and asserted here, across kinds as well as within one.
        """
        plans, T, N = self._plans(chain_graph, ("partial", "scale", "miss"))
        rates = {"partial": 0.02, "scale": 0.02, "miss": 0.02}
        sel, short = select_frames(plans, rates)
        assert not short
        by_slot = {}
        for kind, (ep, t) in sel.items():
            slot = np.asarray(plans[kind]["slot"])[ep]
            for s, tt in zip(slot.tolist(), t.tolist()):
                by_slot.setdefault(s, []).append(tt)
        for s, ts in by_slot.items():
            ts = np.sort(np.asarray(ts))
            assert np.unique(ts).size == ts.size, f"slot {s} has a duplicate"
            # spans are [t, t+1), so the gap rule is b - (a + 1) >= GAP
            assert (np.diff(ts) - 1).min() >= MIN_EPISODE_GAP, \
                f"slot {s} frames {ts.tolist()} are too close"

    def test_a_collision_costs_a_retry_not_a_frame(self, chain_graph):
        """The reason kinds are selected together.

        Selected per kind and reconciled afterwards, every kind lands below its
        quota and the pair kinds land furthest below — the composition skew
        frame selection exists to remove. Sharing the occupancy during
        selection, each kind still hits its quota exactly.
        """
        plans, T, N = self._plans(chain_graph, ("partial", "scale", "miss"))
        rates = {"partial": 0.02, "scale": 0.02, "miss": 0.02}
        sel, short = select_frames(plans, rates)
        assert not short
        for kind, rate in rates.items():
            ep, t = sel[kind]
            assert t.size == int(round(rate * T)) * N, \
                f"{kind} fell short under competition"

    def test_selection_is_deterministic(self, chain_graph):
        plans, _, _ = self._plans(chain_graph, ("partial", "merge"))
        rates = {"partial": 0.02, "merge": 0.01}
        a, _ = select_frames(plans, rates)
        b, _ = select_frames(plans, rates)
        for kind in rates:
            assert np.array_equal(a[kind][0], b[kind][0])
            assert np.array_equal(a[kind][1], b[kind][1])

    def test_every_selected_frame_lies_inside_a_banked_episode(self, chain_graph):
        """Otherwise the bank would hold no re-encoded row for it."""
        plans, _, _ = self._plans(chain_graph)
        ep, t = select_frames(plans, {"partial": 0.03})[0]["partial"]
        t0 = np.asarray(plans["partial"]["t0"])[ep]
        t1 = np.asarray(plans["partial"]["t1"])[ep]
        assert np.all((t >= t0) & (t < t1))

    def test_pair_kinds_gap_their_partner_slot_too(self, chain_graph):
        """id_swap is the kind with the coarsest episodic quantum (0.02 here)."""
        plans, T, N = self._plans(chain_graph, ("id_swap",))
        sel, short = select_frames(plans, {"id_swap": 0.01})
        ep, t = sel["id_swap"]
        assert t.size and not short
        partner = np.asarray(plans["id_swap"]["partner"])[ep]
        assert np.all(partner >= 0)
        # Both roles must respect the gap, or a swapped-into cell could lose
        # fewer than 3 transitions per event.
        by_slot = {}
        for role in (np.asarray(plans["id_swap"]["slot"])[ep], partner):
            for s, tt in zip(role.tolist(), t.tolist()):
                by_slot.setdefault(s, []).append(tt)
        for s, ts in by_slot.items():
            ts = np.sort(np.asarray(ts))
            if ts.size > 1:
                assert (np.diff(ts) - 1).min() >= MIN_EPISODE_GAP, \
                    f"slot {s} (either role) frames {ts.tolist()} too close"

    def test_shortfall_names_the_slots_that_could_not_be_filled(self, chain_graph):
        plans, T, N = self._plans(chain_graph)
        _, short = select_frames(plans, {"partial": 0.12})
        assert short and "partial" in short
        assert all(v < int(round(0.12 * T)) for v in short["partial"].values())


class TestFrameSelectionEndToEnd:

    def _frame_level(self, rates, **kw):
        spec = _level(rates, **kw)
        spec["episodic"] = False
        return normalize_spec(spec)

    def test_episodic_false_without_a_bank_is_fatal(self):
        with pytest.raises(ValueError, match="requires seg_ablation.bank"):
            normalize_spec({"enabled": True, "episodic": False,
                            "kinds": {"partial": {"rate": 0.05}}})

    def test_it_re_keys_the_level(self):
        a = abl_id(_level({"partial": 0.06}))
        b = abl_id(self._frame_level({"partial": 0.06}))
        assert a != b, "a frame-selected level is a different artifact"

    def test_materializes_off_the_same_bank_with_no_re_encode(self, banked):
        """No GPU, no new bank: the rows are already per cell-frame."""
        geo, emb, stats = self._materialize_ok(banked)
        assert stats["n_events"] > 0
        pk = stats["per_kind"]
        for kind in ("partial", "miss"):
            assert pk[kind]["episodic"] is False
            assert pk[kind]["mean_len"] == pytest.approx(1.0), \
                "a frame-selected event is one frame long by construction"

    def _materialize_ok(self, banked):
        # 1 frame per cell of each kind — T is 60 here, and the mini bank holds
        # only ~2.5 gap-separated frames per cell (see the capacity test below).
        return _materialize(banked, self._frame_level({"partial": 1 / 60,
                                                       "miss": 1 / 60}))

    def test_realized_rate_tracks_the_request(self, banked):
        """The property the calibrator existed to work around.

        Episodic joint arms had to be solved for numerically because the
        realized mix lands unevenly below the request. Here `rate` is frames
        per cell over T, so the only loss is cross-kind collisions — and with
        one kind selected there are none, making this EXACT rather than
        approximate.

        One frame per cell is all this fixture can carry: the mini bank's
        thinnest slot holds a single gap-separated frame (see the capacity test
        below). The production check for the higher rates is the materialize
        pass on the real cache, where the ladder_v1 slots hold ~100.
        """
        rate = 1 / 60
        _, _, stats = _materialize(
            banked, self._frame_level({"partial": rate}), save_geo=False)
        got = stats["per_kind"]["partial"]["realized_rate"]
        assert got == pytest.approx(rate, rel=1e-9), \
            f"requested {rate}, realized {got}"

    def test_exceeding_the_frame_capacity_is_fatal(self, banked):
        """Under the bank's nominal rate but over its FRAME capacity.

        MIN_EPISODE_GAP admits only about one frame in three of a banked
        episode, so frame capacity is far below `bank_rate * T` — 0.10 is well
        inside partial's 0.12 bank rate and still unbuildable. Filling what fits
        would quietly break per-slot balance for exactly the cells that ran
        out, which is the confound the stratification exists to remove.
        """
        with pytest.raises(ValueError, match="could only take"):
            _materialize(banked, self._frame_level({"partial": 2 / 60}),
                         save_geo=False)

    def test_severity_still_pins_the_pixels_but_not_mean_len(self, banked):
        """`mean_len` describes the bank's episodes, which frames slice."""
        spec = self._frame_level({"partial": 1 / 60})
        spec["kinds"]["partial"]["mean_len"] = 1.0      # honest for this level
        geo, _, stats = _materialize(banked, spec, save_geo=False)
        assert stats["per_kind"]["partial"]["n_episodes"] > 0

        drifted = self._frame_level({"partial": 1 / 60})
        drifted["kinds"]["partial"]["area_frac_min"] = 0.30
        with pytest.raises(ValueError, match="A bank pins severity"):
            _materialize(banked, drifted, save_geo=False)

    def test_the_episodic_ladder_is_untouched(self, banked):
        """Regression guard: the default path must not have moved at all."""
        _, _, stats = _materialize(banked, _level({"partial": 0.06}),
                                   save_geo=False)
        assert stats["per_kind"]["partial"]["episodic"] is True
        assert stats["per_kind"]["partial"]["mean_len"] > 1.0
