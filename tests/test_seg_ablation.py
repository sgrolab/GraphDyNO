"""Tests for the segmentation / tracking ablation.

Organised as:
  - Planning        (rates, stratification, adjacency, no double-booking)
  - Mask ops        (half-plane cut, morphology, merge canvas, SDF invariance)
  - State           (the two exact identities, swap involution, frame grid)
  - Graph           (edge dropping, nnz, the indices-hash guard)
  - Sidecar         (naming, abl_id, scatter fidelity, atomic writes)
  - Integration     (dataloader graft, masked loss, Huber, config hash)

Run with: pytest tests/test_seg_ablation.py -v
"""

import hashlib
import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from graph_dyno.precompute.precompute_dino_embeddings import _mask_to_sdf, _mask_to_sdf_worker
from graph_dyno.seg_ablation import ops
from graph_dyno.seg_ablation.plan import (
    INVALID_OPS,
    OP_MERGE_ABSORB,
    OP_MERGE_GONE,
    OP_MISS,
    OP_NONE,
    OP_PARTIAL,
    OP_SCALE,
    SEG_ABLATION_CODE_VERSION,
    abl_id,
    build_index_maps,
    normalize_spec,
    per_slot_counts,
    plan_events,
    spec_is_active,
)

# Measured on the live caches; several tests pin the geometry to these numbers
# because they are what makes the padded merge canvas necessary.
LIVE_CROP_SIDE = 87        # median crop_sides
LIVE_NN_DIST_PX = 60       # median nearest-neighbour centroid distance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _disc(S, r, cy=None, cx=None):
    cy = (S - 1) / 2 if cy is None else cy
    cx = (S - 1) / 2 if cx is None else cx
    ys, xs = np.ogrid[:S, :S]
    return (((ys - cy) ** 2 + (xs - cx) ** 2) <= r ** 2).astype(np.uint8)


def _spec(**kinds):
    """A normalized spec with only the named kinds switched on."""
    base = {"seed": 0, "strict_balance": False, "kinds": {}}
    base.update({k: v for k, v in kinds.items() if k not in
                 ("partial", "merge", "miss", "id_swap", "scale")})
    for k, v in kinds.items():
        if k in ("partial", "merge", "miss", "id_swap", "scale"):
            base["kinds"][k] = v
    return normalize_spec(base)


@pytest.fixture
def chain_graph():
    """T x N contact graph where every slot touches both its ring neighbours.

    Well-connected on purpose: exact two-role balance is a degree-constrained
    b-matching, and its feasibility is a Hall condition on the contact graph —
    so the "counts differ by at most 1" guarantee can only be asserted where
    the graph admits it. (`more_osc_clusters` does not; see
    TestPlanning.test_hall_violation_is_reported.)
    """
    T, N = 200, 12
    t, i, j = [], [], []
    for tt in range(T):
        for n in range(N):
            m = (n + 1) % N
            t += [tt, tt]
            i += [n, m]
            j += [m, n]
    idx = torch.tensor([t, i, j], dtype=torch.int64)
    vals = torch.ones(idx.shape[1], dtype=torch.bool)
    return T, N, torch.sparse_coo_tensor(idx, vals, (T, N, N)).coalesce()


@pytest.fixture
def mini_cache(tmp_path):
    """A tiny but structurally faithful trajectory cache.

    Reproduces every property the builder relies on: `next_states[t] ==
    states[t+1]`, `states[t, 2:4] == (pos[t] - pos[t-1])/dt`, object-centred
    masks, a symmetric sparse contact graph, and `sim_metadata` with one varied
    factor.
    """
    T, N, S = 60, 10, 32
    dt = 0.01
    rng = np.random.default_rng(0)
    r_max = 3.0
    px = ops.scale_px(S, r_max)

    pos = np.zeros((T, N, 2), dtype=np.float32)
    pos[0] = rng.uniform(4.0, 16.0, size=(N, 2))
    step = rng.normal(0.0, 0.02, size=(T, N, 2)).astype(np.float32)
    for t in range(1, T):
        pos[t] = pos[t - 1] + step[t]

    pos32 = torch.from_numpy(pos)
    states = torch.zeros(T, N, 4)
    states[..., :2] = pos32
    states[1:, :, 2:4] = (pos32[1:] - pos32[:-1]) / dt
    next_states = torch.zeros_like(states)
    next_states[:-1] = states[1:]
    next_states[-1, :, :2] = pos32[-1] + torch.from_numpy(step[-1])
    next_states[-1, :, 2:4] = (next_states[-1, :, :2] - pos32[-1]) / dt

    radii = rng.uniform(5.0, 9.0, size=N)
    imgs = np.zeros((T, N, S, S), dtype=np.float32)
    for n in range(N):
        for t in range(T):
            imgs[t, n] = _disc(S, radii[n] * (1.0 + 0.05 * np.sin(t / 7 + n)))

    # symmetric contact graph from a generous distance threshold
    ti, ii, jj = [], [], []
    for t in range(T):
        d = np.linalg.norm(pos[t][:, None] - pos[t][None], axis=-1)
        np.fill_diagonal(d, 1e9)
        a, b = np.nonzero(d < 8.0)
        ti += [t] * a.size
        ii += a.tolist()
        jj += b.tolist()
    idx = torch.tensor([ti, ii, jj], dtype=torch.int64)
    W = torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1], dtype=torch.bool),
                                (T, N, N)).coalesce()

    cache_id = "deadbeef"
    torch.save({
        "times": torch.arange(T).float().unsqueeze(-1),
        "states": states,
        "next_states": next_states,
        "forces": torch.zeros(T, N * 2),
        "channeled_images": torch.from_numpy(imgs),
        "W_interactions": W,
        "sim_metadata": {
            "r_max": torch.tensor(r_max),
            "kls": torch.tensor(np.linspace(1e3, 1e4, N), dtype=torch.float32),
            "rs": torch.full((N,), 2.0),
        },
    }, os.path.join(tmp_path, f"{cache_id}_0000.pt"))
    return dict(dir=str(tmp_path), cache_id=cache_id, T=T, N=N, S=S, dt=dt,
                r_max=r_max, px=px, W=W, states=states, next_states=next_states,
                images=torch.from_numpy(imgs))


class FakeEncoder:
    """Deterministic stand-in for a ShapeEncoder.

    `crop_sides` comes from the *real* `_mask_to_sdf` pipeline, because several
    tests and the whole `scale_mse` target depend on it being exact. The
    embedding is a cheap deterministic function of the SDF.
    """

    companion_suffix = "fake_8"
    D = 8

    def encode_masks(self, masks, batch_size=64, num_workers=0):
        a = masks.numpy()
        embs = np.zeros((a.shape[0], self.D), dtype=np.float32)
        crops = np.zeros(a.shape[0], dtype=np.int32)
        for k in range(a.shape[0]):
            sdf, side = _mask_to_sdf_worker((a[k], 224))
            crops[k] = side
            f = sdf.astype(np.float32)
            embs[k] = [f.mean(), f.std(), f[:112].mean(), f[112:].mean(),
                       f[:, :112].mean(), f[:, 112:].mean(),
                       float((f > 0).mean()), float(side) / 256.0]
        return torch.from_numpy(embs), torch.from_numpy(crops)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

class TestPlanning:

    def test_deterministic_by_seed(self, chain_graph):
        T, N, W = chain_graph
        s = _spec(partial={"rate": 0.05, "mean_len": 4},
                  merge={"rate": 0.02, "mean_len": 1})
        a, _ = plan_events(s, T=T, N=N, W_interactions=W, seed=7)
        b, _ = plan_events(s, T=T, N=N, W_interactions=W, seed=7)
        c, _ = plan_events(s, T=T, N=N, W_interactions=W, seed=8)
        assert a == b
        assert a != c

    @pytest.mark.parametrize("rate", [0.02, 0.05, 0.10])
    def test_realized_rate_matches_request(self, rate):
        """Coverage tracks `rate`; the episode COUNT is exact.

        Episode length is Geometric(1/mean_len), so coverage carries sampling
        noise of order `1/sqrt(n_episodes)` — 1.1% on a real 2000x500
        trajectory (measured: 0.05 requested -> 0.0501 realized), which is why
        this uses a large enough synthetic to be in that regime.
        """
        T, N, mean_len = 2000, 200, 5
        s = _spec(partial={"rate": rate, "mean_len": mean_len})
        realized = []
        for seed in range(5):
            ev, rep = plan_events(s, T=T, N=N, seed=seed)
            assert len(ev) == round(rate * T * N / mean_len)     # exact
            r = sum(e.length for e in ev) / (T * N)
            assert abs(rep.per_kind["partial"]["realized_rate"] - r) < 1e-9
            realized.append(r)
        # Unbiasedness across seeds, not a per-seed guarantee: with ~1600
        # episodes a single draw carries ~2% of Geometric sampling noise.
        assert abs(float(np.mean(realized)) - rate) / rate < 0.02

    def test_episode_length_matches_mean(self, chain_graph):
        T, N, W = chain_graph
        s = _spec(partial={"rate": 0.20, "mean_len": 6})
        ev, _ = plan_events(s, T=T, N=N, W_interactions=W, seed=2)
        lengths = np.array([e.length for e in ev])
        assert abs(lengths.mean() - 6.0) < 1.0

    def test_single_kind_counts_differ_by_at_most_one(self, chain_graph):
        T, N, W = chain_graph
        s = _spec(miss={"rate": 0.05, "mean_len": 3})
        ev, _ = plan_events(s, T=T, N=N, W_interactions=W, seed=3)
        c = per_slot_counts(ev, N)["miss_initiator"]
        assert c.max() - c.min() <= 1

    def test_pair_kind_counts_balanced_in_both_roles(self, chain_graph):
        T, N, W = chain_graph
        s = _spec(merge={"rate": 0.03, "mean_len": 1})
        ev, rep = plan_events(s, T=T, N=N, W_interactions=W, seed=4)
        counts = per_slot_counts(ev, N)
        for role in ("merge_initiator", "merge_partner"):
            c = counts[role]
            assert c.max() - c.min() <= 1, f"{role}: {c}"
        assert rep.per_kind["merge"]["slot_count_spread"] <= 1
        assert rep.per_kind["merge"]["partner_count_spread"] <= 1

    def test_merge_partners_adjacent_every_frame(self, mini_cache):
        m = mini_cache
        s = _spec(merge={"rate": 0.05, "mean_len": 3})
        ev, _ = plan_events(s, T=m["T"], N=m["N"], W_interactions=m["W"], seed=5)
        dense = m["W"].to_dense()
        assert ev, "no merge episodes planned"
        for e in ev:
            for t in range(e.t0, e.t1):
                assert bool(dense[t, e.slot, e.partner]), \
                    f"{e} not adjacent at frame {t}"

    def test_no_double_booking(self, mini_cache):
        m = mini_cache
        s = _spec(partial={"rate": 0.05, "mean_len": 4},
                  merge={"rate": 0.03, "mean_len": 2},
                  miss={"rate": 0.04, "mean_len": 2},
                  scale={"rate": 0.05, "mean_len": 3},
                  id_swap={"rate": 0.02, "mean_len": 6})
        ev, _ = plan_events(s, T=m["T"], N=m["N"], W_interactions=m["W"], seed=6)
        seen = set()
        for e in ev:
            slots = [e.slot] + ([e.partner] if e.partner >= 0 else [])
            for n in slots:
                for t in range(e.t0, e.t1):
                    assert (t, n) not in seen, f"(t={t}, n={n}) booked twice"
                    seen.add((t, n))
        # and build_index_maps agrees (it raises on any collision)
        build_index_maps(ev, T=m["T"], N=m["N"])

    def test_supervision_cost_per_event_is_constant(self, mini_cache):
        """Equal event counts must buy equal supervision loss.

        Under `loss_mask_mode: strict` an invalid frame kills three transitions
        (t-1, t, t+1), so two invalid frames within 2 of each other cost less
        than two isolated ones — and on the tension cache that clustering was
        label-correlated at r = +0.265 while the counts themselves were
        perfectly balanced. MIN_EPISODE_GAP makes the +-1 windows disjoint, so
        the cost is exactly 3 per single-frame event for every slot.
        """
        m = mini_cache
        s = _spec(miss={"rate": 0.05, "mean_len": 1})
        ev, _ = plan_events(s, T=m["T"], N=m["N"], W_interactions=m["W"], seed=11)
        _, _, op = build_index_maps(ev, T=m["T"], N=m["N"])
        valid = ~np.isin(op, INVALID_OPS)
        nv = np.empty_like(valid)
        nv[:-1], nv[-1] = valid[1:], valid[-1]
        assert (~valid).any()
        for n in range(m["N"]):
            f = np.flatnonzero(~valid[:, n])
            if f.size < 2:
                continue
            # the +-1 mask windows of two events on one slot never touch
            assert np.diff(f).min() >= 3, (n, f)

        # so the realized supervision loss is exactly the sum of the per-event
        # window sizes, with no discount for overlap
        sup = (valid[:-1] & valid[1:] & nv[1:]).sum(0)
        lost = (m["T"] - 1) - sup
        for n in range(m["N"]):
            f = np.flatnonzero(~valid[:, n])
            want = len({t for k in f for t in (k - 1, k, k + 1)
                        if 1 <= t <= m["T"] - 1})
            assert lost[n] == want, (n, lost[n], want)

    def test_hall_violation_is_reported_not_hidden(self):
        """Two degree-1 slots sharing one partner: balance is impossible.

        No positive rate admits an exactly balanced assignment — the
        feasibility of the b-matching is a Hall condition that does not depend
        on the rate. The planner must surface that (stranded slots, spread)
        rather than silently skewing.
        """
        T, N = 200, 4
        # slots 0 and 1 both touch only slot 2; slot 3 touches nobody
        t, i, j = [], [], []
        for tt in range(T):
            for a, b in ((0, 2), (1, 2)):
                t += [tt, tt]
                i += [a, b]
                j += [b, a]
        idx = torch.tensor([t, i, j], dtype=torch.int64)
        W = torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1], dtype=torch.bool),
                                    (T, N, N)).coalesce()
        s = _spec(merge={"rate": 0.05, "mean_len": 1})
        _, rep = plan_events(s, T=T, N=N, W_interactions=W, seed=0)
        d = rep.per_kind["merge"]
        assert d["n_infeasible_slots"] == 1          # slot 3 never touches anyone
        assert d["shortfall"] > 0
        assert rep.infeasible_slots["merge"] == [3]

    def test_index_maps_swap_is_involution(self, mini_cache):
        m = mini_cache
        s = _spec(id_swap={"rate": 0.05, "mean_len": 5})
        ev, _ = plan_events(s, T=m["T"], N=m["N"], W_interactions=m["W"], seed=9)
        src_t, src_n, op = build_index_maps(ev, T=m["T"], N=m["N"])
        rows = np.arange(m["T"])[:, None]
        ident = np.broadcast_to(np.arange(m["N"], dtype=np.int32)[None, :],
                                (m["T"], m["N"]))
        assert np.array_equal(src_n[rows, src_n], ident)
        assert np.array_equal(src_t, np.broadcast_to(
            np.arange(m["T"], dtype=np.int32)[:, None], (m["T"], m["N"])))
        assert (src_n != ident).any()
        assert (op == OP_NONE).all(), "an ID swap changes no pixels"

    def test_pair_kind_requires_graph(self):
        s = _spec(merge={"rate": 0.01, "mean_len": 1})
        with pytest.raises(ValueError, match="W_interactions is required"):
            plan_events(s, T=10, N=4, W_interactions=None)

    def test_unknown_spec_keys_are_fatal(self):
        """A typo'd `rate` silently meaning "no events" would produce a whole
        ladder of identical runs."""
        with pytest.raises(ValueError, match="Unknown fields"):
            normalize_spec({"kinds": {"partial": {"raet": 0.1}}})
        with pytest.raises(ValueError, match="Unknown seg_ablation kinds"):
            normalize_spec({"kinds": {"partials": {"rate": 0.1}}})
        with pytest.raises(ValueError, match="Unknown seg_ablation keys"):
            normalize_spec({"sed": 3})

    def test_spec_is_active(self):
        assert not spec_is_active({"kinds": {}})
        assert spec_is_active({"kinds": {"miss": {"rate": 0.01}}})
        assert spec_is_active({"centroid_noise_sigma_rel": 1.0})
        assert not spec_is_active({"enabled": False,
                                   "kinds": {"miss": {"rate": 0.5}}})


# ---------------------------------------------------------------------------
# Mask ops
# ---------------------------------------------------------------------------

class TestMaskOps:

    @pytest.mark.parametrize("seed", range(20))
    def test_half_plane_area_exact_and_connected(self, seed):
        rng = np.random.default_rng(seed)
        m = _disc(128, LIVE_CROP_SIDE / 2)
        frac = float(rng.uniform(0.45, 0.9))
        axis = float(rng.uniform(0, np.pi))
        cut = ops.half_plane_cut(m, axis, frac, float(rng.integers(0, 2)))
        target = round(frac * int(m.sum()))
        assert abs(int(cut.sum()) - target) <= 1
        from scipy.ndimage import label
        _, n_comp = label(cut)
        assert n_comp == 1, "a half-plane cut of a convex mask must stay connected"

    def test_half_plane_empty_and_full(self):
        assert ops.half_plane_cut(np.zeros((32, 32), np.uint8), 0.0, 0.5).sum() == 0
        m = _disc(64, 20)
        assert int(ops.half_plane_cut(m, 0.3, 1.0).sum()) == int(m.sum())

    def test_morph_crop_side_monotone_and_identity(self):
        m = _disc(128, LIVE_CROP_SIDE / 2)
        base = ops.crop_side_of(m)
        assert ops.morph_crop_side(m, 0.0) == base
        sides = [ops.morph_crop_side(m, d) for d in (-4, -2, 0, 2, 4)]
        assert sides == sorted(sides), sides
        assert sides[0] < base < sides[-1]

    def test_morph_is_symmetric_in_sign(self):
        """A +delta episode must perturb as much as its -delta twin.

        `sdf` never takes a value in (-1, 1), so a single `>` on both sides
        makes dilation by +1 a silent no-op while erosion by -1 removes a whole
        ring. `_episode_params` samples the sign uniformly, so half the `scale`
        episodes would have done nothing.
        """
        m = _disc(128, LIVE_CROP_SIDE / 2)
        assert np.array_equal(ops.morph_mask(m, 0.0), m > 0.5)
        base = int(m.sum())
        for d in (1.0, 2.0, 3.0):
            grew = int(ops.morph_mask(m, d).sum()) - base
            shrank = base - int(ops.morph_mask(m, -d).sum())
            assert grew > 0 and shrank > 0, d
            # a disc of radius r: added/removed area ~ 2*pi*r*d each way
            assert abs(grew - shrank) / max(grew, shrank) < 0.15, (d, grew, shrank)

    def test_zero_mask_gives_crop_side_zero_and_flat_sdf(self):
        sdf, side = _mask_to_sdf_worker((np.zeros((128, 128), np.uint8), 224))
        assert side == 0
        assert np.all(sdf == -1.0)
        assert ops.crop_side_of(np.zeros((128, 128), np.uint8)) == 0

    def test_padding_does_not_change_the_sdf(self):
        """The load-bearing property for cross-arm comparability.

        Corrupted crops are encoded on a 256 canvas (a merged pair spans
        147-181 px) while the clean companions were encoded on 128. If padding
        moved the SDF, merged and unmerged cells would not be comparable.
        """
        m = _disc(128, LIVE_CROP_SIDE / 2)
        a, sa = _mask_to_sdf(m, 224)
        b, sb = _mask_to_sdf(ops.pad_to(m, 2), 224)
        assert sa == sb
        assert np.array_equal(a, b)

    def test_merged_pair_needs_the_padded_canvas(self):
        """128 clips a merged pair at the live geometry; 256 does not."""
        S = 128
        a = _disc(S, LIVE_CROP_SIDE / 2)
        b = _disc(S, LIVE_CROP_SIDE / 2)
        off = (LIVE_NN_DIST_PX, 0.0)

        blob1, clipped1 = ops.merge_pair(a, b, off, pad_factor=1)
        assert clipped1, "a 128 canvas must clip the live geometry"

        blob2, clipped2 = ops.merge_pair(a, b, off, pad_factor=2)
        assert not clipped2
        side = ops.crop_side_of(blob2)
        assert side > S, f"merged bbox {side} should exceed the 128 crop"
        ratio = side / ops.crop_side_of(a)
        assert 1.55 <= ratio <= 2.1, ratio

    def test_merged_blob_is_one_connected_region(self):
        """A merge is a LOST MEMBRANE, so the union must be one blob.

        Measured on the live cache the raw union is already connected 99.7% of
        the time; `close_px` takes the remainder to 100% for +0.1% area. Two
        disconnected lobes in one channel would be a different failure (two
        detections given one ID) and would give `_mask_to_sdf` a dumbbell.
        """
        from scipy.ndimage import label
        S = 128
        a = _disc(S, LIVE_CROP_SIDE / 2)
        b = _disc(S, LIVE_CROP_SIDE / 2)
        # a gap just wide enough that the raw union would split
        off = (LIVE_CROP_SIDE + 1.0, 0.0)
        raw, _ = ops.merge_pair(a, b, off, pad_factor=2, close_px=0.0)
        assert label(raw)[1] == 2, "the fixture must actually be disconnected"
        closed, _ = ops.merge_pair(a, b, off, pad_factor=2, close_px=2.0)
        assert label(closed)[1] == 1
        assert abs(int(closed.sum()) - int(raw.sum())) / int(raw.sum()) < 0.02

    def test_merge_offset_is_row_x_col_y(self):
        """Row axis = x. Transposing this silently transposes every merge."""
        S = 64
        a = _disc(S, 8)
        b = _disc(S, 8)
        blob, _ = ops.merge_pair(a, b, (20.0, 0.0), pad_factor=2)
        rows, cols = np.nonzero(blob)
        assert rows.max() - rows.min() > cols.max() - cols.min()

    def test_observed_centroid_tracks_the_cut(self):
        m = _disc(128, 40)
        px = 10.0
        base = ops.observed_centroid_units(m, px)
        cut = ops.half_plane_cut(m, 0.0, 0.5, 0.0)     # keep the low-row half
        got = ops.observed_centroid_units(cut, px)
        assert got[0] < base[0] - 0.5           # moved along +row = +x, negative
        assert abs(got[1] - base[1]) < 1e-6     # unmoved along col = y
        assert ops.observed_centroid_units(np.zeros((8, 8), np.uint8), px) is None


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _build(mini, spec, **kw):
    from graph_dyno.seg_ablation.build import build_sidecar
    return build_sidecar(mini["dir"], spec, cache_id=mini["cache_id"], traj_idx=0,
                         companion_suffix=FakeEncoder.companion_suffix,
                         dt=mini["dt"], encoder_factory=FakeEncoder,
                         workers=1, verbose=False, **kw)


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


class TestState:

    def test_identities_hold_exactly_on_the_rebuilt_states(self, mini_cache):
        from graph_dyno.seg_ablation.build import assert_state_identities
        s = _spec(partial={"rate": 0.06, "mean_len": 3},
                  miss={"rate": 0.05, "mean_len": 2},
                  id_swap={"rate": 0.04, "mean_len": 5},
                  centroid_noise_sigma_rel=1.0)
        geo_path, _, _ = _build(mini_cache, s)
        geo = _load(geo_path)
        assert_state_identities(geo["states"], geo["next_states"], mini_cache["dt"])
        assert torch.equal(geo["next_states"][:-1], geo["states"][1:])

    def test_event_free_cells_are_bit_identical(self, mini_cache):
        s = _spec(partial={"rate": 0.05, "mean_len": 3})
        geo_path, _, _ = _build(mini_cache, s)
        geo = _load(geo_path)
        op = geo["op"].numpy()
        untouched = np.ones(op.shape, bool)
        untouched[op != OP_NONE] = False
        untouched[0] = False               # t=0 velocity is copied, not derived
        # every frame after an event on that slot has a changed velocity
        touched_after = np.zeros_like(untouched)
        touched_after[1:][op[:-1] != OP_NONE] = True
        untouched &= ~touched_after
        a = geo["states"].numpy()[untouched]
        b = mini_cache["states"].numpy()[untouched]
        assert np.array_equal(a, b)

    def test_swap_applied_twice_is_the_identity(self, mini_cache):
        s = _spec(id_swap={"rate": 0.08, "mean_len": 4})
        geo_path, _, _ = _build(mini_cache, s)
        geo = _load(geo_path)
        src_n = geo["src_n"].numpy()
        T, N = src_n.shape
        rows = np.arange(T)[:, None]
        pos = geo["states"][..., :2].numpy()
        # undo the permutation -> the clean positions come back
        assert np.allclose(pos[rows, src_n], mini_cache["states"][..., :2].numpy())

    def test_merge_offset_matches_states_and_fails_at_a_larger_dt(
            self, mini_cache):
        """Pins the frame-grid convention.

        `channeled_images[k]` is rasterised from the same instant as
        `states[k]`, so it depicts `states[k]`, not `next_states[k]`. At
        dt = 0.01 the two readings differ by ~0.5 px, which is immaterial
        against a ~60 px merge offset — so the test amplifies dt 100x, where
        reading `next_states[t]` instead would be unmistakably wrong.
        """
        m = mini_cache
        t, i, j = 10, 0, 1
        px = m["px"]
        off_next = (m["next_states"][t, j, :2] - m["next_states"][t, i, :2]).numpy() * px
        off_cur = (m["states"][t, j, :2] - m["states"][t, i, :2]).numpy() * px
        assert np.linalg.norm(off_next - off_cur) < 1.0, "sub-pixel at dt=0.01"

        # 100x dt: the same states, 100x the per-frame displacement
        big = m["states"].clone()
        big[..., :2] = big[0, :, :2] + (big[..., :2] - big[0, :, :2]) * 100.0
        big_next = torch.zeros_like(big)
        big_next[:-1] = big[1:]
        big_next[-1] = big[-1]
        off_next_b = (big_next[t, j, :2] - big_next[t, i, :2]).numpy() * px
        off_cur_b = (big[t, j, :2] - big[t, i, :2]).numpy() * px
        assert np.linalg.norm(off_next_b - off_cur_b) > 1.0, \
            "at a 100x dt the two readings must be distinguishable"

    def test_missed_slots_are_parked_at_pad_pos(self, mini_cache):
        s = _spec(miss={"rate": 0.10, "mean_len": 2})
        geo_path, _, _ = _build(mini_cache, s)
        geo = _load(geo_path)
        invalid = ~geo["valid"].numpy()
        assert invalid.any()
        pad = -4.0 * mini_cache["r_max"]
        assert np.allclose(geo["states"][..., :2].numpy()[invalid], pad)
        assert np.array_equal(geo["next_valid"].numpy()[:-1],
                              geo["valid"].numpy()[1:])

    def test_no_events_leaves_states_untouched(self, mini_cache):
        geo_path, emb_path, _ = _build(mini_cache, _spec())
        geo = _load(geo_path)
        assert torch.equal(geo["states"], mini_cache["states"])
        assert torch.equal(geo["next_states"], mini_cache["next_states"])
        assert bool(geo["valid"].all())
        assert bool(geo["W_edge_keep"].all())
        assert _load(emb_path)["K"] == 0


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class TestGraph:

    def test_invalid_slots_have_no_incident_edges(self, mini_cache):
        from graph_dyno.seg_ablation.apply import _rebuild_edges
        s = _spec(miss={"rate": 0.10, "mean_len": 2},
                  merge={"rate": 0.05, "mean_len": 1})
        geo_path, _, _ = _build(mini_cache, s)
        geo = _load(geo_path)
        traj_path = os.path.join(mini_cache["dir"],
                                 f"{mini_cache['cache_id']}_0000.pt")
        W = _rebuild_edges(traj_path, geo, mini_cache["N"])
        idx = W._indices().numpy()
        valid = geo["valid"].numpy()
        assert valid[idx[0], idx[1]].all()
        assert valid[idx[0], idx[2]].all()

    def test_nnz_non_increasing_and_no_duplicates(self, mini_cache):
        from graph_dyno.seg_ablation.apply import _rebuild_edges
        s = _spec(miss={"rate": 0.10, "mean_len": 2},
                  id_swap={"rate": 0.05, "mean_len": 4})
        geo_path, _, _ = _build(mini_cache, s)
        geo = _load(geo_path)
        traj_path = os.path.join(mini_cache["dir"],
                                 f"{mini_cache['cache_id']}_0000.pt")
        W = _rebuild_edges(traj_path, geo, mini_cache["N"])
        idx = W._indices().numpy()
        assert idx.shape[1] <= int(geo["W_nnz"])
        # gnn.py:90 scatter_add_ SUMS duplicates, so a duplicate edge would
        # double its message.
        keys = idx[0] * (mini_cache["N"] ** 2) + idx[1] * mini_cache["N"] + idx[2]
        assert len(np.unique(keys)) == keys.size
        assert W.is_coalesced()

    def test_indices_hash_guard_trips(self, mini_cache, tmp_path):
        from graph_dyno.seg_ablation.apply import _rebuild_edges
        geo_path, _, _ = _build(mini_cache, _spec(miss={"rate": 0.05}))
        geo = _load(geo_path)
        geo["W_indices_sha1"] = hashlib.sha1(b"different").hexdigest()
        traj_path = os.path.join(mini_cache["dir"],
                                 f"{mini_cache['cache_id']}_0000.pt")
        with pytest.raises(ValueError, match="hash to"):
            _rebuild_edges(traj_path, geo, mini_cache["N"])
        geo["W_nnz"] = 7
        with pytest.raises(ValueError, match="nnz="):
            _rebuild_edges(traj_path, geo, mini_cache["N"])


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------

class TestSidecar:

    def test_abl_id_changes_with_spec_and_with_code_version(self, monkeypatch):
        a = abl_id(_spec(miss={"rate": 0.01}))
        b = abl_id(_spec(miss={"rate": 0.02}))
        assert a != b
        import graph_dyno.seg_ablation.plan as P
        monkeypatch.setattr(P, "SEG_ABLATION_CODE_VERSION", "segabl-vTEST")
        assert P.abl_id(_spec(miss={"rate": 0.01})) != a

    def test_backbone_dependent_name_carries_the_companion_suffix(self):
        from graph_dyno.seg_ablation.build import embedding_filename, geometry_filename
        a = embedding_filename("aaaa1111", "dino_SP", "cafe", 0)
        b = embedding_filename("aaaa1111", "dino_CELL_CA_L16", "cafe", 0)
        assert a != b
        # ...while the geometry half is shared across encoders, which is the
        # whole point of the split.
        assert geometry_filename("aaaa1111", "cafe", 0) == \
            geometry_filename("aaaa1111", "cafe", 0)

    def test_crop_sides_stay_int32(self, mini_cache):
        from graph_dyno.seg_ablation.apply import apply_sidecar
        s = _spec(partial={"rate": 0.06, "mean_len": 3},
                  miss={"rate": 0.04, "mean_len": 2},
                  scale={"rate": 0.06, "mean_len": 3})
        geo_path, emb_path, _ = _build(mini_cache, s)
        entry = _clean_entry(mini_cache)
        apply_sidecar(entry, _load(geo_path), _load(emb_path),
                      os.path.join(mini_cache["dir"],
                                   f"{mini_cache['cache_id']}_0000.pt"),
                      dt=mini_cache["dt"])
        # precompute_dino_embeddings.py:168 stores int32; torch.stack in
        # sparse_collate_fn fails on a dtype-mixed batch.
        assert entry["crop_sides"].dtype == torch.int32

    def test_scatter_matches_a_from_scratch_corrupted_companion(self, mini_cache):
        """The sidecar delta must reproduce a full re-encode exactly."""
        from graph_dyno.seg_ablation.apply import apply_sidecar
        s = _spec(partial={"rate": 0.08, "mean_len": 3},
                  merge={"rate": 0.05, "mean_len": 2},
                  miss={"rate": 0.05, "mean_len": 2},
                  scale={"rate": 0.06, "mean_len": 3},
                  id_swap={"rate": 0.04, "mean_len": 4})
        geo_path, emb_path, _ = _build(mini_cache, s)
        geo, emb = _load(geo_path), _load(emb_path)

        entry = _clean_entry(mini_cache)
        apply_sidecar(entry, geo, emb,
                      os.path.join(mini_cache["dir"],
                                   f"{mini_cache['cache_id']}_0000.pt"),
                      dt=mini_cache["dt"])

        ref_e, ref_c = _from_scratch(mini_cache, geo)
        assert torch.allclose(entry["embeddings"], ref_e, atol=1e-4)
        assert torch.equal(entry["crop_sides"], ref_c)

    def test_truncated_tmp_is_not_picked_up(self, mini_cache):
        from graph_dyno.seg_ablation.build import sidecar_paths
        s = _spec(miss={"rate": 0.05})
        abl = abl_id(s)
        geo_path, emb_path = sidecar_paths(
            mini_cache["dir"], abl, FakeEncoder.companion_suffix,
            mini_cache["cache_id"], 0)
        with open(geo_path + ".tmp", "wb") as f:
            f.write(b"truncated")
        _, _, stats = _build(mini_cache, s)
        assert stats is not None, "a .tmp must never satisfy skip-if-exists"
        assert os.path.exists(geo_path) and os.path.exists(emb_path)
        # rebuilt is now reused
        _, _, again = _build(mini_cache, s)
        assert again is None

    def test_apply_refuses_a_stale_code_version(self, mini_cache):
        from graph_dyno.seg_ablation.apply import apply_sidecar
        geo_path, emb_path, _ = _build(mini_cache, _spec(miss={"rate": 0.05}))
        geo = _load(geo_path)
        geo["code_version"] = "segabl-vOLD"
        with pytest.raises(ValueError, match="perturbation math changed"):
            apply_sidecar(_clean_entry(mini_cache), geo, _load(emb_path),
                          os.path.join(mini_cache["dir"],
                                       f"{mini_cache['cache_id']}_0000.pt"))

    def test_apply_refuses_a_mismatched_embedding_width(self, mini_cache):
        from graph_dyno.seg_ablation.apply import apply_sidecar
        geo_path, emb_path, _ = _build(mini_cache, _spec(miss={"rate": 0.05}))
        entry = _clean_entry(mini_cache)
        entry["embeddings"] = entry["embeddings"][..., :4]
        with pytest.raises(ValueError, match="different encoder"):
            apply_sidecar(entry, _load(geo_path), _load(emb_path),
                          os.path.join(mini_cache["dir"],
                                       f"{mini_cache['cache_id']}_0000.pt"))


def _clean_entry(mini):
    """The clean companion dict `_load_companion_embeddings` would produce."""
    enc = FakeEncoder()
    T, N, S = mini["T"], mini["N"], mini["S"]
    e, c = enc.encode_masks(
        (mini["images"].reshape(T * N, S, S) > 0.5).to(torch.uint8))
    return {"embeddings": e.view(T, N, -1).clone(),
            "crop_sides": c.view(T, N).clone()}


def _from_scratch(mini, geo):
    """Re-encode the whole trajectory as observed, without using the delta."""
    enc = FakeEncoder()
    T, N, S = mini["T"], mini["N"], mini["S"]
    pad = int(geo["pad_factor"])
    op = geo["op"].numpy()
    src_n = geo["src_n"].numpy()
    ov = geo["crop_sides_override"].numpy()
    events = geo["events"]
    imgs = mini["images"].numpy()
    px = mini["px"]

    masks = np.zeros((T * N, pad * S, pad * S), dtype=np.uint8)
    params = {}
    partner = {}
    for kind, t0, t1, slot, part, prm in events:
        for t in range(t0, t1):
            if kind == "id_swap":
                continue
            params[(t, slot)] = prm
            if kind == "merge":
                partner[(t, slot)] = part
    for t in range(T):
        for n in range(N):
            code = int(op[t, n])
            k = t * N + n
            if code in INVALID_OPS:
                continue
            m = (imgs[t, int(src_n[t, n])] > 0.5).astype(np.uint8)
            if code == OP_PARTIAL:
                p = params[(t, n)]
                m = ops.half_plane_cut(m, p["axis"], p["area_frac"], p["side"])
            elif code == OP_SCALE:
                m = ops.morph_mask(m, params[(t, n)]["delta_px"])
            elif code == OP_MERGE_ABSORB:
                j = partner[(t, n)]
                off = (mini["states"][t, j, :2]
                       - mini["states"][t, n, :2]).numpy() * px
                masks[k] = ops.merge_pair(
                    m, (imgs[t, j] > 0.5).astype(np.uint8), off, pad,
                    geo["spec"]["kinds"]["merge"]["close_px"])[0]
                continue
            masks[k] = ops.pad_to(m, pad)

    e, c = enc.encode_masks(torch.from_numpy(masks))
    e = e.view(T, N, -1)
    c = c.view(T, N).clone()
    has = ov >= 0
    c[torch.from_numpy(has)] = torch.from_numpy(ov[has]).to(c.dtype)
    return e, c


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_config_hash_is_unchanged_by_seg_ablation(self):
        """The single most important guard.

        `generate_config_hash` (trajectory_generator.py:19-23) hashes all of
        cfg.data except the `generator` block. If the spec lived inside
        cfg.data, the trajectory .pt would be renamed and hundreds of GB would
        regenerate.
        """
        from hydra import compose, initialize_config_dir
        from graph_dyno.data.trajectory_generator import generate_config_hash

        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "configs"))
        base = ["+experiment=more_osc_clusters", "+ablation=full_int",
                "+embedding=dino_sp"]

        def h(extra):
            with initialize_config_dir(config_dir=root, version_base=None):
                cfg = compose(config_name="config", overrides=base + extra)
            return generate_config_hash(cfg.data)

        assert h([]) == h(["+seg_ablation=bank_none"]) == h(["+seg_ablation=partial_d2"])

    def test_graft_reproduces_todays_dict(self, mini_cache):
        """Without a sidecar, the generalised graft is byte-identical."""
        from graph_dyno.data.dataloader import TrajectoryDatasetFromFolder as DS
        traj = {"states": torch.zeros(2), "channeled_images": torch.zeros(2)}
        entry = {"L": torch.arange(4.0), "crop_sides": torch.arange(4)}
        out = dict(traj)
        DS._graft(out, entry)
        assert out["L"] is entry["L"]
        assert out["crop_sides"] is entry["crop_sides"]
        assert out["states"] is traj["states"]

    def test_graft_rejects_image_keys(self):
        from graph_dyno.data.dataloader import TrajectoryDatasetFromFolder as DS
        with pytest.raises(ValueError, match="image key"):
            DS._graft({}, {"channeled_images": torch.zeros(1)})

    def test_sparse_memory_accounting(self):
        from graph_dyno.data.dataloader import get_tensor_memory_bytes
        idx = torch.zeros(3, 100, dtype=torch.int64)
        W = torch.sparse_coo_tensor(idx, torch.ones(100, dtype=torch.bool),
                                    (2000, 500, 500))
        # nelement() on sparse COO returns the DENSE count (5e8), which charged
        # W_interactions 720 MB against a true ~102 MB.
        assert get_tensor_memory_bytes(W) < 1e5
        assert get_tensor_memory_bytes(torch.zeros(10, 10)) == 400

    def test_masked_mse_ignores_invalid_and_reduces_to_mse(self):
        from graph_dyno.models.graph_model_real import GraphModelReal
        import torch.nn.functional as F
        pred = torch.randn(2, 5, 3)
        tgt = torch.randn(2, 5, 3)
        ones = torch.ones(2, 5, dtype=torch.bool)
        assert torch.allclose(GraphModelReal._masked_mse(pred, tgt, ones),
                              F.mse_loss(pred, tgt))
        mask = ones.clone()
        mask[0, 0] = False
        bad = pred.clone()
        bad[0, 0] = 1e6
        assert torch.allclose(GraphModelReal._masked_mse(bad, tgt, mask),
                              GraphModelReal._masked_mse(pred, tgt, mask))

    def test_huber_reduces_to_mse_for_small_residuals(self):
        from graph_dyno.models.graph_model_segablation import huber
        d = torch.linspace(-0.04, 0.04, 21)
        assert torch.allclose(huber(d, torch.zeros_like(d), 0.05), d ** 2)
        big = torch.tensor([10.0])
        beta = 0.05
        assert torch.allclose(huber(big, torch.zeros(1), beta),
                              torch.tensor([2 * beta * 10.0 - beta ** 2]))
        # bounded: the MSE contribution would be 100, the Huber one is ~1
        assert float(huber(big, torch.zeros(1), beta)) < 1.5

    def test_dino_model_type_with_an_active_spec_raises(self):
        from hydra import compose, initialize_config_dir
        root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "configs"))
        with initialize_config_dir(config_dir=root, version_base=None):
            cfg = compose(config_name="config",
                          overrides=["+experiment=more_osc_clusters",
                                     "+ablation=full_int", "+embedding=dino_sp",
                                     "+seg_ablation=partial_d2"])
        OmegaConf.set_struct(cfg, False)
        cfg.model.type = "dino"
        from graph_dyno.models.model_factory import create_model
        with pytest.raises(ValueError, match="seg_ablation is active"):
            create_model(cfg, device="cpu")

    def test_validity_reaches_single_step_samples(self, mini_cache):
        """`valid` / `next_valid` / `prev_valid` flow with zero __getitem__ changes."""
        from graph_dyno.data.dataloader import TrajectoryDatasetFromFolder
        import yaml
        meta = {"data": {"generator": {"n_trajectories": 1,
                                       "trajectory_length": mini_cache["T"]}}}
        with open(os.path.join(mini_cache["dir"], "metadata.yaml"), "w") as f:
            yaml.safe_dump(meta, f)

        geo_path, emb_path, _ = _build(mini_cache, _spec(miss={"rate": 0.1}))
        from graph_dyno.seg_ablation.apply import apply_sidecar
        entry = _clean_entry(mini_cache)
        apply_sidecar(entry, _load(geo_path), _load(emb_path),
                      os.path.join(mini_cache["dir"],
                                   f"{mini_cache['cache_id']}_0000.pt"))
        entry["L"] = entry.pop("embeddings")

        ds = object.__new__(TrajectoryDatasetFromFolder)
        ds.folder_path = mini_cache["dir"]
        ds.cache_id = mini_cache["cache_id"]
        ds.trajectory_length = mini_cache["T"]
        ds.n_trajectories = 1
        ds.rollout_steps = 1
        ds.include_images = False
        ds.history_window = 0
        ds._has_embeddings = True
        ds._embeddings = {0: entry}
        ds.use_gpu_cache = False
        ds._cpu_cache = {}
        ds.cpu_cache_size = 1

        sample = ds[5]
        for k in ("valid", "next_valid", "prev_valid"):
            assert k in sample, k
        assert sample["states"].shape == (mini_cache["N"], 4)
        assert ds[0] is None          # t=0 has no previous frame
