"""GPU + live-cache tests for the segmentation / tracking ablation.

Skipped by default (`pytest.ini` excludes `-m gpu`). This box has **one** GPU,
and two trainings just time-slice while each preloads the dataset to an 80 GB
GPU cache, so run these only when nothing else is on the card:

    nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # must be empty
    CUDA_VISIBLE_DEVICES=0 pytest tests/test_seg_ablation_gpu.py -m gpu -v -s

`CUDA_VISIBLE_DEVICES=0` is the Blackwell (95 GB) — CUDA's FASTEST_FIRST
ordering is the inverse of nvidia-smi's, and the tests assert that.

They exercise what `tests/test_seg_ablation.py` structurally cannot: the real
DINOv3 encoder, the real 78 GB trajectory cache, and `train.py` end to end
including the paths that only fire under a seg-ablation — the Risk-10
zero-gradient check at `train.py:486` (which can `raise RuntimeError` or write
`untrained_slots.json` into cwd), the clean-target val loss, and the grad-clip
fraction.

Requires the `moderate` sidecars to exist:

    CUDA_VISIBLE_DEVICES=0 python scripts/build_seg_ablation.py \\
        +experiment=more_osc_clusters +embedding=dino_sp +seg_ablation=moderate

The end-to-end test runs with `finetune_decoder.enabled=false`: the decoder
finetune is orthogonal to the seg-ablation path and costs 300 GB of RAM and
~15 min. Warming the real decoder
cache is a job for the ladder's first arm, not for a test.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPERIMENT = "more_osc_clusters"
LEVEL = "moderate"
OVERRIDES = [f"+experiment={EXPERIMENT}", "+ablation=full_int",
             "+embedding=dino_sp", f"+seg_ablation={LEVEL}"]

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    from hydra import compose, initialize_config_dir
    from graph_dyno.models.model_factory import resolve_shape_config
    with initialize_config_dir(config_dir=os.path.join(ROOT, "configs"),
                               version_base=None):
        c = compose(config_name="config", overrides=OVERRIDES)
    resolve_shape_config(c)
    return c


@pytest.fixture(scope="module")
def live(cfg):
    """The live cache's coordinates. Deliberately does NOT require sidecars —
    the bank tests build their own artifacts and must not be gated on a
    `moderate` build that may predate the current `abl_id`."""
    from graph_dyno.encoders.factory import get_companion_suffix
    from graph_dyno.data.trajectory_generator import generate_config_hash

    cache_dir = cfg.data.generator.cache_dir
    cache_id = generate_config_hash(cfg.data)
    traj = os.path.join(cache_dir, f"{cache_id}_0000.pt")
    if not os.path.exists(traj):
        pytest.skip(f"no live trajectory at {traj}")
    return dict(cache_dir=cache_dir, cache_id=cache_id,
                suffix=get_companion_suffix(cfg), traj=traj)


@pytest.fixture(scope="module")
def paths(cfg, live):
    from graph_dyno.seg_ablation.build import sidecar_paths
    from graph_dyno.seg_ablation.plan import abl_id

    abl = abl_id(cfg.seg_ablation)
    geo, emb = sidecar_paths(live["cache_dir"], abl, live["suffix"],
                             live["cache_id"], 0)
    if not (os.path.exists(geo) and os.path.exists(emb)):
        pytest.skip(f"sidecars for {LEVEL} ({abl}) not built — run "
                    f"scripts/build_seg_ablation.py first")
    return dict(live, abl=abl, geo=geo, emb=emb)


# ---------------------------------------------------------------------------

class TestEnvironment:

    def test_device_zero_is_the_big_card(self):
        """CUDA's FASTEST_FIRST ordering is the INVERSE of nvidia-smi's.

        Running on the 3.7 GB A400 by mistake makes DINO precompute ~24x
        slower (hours vs minutes), which is why this is asserted rather than
        assumed.
        """
        assert torch.cuda.is_available()
        p = torch.cuda.get_device_properties(0)
        gb = round(p.total_memory / 1024 ** 3)
        assert gb > 80, f"cuda:0 is {p.name} ({gb} GB) — set CUDA_VISIBLE_DEVICES=0"


class TestRealEncoder:

    def test_empty_mask_row_is_the_documented_fallback(self, cfg):
        """Every miss and every absorbed cell shares this one encoded row.

        `_mask_to_sdf_worker` (precompute_dino_embeddings.py:106-110) returns
        an all-(-1) SDF and crop_side 0 for an empty mask. The builder encodes
        it exactly once and broadcasts it, so if that contract ever changed,
        every invalid slot in the dataset would silently get a different
        embedding.
        """
        from graph_dyno.encoders.factory import create_shape_encoder
        enc = create_shape_encoder(cfg, device="cuda")
        pad = int(cfg.seg_ablation.kinds.merge.pad_factor)
        S = int(cfg.data.image_size)
        e, c = enc.encode_masks(
            torch.zeros((2, pad * S, pad * S), dtype=torch.uint8), batch_size=2)
        assert int(c[0]) == 0 and int(c[1]) == 0
        assert torch.allclose(e[0], e[1]), "the fallback must be deterministic"

    def test_padding_the_canvas_does_not_move_the_embedding(self, cfg, paths):
        """The load-bearing property for cross-arm comparability.

        Corrupted crops are encoded on a 256 canvas (a merged pair spans
        147-181 px) while the clean companions were encoded on 128. If padding
        moved the *embedding*, merged and unmerged cells would not be
        comparable — and the SDF-level test in test_seg_ablation.py cannot see
        a backbone that is sensitive to canvas size.
        """
        from graph_dyno.encoders.factory import create_shape_encoder
        from graph_dyno.seg_ablation import ops

        traj = torch.load(paths["traj"], map_location="cpu", mmap=True)
        imgs = traj["channeled_images"]
        rng = np.random.default_rng(0)
        ts, ns = rng.integers(200, 2000, 32), rng.integers(0, 500, 32)
        raw = [(imgs[int(t), int(n)].numpy() > 0.5).astype(np.uint8)
               for t, n in zip(ts, ns)]
        # skip any mask touching the 128 border: there the tight bbox is
        # clamped differently and the two are legitimately not identical
        raw = [m for m in raw if not (m[0].any() or m[-1].any()
                                      or m[:, 0].any() or m[:, -1].any())]
        assert len(raw) > 8

        enc = create_shape_encoder(cfg, device="cuda")
        e128, c128 = enc.encode_masks(torch.from_numpy(np.stack(raw)), batch_size=32)
        padded = np.stack([ops.pad_to(m, 2) for m in raw])
        e256, c256 = enc.encode_masks(torch.from_numpy(padded), batch_size=32)
        assert torch.equal(c128, c256)
        assert torch.allclose(e128, e256, atol=1e-4), \
            (e128 - e256).abs().max().item()


class TestSidecarOnLiveCache:

    def test_apply_preserves_every_invariant(self, cfg, paths):
        """`apply_sidecar` against the real 78 GB cache."""
        from graph_dyno.seg_ablation.apply import apply_sidecar, load_sidecars
        from graph_dyno.seg_ablation.build import assert_state_identities
        from graph_dyno.seg_ablation.plan import INVALID_OPS, normalize_spec

        spec = normalize_spec(cfg.seg_ablation)
        clean = torch.load(
            os.path.join(paths["cache_dir"],
                         f"{paths['suffix']}_{paths['cache_id']}_0000.pt"),
            map_location="cpu")
        entry = {"embeddings": clean["embeddings"].clone(),
                 "crop_sides": clean["crop_sides"].clone()}
        geo, emb = load_sidecars(paths["cache_dir"], spec, cache_id=paths["cache_id"],
                                 traj_idx=0, companion_suffix=paths["suffix"])
        apply_sidecar(entry, geo, emb, paths["traj"], spec=spec, dt=cfg.data.dt)

        T, N = entry["crop_sides"].shape
        assert_state_identities(entry["states"], entry["next_states"],
                                cfg.data.dt, where="live apply")

        op = geo["op"].numpy()
        src_n = geo["src_n"].numpy()
        ident = np.arange(N, dtype=np.int32)[None, :]
        expect = (op != 0) | (src_n != ident)
        changed = (entry["embeddings"] != clean["embeddings"]).any(-1).numpy()
        assert not (changed & ~expect).any(), "an event-free (t, n) moved"
        un = torch.from_numpy(~expect)
        assert torch.equal(entry["embeddings"][un], clean["embeddings"][un])
        assert torch.equal(entry["crop_sides"][un], clean["crop_sides"][un])
        assert entry["crop_sides"].dtype == torch.int32

        valid = entry["valid"].numpy()
        assert np.array_equal(
            valid, ~np.isin(op, INVALID_OPS)[np.arange(T)[:, None], src_n])
        idx = entry["W_interactions"]._indices().numpy()
        assert valid[idx[0], idx[1]].all() and valid[idx[0], idx[2]].all()
        assert entry["W_interactions"].is_coalesced()
        keys = idx[0].astype(np.int64) * N * N + idx[1] * N + idx[2]
        assert len(np.unique(keys)) == keys.size, "a duplicate edge doubles its message"

        pad = -4.0 * float(geo["r_max"])
        inv = ~valid
        assert np.allclose(entry["states"][..., :2].numpy()[inv], pad)
        assert int(entry["crop_sides"].numpy()[inv].max()) == 0

        # the guard band: every invalid frame costs its full +-1 window
        nv = entry["next_valid"].numpy()
        sup = (valid[:-1] & valid[1:] & nv[1:]).sum(0)
        for n in range(0, N, 7):
            f = np.flatnonzero(~valid[:, n])
            want = len({t for k in f for t in (k - 1, k, k + 1)
                        if 1 <= t <= T - 1})
            assert (T - 1) - sup[n] == want, (n, (T - 1) - sup[n], want)

    def test_label_correlation_is_under_the_gate(self, paths):
        geo = torch.load(paths["geo"], map_location="cpu", weights_only=False)
        worst = geo["stats"].get("worst_label_correlation")
        limit = geo["spec"]["max_label_correlation"]
        if worst is not None:
            assert abs(worst[2]) <= limit, worst


class TestTrainEndToEnd:

    @pytest.mark.slow
    def test_two_epochs(self, tmp_path, paths):
        """`train.py` end to end — the paths nothing else can reach.

        Asserts, beyond "it ran": the Risk-10 check is a clean no-op on the sim
        path (always-present slots => no RuntimeError, no untrained_slots.json),
        the final checkpoint self-identifies its ablation, and the ablation-only
        metrics reach MLflow.
        """
        import mlflow

        run_dir = tmp_path / "hydra"
        mlruns = tmp_path / "mlruns"
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        cmd = [sys.executable, os.path.join(ROOT, "train.py"), *OVERRIDES,
               "+experiment_name=segabl_gputest",
               f"+mlflow_uri={mlruns}",
               "training.n_epochs=2",
               # The decoder finetune is orthogonal to everything this test
               # asserts, and it is the whole cost: `_setup_dino` loads every
               # `sdf_*.npy` into one 300 GB tensor (311 GB read, ~7 min on
               # NFS) before the GPU does anything, then trains 4 epochs.
               # The seg-ablation path does not care whether the shape decoder
               # was finetuned — it consumes `model.shape_decoder` either way.
               # Leave it off so this stays a minutes-long test.
               "training.finetune_decoder.enabled=false",
               f"hydra.run.dir={run_dir}"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONPATH=ROOT)

        # Stream to a file rather than capture_output=True. A first run whose
        # decoder cache misses spends ~7 min reading 311 GB of sdf_*.npy before
        # the GPU does anything, and
        # buffering all of that until exit turns a 15-minute test into a black
        # box — diagnosing a hang meant reading /proc/<pid>/io.
        log = tmp_path / "train_stdout.log"
        print(f"\n  streaming train.py output to {log}", flush=True)
        with open(log, "w") as fh:
            r = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh,
                               stderr=subprocess.STDOUT, text=True, timeout=7200)
        out = log.read_text()
        assert r.returncode == 0, out[-6000:]

        # Risk 10: sim slots are always present, so nothing can be untrained.
        assert not (cwd / "untrained_slots.json").exists(), \
            (cwd / "untrained_slots.json").read_text()
        assert "grad reaches all" in out, \
            "the epoch-0 gradient check never ran — is `valid` reaching the batch?"

        final = run_dir / "graph_decoder_model.pth"
        assert final.exists()
        sd = torch.load(final, map_location="cpu", weights_only=False)
        assert "_segabl_id" in sd, "the checkpoint cannot identify its ablation"
        assert f"{int(sd['_segabl_id']):08x}" == paths["abl"]

        mlflow.set_tracking_uri(f"file://{mlruns}")
        client = mlflow.tracking.MlflowClient()
        exp = [e for e in client.search_experiments()
               if e.name == "segabl_gputest"][0]
        run = mlflow.search_runs(experiment_ids=[exp.experiment_id],
                                 output_format="list")[0]
        for k in ("val_clean_dino_mse", "val_position_huber_linear_frac",
                  "val_supervised_frac", "grad_clip_frac"):
            assert k in run.data.metrics, f"{k} not logged"
        assert run.data.tags.get("seg_ablation") == LEVEL
        assert run.data.params.get("seg_ablation.abl_id") == paths["abl"]
        # sim slots are always observed somewhere, but a moderate ablation
        # must actually mask some transitions
        assert 0.5 < run.data.metrics["val_supervised_frac"] < 1.0


# ---------------------------------------------------------------------------
# Embedding bank, on the real encoder and the real cache
# ---------------------------------------------------------------------------

#: A deliberately TINY bank: ~2500 crops through DINOv3, i.e. seconds, against
#: the ~375k rows/trajectory the real `ladder_v1` wave encodes. What it proves
#: is structural — that banked rows and freshly-encoded rows are the same
#: corrupted crop under the real backbone — and that does not need volume.
TINY_BANK = {
    "name": "gputestbank",
    "seed": 5,
    "max_episode_len": 200,
    "strict_balance": False,
    "kinds": {
        "partial": {"rate": 0.001, "mean_len": 8,
                    "area_frac_min": 0.55, "area_frac_max": 0.85},
        "scale": {"rate": 0.001, "mean_len": 8, "delta_px": 3.0},
        "merge": {"rate": 0.0005, "mean_len": 1, "pad_factor": 2,
                  "close_px": 1.0},
        "miss": {"rate": 0.001, "mean_len": 3},
        "id_swap": {"rate": 0.0005, "mean_len": 40},
    },
}


@pytest.fixture(scope="module")
def tiny_bank(cfg, live, tmp_path_factory):
    """Plan + encode TINY_BANK against a symlinked copy of trajectory 0.

    The cache dir is a tmp directory holding a symlink to the real 78 GB
    trajectory, so every bank artifact lands in tmp and nothing is written into
    the live cache.
    """
    from omegaconf import OmegaConf

    from graph_dyno.encoders.factory import create_shape_encoder
    from graph_dyno.seg_ablation.bank import (BANK_KINDS, PLAN_KINDS, bank_id,
                                   build_bank_plan, merge_chunks,
                                   plan_filename)
    from graph_dyno.seg_ablation.bank_build import encode_chunk
    from graph_dyno.seg_ablation.build import _atomic_save

    root = tmp_path_factory.mktemp("bank")
    cache_dir = root / "cache"
    cache_dir.mkdir()
    cache_id = live["cache_id"]
    os.symlink(live["traj"], cache_dir / f"{cache_id}_0000.pt")

    configs = root / "seg_bank"
    configs.mkdir()
    OmegaConf.save(OmegaConf.create({"seg_bank": TINY_BANK}),
                   configs / f"{TINY_BANK['name']}.yaml")

    traj = torch.load(live["traj"], map_location="cpu", weights_only=False,
                      mmap=True)
    T, N, _ = traj["states"].shape
    bid = bank_id(TINY_BANK)
    for kind in PLAN_KINDS:
        plan = build_bank_plan(TINY_BANK, kind, cache_id=cache_id, traj_idx=0,
                               T=T, N=N,
                               states=traj["states"].numpy(),
                               W_interactions=traj["W_interactions"])
        _atomic_save(plan, str(cache_dir / plan_filename(bid, kind, cache_id, 0)))

    encoder = create_shape_encoder(cfg, device="cuda")
    for kind in BANK_KINDS:
        for c in (0, 1):
            encode_chunk(str(cache_dir), TINY_BANK, kind, cache_id=cache_id,
                         traj_idx=0, companion_suffix=live["suffix"],
                         encoder=encoder, n_chunks=2, chunk_idx=c, workers=12,
                         verbose=True)
        merge_chunks(str(cache_dir), bid, kind, live["suffix"], cache_id, 0,
                     expect_n_chunks=2, verbose=True)
    del encoder
    torch.cuda.empty_cache()
    return dict(cache_dir=str(cache_dir), configs=str(configs), bid=bid,
                cache_id=cache_id, suffix=live["suffix"], T=T, N=N)


class TestBankOnTheRealEncoder:

    def test_banked_rows_are_finite_and_shaped_like_the_companion(
            self, tiny_bank, cfg):
        from graph_dyno.seg_ablation.bank import BANK_KINDS, load_rows

        for kind in BANK_KINDS:
            r = load_rows(tiny_bank["cache_dir"], tiny_bank["bid"], kind,
                          tiny_bank["suffix"], tiny_bank["cache_id"], 0,
                          mmap=False)
            assert r["K"] > 0
            assert r["embeddings"].shape == (r["K"], r["D"])
            assert torch.isfinite(r["embeddings"]).all()
            assert float(r["embeddings"].std()) > 1e-3
            assert r["crop_sides"].dtype == torch.int32
            assert int(r["crop_sides"].min()) > 0
            assert int(r["empty_emb"].shape[0]) == r["D"]

    def test_a_bank_level_matches_a_direct_build_on_the_real_backbone(
            self, tiny_bank, cfg, monkeypatch):
        """The equivalence check from the CPU suite, under real DINOv3.

        Both paths are pinned to the same episodes by forcing the bank's
        per-(traj, kind) seed to the one `build_sidecar` derives from the spec.
        """
        from graph_dyno.encoders.factory import create_shape_encoder
        from graph_dyno.seg_ablation import bank as bank_mod
        from graph_dyno.seg_ablation.bank import (BANK_KINDS, PLAN_KINDS, bank_id,
                                       build_bank_plan, merge_chunks,
                                       plan_filename)
        from graph_dyno.seg_ablation.bank_apply import materialize_level
        from graph_dyno.seg_ablation.bank_build import encode_chunk
        from graph_dyno.seg_ablation.build import _atomic_save, build_sidecar
        from graph_dyno.seg_ablation.plan import abl_id, normalize_spec

        rate = TINY_BANK["kinds"]["partial"]["rate"]

        def _spec(bank):
            kinds = {k: dict(TINY_BANK["kinds"][k], rate=0.0)
                     for k in PLAN_KINDS}
            kinds["partial"]["rate"] = rate
            return normalize_spec({
                "enabled": True, "seed": 0, "bank": bank,
                "strict_balance": False, "max_episode_len": 200,
                "centroid_noise_sigma_rel": 0.0, "kinds": kinds})

        direct_spec = _spec(None)
        seed = int(direct_spec["seed"]) * 1_000_003
        monkeypatch.setattr(bank_mod, "kind_seed", lambda s, t, k: seed)

        # Re-plan and re-encode under the pinned seed, in a directory of their
        # own so the module-scoped fixture stays intact for the other tests.
        cache_dir = tiny_bank["cache_dir"]
        cache_id = tiny_bank["cache_id"]
        traj = torch.load(os.path.join(cache_dir, f"{cache_id}_0000.pt"),
                          map_location="cpu", weights_only=False, mmap=True)
        T, N, _ = traj["states"].shape
        pinned = dict(TINY_BANK, name="gputestpinned")
        from omegaconf import OmegaConf
        OmegaConf.save(OmegaConf.create({"seg_bank": pinned}),
                       os.path.join(tiny_bank["configs"], "gputestpinned.yaml"))
        bid = bank_id(pinned)
        for kind in PLAN_KINDS:
            plan = build_bank_plan(pinned, kind, cache_id=cache_id, traj_idx=0,
                                   T=T, N=N,
                                   states=traj["states"].numpy(),
                                   W_interactions=traj["W_interactions"])
            _atomic_save(plan, os.path.join(
                cache_dir, plan_filename(bid, kind, cache_id, 0)))
        encoder = create_shape_encoder(cfg, device="cuda")
        for kind in BANK_KINDS:
            encode_chunk(cache_dir, pinned, kind, cache_id=cache_id,
                         traj_idx=0, companion_suffix=tiny_bank["suffix"],
                         encoder=encoder, n_chunks=1, chunk_idx=0, workers=12,
                         verbose=False)
            merge_chunks(cache_dir, bid, kind, tiny_bank["suffix"], cache_id,
                         0, expect_n_chunks=1, verbose=False)

        _, _, d_stats = build_sidecar(
            cache_dir, direct_spec, cache_id=cache_id, traj_idx=0,
            companion_suffix=tiny_bank["suffix"], dt=cfg.data.dt,
            encoder_factory=lambda: encoder, workers=12, verbose=False)
        del encoder
        torch.cuda.empty_cache()

        from graph_dyno.seg_ablation.build import sidecar_paths
        d_geo_p, d_emb_p = sidecar_paths(cache_dir, abl_id(direct_spec),
                                         tiny_bank["suffix"], cache_id, 0)
        d_geo = torch.load(d_geo_p, map_location="cpu", weights_only=False)
        d_emb = torch.load(d_emb_p, map_location="cpu", weights_only=False)

        b_geo, b_emb, b_stats = materialize_level(
            cache_dir, _spec("gputestpinned"), cache_id=cache_id, traj_idx=0,
            companion_suffix=tiny_bank["suffix"], dt=cfg.data.dt,
            bank_configs_dir=tiny_bank["configs"], verbose=False)

        assert b_geo["abl_id"] != d_geo["abl_id"]
        for key in ("op", "src_n", "states", "next_states", "valid",
                    "next_valid", "W_edge_keep"):
            assert torch.equal(b_geo[key], d_geo[key]), f"{key} differs"
        assert b_stats["n_reencoded"] == d_stats["n_reencoded"] > 0

        def _table(e):
            return {(int(t), int(n)): (v.clone(), int(c))
                    for t, n, v, c in zip(e["idx_t"], e["idx_n"],
                                          e["embeddings"], e["crop_sides"])}
        bt, dt_ = _table(b_emb), _table(d_emb)
        assert set(bt) == set(dt_)
        for k in bt:
            assert torch.equal(bt[k][0], dt_[k][0]), f"embedding differs at {k}"
            assert bt[k][1] == dt_[k][1]

    def test_apply_bank_loads_no_encoder(self, tiny_bank, cfg, monkeypatch):
        """`apply_bank` must stay GPU-free.

        290 concurrent training jobs each loading DINOv3 to re-encode is the
        N-times-GPU failure the whole design exists to avoid, and the only
        encode a level could still want — the single all-zero-mask row — is
        cached in the bank file.
        """
        import graph_dyno.encoders.factory as ef
        from graph_dyno.seg_ablation.bank import PLAN_KINDS
        from graph_dyno.seg_ablation.bank_apply import materialize_level
        from graph_dyno.seg_ablation.plan import normalize_spec

        def _boom(*a, **k):
            raise AssertionError("apply_bank loaded a shape encoder")
        monkeypatch.setattr(ef, "create_shape_encoder", _boom)

        kinds = {k: dict(TINY_BANK["kinds"][k]) for k in PLAN_KINDS}
        spec = normalize_spec({
            "enabled": True, "seed": 0, "bank": TINY_BANK["name"],
            "strict_balance": False, "max_episode_len": 200,
            "centroid_noise_sigma_rel": 1.0, "kinds": kinds})
        _, emb, stats = materialize_level(
            tiny_bank["cache_dir"], spec, cache_id=tiny_bank["cache_id"],
            traj_idx=0, companion_suffix=tiny_bank["suffix"], dt=cfg.data.dt,
            bank_configs_dir=tiny_bank["configs"], verbose=False)
        assert stats["n_invalid_cell_frames"] > 0, \
            "no invalid slots — the empty-mask path was not exercised"
        assert int(emb["K"]) > 0
        assert emb["embeddings"].shape[1] == emb["D"]

    def test_live_cache_invariants_on_a_materialized_level(self, tiny_bank,
                                                           cfg):
        """The sidecar invariants, on the real cache, via the bank path.

        These are what `TestSidecarOnLiveCache` asserts for a directly-built
        level; they matter just as much for a bank-derived one, and the two
        builders now share `_finalize_sidecar`, so this is where that sharing
        is checked against real geometry rather than a 10-cell fixture.
        """
        import numpy as np

        from graph_dyno.seg_ablation.bank import PLAN_KINDS
        from graph_dyno.seg_ablation.bank_apply import materialize_level
        from graph_dyno.seg_ablation.build import assert_state_identities
        from graph_dyno.seg_ablation.plan import INVALID_OPS, normalize_spec

        spec = normalize_spec({
            "enabled": True, "seed": 0, "bank": TINY_BANK["name"],
            "strict_balance": True, "max_episode_len": 200,
            "centroid_noise_sigma_rel": 0.0,
            "kinds": {k: dict(TINY_BANK["kinds"][k]) for k in PLAN_KINDS}})
        geo, _, stats = materialize_level(
            tiny_bank["cache_dir"], spec, cache_id=tiny_bank["cache_id"],
            traj_idx=0, companion_suffix=tiny_bank["suffix"], dt=cfg.data.dt,
            bank_configs_dir=tiny_bank["configs"], verbose=False)

        T, N = tiny_bank["T"], tiny_bank["N"]
        assert_state_identities(geo["states"], geo["next_states"], cfg.data.dt,
                                where="bank level on the live cache")

        op = geo["op"].numpy()
        src_n = geo["src_n"].numpy()
        valid = geo["valid"].numpy()
        assert np.array_equal(
            valid, ~np.isin(op, INVALID_OPS)[np.arange(T)[:, None], src_n])
        # A swapped slot never also carries a pixel op.
        ident = np.arange(N, dtype=np.int32)[None, :]
        assert not np.any((src_n != ident) & (op != 0))

        # No surviving edge touches an invalid slot — this is what makes a
        # padded slot harmless in gnn.py.
        traj = torch.load(os.path.join(
            tiny_bank["cache_dir"], f"{tiny_bank['cache_id']}_0000.pt"),
            map_location="cpu", weights_only=False, mmap=True)
        idx = traj["W_interactions"]._indices().numpy()
        keep = geo["W_edge_keep"].numpy()
        valid_true = ~np.isin(op, INVALID_OPS)
        assert valid_true[idx[0][keep], idx[1][keep]].all()
        assert valid_true[idx[0][keep], idx[2][keep]].all()

        # The guard band: every invalid frame costs its full +-1 window, so
        # lost transitions are exactly 3 per isolated event.
        nv = geo["next_valid"].numpy()
        sup = (valid[:-1] & valid[1:] & nv[1:]).sum(0)
        for n in range(0, N, 97):
            f = np.flatnonzero(~valid[:, n])
            want = len({t for k in f for t in (k - 1, k, k + 1)
                        if 1 <= t <= T - 1})
            assert (T - 1) - sup[n] == want, (n, (T - 1) - sup[n], want)

        worst = stats.get("worst_label_correlation")
        if worst is not None:
            assert abs(worst[2]) <= spec["max_label_correlation"], worst


class TestBankTrainEndToEnd:
    """`train.py` on a committed bank-derived level.

    Conditional on the real `ladder_v1` bank existing in the live cache, i.e.
    on the GPU wave having run. Before that it skips rather than failing —
    building the real bank is a 36-job cluster wave, not a test fixture.
    """

    @pytest.mark.slow
    def test_two_epochs_on_a_banked_level(self, tmp_path, cfg):
        import mlflow
        from hydra import compose, initialize_config_dir

        from graph_dyno.encoders.factory import get_companion_suffix
        from graph_dyno.seg_ablation.bank import bank_id, load_bank_spec, rows_filename
        from graph_dyno.seg_ablation.plan import abl_id, normalize_spec
        from graph_dyno.data.trajectory_generator import generate_config_hash

        level = "partial_d1"
        overrides = [f"+experiment={EXPERIMENT}", "+ablation/int=full",
                     "+embedding=dino_sp", f"+seg_ablation={level}"]
        with initialize_config_dir(config_dir=os.path.join(ROOT, "configs"),
                                   version_base=None):
            c = compose(config_name="config", overrides=overrides)
        from graph_dyno.models.model_factory import resolve_shape_config
        resolve_shape_config(c)
        cache_dir = c.data.generator.cache_dir
        cache_id = generate_config_hash(c.data)
        suffix = get_companion_suffix(c)
        spec = normalize_spec(c.seg_ablation)
        bid = bank_id(load_bank_spec(spec["bank"]))
        rows = os.path.join(cache_dir, rows_filename(
            bid, "partial", suffix, cache_id, 0))
        if not os.path.exists(rows):
            pytest.skip(f"the {spec['bank']} bank ({bid}) is not built on this "
                        f"cache yet — run the GPU wave first")

        run_dir = tmp_path / "hydra"
        mlruns = tmp_path / "mlruns"
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        cmd = [sys.executable, os.path.join(ROOT, "train.py"), *overrides,
               "+experiment_name=segabl_bank_gputest",
               f"+mlflow_uri={mlruns}",
               "training.n_epochs=2",
               "training.finetune_decoder.enabled=false",
               f"hydra.run.dir={run_dir}"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONPATH=ROOT)
        log = tmp_path / "train_stdout.log"
        print(f"\n  streaming train.py output to {log}", flush=True)
        with open(log, "w") as fh:
            r = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh,
                               stderr=subprocess.STDOUT, text=True, timeout=7200)
        out = log.read_text()
        assert r.returncode == 0, out[-6000:]
        assert "from bank" in out, "train.py did not take the bank path"

        final = run_dir / "graph_decoder_model.pth"
        sd = torch.load(final, map_location="cpu", weights_only=False)
        assert f"{int(sd['_segabl_id']):08x}" == abl_id(spec)

        mlflow.set_tracking_uri(f"file://{mlruns}")
        client = mlflow.tracking.MlflowClient()
        exp = [e for e in client.search_experiments()
               if e.name == "segabl_bank_gputest"][0]
        run = mlflow.search_runs(experiment_ids=[exp.experiment_id],
                                 output_format="list")[0]
        assert run.data.tags.get("seg_ablation") == level
        assert float(run.data.params["segabl.t0.partial.realized_rate"]) > 0
        for k in ("val_clean_dino_mse", "val_supervised_frac", "grad_clip_frac"):
            assert k in run.data.metrics, f"{k} not logged"

        # The geometry sidecar is written lazily by the first run of a level,
        # so a sibling seed reads it instead of rebuilding.
        from graph_dyno.seg_ablation.build import sidecar_paths
        geo, _ = sidecar_paths(cache_dir, abl_id(spec), suffix, cache_id, 0)
        assert os.path.exists(geo), "the lazy geometry cache was not written"
