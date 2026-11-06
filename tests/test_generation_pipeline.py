"""Integration tests for per-index trajectory generation and the cache it writes.

These cover the seams the LSF generation wave depends on, none of which the
rasterizer tests in test_raster.py touch:

  1. a cache written by the NEW generate() path survives the whole downstream
     chain (precompute_all -> companion files -> dataloader -> batch),
  2. generate_one is idempotent and fills holes without touching its neighbours,
  3. metadata.yaml survives concurrent writers,
  4. the config hash ignores thread knobs but not physics.

Everything uses the Fourier shape backend, which is analytic and weight-free, so
the tests are hermetic (no gated checkpoints, no GPU).
"""

import multiprocessing as mp
import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from graph_dyno.data.trajectory_generator import (
    create_generator,
    generate_config_hash,
    generate_one,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _polyhoop_cfg(cache_dir, n_trajectories=2, trajectory_length=4, n_objects=6):
    """A polyhoop config small enough to generate in seconds."""
    return OmegaConf.create({
        "data": {
            "type": "simulation",
            "n_objects": n_objects,
            "image_size": 32,
            "domain_size": 20,
            "dt": 0.01,
            "seed": 42,
            "generator": {
                "n_trajectories": n_trajectories,
                "trajectory_length": trajectory_length,
                "seed_offset": 0,
                "cache_dir": str(cache_dir),
                "overwrite_existing": False,
            },
            "params": {
                "name": "polyhoop",
                "num_substeps": 10,
                "gam": 1000,
                "ka": 100000,
                "kr": 10000000,
                "cv": 10,
                "fp": 0,
                "num_vertices": 20,
                "radius": 1.0,
                "density_fraction": 0.3,
                "r_cutoff": 5.0,
                "growth_rate": 0,
                "oscillatory_growth": False,
                "area_fluct_period": 0,
                "area_fluct_amp": 0,
                # Keep the test single-threaded: polyhoop is only bit-reproducible
                # at low OMP thread counts (atomic force accumulation reorders).
                "omp_threads": 1,
                "raster_threads": 1,
            },
        },
        "shape": {
            "type": "fourier",
            "n_coefficients": 8,
            "output_size": 32,
        },
    })


@pytest.fixture
def generated_cache(tmp_path):
    """A real 2-trajectory cache produced by the per-index generation path."""
    cache_dir = tmp_path / "cache"
    cfg = _polyhoop_cfg(cache_dir)
    for i in range(cfg.data.generator.n_trajectories):
        generate_one(cfg, i, device="cpu")
    return cfg, str(cache_dir)


# ---------------------------------------------------------------------------
# 1. Round trip: generated cache -> embeddings -> dataloader -> batch
# ---------------------------------------------------------------------------

class TestCacheRoundTrip:
    """generate() now fills a preallocated buffer instead of torch.stack-ing a
    list. The tensor is equivalent, but nothing else asserted that a freshly
    generated cache still flows through the real consumers."""

    def test_trajectory_files_have_expected_structure(self, generated_cache):
        cfg, cache_dir = generated_cache
        gen = create_generator(cfg, device="cpu")
        T = cfg.data.generator.trajectory_length
        N = gen.n_trajectories

        for i in range(N):
            data = torch.load(gen._get_cache_path(i), map_location="cpu")
            for key in ("times", "states", "next_states", "forces",
                        "channeled_images", "W_interactions", "sim_metadata"):
                assert key in data, f"traj {i} missing {key}"
            imgs = data["channeled_images"]
            assert imgs.shape[0] == T
            assert imgs.shape[2:] == (cfg.data.image_size, cfg.data.image_size)
            assert imgs.dtype == torch.float32
            assert torch.isfinite(imgs).all()
            # Rasterizer output is a binary mask, and no frame may be empty.
            assert set(imgs.unique().tolist()) <= {0.0, 1.0}
            assert (imgs.sum(dim=(2, 3)) > 0).all(), "a cell rasterized to nothing"
            assert data["states"].shape[0] == T

    def test_precompute_all_consumes_generated_cache(self, generated_cache):
        from graph_dyno.precompute.precompute_embeddings import precompute_all, _companion_filename
        from graph_dyno.encoders.factory import get_companion_suffix

        cfg, cache_dir = generated_cache
        precompute_all(cache_dir=cache_dir, cfg=cfg, device="cpu", batch_size=16)

        suffix = get_companion_suffix(cfg)
        cache_id = generate_config_hash(cfg.data)
        T = cfg.data.generator.trajectory_length
        N = cfg.data.n_objects

        for i in range(cfg.data.generator.n_trajectories):
            path = os.path.join(cache_dir, _companion_filename(suffix, cache_id, i))
            assert os.path.exists(path), f"no companion file for traj {i}"
            comp = torch.load(path, map_location="cpu")
            assert comp["embeddings"].shape[:2] == (T, N)
            assert comp["crop_sides"].shape == (T, N)
            assert torch.isfinite(comp["embeddings"]).all()
            # crop_sides feeds true scale to the GNN; zero means a lost cell.
            assert (comp["crop_sides"] > 0).all()

    def test_dataloader_serves_batches_from_generated_cache(self, generated_cache):
        """The images path — what the new generate() actually changed."""
        from graph_dyno.data.dataloader import create_split_dataloaders_from_folder

        cfg, cache_dir = generated_cache
        train, val, test = create_split_dataloaders_from_folder(
            folder_path=cache_dir, batch_size=2, num_workers=0,
            train_ratio=0.5, val_ratio=0.25, test_ratio=0.25,
            persistent_workers=False, val_split_mode="time",
        )
        batch = next(iter(train))
        assert "channeled_images" in batch
        imgs = batch["channeled_images"]
        assert imgs.shape[-2:] == (cfg.data.image_size, cfg.data.image_size)
        assert torch.isfinite(imgs).all()

    def test_dataloader_serves_precomputed_embeddings(self, generated_cache):
        """The full train.py chain: companions -> bottlenecks -> dataloader."""
        from graph_dyno.precompute.precompute_embeddings import precompute_all
        from graph_dyno.encoders.factory import get_companion_suffix
        from graph_dyno.decoders.factory import create_shape_decoder
        from graph_dyno.data.dataloader import create_split_dataloaders_from_folder
        from train import _load_companion_embeddings, _precompute_bottlenecks

        cfg, cache_dir = generated_cache
        precompute_all(cache_dir=cache_dir, cfg=cfg, device="cpu", batch_size=16)

        cache_id = generate_config_hash(cfg.data)
        traj_indices = list(range(cfg.data.generator.n_trajectories))
        embeddings = _load_companion_embeddings(
            cache_dir, cache_id, get_companion_suffix(cfg), traj_indices)

        decoder = create_shape_decoder(cfg, device="cpu")
        embeddings = _precompute_bottlenecks(embeddings, decoder, "cpu")
        assert "L" in embeddings[0] and "embeddings" not in embeddings[0]

        train, _, _ = create_split_dataloaders_from_folder(
            folder_path=cache_dir, cfg=None, embeddings=embeddings,
            batch_size=2, num_workers=0,
            train_ratio=0.5, val_ratio=0.25, test_ratio=0.25,
            persistent_workers=False, val_split_mode="time",
        )
        # Mirrors train.py: with embeddings present the images are dropped from
        # the batch (and from the cache), so reading them would KeyError.
        underlying = train.dataset.dataset
        assert underlying.has_embeddings
        underlying.include_images = False

        batch = next(iter(train))
        assert "L" in batch and "crop_sides" in batch
        assert torch.isfinite(batch["L"]).all()
        assert (batch["crop_sides"] > 0).all()


# ---------------------------------------------------------------------------
# 2. generate_one idempotency / hole filling
# ---------------------------------------------------------------------------

class TestGenerateOneResume:
    """generate_all's resume is contiguous — _check_existing_dataset returns
    min(missing_trajectories) — so a hole at index 0 regenerates everything.
    generate_one must be per-index instead."""

    def test_second_call_is_a_noop(self, tmp_path):
        cfg = _polyhoop_cfg(tmp_path / "cache", n_trajectories=1)
        generate_one(cfg, 0, device="cpu")
        path = create_generator(cfg, device="cpu")._get_cache_path(0)
        before = os.stat(path).st_mtime_ns

        generate_one(cfg, 0, device="cpu")
        assert os.stat(path).st_mtime_ns == before, "existing trajectory rewritten"

    def test_fills_a_hole_without_touching_neighbours(self, tmp_path):
        """Generate index 1 into a cache missing index 0 — the exact case that
        would make generate_all redo the whole dataset."""
        cfg = _polyhoop_cfg(tmp_path / "cache", n_trajectories=3)
        gen = create_generator(cfg, device="cpu")

        generate_one(cfg, 2, device="cpu")
        p2 = gen._get_cache_path(2)
        mtime2 = os.stat(p2).st_mtime_ns

        generate_one(cfg, 1, device="cpu")

        assert not os.path.exists(gen._get_cache_path(0)), "index 0 was generated"
        assert os.path.exists(gen._get_cache_path(1))
        assert os.stat(p2).st_mtime_ns == mtime2, "index 2 was rewritten"

    def test_overwrite_existing_forces_regeneration(self, tmp_path):
        cfg = _polyhoop_cfg(tmp_path / "cache", n_trajectories=1)
        generate_one(cfg, 0, device="cpu")
        path = create_generator(cfg, device="cpu")._get_cache_path(0)
        before = os.stat(path).st_mtime_ns

        cfg.data.generator.overwrite_existing = True
        generate_one(cfg, 0, device="cpu")
        assert os.stat(path).st_mtime_ns != before

    def test_out_of_range_index_raises(self, tmp_path):
        cfg = _polyhoop_cfg(tmp_path / "cache", n_trajectories=2)
        with pytest.raises(IndexError):
            generate_one(cfg, 2, device="cpu")
        with pytest.raises(IndexError):
            generate_one(cfg, -1, device="cpu")

    def test_mismatched_cache_dir_raises_before_simulating(self, tmp_path):
        """A wrong cache_dir must fail fast, not silently mix two datasets."""
        cache = tmp_path / "cache"
        cfg = _polyhoop_cfg(cache, n_trajectories=1)
        generate_one(cfg, 0, device="cpu")

        other = _polyhoop_cfg(cache, n_trajectories=1)
        other.data.params.cv = 999  # physics change -> different hash
        with pytest.raises(ValueError, match="different dataset"):
            generate_one(other, 0, device="cpu")

    def test_seed_depends_only_on_index(self, tmp_path):
        """Jobs run in any order, so index -> data must be order-independent."""
        a = _polyhoop_cfg(tmp_path / "a", n_trajectories=2)
        b = _polyhoop_cfg(tmp_path / "b", n_trajectories=2)
        generate_one(a, 1, device="cpu")           # alone
        generate_one(b, 0, device="cpu")           # after a sibling
        generate_one(b, 1, device="cpu")

        ta = torch.load(create_generator(a, device="cpu")._get_cache_path(1),
                        map_location="cpu")
        tb = torch.load(create_generator(b, device="cpu")._get_cache_path(1),
                        map_location="cpu")
        assert torch.equal(ta["states"], tb["states"])
        assert torch.equal(ta["channeled_images"], tb["channeled_images"])


# ---------------------------------------------------------------------------
# 3. Concurrent metadata writes
# ---------------------------------------------------------------------------

def _write_metadata_worker(cache_dir, n_reps):
    """Hammer save_metadata_atomic from a separate process."""
    cfg = _polyhoop_cfg(cache_dir, n_trajectories=3)
    gen = create_generator(cfg, device="cpu")
    for _ in range(n_reps):
        gen.save_metadata_atomic()


class TestConcurrentMetadata:
    """Every job in a wave writes the same metadata.yaml. save_metadata()
    truncates then writes, so a reader could see a partial file; the atomic
    variant renames into place instead."""

    def test_concurrent_writers_leave_a_valid_file(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir, exist_ok=True)
        ctx = mp.get_context("spawn")   # fork + OpenMP threads deadlocks
        procs = [ctx.Process(target=_write_metadata_worker, args=(cache_dir, 20))
                 for _ in range(4)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=180)
            assert p.exitcode == 0, f"writer failed with exitcode {p.exitcode}"

        path = os.path.join(cache_dir, "metadata.yaml")
        loaded = OmegaConf.load(path)   # raises if truncated/interleaved
        expected = _polyhoop_cfg(cache_dir, n_trajectories=3)
        assert generate_config_hash(loaded.data) == generate_config_hash(expected.data)
        assert loaded.data.generator.n_trajectories == 3

    def test_no_temp_files_left_behind(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        cfg = _polyhoop_cfg(cache_dir, n_trajectories=1)
        gen = create_generator(cfg, device="cpu")
        gen.save_metadata_atomic()
        leftovers = [f for f in os.listdir(cache_dir) if ".tmp." in f]
        assert leftovers == [], f"temp files left: {leftovers}"


# ---------------------------------------------------------------------------
# 4. Config hash: thread knobs excluded, physics not
# ---------------------------------------------------------------------------

class TestConfigHashThreadKnobs:
    """omp_threads/raster_threads live in data.params, which is hashed. If they
    entered the cache key, tuning threads per node would orphan the cache."""

    def test_thread_knobs_do_not_change_the_hash(self, tmp_path):
        base = _polyhoop_cfg(tmp_path / "c")
        del base.data.params.omp_threads
        del base.data.params.raster_threads
        h0 = generate_config_hash(base.data)

        for omp, raster in [(1, 1), (8, 4), (48, 16), (128, 32)]:
            tuned = _polyhoop_cfg(tmp_path / "c")
            tuned.data.params.omp_threads = omp
            tuned.data.params.raster_threads = raster
            assert generate_config_hash(tuned.data) == h0, (
                f"omp={omp} raster={raster} changed the cache key")

    def test_generator_block_still_excluded(self, tmp_path):
        a = _polyhoop_cfg(tmp_path / "c", n_trajectories=2, trajectory_length=4)
        b = _polyhoop_cfg(tmp_path / "c", n_trajectories=9, trajectory_length=99)
        assert generate_config_hash(a.data) == generate_config_hash(b.data)

    @pytest.mark.parametrize("key,value", [
        ("cv", 999), ("ka", 1), ("num_vertices", 33), ("radius", 2.0),
    ])
    def test_physics_params_still_change_the_hash(self, tmp_path, key, value):
        base = _polyhoop_cfg(tmp_path / "c")
        h0 = generate_config_hash(base.data)
        changed = _polyhoop_cfg(tmp_path / "c")
        changed.data.params[key] = value
        assert generate_config_hash(changed.data) != h0, (
            f"changing {key} did not change the cache key")

    def test_top_level_data_fields_still_change_the_hash(self, tmp_path):
        base = _polyhoop_cfg(tmp_path / "c")
        h0 = generate_config_hash(base.data)
        for field, value in [("n_objects", 7), ("image_size", 64), ("seed", 1)]:
            changed = _polyhoop_cfg(tmp_path / "c")
            changed.data[field] = value
            assert generate_config_hash(changed.data) != h0, f"{field} ignored"
