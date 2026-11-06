"""Bit-exactness tests for the batched rasterizer in raster.py.

This is the load-bearing test for the raster rewrite: the batched crossing test
must reproduce ``skimage.draw.polygon2mask`` *exactly*, not approximately, since
its output feeds the SDF crops and DINO embeddings.

Trajectory-level comparison is NOT a valid substitute — polyhoop's
``#pragma omp atomic`` force accumulation reorders at high thread counts, so
even a rerun of unmodified code diverges (max |dpos| 1.3e-2 over 25 frames at
omp=32). Equivalence has to be asserted at the snapshot level.
"""

import numpy as np
import pytest
import torch
from skimage.draw import polygon2mask

from graph_dyno.data.raster import rasterize_batch, rasterize_batch_torch


def _reference(polys, shape):
    return np.stack([polygon2mask(shape, p) for p in polys]).astype(np.float32)


def _random_polys(n, rng, size=128):
    """Blobby star-shaped polygons, roughly the shape/scale of real cells."""
    polys = []
    for _ in range(n):
        n_verts = rng.integers(6, 40)
        cy, cx = rng.uniform(0.3, 0.7, size=2) * size
        r = rng.uniform(8, size * 0.45, size=n_verts)
        theta = np.sort(rng.uniform(0, 2 * np.pi, size=n_verts))
        polys.append(np.stack((cy + r * np.sin(theta), cx + r * np.cos(theta)), axis=-1))
    return polys


def test_matches_skimage_on_random_polygons():
    rng = np.random.default_rng(0)
    polys = _random_polys(400, rng)
    shape = (128, 128)
    assert np.array_equal(rasterize_batch(polys, shape), _reference(polys, shape))


def test_matches_skimage_with_partly_out_of_bounds_polygons():
    """polygon2mask clips to the image; the batched version must clip identically."""
    rng = np.random.default_rng(1)
    polys = [p + rng.uniform(-90, 90, size=2) for p in _random_polys(100, rng)]
    shape = (128, 128)
    assert np.array_equal(rasterize_batch(polys, shape), _reference(polys, shape))


@pytest.mark.parametrize("chunk", [1, 7, 64, 1000])
def test_chunking_does_not_change_output(chunk):
    rng = np.random.default_rng(2)
    polys = _random_polys(50, rng)
    shape = (64, 64)
    assert np.array_equal(
        rasterize_batch(polys, shape, chunk=chunk), _reference(polys, shape)
    )


def test_ragged_vertex_counts():
    """Padding repeats the last vertex, which must not add spurious crossings."""
    rng = np.random.default_rng(3)
    polys = _random_polys(30, rng)
    assert len({len(p) for p in polys}) > 1, "fixture should be ragged"
    shape = (96, 96)
    assert np.array_equal(rasterize_batch(polys, shape), _reference(polys, shape))


def test_torch_variant_matches_skimage():
    rng = np.random.default_rng(4)
    polys = _random_polys(60, rng)
    shape = (64, 64)
    got = rasterize_batch_torch(polys, shape).numpy()
    assert np.array_equal(got, _reference(polys, shape))


def test_empty_input():
    assert rasterize_batch([], (32, 32)).shape == (0, 32, 32)


def test_out_parameter_and_strided_views():
    """Threading relies on rasterize_batch filling a strided view in place."""
    rng = np.random.default_rng(5)
    polys = _random_polys(24, rng)
    shape = (64, 64)
    out = np.zeros((24, 64, 64), dtype=np.float32)
    n_threads = 4
    for k in range(n_threads):
        rasterize_batch(polys[k::n_threads], shape, out=out[k::n_threads])
    assert np.array_equal(out, _reference(polys, shape))


def test_matches_skimage_on_real_snapshot(tmp_path):
    """End-to-end on real polyhoop cells, through the production code path."""
    from omegaconf import OmegaConf
    from graph_dyno.data.polyhoop_simulation import (
        PolyhoopSimulation, _snapshot_to_polys, _generate_image_from_snapshot,
    )
    from tests.test_polyhoop_simulation import base_config

    cfg = base_config.__wrapped__()
    cfg.data.n_objects = 12
    cfg.data.image_size = 64
    cfg.data.params.omp_threads = 1
    cfg.data.params.raster_threads = 1

    sim = PolyhoopSimulation(cfg, device="cpu")
    for _ in range(3):
        sim.step()

    snapshot = sim._snapshot_polygons()
    image_size = cfg.data.image_size
    polys = _snapshot_to_polys(snapshot, image_size, sim.max_possible_radius)
    reference = _reference(polys, (image_size, image_size))

    got = _generate_image_from_snapshot(
        snapshot, image_size, sim.max_possible_radius, "cpu",
    )
    assert np.array_equal(got.numpy(), reference)


def _legacy_images_from_snapshot(snapshot, image_size, r_max):
    """The pre-batching implementation: one skimage polygon2mask per polygon."""
    from graph_dyno.data.polyhoop_simulation import _rasterize_polygon_from_snapshot

    masks = np.zeros((len(snapshot), image_size, image_size), dtype=np.float32)
    for i, (cx, cy, verts_np) in enumerate(snapshot):
        centered = verts_np.copy()
        centered[:, 0] -= cx
        centered[:, 1] -= cy
        masks[i] = _rasterize_polygon_from_snapshot(centered, image_size, r_max)
    return masks


@pytest.mark.slow
def test_end_to_end_matches_legacy_rasterizer_at_omp1():
    """Threaded batched raster == old per-polygon skimage, over an evolving sim.

    Pinned to omp_threads=1 because polyhoop's `#pragma omp atomic` force
    accumulation reorders at higher thread counts: at omp=32 even a rerun of
    unmodified code diverges. At omp<=2 the physics is bit-reproducible, so any
    pixel difference here is attributable to the rasterizer.
    """
    from concurrent.futures import ThreadPoolExecutor
    from graph_dyno.data.polyhoop_simulation import PolyhoopSimulation, _generate_image_from_snapshot
    from tests.test_polyhoop_simulation import base_config

    cfg = base_config.__wrapped__()
    cfg.data.n_objects = 20
    cfg.data.image_size = 64
    cfg.data.params.omp_threads = 1
    cfg.data.params.raster_threads = 6

    sim = PolyhoopSimulation(cfg, device="cpu")
    image_size = cfg.data.image_size
    r_max = sim.max_possible_radius

    with ThreadPoolExecutor(max_workers=6) as ex:
        for _ in range(25):
            sim.step()
            snapshot = sim._snapshot_polygons()
            got = _generate_image_from_snapshot(
                snapshot, image_size, r_max, "cpu", executor=ex, n_threads=6,
            )
            expected = _legacy_images_from_snapshot(snapshot, image_size, r_max)
            assert np.array_equal(got.numpy(), expected)


@pytest.mark.slow
def test_generate_is_deterministic_across_raster_thread_counts():
    """generate()'s preallocated buffer + threading must not perturb the output."""
    from graph_dyno.data.polyhoop_simulation import PolyhoopSimulation
    from tests.test_polyhoop_simulation import base_config

    def run(raster_threads):
        cfg = base_config.__wrapped__()
        cfg.data.n_objects = 12
        cfg.data.image_size = 48
        cfg.data.generator.trajectory_length = 8
        cfg.data.params.omp_threads = 1
        cfg.data.params.raster_threads = raster_threads
        sim = PolyhoopSimulation(cfg, device="cpu")
        return sim.generate()

    serial = run(1)
    threaded = run(8)

    assert serial[4].shape == (8, 12, 48, 48)
    assert torch.equal(serial[4], threaded[4]), "images differ across raster_threads"
    assert torch.equal(serial[1], threaded[1]), "states differ across raster_threads"


def test_threaded_path_matches_serial_path():
    """The inner ThreadPoolExecutor split must not change a single pixel."""
    from concurrent.futures import ThreadPoolExecutor
    from graph_dyno.data.polyhoop_simulation import _generate_image_from_snapshot

    rng = np.random.default_rng(6)
    # Snapshot format: (cx, cy, verts) — center at origin so the scaling is a no-op.
    snapshot = [
        (0.0, 0.0, (p - 64.0) / 63.0 * 2.0) for p in _random_polys(37, rng)
    ]
    serial = _generate_image_from_snapshot(snapshot, 128, 1.0, "cpu")
    with ThreadPoolExecutor(max_workers=5) as ex:
        threaded = _generate_image_from_snapshot(
            snapshot, 128, 1.0, "cpu", executor=ex, n_threads=5,
        )
    assert np.array_equal(serial.numpy(), threaded.numpy())
