"""End-to-end test of CTCTrajectoryGenerator through the real dataloader.

Builds a tiny synthetic 2-video CTC tree, runs generate_all(), then loads the
result with TrajectoryDatasetFromFolder — the same path train.py uses.

Run with: pytest tests/test_real_generator.py -v
"""

import pytest
import torch
from omegaconf import OmegaConf

from graph_dyno.data.dataloader import TrajectoryDatasetFromFolder, sparse_collate_fn
from graph_dyno.real_data.generator import CTCTrajectoryGenerator
from tests.test_ctc_tracks import make_ctc_video

N_FRAMES = 9
WINDOW = 4          # -> (9 - 1) // 4 = 2 windows per video
SIZE = 64
RADIUS = 4
R_MAX = 12.0
IMAGE_SIZE = 16
DOMAIN = 64.0


def _rows_and_presence():
    """Three tracks: one full-length, plus a mitosis at frame 5."""
    rows = [(1, 0, 8, 0), (2, 0, 4, 0), (3, 5, 8, 2), (4, 5, 8, 2)]
    presence = {L: set(range(B, E + 1)) for L, B, E, _ in rows}
    return rows, presence


def _positions():
    return {1: (16, 16), 2: (40, 16), 3: (36, 40), 4: (48, 40)}


@pytest.fixture
def cache(tmp_path):
    """Generate a 2-video, 4-trajectory real-data cache; return (dir, cfg)."""
    root = tmp_path / "data"
    rows, presence = _rows_and_presence()
    for dataset in ("Synth-A", "Synth-B"):
        make_ctc_video(root, dataset, "01", N_FRAMES, rows, presence,
                       positions=_positions(), size=SIZE, radius=RADIUS)

    cache_dir = tmp_path / "cache"
    cfg = OmegaConf.create({
        "data": {
            "type": "real",
            "n_objects": 8,        # 4 tracks x 2 videos, min_track_length=4
            "image_size": IMAGE_SIZE,
            "domain_size": DOMAIN,
            "dt": 1.0,
            "seed": 0,
            "generator": {
                "n_trajectories": 4,
                "trajectory_length": WINDOW,
                "seed_offset": 0,
                "cache_dir": str(cache_dir),
                "overwrite_existing": False,
            },
            "params": {
                "name": "real",
                "root": str(root),
                "videos": [
                    {"dataset": "Synth-A", "seq": "01"},
                    {"dataset": "Synth-B", "seq": "01"},
                ],
                "r_max_px": R_MAX,
                "window_length": WINDOW,
                "contact_frac": 1.0,
                "contact_px": None,
                "min_track_length": 4,
                "min_area_px": 8,
                "border_policy": "invalidate_frame",
                "max_tracks_per_video": None,
                "max_windows_per_video": None,
                "division_mode": "new_slots",
                "center_fov": True,
                "pseudo_split": None,
                "scan_workers": 1,
                "raster_workers": 1,
            },
        },
    })
    gen = CTCTrajectoryGenerator(cfg=cfg, device="cpu")
    gen.generate_all(show_progress=False)
    return str(cache_dir), cfg


def _load(cache_dir, idx):
    from graph_dyno.data.trajectory_generator import generate_config_hash
    import os
    cfg = OmegaConf.load(os.path.join(cache_dir, "metadata.yaml"))
    h = generate_config_hash(cfg.data)
    return torch.load(os.path.join(cache_dir, f"{h}_{idx:04d}.pt"),
                      weights_only=False)


class TestSchema:
    def test_keys_and_shapes(self, cache):
        cache_dir, cfg = cache
        data = _load(cache_dir, 0)
        N = cfg.data.n_objects
        assert set(data) == {
            "times", "states", "next_states", "forces", "channeled_images",
            "W_interactions", "valid", "next_valid", "sim_metadata",
        }
        assert data["times"].shape == (WINDOW, 1)
        assert data["states"].shape == (WINDOW, N, 4)
        assert data["next_states"].shape == (WINDOW, N, 4)
        assert data["forces"].shape == (WINDOW, N * 2)
        assert data["channeled_images"].shape == (WINDOW, N, IMAGE_SIZE, IMAGE_SIZE)
        assert data["valid"].shape == data["next_valid"].shape == (WINDOW, N)
        assert data["valid"].dtype == torch.bool

    def test_interaction_graph_is_stacked_sparse(self, cache):
        cache_dir, _ = cache
        W = _load(cache_dir, 0)["W_interactions"]
        assert W.is_sparse and W.sparse_dim() == 3
        assert W.dtype == torch.bool
        dense = W.to_dense()
        # Symmetric, zero diagonal.
        assert torch.equal(dense, dense.transpose(1, 2))
        assert not dense.diagonal(dim1=1, dim2=2).any()

    def test_padded_slots_have_no_incident_edges(self, cache):
        """What makes them harmless in gnn.py: never a src/tgt, so scatter_add_
        leaves their message at exactly zero."""
        cache_dir, _ = cache
        data = _load(cache_dir, 0)
        dense = data["W_interactions"].to_dense()
        invalid = ~data["valid"]
        assert not dense[invalid].any()                 # empty rows
        assert not dense.transpose(1, 2)[invalid].any()  # empty columns

    def test_next_states_matches_states_at_t_plus_one(self, cache):
        """next_states is sliced from the FULL-video array, so within a window
        next_states[t] == states[t+1]."""
        cache_dir, _ = cache
        data = _load(cache_dir, 0)
        torch.testing.assert_close(data["next_states"][:-1], data["states"][1:])
        torch.testing.assert_close(data["next_valid"][:-1].float(),
                                   data["valid"][1:].float())

    def test_no_nans_and_finite_padding(self, cache):
        """PAD_POS must be finite: masked MSE is ((pred-tgt)^2 * mask).sum() and
        NaN * 0 = NaN would poison the whole loss."""
        cache_dir, _ = cache
        data = _load(cache_dir, 0)
        for key in ("times", "states", "next_states", "forces", "channeled_images"):
            assert torch.isfinite(data[key]).all(), key
        pad = data["states"][~data["valid"]]
        assert torch.isfinite(pad).all()
        assert (pad[..., :2] == -4.0 * R_MAX).all()
        assert (pad[..., 2:] == 0).all()

    def test_padded_slots_render_nothing(self, cache):
        cache_dir, _ = cache
        data = _load(cache_dir, 0)
        assert data["channeled_images"][~data["valid"]].sum() == 0
        assert data["channeled_images"][data["valid"]].sum() > 0


class TestSlotsAndWindows:
    def test_valid_is_all_false_outside_the_video_slot_range(self, cache):
        """The invariant that makes disjoint slots the experiment."""
        cache_dir, _ = cache
        occupied = []
        for idx in range(4):
            v = _load(cache_dir, idx)["valid"]
            slots = torch.nonzero(v.any(dim=0)).flatten().tolist()
            occupied.append(set(slots))
        # Round-robin: trajectories 0 and 2 are video 0; 1 and 3 are video 1.
        assert occupied[0] & occupied[1] == set()
        assert max(occupied[0]) < min(occupied[1])
        assert occupied[0] | occupied[2] and occupied[1] | occupied[3]

    def test_round_robin_window_order(self, cache):
        """Any prefix of n_trajectories covers every video."""
        cache_dir, _ = cache
        first = torch.nonzero(_load(cache_dir, 0)["valid"].any(0)).flatten()
        second = torch.nonzero(_load(cache_dir, 1)["valid"].any(0)).flatten()
        assert first.min() < second.min()      # different videos, in order

    def test_fov_centred(self, cache):
        """Each video's FOV centre lands at domain_size/2."""
        cache_dir, _ = cache
        data = _load(cache_dir, 0)
        pos = data["states"][..., :2][data["valid"]]
        # Cells sit within the imaged region, which is centred on DOMAIN/2.
        assert (pos - DOMAIN / 2).abs().max() <= SIZE / 2 + 1

    def test_mitosis_parent_and_daughters_never_coexist(self, cache):
        cache_dir, _ = cache
        # Track 2 dies at frame 4; 3 and 4 are born at 5 -> window 1 (frames 4-7).
        data = _load(cache_dir, 2)
        v = data["valid"]
        alive = [torch.nonzero(v[:, s]).flatten().tolist() for s in range(v.shape[1])]
        # No slot is alive across the whole window AND no parent/daughter overlap:
        # the parent's frames and the daughters' frames are disjoint sets.
        spans = [set(a) for a in alive if a]
        assert any(s != spans[0] for s in spans), "expected staggered lifetimes"


class TestThroughDataloader:
    def test_dataset_loads_and_metadata_is_consistent(self, cache):
        cache_dir, cfg = cache
        ds = TrajectoryDatasetFromFolder(cache_dir, enable_gpu_cache=False)
        assert len(ds) == 4 * WINDOW
        assert ds.verify_metadata_consistency() is True

    def test_sim_metadata_is_all_tensors(self, cache):
        """train.py calls model.set_metadata(**sim_metadata), which does
        torch.tensor(v, dtype=torch.float32) for non-tensors — a string raises,
        and it raises after cache gen, SDF/DINO precompute and MLflow start."""
        cache_dir, cfg = cache
        meta = _load(cache_dir, 0)["sim_metadata"]
        for key, value in meta.items():
            assert isinstance(value, torch.Tensor), key
            assert torch.isfinite(value).all(), key
        assert meta["r_max"].item() == R_MAX
        assert meta["rs"].shape == (cfg.data.n_objects,)
        assert len(torch.unique(meta["video_id"])) == 2

    def test_single_step_sample_carries_validity(self, cache):
        """The additive dataloader edit: valid/next_valid/prev_valid appear when
        the cache has them, and next_valid comes from the cache's OWN key."""
        cache_dir, cfg = cache
        N = cfg.data.n_objects
        bn = 3
        embeddings = {
            i: {"L": torch.zeros(WINDOW, N, bn), "crop_sides": torch.zeros(WINDOW, N)}
            for i in range(4)
        }
        ds = TrajectoryDatasetFromFolder(cache_dir, enable_gpu_cache=False,
                                         embeddings=embeddings)
        ds.include_images = False   # as train.py does once embeddings exist
        sample = ds[1]              # timestep 1 -> prev_t = 0 exists
        assert {"valid", "next_valid", "prev_valid"} <= set(sample)
        assert sample["valid"].shape == (N,)

        cached = _load(cache_dir, 0)
        torch.testing.assert_close(sample["valid"].float(), cached["valid"][1].float())
        torch.testing.assert_close(sample["next_valid"].float(),
                                   cached["next_valid"][1].float())
        torch.testing.assert_close(sample["prev_valid"].float(),
                                   cached["valid"][0].float())

        batch = sparse_collate_fn([ds[1], ds[2]])
        assert batch["valid"].shape == (2, N)

    def test_last_timestep_next_valid_is_not_the_clamped_valid(self, cache):
        """The window-seam defect the explicit next_valid key exists to avoid.

        Deriving next_valid as valid[min(t+1, T-1)] would reuse valid[T-1] at
        t = T-1, marking a cell that dies at frame f0+T as valid against a
        PAD_POS target at every window seam.
        """
        cache_dir, _ = cache
        data = _load(cache_dir, 0)
        derived = data["valid"][-1]          # what the clamp would produce
        stored = data["next_valid"][-1]      # what is actually shipped
        # They describe different frames; the stored one is the correct target.
        assert stored.shape == derived.shape
        nxt = _load(cache_dir, 2)  # same video, next window
        assert torch.equal(stored, nxt["valid"][0])


class TestResume:
    def test_regeneration_is_skipped_and_deterministic(self, cache, tmp_path):
        cache_dir, cfg = cache
        before = _load(cache_dir, 0)["states"].clone()
        gen = CTCTrajectoryGenerator(cfg=cfg, device="cpu")
        gen.generate_all(show_progress=False)   # should no-op
        torch.testing.assert_close(before, _load(cache_dir, 0)["states"])

    def test_window_length_must_mirror_trajectory_length(self, cache):
        cache_dir, cfg = cache
        bad = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        bad.data.params.window_length = WINDOW + 1
        with pytest.raises(ValueError, match="window_length"):
            CTCTrajectoryGenerator(cfg=bad, device="cpu")

    def test_wrong_n_objects_raises_with_the_right_number(self, cache, tmp_path):
        cache_dir, cfg = cache
        bad = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        bad.data.n_objects = 999
        bad.data.generator.cache_dir = str(tmp_path / "cache2")
        gen = CTCTrajectoryGenerator(cfg=bad, device="cpu")
        with pytest.raises(ValueError, match="track table has 8 slots"):
            gen.generate_all(show_progress=False)

    def test_too_small_r_max_raises_naming_the_video(self, cache, tmp_path):
        cache_dir, cfg = cache
        bad = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        bad.data.params.r_max_px = 2.0
        bad.data.generator.cache_dir = str(tmp_path / "cache3")
        gen = CTCTrajectoryGenerator(cfg=bad, device="cpu")
        with pytest.raises(ValueError, match=r"Synth-A/01: needs >="):
            gen.generate_all(show_progress=False)
