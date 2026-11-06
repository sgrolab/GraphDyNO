"""Tests for GraphModelReal (masked losses over variable cell counts).

Run with: pytest tests/test_graph_model_real.py -v
"""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.nn.functional import mse_loss

from graph_dyno.decoders.dino import DINODecoder
from graph_dyno.models.graph_model_dino import GraphModelDINO
from graph_dyno.models.graph_model_real import GraphModelReal

DINO_DIM = 384
BOTTLENECK_DIM = 16
BATCH_SIZE = 4
N_NODES = 6
CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def _make_cfg(model_overrides=None, **top_overrides):
    base = {
        "model": {
            "type": "real",
            "loss_mask_mode": "strict",
            "embedding_dim": 3,
            "num_nodes": N_NODES,
            "interaction_dim": 2,
            "spatial_dim": 2,
            "hidden_dim": 32,
            "hidden_layers": 2,
            "activation": "PReLU",
            "feature_dim": 4,
            "indices_of_mlp_features": [2, 3],
            "skip_connection": True,
            "embedding_noise_std": 0.0,
            "seed": 0,
            "gnn_inputs": ["L", "scale"],
            "position_inputs": ["x_features", "interactions"],
            "shape_mlps": [["L", "b", "interactions"]],
            "scale_mlps": [["L", "b", "interactions", "scale"]],
        },
        "data": {
            "domain_size": 64.0,
            "type": "real",
            "dt": 1.0,
            "image_size": 32,
            "params": {"name": "real"},
        },
        "training": {"rollout": {"enabled": False}},
        "augmentation": {"enabled": False},
    }
    base["model"].update(model_overrides or {})
    base.update(top_overrides)
    return OmegaConf.create(base)


def _make_model(cfg=None, cls=GraphModelReal):
    cfg = cfg if cfg is not None else _make_cfg()
    decoder = DINODecoder(input_dim=DINO_DIM, bottleneck_dim=BOTTLENECK_DIM)
    return cls(cfg, shape_decoder=decoder, device="cpu")


def _make_W(batch_size, N):
    W = []
    for _ in range(batch_size):
        idx = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        W.append(torch.sparse_coo_tensor(idx, torch.ones(2), (N, N)))
    return torch.stack(W)


def _make_batch(valid=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    b = {
        "states": torch.randn(BATCH_SIZE, N_NODES, 4, generator=g),
        "next_states": torch.randn(BATCH_SIZE, N_NODES, 4, generator=g),
        "W_interactions": _make_W(BATCH_SIZE, N_NODES),
        "L": torch.randn(BATCH_SIZE, N_NODES, BOTTLENECK_DIM, generator=g),
        "next_L": torch.randn(BATCH_SIZE, N_NODES, BOTTLENECK_DIM, generator=g),
        "prev_L": torch.randn(BATCH_SIZE, N_NODES, BOTTLENECK_DIM, generator=g),
        "crop_sides": torch.rand(BATCH_SIZE, N_NODES, generator=g) * 30,
        "next_crop_sides": torch.rand(BATCH_SIZE, N_NODES, generator=g) * 30,
        "prev_crop_sides": torch.rand(BATCH_SIZE, N_NODES, generator=g) * 30,
    }
    if valid is not None:
        b["valid"] = valid
        b["next_valid"] = valid.clone()
        b["prev_valid"] = valid.clone()
    return b


# ---------------------------------------------------------------------------

class TestMaskedMSE:
    def test_all_true_mask_equals_mse_loss(self):
        """The guarantee that nothing about the simulation path changes."""
        pred = torch.randn(BATCH_SIZE, N_NODES, 5)
        target = torch.randn(BATCH_SIZE, N_NODES, 5)
        mask = torch.ones(BATCH_SIZE, N_NODES, dtype=torch.bool)
        torch.testing.assert_close(
            GraphModelReal._masked_mse(pred, target, mask),
            mse_loss(pred, target),
        )

    def test_all_false_mask_is_zero_with_finite_grads(self):
        pred = torch.randn(BATCH_SIZE, N_NODES, 5, requires_grad=True)
        target = torch.randn(BATCH_SIZE, N_NODES, 5)
        mask = torch.zeros(BATCH_SIZE, N_NODES, dtype=torch.bool)
        loss = GraphModelReal._masked_mse(pred, target, mask)
        assert loss.item() == 0.0
        loss.backward()
        assert torch.isfinite(pred.grad).all()

    def test_partial_mask_matches_manual_subset_mse(self):
        pred = torch.randn(BATCH_SIZE, N_NODES, 5)
        target = torch.randn(BATCH_SIZE, N_NODES, 5)
        mask = torch.zeros(BATCH_SIZE, N_NODES, dtype=torch.bool)
        mask[0, :3] = True
        mask[2, 4] = True
        torch.testing.assert_close(
            GraphModelReal._masked_mse(pred, target, mask),
            mse_loss(pred[mask], target[mask]),
        )

    def test_padded_targets_do_not_affect_the_loss(self):
        """Junk in padded slots is invisible — the point of masking."""
        pred = torch.randn(BATCH_SIZE, N_NODES, 5)
        target = torch.randn(BATCH_SIZE, N_NODES, 5)
        mask = torch.zeros(BATCH_SIZE, N_NODES, dtype=torch.bool)
        mask[:, :2] = True
        clean = GraphModelReal._masked_mse(pred, target, mask)
        dirty_target = target.clone()
        dirty_target[:, 2:] = 1e6
        torch.testing.assert_close(
            GraphModelReal._masked_mse(pred, dirty_target, mask), clean)

    def test_finite_pad_required(self):
        """Documents WHY PAD_POS is finite: NaN * 0 = NaN poisons the sum."""
        pred = torch.randn(BATCH_SIZE, N_NODES, 5)
        target = torch.randn(BATCH_SIZE, N_NODES, 5)
        target[:, 3:] = float("nan")
        mask = torch.zeros(BATCH_SIZE, N_NODES, dtype=torch.bool)
        mask[:, :2] = True
        assert torch.isnan(GraphModelReal._masked_mse(pred, target, mask))


class TestTrainingStep:
    def test_matches_dino_when_everything_is_valid(self):
        cfg = _make_cfg()
        real = _make_model(cfg)
        dino = _make_model(_make_cfg(model_overrides={"type": "dino"}),
                           cls=GraphModelDINO)
        dino.load_state_dict(real.state_dict())
        dino.eval(); real.eval()

        valid = torch.ones(BATCH_SIZE, N_NODES, dtype=torch.bool)
        batch = _make_batch(valid)
        r = real.training_step(batch)
        d = dino.training_step({k: v for k, v in batch.items()
                                if k not in {"valid", "next_valid", "prev_valid"}})
        for key in ("position_mse", "velocity_mse", "dino_mse", "scale_mse",
                    "parameter_embedding_l2"):
            torch.testing.assert_close(r[key], d[key], msg=key)

    def test_works_without_validity_keys(self):
        """Falls back to all-True, so the class still runs on sim caches."""
        model = _make_model()
        losses = model.training_step(_make_batch())
        for v in losses.values():
            assert torch.isfinite(v)

    def test_padded_slots_get_zero_gradient(self):
        model = _make_model()
        valid = torch.zeros(BATCH_SIZE, N_NODES, dtype=torch.bool)
        valid[:, :3] = True            # slots 3, 4, 5 are never real
        batch = _make_batch(valid)
        batch["W_interactions"] = torch.stack([
            torch.sparse_coo_tensor(torch.tensor([[0, 1], [1, 0]]),
                                    torch.ones(2), (N_NODES, N_NODES))
            for _ in range(BATCH_SIZE)
        ])
        losses = model.training_step(batch)
        total = sum(losses[k] for k in ("position_mse", "velocity_mse",
                                        "dino_mse", "scale_mse"))
        total.backward()
        grad = model.parameter_embedding.grad
        assert grad is not None
        assert (grad[3:] == 0).all(), "padded slots received gradient"
        assert grad[:3].abs().sum() > 0, "real slots received no gradient"

    def test_loss_mask_for_matches_the_mask_training_step_applies(self):
        """train.py's Risk 10 check reads `loss_mask_for` to decide whether a
        gradient-starved slot is a wiring bug or a short track at a window edge.
        If it ever drifted from the mask `training_step` really uses, that check
        would start accusing the wiring of a data property (or worse, stop
        catching a real bug)."""
        model = _make_model()
        valid = torch.zeros(BATCH_SIZE, N_NODES, dtype=torch.bool)
        valid[:, :4] = True
        batch = _make_batch(valid)
        batch["prev_valid"] = valid.clone()
        batch["prev_valid"][:, 3] = False      # slot 3: valid, but no transition
        expected = valid & batch["next_valid"] & batch["prev_valid"]
        torch.testing.assert_close(model.loss_mask_for(batch), expected)
        assert bool(batch["valid"][:, 3].any())
        assert not bool(model.loss_mask_for(batch)[:, 3].any())

    def test_valid_but_unmasked_slot_gets_no_gradient(self):
        """The exact situation on Mix B: PhC-C2DL-PSC tracks born in the last
        few frames of a window are valid there, but never have prev/cur/next all
        valid, so no loss term can reach them and they keep their random init.
        Legitimate — but they must be excluded from the separation analysis."""
        model = _make_model()
        valid = torch.zeros(BATCH_SIZE, N_NODES, dtype=torch.bool)
        valid[:, :4] = True
        batch = _make_batch(valid)
        batch["prev_valid"] = valid.clone()
        batch["prev_valid"][:, 3] = False
        losses = model.training_step(batch)
        total = sum(losses[k] for k in ("position_mse", "velocity_mse",
                                        "dino_mse", "scale_mse"))
        total.backward()
        grad = model.parameter_embedding.grad
        assert grad is not None
        assert grad[3].abs().sum() == 0, "unmasked slot should get no gradient"
        assert grad[:3].abs().sum() > 0, "masked slots received no gradient"

    def test_embedding_l2_is_masked_to_seen_slots(self):
        """Unmasked mean(pe**2) would over-shrink rarely-seen slots, making the
        embedding-norm gradient a direct function of video size — a
        video-correlated artifact in the exact quantity being clustered."""
        model = _make_model()
        with torch.no_grad():
            model.parameter_embedding.zero_()
            model.parameter_embedding[3:] = 10.0     # huge, but never seen
        valid = torch.zeros(BATCH_SIZE, N_NODES, dtype=torch.bool)
        valid[:, :3] = True
        l2 = model.training_step(_make_batch(valid))["parameter_embedding_l2"]
        assert l2.item() == pytest.approx(0.0, abs=1e-9)

    def test_transition_mode_ignores_prev_valid(self):
        cfg = _make_cfg(model_overrides={"loss_mask_mode": "transition"})
        model = _make_model(cfg)
        valid = torch.ones(BATCH_SIZE, N_NODES, dtype=torch.bool)
        batch = _make_batch(valid)
        with_prev = model.training_step(batch)["position_mse"]
        batch["prev_valid"] = torch.zeros_like(valid)
        torch.testing.assert_close(model.training_step(batch)["position_mse"],
                                   with_prev)

    def test_strict_mode_uses_prev_valid(self):
        model = _make_model()
        valid = torch.ones(BATCH_SIZE, N_NODES, dtype=torch.bool)
        batch = _make_batch(valid)
        batch["prev_valid"] = torch.zeros_like(valid)
        assert model.training_step(batch)["position_mse"].item() == 0.0

    def test_loss_keys_are_all_in_the_shipped_loss_terms(self):
        """configs/training/default.yaml lists image_bce / image_dice /
        shape_mse — NOT dino_mse or scale_mse. train.py sums only keys present
        in loss_terms, so a missing override silently zeroes two heads."""
        model = _make_model()
        keys = set(model.training_step(_make_batch()))
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            for name in ("real_dic_psc_late",):
                cfg = compose(config_name="config", overrides=[f"+experiment={name}"])
                terms = set(cfg.training.loss_terms)
                missing = keys - terms
                assert not missing, f"{name} loss_terms is missing {missing}"


class TestGuards:
    def test_absolute_position_in_mlp_features_is_rejected(self):
        """states are [x, y, vx, vy]; 0/1 would be a per-video FOV fingerprint."""
        cfg = _make_cfg(model_overrides={"indices_of_mlp_features": [0, 1, 2, 3]})
        with pytest.raises(ValueError, match="absolute position"):
            _make_model(cfg)

    def test_augmentation_is_rejected(self):
        cfg = _make_cfg()
        cfg.augmentation.enabled = True
        with pytest.raises(ValueError, match="augmentation"):
            _make_model(cfg)

    def test_bad_loss_mask_mode_is_rejected(self):
        cfg = _make_cfg(model_overrides={"loss_mask_mode": "nope"})
        with pytest.raises(ValueError, match="loss_mask_mode"):
            _make_model(cfg)

    def test_rollout_warns_but_does_not_crash(self, capsys):
        cfg = _make_cfg()
        cfg.training.rollout.enabled = True
        _make_model(cfg)
        assert "rollout" in capsys.readouterr().out.lower()


class TestConfigComposition:
    def test_unmodified_unified_dino_still_returns_graph_model_dino(self):
        """create_model dispatches on cfg.model.type, which unified_dino lacks."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["model=unified_dino"])
        assert cfg.model.get("type", "dino") == "dino"

    def test_unified_real_selects_the_real_model(self):
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config",
                          overrides=["data=real", "model=unified_real"])
        assert cfg.model.type == "real"
        assert cfg.model.indices_of_mlp_features == [2, 3]

    @pytest.mark.parametrize("ablation", [
        "full", "full_no_history", "full_velocity", "full_int", "scale_only",
        "int/static", "int/spatial", "int/temporal", "int/full",
        "b_sweep/b_shape_scale_int",
    ])
    def test_every_ablation_composes_and_instantiates(self, ablation):
        """Ablations are `# @package _global_` overlays patching the four MLP
        input lists, so they apply to unified_real unchanged."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=[
                "data=real", "model=unified_real", f"+ablation={ablation}",
            ])
        cfg.model.num_nodes = N_NODES
        cfg.model.hidden_dim = 16
        cfg.model.hidden_layers = 1
        cfg.data.domain_size = 64.0
        cfg.training.rollout.enabled = False
        model = _make_model(cfg)
        assert isinstance(model, GraphModelReal)
