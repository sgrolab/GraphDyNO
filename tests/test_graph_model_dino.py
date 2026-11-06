import torch
import pytest
from omegaconf import OmegaConf

from graph_dyno.decoders.dino import DINODecoder
from graph_dyno.models.graph_model_dino import GraphModelDINO


DINO_DIM = 384
BOTTLENECK_DIM = 64
BATCH_SIZE = 4
N_NODES = 5


def _make_cfg(**overrides):
    base = {
        "model": {
            "embedding_dim": 2,
            "num_nodes": N_NODES,
            "interaction_dim": 2,
            "spatial_dim": 2,
            "hidden_dim": 64,
            "hidden_layers": 3,
            "activation": "PReLU",
            "feature_dim": 4,
            "indices_of_mlp_features": [2, 3],
            "skip_connection": True,
            "embedding_noise_std": 0.0,
            "seed": 0,
            "embedding_noise_std": 0.0,
            "gnn_inputs": ["L", "scale", "L_velocity", "scale_velocity"],
            "position_inputs": ["x_features", "interactions"],
            "shape_mlps": [["L", "b", "interactions", "L_velocity"]],
            "scale_mlps": [["L", "b", "interactions", "L_velocity", "scale", "scale_velocity"]],
        },
        "data": {
            "domain_size": 1.0,
            "type": "simulation",
            "dt": 0.01,
            "image_size": 64,
            "params": {"periodic_boundary": False},
        },
    }
    base["model"].update(overrides)
    return OmegaConf.create(base)


def _make_model(cfg=None):
    if cfg is None:
        cfg = _make_cfg()
    decoder = DINODecoder(input_dim=DINO_DIM, bottleneck_dim=BOTTLENECK_DIM)
    return GraphModelDINO(cfg, shape_decoder=decoder, device="cpu")


def _make_W(batch_size, N):
    W_list = []
    for _ in range(batch_size):
        idx = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        vals = torch.tensor([1.0, 0.5, 0.8])
        W_list.append(torch.sparse_coo_tensor(idx, vals, (N, N)))
    return torch.stack(W_list)


class TestGraphModelDINO:
    def test_instantiation(self):
        model = _make_model()
        assert model.dino_dim == DINO_DIM
        assert model.bottleneck_dim == BOTTLENECK_DIM

    def test_dino_decoder_alias(self):
        model = _make_model()
        assert model.dino_decoder is model.shape_decoder

    def test_forward(self):
        model = _make_model()
        cfg = _make_cfg()
        x = torch.randn(BATCH_SIZE, N_NODES, cfg.model.feature_dim)
        L = torch.randn(BATCH_SIZE, N_NODES, BOTTLENECK_DIM)
        W = _make_W(BATCH_SIZE, N_NODES)
        scale = torch.rand(BATCH_SIZE, N_NODES) * 100
        L_vel = torch.randn(BATCH_SIZE, N_NODES, BOTTLENECK_DIM)
        s_vel = torch.randn(BATCH_SIZE, N_NODES)

        new_x, L_out, next_scale = model.forward(x, W, L, scale, L_vel, s_vel)
        assert new_x.shape == (BATCH_SIZE, N_NODES, cfg.model.feature_dim)
        assert L_out.shape == (BATCH_SIZE, N_NODES, BOTTLENECK_DIM)
        assert next_scale.shape == (BATCH_SIZE, N_NODES)

    def test_training_step(self):
        model = _make_model()
        cfg = _make_cfg()
        x = torch.randn(BATCH_SIZE, N_NODES, cfg.model.feature_dim)
        W = _make_W(BATCH_SIZE, N_NODES)
        scale = torch.rand(BATCH_SIZE, N_NODES) * 100

        batch = {
            "states": x,
            "next_states": torch.randn(BATCH_SIZE, N_NODES, cfg.model.feature_dim),
            "W_interactions": W,
            "L": torch.randn(BATCH_SIZE, N_NODES, BOTTLENECK_DIM),
            "next_L": torch.randn(BATCH_SIZE, N_NODES, BOTTLENECK_DIM),
            "prev_L": torch.randn(BATCH_SIZE, N_NODES, BOTTLENECK_DIM),
            "crop_sides": scale,
            "next_crop_sides": torch.rand(BATCH_SIZE, N_NODES) * 100,
            "prev_crop_sides": torch.rand(BATCH_SIZE, N_NODES) * 100,
        }
        losses = model.training_step(batch)
        for key in ["position_mse", "velocity_mse", "dino_mse", "scale_mse"]:
            assert key in losses
            assert losses[key].ndim == 0

    def test_generate(self):
        model = _make_model()
        cfg = _make_cfg()
        x = torch.randn(N_NODES, cfg.model.feature_dim)
        L = torch.randn(N_NODES, BOTTLENECK_DIM)
        b = model.parameter_embedding.detach()
        idx = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        vals = torch.tensor([1.0, 0.5, 0.8])
        W = torch.sparse_coo_tensor(idx, vals, (N_NODES, N_NODES))
        scale = torch.rand(N_NODES) * 100
        L_vel = torch.randn(N_NODES, BOTTLENECK_DIM)
        s_vel = torch.randn(N_NODES)

        new_x, L_out, next_scale = model.generate(x, W, b, L, scale, L_vel, s_vel)
        assert new_x.shape == (N_NODES, cfg.model.feature_dim)
        assert L_out.shape == (N_NODES, BOTTLENECK_DIM)
        assert next_scale.shape == (N_NODES,)

    def test_decode_to_image(self):
        model = _make_model()
        L_vis = torch.randn(BATCH_SIZE, N_NODES, BOTTLENECK_DIM)
        images = model.decode_to_image(L_vis)
        assert images.shape == (BATCH_SIZE, N_NODES, 224, 224)

    def test_multi_mlp(self):
        cfg = _make_cfg(
            shape_mlps=[["L", "interactions", "L_velocity"], ["L", "b", "L_velocity"]],
            scale_mlps=[["interactions", "scale", "scale_velocity"], ["b", "scale", "scale_velocity"]],
        )
        model = _make_model(cfg)
        assert len(model.shape_mlps) == 2
        assert len(model.scale_mlps) == 2

        x = torch.randn(BATCH_SIZE, N_NODES, cfg.model.feature_dim)
        L = torch.randn(BATCH_SIZE, N_NODES, BOTTLENECK_DIM)
        W = _make_W(BATCH_SIZE, N_NODES)
        scale = torch.rand(BATCH_SIZE, N_NODES) * 100
        L_vel = torch.randn(BATCH_SIZE, N_NODES, BOTTLENECK_DIM)
        s_vel = torch.randn(BATCH_SIZE, N_NODES)

        new_x, L_out, next_scale = model.forward(x, W, L, scale, L_vel, s_vel)
        assert new_x.shape == (BATCH_SIZE, N_NODES, cfg.model.feature_dim)
        assert L_out.shape == (BATCH_SIZE, N_NODES, BOTTLENECK_DIM)
        assert next_scale.shape == (BATCH_SIZE, N_NODES)
