import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import pytest
from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from graph_dyno.decoders.base import ShapeDecoder
from graph_dyno.decoders.dino import DINODecoder
from graph_dyno.decoders.fourier import FourierDecoder
from graph_dyno.encoders.base import ShapeEncoder
from graph_dyno.encoders.fourier import FourierEncoder, _mask_to_fourier


# ---------------------------------------------------------------------------
# ShapeDecoder ABC
# ---------------------------------------------------------------------------

class TestShapeDecoderABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ShapeDecoder()

    def test_dino_is_shape_decoder(self):
        d = DINODecoder(input_dim=384, bottleneck_dim=64)
        assert isinstance(d, ShapeDecoder)

    def test_fourier_is_shape_decoder(self):
        d = FourierDecoder(n_coefficients=16)
        assert isinstance(d, ShapeDecoder)


class TestShapeEncoderABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ShapeEncoder()

    def test_fourier_is_shape_encoder(self):
        e = FourierEncoder(n_coefficients=16)
        assert isinstance(e, ShapeEncoder)


# ---------------------------------------------------------------------------
# DINODecoder
# ---------------------------------------------------------------------------

class TestDINODecoder:
    def test_properties(self):
        d = DINODecoder(input_dim=384, bottleneck_dim=64)
        assert d.input_dim == 384
        assert d.bottleneck_dim == 64

    def test_get_bottleneck_shape(self):
        d = DINODecoder(input_dim=384, bottleneck_dim=64)
        emb = torch.randn(8, 384)
        out = d.get_bottleneck(emb)
        assert out.shape == (8, 64)

    def test_decode_from_bottleneck_shape(self):
        d = DINODecoder(input_dim=384, bottleneck_dim=64)
        bn = torch.randn(4, 64)
        img = d.decode_from_bottleneck(bn)
        assert img.shape == (4, 224, 224)
        assert img.min() >= -1.0 and img.max() <= 1.0

    def test_forward_shape(self):
        d = DINODecoder(input_dim=384, bottleneck_dim=64)
        emb = torch.randn(4, 384)
        img = d(emb)
        assert img.shape == (4, 224, 224)

    def test_training_step(self):
        d = DINODecoder(input_dim=384, bottleneck_dim=64)
        emb = torch.randn(4, 384)
        targets = torch.randn(4, 224, 224).clamp(-1, 1)
        losses = d.training_step(emb, targets)
        assert "image_mse" in losses
        assert "embedding_l2" in losses
        assert losses["image_mse"].ndim == 0


# ---------------------------------------------------------------------------
# FourierEncoder
# ---------------------------------------------------------------------------

class TestFourierEncoder:
    def test_properties(self):
        e = FourierEncoder(n_coefficients=32)
        assert e.embedding_dim == 64
        assert e.companion_suffix == "fourier_32"

    def test_encode_circle_mask(self):
        """A circle mask should produce valid Fourier coefficients."""
        H, W = 128, 128
        Y, X = np.ogrid[:H, :W]
        mask = ((X - 64)**2 + (Y - 64)**2 < 30**2).astype(np.float32)
        masks = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        e = FourierEncoder(n_coefficients=16)
        embeddings, crop_sides = e.encode_masks(masks, num_workers=1)

        assert embeddings.shape == (1, 32)
        assert crop_sides.shape == (1,)
        assert crop_sides[0] > 0
        assert not torch.isnan(embeddings).any()

    def test_encode_empty_mask(self):
        """Empty mask should return zero coefficients."""
        masks = torch.zeros(1, 64, 64)
        e = FourierEncoder(n_coefficients=16)
        embeddings, crop_sides = e.encode_masks(masks, num_workers=1)

        assert embeddings.shape == (1, 32)
        assert torch.allclose(embeddings, torch.zeros_like(embeddings))
        assert crop_sides[0] == 0

    def test_encode_batch(self):
        """Multiple masks should be processed correctly."""
        H, W = 64, 64
        Y, X = np.ogrid[:H, :W]
        masks = []
        for r in [10, 15, 20]:
            mask = ((X - 32)**2 + (Y - 32)**2 < r**2).astype(np.float32)
            masks.append(mask)
        masks = torch.from_numpy(np.stack(masks))  # (3, H, W)

        e = FourierEncoder(n_coefficients=8)
        embeddings, crop_sides = e.encode_masks(masks, num_workers=1)

        assert embeddings.shape == (3, 16)
        assert crop_sides.shape == (3,)
        assert (crop_sides > 0).all()


# ---------------------------------------------------------------------------
# FourierDecoder
# ---------------------------------------------------------------------------

class TestFourierDecoder:
    def test_properties(self):
        d = FourierDecoder(n_coefficients=32, output_size=224)
        assert d.bottleneck_dim == 64
        assert d.input_dim == 64

    def test_get_bottleneck_identity(self):
        d = FourierDecoder(n_coefficients=16)
        emb = torch.randn(4, 32)
        assert torch.equal(d.get_bottleneck(emb), emb)

    def test_decode_from_bottleneck_shape(self):
        d = FourierDecoder(n_coefficients=16, output_size=64)
        bn = torch.randn(2, 32)
        img = d.decode_from_bottleneck(bn)
        assert img.shape == (2, 64, 64)

    def test_training_step(self):
        d = FourierDecoder(n_coefficients=16, output_size=64)
        emb = torch.randn(2, 32)
        targets = torch.randn(2, 64, 64)
        losses = d.training_step(emb, targets)
        assert "image_mse" in losses

    def test_no_learnable_parameters(self):
        d = FourierDecoder(n_coefficients=16)
        assert sum(p.numel() for p in d.parameters()) == 0


# ---------------------------------------------------------------------------
# Fourier roundtrip
# ---------------------------------------------------------------------------

class TestFourierRoundtrip:
    def test_circle_roundtrip(self):
        """Encode a circle, decode it, verify the reconstruction is reasonable."""
        H, W = 128, 128
        Y, X = np.ogrid[:H, :W]
        mask = ((X - 64)**2 + (Y - 64)**2 < 30**2).astype(np.float32)
        masks = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

        n_coeff = 32
        encoder = FourierEncoder(n_coefficients=n_coeff)
        decoder = FourierDecoder(n_coefficients=n_coeff, output_size=128)

        embeddings, crop_sides = encoder.encode_masks(masks, num_workers=1)
        reconstruction = decoder.decode_from_bottleneck(embeddings)

        assert reconstruction.shape == (1, 128, 128)
        # Reconstruction of a circle should have nonzero pixels
        assert reconstruction.sum() > 0


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

class TestEncoderFactory:
    def test_get_companion_suffix_fourier(self):
        from graph_dyno.encoders.factory import get_companion_suffix
        cfg = OmegaConf.create({"shape": {"type": "fourier", "n_coefficients": 16}})
        assert get_companion_suffix(cfg) == "fourier_16"

    def test_get_companion_suffix_dino(self):
        from graph_dyno.encoders.factory import get_companion_suffix
        cfg = OmegaConf.create({"shape": {"type": "dino", "dino_model": "B"}})
        assert get_companion_suffix(cfg) == "dino_B"

    def test_create_fourier_encoder(self):
        from graph_dyno.encoders.factory import create_shape_encoder
        cfg = OmegaConf.create({"shape": {"type": "fourier", "n_coefficients": 16}})
        encoder = create_shape_encoder(cfg, device="cpu")
        assert isinstance(encoder, FourierEncoder)
        assert encoder.embedding_dim == 32


class TestDecoderFactory:
    def test_create_fourier_decoder(self):
        from graph_dyno.decoders.factory import create_shape_decoder
        cfg = OmegaConf.create({
            "shape": {"type": "fourier", "n_coefficients": 16, "output_size": 64}
        })
        decoder = create_shape_decoder(cfg, device="cpu")
        assert isinstance(decoder, FourierDecoder)
        assert decoder.bottleneck_dim == 32


# ---------------------------------------------------------------------------
# _mask_to_fourier unit tests
# ---------------------------------------------------------------------------

class TestMaskToFourier:
    def test_circle(self):
        H, W = 128, 128
        Y, X = np.ogrid[:H, :W]
        mask = ((X - 64)**2 + (Y - 64)**2 < 30**2).astype(np.float32)
        result = _mask_to_fourier(mask, n_coefficients=16)
        assert result is not None
        coeffs, crop_side = result
        assert coeffs.shape == (32,)
        assert crop_side > 0

    def test_empty_mask(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        result = _mask_to_fourier(mask, n_coefficients=16)
        assert result is None

    def test_small_mask(self):
        """A very small mask (few pixels) should still work or return None gracefully."""
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[30:33, 30:33] = 1.0
        result = _mask_to_fourier(mask, n_coefficients=8)
        # Either valid or None for tiny masks
        if result is not None:
            coeffs, crop_side = result
            assert coeffs.shape == (16,)


# ---------------------------------------------------------------------------
# Cell-DINO backend
#
# Every test here runs without the dinov2 clone and without the gated weights:
# the only thing that needs them is an actual backbone load, which is stubbed
# out at the _load_cell_backbone seam.
# ---------------------------------------------------------------------------

class _StubBackbone(nn.Module):
    """Minimal stand-in for a Cell-DINO ViT: no parameters, named outputs."""

    embed_dim = 32

    def __init__(self, n_patches=196):
        super().__init__()
        self.n_patches = n_patches
        self.last_input = None

    def forward_features(self, x):
        self.last_input = x
        B = x.shape[0]
        D = self.embed_dim
        return {
            "x_norm_clstoken": torch.zeros(B, D),
            "x_norm_regtokens": torch.zeros(B, 0, D),
            "x_norm_patchtokens": torch.arange(
                B * self.n_patches * D, dtype=torch.float32
            ).view(B, self.n_patches, D),
            "x_prenorm": torch.zeros(B, self.n_patches + 1, D),
            "masks": None,
        }


@pytest.fixture
def stub_backbone(monkeypatch):
    """Replace the one seam that needs the clone + gated weights."""
    import dino.feature_extractor as fe_mod

    backbone = _StubBackbone()
    monkeypatch.setattr(fe_mod, "_load_cell_backbone", lambda *a, **k: backbone)
    return backbone


class TestCellModelRegistry:
    def test_keys_disjoint_from_modeltype(self):
        from dino.feature_extractor import CELL_MODELS, ModelType
        assert not (set(CELL_MODELS) & set(ModelType.__members__))

    def test_specs_well_formed(self):
        from dino.feature_extractor import CELL_MODELS, InputSpec
        for key, spec in CELL_MODELS.items():
            assert spec.norm in ("self", "fixed"), key
            if spec.norm == "fixed":
                assert spec.mean is not None and spec.std is not None, key
                assert len(spec.mean) == spec.n_channels, key
                assert len(spec.std) == spec.n_channels, key
            assert spec.n_channels >= 1, key
            assert spec.img_size > 0, key
            assert isinstance(spec.input_spec, InputSpec), key

    def test_primary_variant(self):
        from dino.feature_extractor import CELL_MODELS
        spec = CELL_MODELS["CELL_CA_L16"]
        assert spec.entrypoint == "channel_adaptive_dino_vitl16"
        assert (spec.img_size, spec.n_channels) == (224, 1)


class TestInputSpec:
    def test_replicate_3ch_matches_legacy(self):
        """Bit-identical to the old expand(3,-1,-1) — guards every dino_* cache."""
        from dino.feature_extractor import InputSpec
        sdf = torch.randn(224, 224).clamp(-1, 1)
        legacy = ((sdf.float() + 1.0) * 0.5).unsqueeze(0).expand(3, -1, -1)
        assert torch.equal(InputSpec(3, "replicate")(sdf), legacy)

    def test_single_channel(self):
        from dino.feature_extractor import InputSpec
        sdf = torch.randn(224, 224).clamp(-1, 1)
        out = InputSpec(1, "replicate")(sdf)
        assert out.shape == (1, 224, 224)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_replicate_vs_zeros_5ch(self):
        from dino.feature_extractor import InputSpec
        sdf = torch.randn(16, 16).clamp(-1, 1)
        rep = InputSpec(5, "replicate")(sdf)
        zer = InputSpec(5, "zeros")(sdf)
        assert rep.shape == zer.shape == (5, 16, 16)
        assert torch.equal(rep[0], zer[0])
        for c in range(1, 5):
            assert torch.equal(rep[c], rep[0])
            assert torch.equal(zer[c], torch.zeros_like(zer[0]))

    def test_rejects_bad_mode(self):
        from dino.feature_extractor import InputSpec
        with pytest.raises(ValueError):
            InputSpec(3, "rgb")

    def test_pickle_roundtrip(self):
        """Specs cross into DataLoader workers when num_workers > 0."""
        from dino.feature_extractor import InputSpec
        spec = InputSpec(5, "zeros")
        assert pickle.loads(pickle.dumps(spec)) == spec


class TestInputSpecResize:
    """out_size feeds a 128 px backbone (CELL_CP_S8) from the 224 px pipeline."""

    def test_noop_when_sizes_match(self):
        from dino.feature_extractor import InputSpec
        sdf = torch.randn(224, 224).clamp(-1, 1)
        assert torch.equal(InputSpec(1, "replicate", 224)(sdf),
                           InputSpec(1, "replicate", None)(sdf))

    def test_downsamples(self):
        from dino.feature_extractor import InputSpec
        out = InputSpec(5, "replicate", 128)(torch.randn(224, 224).clamp(-1, 1))
        assert out.shape == (5, 128, 128)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_matches_recomputing_the_sdf_at_128(self):
        """Resampling the normalized SDF ~= _mask_to_sdf(mask, 128).

        Both divide distances by their own half-width, so the field is
        resolution-independent up to interpolation error. This is the check the
        plan deferred until a 128-input variant was actually run.
        """
        from graph_dyno.precompute.precompute_dino_embeddings import _mask_to_sdf
        from dino.feature_extractor import InputSpec

        H = W = 256
        Y, X = np.ogrid[:H, :W]
        theta = np.arctan2(Y - 128, X - 128)
        r = np.sqrt((X - 128.0) ** 2 + (Y - 128.0) ** 2)
        mask = (r < 55 + 12 * np.sin(5 * theta) + 6 * np.cos(3 * theta)).astype(np.float32)

        sdf224, _ = _mask_to_sdf(mask, 224)
        sdf128, _ = _mask_to_sdf(mask, 128)
        direct = torch.from_numpy(sdf128.astype(np.float32))
        # undo the [0,1] remap so the two are on the same [-1,1] scale
        resampled = InputSpec(1, "replicate", 128)(
            torch.from_numpy(sdf224.astype(np.float32)))[0] * 2.0 - 1.0

        assert (resampled - direct).abs().max() < 0.05
        inter = ((resampled > 0) & (direct > 0)).sum()
        union = ((resampled > 0) | (direct > 0)).sum()
        assert inter / union > 0.99

    def test_spec_carries_img_size(self):
        from dino.feature_extractor import CELL_MODELS
        assert CELL_MODELS["CELL_CP_S8"].input_spec.out_size == 128
        assert CELL_MODELS["CELL_CA_L16"].input_spec.out_size == 224


class TestSimMaskDatasetBackCompat:
    def test_single_arg_still_3ch(self):
        from graph_dyno.precompute.precompute_dino_embeddings import _SimMaskDataset
        ds = _SimMaskDataset(torch.randn(4, 224, 224).clamp(-1, 1))
        (img,) = ds[0]
        assert img.shape == (3, 224, 224)


class TestSelfNormalize:
    def test_zero_mean_unit_std(self):
        from dino.feature_extractor import _self_normalize
        x = torch.rand(4, 1, 32, 32) * 7.0 + 3.0
        out = _self_normalize(x)
        assert torch.allclose(out.mean(dim=(-2, -1)), torch.zeros(4, 1), atol=1e-4)
        assert torch.allclose(
            out.std(dim=(-2, -1), unbiased=False), torch.ones(4, 1), atol=1e-4)

    def test_affine_invariant(self):
        """Removing per-crop mean and scale is the whole point — check it holds."""
        from dino.feature_extractor import _self_normalize
        x = torch.rand(3, 1, 16, 16)
        assert torch.allclose(_self_normalize(x), _self_normalize(x * 4.0 + 2.0),
                              atol=1e-4)


class TestCellDINOFeatureExtractor:
    def test_patch_tokens_unsliced(self, stub_backbone):
        from dino.feature_extractor import CellDINOFeatureExtractor
        fe = CellDINOFeatureExtractor("CELL_CA_L16")
        cls, patches = fe(torch.rand(2, 1, 224, 224))
        assert cls.shape == (2, 32)
        # 196 patches at 224/16, and 0 register tokens — nothing to skip.
        assert patches.shape == (2, 196, 32)

    def test_embed_dim_and_input_spec(self, stub_backbone):
        from dino.feature_extractor import CellDINOFeatureExtractor, InputSpec
        fe = CellDINOFeatureExtractor("CELL_CA_L16")
        assert fe.embed_dim == 32
        assert fe.input_spec == InputSpec(1, "replicate", 224)

    def test_input_is_self_normalized(self, stub_backbone):
        from dino.feature_extractor import CellDINOFeatureExtractor
        fe = CellDINOFeatureExtractor("CELL_CA_L16")
        fe(torch.rand(2, 1, 8, 8))
        seen = stub_backbone.last_input
        assert torch.allclose(seen.mean(dim=(-2, -1)), torch.zeros(2, 1), atol=1e-4)

    def test_buffers_move_with_to(self, stub_backbone):
        """encoders/dino.py reads the device off buffers(), so one must exist."""
        from dino.feature_extractor import CellDINOFeatureExtractor
        fe = CellDINOFeatureExtractor("CELL_CA_L16")
        assert "_device_anchor" in dict(fe.named_buffers())
        assert next(fe.buffers()).dtype == torch.float32
        fe.to(torch.float64)
        assert next(fe.buffers()).dtype == torch.float64

    def test_unknown_key(self, stub_backbone):
        from dino.feature_extractor import CellDINOFeatureExtractor
        with pytest.raises(KeyError):
            CellDINOFeatureExtractor("CELL_NOT_A_MODEL")


class TestCellDINOEncoder:
    def test_embedding_dim_from_extractor(self, stub_backbone):
        from graph_dyno.encoders.dino import DINOEncoder
        e = DINOEncoder("CELL_CA_L16", use_patches=False, backend="cell_dino")
        assert e.embedding_dim == 32
        assert e.companion_suffix == "dino_CELL_CA_L16"

    def test_embedding_dim_with_patches(self, stub_backbone):
        from graph_dyno.encoders.dino import DINOEncoder
        e = DINOEncoder("CELL_CA_L16", use_patches=True, backend="cell_dino")
        assert e.embedding_dim == 64


class TestCellDINOFactories:
    def test_companion_suffix(self):
        from graph_dyno.encoders.factory import get_companion_suffix
        cfg = OmegaConf.create(
            {"shape": {"type": "dino", "dino_model": "CELL_CA_L16",
                       "dino_backend": "cell_dino"}})
        assert get_companion_suffix(cfg) == "dino_CELL_CA_L16"

    def test_missing_clone_raises(self):
        from graph_dyno.encoders.factory import create_shape_encoder
        cfg = OmegaConf.create({"shape": {
            "type": "dino", "dino_model": "CELL_CA_L16",
            "dino_backend": "cell_dino", "dino_repo_dir": "/nonexistent",
            "dino_weights_path": "/nonexistent/w.pth",
        }})
        with pytest.raises(FileNotFoundError, match="dinov2"):
            create_shape_encoder(cfg, device="cpu")

    def test_backend_key_mismatch_raises(self):
        from graph_dyno.encoders.factory import create_shape_encoder
        cfg = OmegaConf.create({"shape": {
            "type": "dino", "dino_model": "CELL_CA_L16", "dino_backend": "hf",
        }})
        with pytest.raises(ValueError, match="does not match"):
            create_shape_encoder(cfg, device="cpu")

    def test_missing_weights_raises(self, tmp_path):
        """With a clone present but no weights, the error must name the gating."""
        from dino.feature_extractor import _load_cell_backbone, CELL_MODELS
        (tmp_path / "hubconf.py").write_text("")
        with pytest.raises(FileNotFoundError, match="gated"):
            _load_cell_backbone(CELL_MODELS["CELL_CA_L16"], repo_dir=str(tmp_path))

    def test_wrong_weights_path_raises(self, tmp_path):
        from dino.feature_extractor import _load_cell_backbone, CELL_MODELS
        (tmp_path / "hubconf.py").write_text("")
        with pytest.raises(FileNotFoundError, match="not found"):
            _load_cell_backbone(CELL_MODELS["CELL_CA_L16"], repo_dir=str(tmp_path),
                                weights_path=str(tmp_path / "missing.pth"))


class TestCellDINOComposition:
    def test_embedding_overlay(self):
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.join(REPO_ROOT, "configs"),
                                   version_base=None):
            cfg = compose(config_name="config",
                          overrides=["model=unified_dino",
                                     "+embedding=cell_dino",
                                     "+ablation=full_no_history"])
        assert cfg.shape.type == "dino"
        assert cfg.shape.dino_backend == "cell_dino"
        assert cfg.shape.dino_model == "CELL_CA_L16"
        assert cfg.shape.dino_use_patches is False
        assert isinstance(cfg.shape.checkpoint_path, str)
        assert isinstance(cfg.shape.dino_repo_dir, str)
        assert isinstance(cfg.shape.dino_weights_path, str)

    def test_hf_overlay_unchanged(self):
        """Existing DINOv3 runs must resolve exactly as before."""
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=os.path.join(REPO_ROOT, "configs"),
                                   version_base=None):
            cfg = compose(config_name="config",
                          overrides=["model=unified_dino",
                                     "+embedding=dino_sp",
                                     "+ablation=full_no_history"])
        assert cfg.shape.dino_backend == "hf"
        assert cfg.shape.dino_repo_dir is None


def _resolved_cell_paths(model_key):
    """(repo_dir, weights_path) from the pth registry, or None if unavailable."""
    from hydra import compose, initialize_config_dir
    overlay = {"CELL_CA_L16": "cell_dino"}[model_key]
    with initialize_config_dir(config_dir=os.path.join(REPO_ROOT, "configs"),
                               version_base=None):
        cfg = compose(config_name="config",
                      overrides=["model=unified_dino", f"+embedding={overlay}",
                                 "+ablation=full_no_history"])
    repo, weights = str(cfg.shape.dino_repo_dir), str(cfg.shape.dino_weights_path)
    if not os.path.exists(os.path.join(repo, "hubconf.py")) or not os.path.exists(weights):
        return None
    return repo, weights


@pytest.mark.slow
@pytest.mark.parametrize("model_key,embed_dim,n_patches", [
    ("CELL_CA_L16", 1024, 196),   # 224/16
])
class TestCellDINORealWeights:
    """Runs only where the gated weights and the dinov2 clone are present."""

    def test_loads_strict_and_forwards(self, model_key, embed_dim, n_patches):
        from dino.feature_extractor import create_feature_extractor, CELL_MODELS

        paths = _resolved_cell_paths(model_key)
        if paths is None:
            pytest.skip(f"Cell-DINO weights/clone not available for {model_key}")
        repo, weights = paths

        spec = CELL_MODELS[model_key]
        fe = create_feature_extractor(model_key, backend="cell_dino",
                                      repo_dir=repo, weights_path=weights)
        assert fe.embed_dim == embed_dim
        sdf = torch.randn(2, spec.img_size, spec.img_size).clamp(-1, 1)
        imgs = torch.stack([fe.input_spec(s) for s in sdf])
        cls, patches = fe(imgs)
        assert cls.shape == (2, embed_dim)
        assert patches.shape == (2, n_patches, embed_dim)
        assert torch.isfinite(cls).all()
