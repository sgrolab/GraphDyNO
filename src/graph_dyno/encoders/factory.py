import torch
from omegaconf import DictConfig

from graph_dyno.encoders.base import ShapeEncoder


def get_companion_suffix(cfg: DictConfig) -> str:
    """Derive the companion file suffix from config without instantiating an encoder."""
    shape_type = cfg.shape.type

    if shape_type == "dino":
        # Weight-free by design: this must be derivable before any checkpoint is
        # loaded. Preprocessing is frozen per dino_model key (see
        # dino/feature_extractor.py), so the key alone identifies the artifact.
        return f"dino_{cfg.shape.dino_model}"
    elif shape_type == "fourier":
        return f"fourier_{cfg.shape.n_coefficients}"
    elif shape_type == "conv":
        from graph_dyno.encoders.conv import _hash_file
        ckpt_hash = _hash_file(cfg.shape.checkpoint_path)
        return f"conv_{cfg.shape.bottleneck_dim}_{ckpt_hash}"
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")


def create_shape_encoder(cfg: DictConfig, device: str = "cuda") -> ShapeEncoder:
    """Create a shape encoder based on config.

    Args:
        cfg: Hydra config with shape section
        device: Device to place the encoder on

    Returns:
        ShapeEncoder instance
    """
    shape_type = cfg.shape.type

    if shape_type == "dino":
        from dino.feature_extractor import CELL_MODELS, ModelType
        from graph_dyno.encoders.dino import DINOEncoder

        model_key = cfg.shape.dino_model
        backend = cfg.shape.get("dino_backend", "hf")
        use_patches = cfg.shape.get("dino_use_patches", False)
        sdf_workers = cfg.shape.get("sdf_workers", 32)
        target_size = cfg.shape.get("target_size", 224)

        # Check the backend/key pair before doing hours of SDF work. (The
        # extractor factory checks this too, but only once the encoder is
        # actually built — which for the precompute path is after the SDFs.)
        if (model_key in CELL_MODELS) != (backend == "cell_dino"):
            raise ValueError(
                f"shape.dino_backend={backend!r} does not match "
                f"shape.dino_model={model_key!r}: {sorted(CELL_MODELS)} require "
                f"dino_backend='cell_dino', everything else requires 'hf'."
            )

        model_type = ModelType[model_key] if model_key in ModelType.__members__ else model_key
        return DINOEncoder(
            model_type=model_type, use_patches=use_patches,
            sdf_workers=sdf_workers, target_size=target_size,
            backend=backend,
            repo_dir=cfg.shape.get("dino_repo_dir", None),
            weights_path=cfg.shape.get("dino_weights_path", None),
            weights_url=cfg.shape.get("dino_weights_url", None),
            trust_weights=cfg.shape.get("dino_trust_weights", False),
        ).to(device)

    elif shape_type == "fourier":
        from graph_dyno.encoders.fourier import FourierEncoder

        n = cfg.shape.get("n_coefficients", 32)
        return FourierEncoder(n_coefficients=n)

    elif shape_type == "conv":
        from graph_dyno.encoders.conv import ConvEncoder

        ckpt = torch.load(cfg.shape.checkpoint_path, map_location=device,
                          weights_only=True)
        encoder = ConvEncoder(
            bottleneck_dim=ckpt["bottleneck_dim"],
            output_size=ckpt.get("output_size", 224),
            channels=tuple(ckpt.get("channels", [256, 128, 64, 32, 16])),
            intermediate_conv_layers=ckpt.get("intermediate_conv_layers", 0),
            checkpoint_path=cfg.shape.checkpoint_path,
            sdf_workers=cfg.shape.get("sdf_workers", 32),
            target_size=cfg.shape.get("target_size", 224),
        ).to(device)
        encoder.load_state_dict(ckpt["encoder"])
        encoder.eval()
        return encoder

    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
