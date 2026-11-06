import torch
from omegaconf import DictConfig

from graph_dyno.decoders.base import ShapeDecoder


def create_shape_decoder(cfg: DictConfig, device: str = "cuda") -> ShapeDecoder:
    """Create a shape decoder based on config.

    Args:
        cfg: Hydra config with shape section
        device: Device to place the decoder on

    Returns:
        ShapeDecoder instance
    """
    shape_type = cfg.shape.type

    if shape_type == "dino":
        from graph_dyno.decoders.dino import DINODecoder

        ckpt = torch.load(cfg.shape.checkpoint_path, map_location=device, weights_only=True)
        decoder = DINODecoder(
            input_dim=ckpt["input_dim"],
            bottleneck_dim=ckpt["bottleneck_dim"],
            output_size=ckpt.get("output_size", 224),
            channels=tuple(ckpt.get("channels", (256, 128, 64, 32, 16))),
            intermediate_conv_layers=ckpt.get("intermediate_conv_layers", 0),
            proj_hidden_layers=ckpt.get("proj_hidden_layers", 0),
        )
        decoder.load_state_dict(ckpt["decoder"])
        return decoder.to(device)

    elif shape_type == "fourier":
        from graph_dyno.decoders.fourier import FourierDecoder

        n = cfg.shape.get("n_coefficients", 32)
        output_size = cfg.shape.get("output_size", 224)
        return FourierDecoder(n_coefficients=n, output_size=output_size).to(device)

    elif shape_type == "conv":
        from graph_dyno.decoders.conv import ConvDecoder

        ckpt = torch.load(cfg.shape.checkpoint_path, map_location=device, weights_only=True)
        decoder = ConvDecoder(
            bottleneck_dim=ckpt["bottleneck_dim"],
            channels=tuple(ckpt.get("channels", (256, 128, 64, 32, 16))),
            intermediate_conv_layers=ckpt.get("intermediate_conv_layers", 0),
        )
        decoder.load_state_dict(ckpt["decoder"])
        return decoder.to(device)

    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
