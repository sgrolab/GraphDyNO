"""Factory function for creating the appropriate model based on config."""

import torch
from omegaconf import DictConfig, OmegaConf, open_dict


def resolve_shape_config(cfg: DictConfig) -> None:
    """Auto-resolve shape config values from checkpoint.

    Loads the checkpoint metadata and writes resolved values back into
    cfg.shape so downstream code (factories, companion suffix) picks them up.

    For DINO: explicit config values take precedence over checkpoint values
    (with a warning on mismatch).
    For Conv: the checkpoint is authoritative — `create_shape_encoder` and
    `create_shape_decoder` always read these fields from the ckpt, so the
    cfg is overwritten unconditionally to keep them in sync.
    """
    if cfg.shape.type not in ("dino", "conv"):
        return

    ckpt_path = cfg.shape.checkpoint_path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    was_readonly = OmegaConf.is_readonly(cfg)
    OmegaConf.set_readonly(cfg, False)
    try:
        if cfg.shape.type == "dino":
            # Resolve dino_model
            config_model = cfg.shape.get("dino_model", None)
            ckpt_model = ckpt.get("model", None)
            if config_model is None:
                if ckpt_model is None:
                    raise KeyError(f"Checkpoint {ckpt_path} missing 'model' key and dino_model not set in config")
                cfg.shape.dino_model = ckpt_model
                print(f"Auto-resolved dino_model='{ckpt_model}' from checkpoint")
            elif ckpt_model is not None and config_model != ckpt_model:
                print(f"WARNING: config dino_model='{config_model}' differs from checkpoint ('{ckpt_model}')")

            # Resolve dino_backend. Checkpoints predating the Cell-DINO backend
            # have no 'dino_backend' key, so they resolve to 'hf' unchanged.
            # open_dict: shape configs that predate the backend have no such key.
            config_backend = cfg.shape.get("dino_backend", None)
            ckpt_backend = ckpt.get("dino_backend", "hf")
            if config_backend is None:
                with open_dict(cfg.shape):
                    cfg.shape.dino_backend = ckpt_backend
                print(f"Auto-resolved dino_backend='{ckpt_backend}' from checkpoint")
            elif config_backend != ckpt_backend:
                print(f"WARNING: config dino_backend='{config_backend}' differs from "
                      f"checkpoint ('{ckpt_backend}')")

            # Resolve dino_use_patches
            config_patches = cfg.shape.get("dino_use_patches", None)
            ckpt_patches = ckpt.get("use_patches", None)
            if config_patches is None:
                if ckpt_patches is None:
                    raise KeyError(f"Checkpoint {ckpt_path} missing 'use_patches' key and dino_use_patches not set")
                cfg.shape.dino_use_patches = ckpt_patches
                print(f"Auto-resolved dino_use_patches={ckpt_patches} from checkpoint")
            elif ckpt_patches is not None and config_patches != ckpt_patches:
                # Fatal, not a warning: use_patches sets the encoder's output
                # width (D vs 2D) and the decoder's linear_proj.in_features. A
                # disagreement only surfaces in _precompute_bottlenecks — after
                # the multi-hour precompute has written ~10 GB/traj under a
                # perfectly valid companion suffix.
                raise ValueError(
                    f"shape.dino_use_patches={config_patches} contradicts checkpoint "
                    f"{ckpt_path} (use_patches={ckpt_patches}). The encoder would emit "
                    f"a {'2x ' if config_patches else ''}wider embedding than the decoder's "
                    f"linear_proj accepts. Set shape.dino_use_patches={ckpt_patches} "
                    f"(or null to auto-resolve)."
                )

        elif cfg.shape.type == "conv":
            cfg.shape.bottleneck_dim = ckpt["bottleneck_dim"]
            cfg.shape.output_size = ckpt.get("output_size", 224)
            cfg.shape.channels = list(ckpt.get("channels", [256, 128, 64, 32, 16]))
            cfg.shape.intermediate_conv_layers = ckpt.get("intermediate_conv_layers", 0)
            print(f"Auto-resolved conv params from checkpoint: "
                  f"bottleneck_dim={cfg.shape.bottleneck_dim}, "
                  f"channels={cfg.shape.channels}, "
                  f"intermediate_conv_layers={cfg.shape.intermediate_conv_layers}")
    finally:
        OmegaConf.set_readonly(cfg, was_readonly)


def create_model(cfg: DictConfig, device: str = "cuda"):
    """Create a GraphModel or GraphModelDINO based on config.

    Use ``hasattr(model, 'dino_decoder')`` to detect shape-decoder models downstream.

    Returns:
        model: GraphModel or GraphModelDINO instance
    """
    from graph_dyno.decoders.factory import create_shape_decoder

    shape_decoder = create_shape_decoder(cfg, device=device)

    # Keyed on cfg.model.type, not cfg.data.type, so plain GraphModelDINO
    # can still be run on real data as a control. unified_dino.yaml has no
    # `type` key, so existing runs are bit-identical.
    model_type = cfg.model.get("type", "dino")
    if model_type == "real":
        from graph_dyno.models.graph_model_real import GraphModelReal
        model = GraphModelReal(cfg, shape_decoder=shape_decoder, device=device)
    elif model_type == "segablation":
        from graph_dyno.models.graph_model_segablation import GraphModelSegAblation
        model = GraphModelSegAblation(cfg, shape_decoder=shape_decoder,
                                      device=device)
    elif model_type == "dino":
        from graph_dyno.models.graph_model_dino import GraphModelDINO
        model = GraphModelDINO(cfg, shape_decoder=shape_decoder, device=device)
    else:
        raise ValueError(
            f"Unknown model.type: {model_type!r}. "
            f"Available: 'dino', 'real', 'segablation'."
        )

    # GraphModelDINO ignores `valid` entirely (training_step uses plain
    # mse_loss), so an active seg_ablation spec under model.type=dino
    # would train on padded slots and silently produce a result that looks
    # like a segmentation effect but is a masking bug.
    if model_type != "segablation" and cfg.get("seg_ablation", None) is not None:
        from graph_dyno.seg_ablation.plan import spec_is_active
        if spec_is_active(cfg.seg_ablation):
            raise ValueError(
                f"seg_ablation is active but model.type={model_type!r}. "
                f"Only 'segablation' carries the masked losses the "
                f"oracle-masked-miss convention needs. Set "
                f"model.type=segablation (the configs/seg_ablation/* "
                f"overlays do this for you)."
            )

    return model
