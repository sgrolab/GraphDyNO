"""Frozen vision-transformer backbones that turn SDF crops into shape features.

Two backbones live behind one interface:

  ``hf``        - DINOv3, natural-image pretraining, loaded through
                  ``transformers.AutoModel``. Keyed by :class:`ModelType`.
  ``cell_dino`` - Meta's Cell-DINO (a DINOv2 variant pretrained on cell
                  fluorescence microscopy), loaded through ``torch.hub`` from an
                  out-of-tree clone of ``facebookresearch/dinov2``. Keyed by
                  :data:`CELL_MODELS`.

The clone is NOT vendored: the Cell-DINO code and weights are CC-by-NC
(research use only, non-commercial), so they must stay outside this repo. Point
``shape.dino_repo_dir`` (``pth.DINOV2_REPO``) at the clone and
``shape.dino_weights_path`` at the gated ``.pth``.

**Preprocessing is frozen per model key.** The companion embedding files are
named ``dino_{model_key}_{hash}_{traj}.pt`` and that name is their only cache
key — a multi-GB artifact with no record of how its pixels were built. So the
channel layout and the normalization scheme are baked into the immutable
registry entry for a key and are never run-time overridable. To change either,
add a new key.
"""

import os
import pickle
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelType(Enum):
    S = "facebook/dinov3-vits16-pretrain-lvd1689m"
    SP = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    B = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    L = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    H = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
    F = "facebook/dinov3-vit7b16-pretrain-lvd1689m"


# ---------------------------------------------------------------------------
# Pixel construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InputSpec:
    """Turns one (H, W) SDF into the (C, H, W) tensor a backbone expects.

    Deliberately free of torch.hub, CUDA and model state: instances are pickled
    into DataLoader workers when ``num_workers > 0``.

    Args:
        n_channels: Channel count the backbone's ``patch_embed`` was built for.
        mode: How to fill channels beyond the first.
            ``"replicate"`` copies the image into every channel — a physically
            realizable "perfectly co-localized stain" image.
            ``"zeros"`` writes the image into channel 0 and leaves the rest
            blank, i.e. a constant in slots the backbone expects real stains in.
            For the multi-channel variants neither is *correct* (those models
            were trained on a fixed stain order); replicate is the default as
            the less degenerate of the two. Only the C=1 variant is a clean map.
    """

    n_channels: int = 3
    mode: str = "replicate"
    out_size: int | None = None

    def __post_init__(self):
        if self.mode not in ("replicate", "zeros"):
            raise ValueError(f"InputSpec.mode must be 'replicate' or 'zeros', got {self.mode!r}")
        if self.n_channels < 1:
            raise ValueError(f"InputSpec.n_channels must be >= 1, got {self.n_channels}")
        if self.out_size is not None and self.out_size < 1:
            raise ValueError(f"InputSpec.out_size must be >= 1, got {self.out_size}")

    def __call__(self, sdf_hw: torch.Tensor) -> torch.Tensor:
        """Map an (H, W) SDF in [-1, 1] to a (C, out_size, out_size) image in [0, 1].

        The ``(sdf + 1) * 0.5`` mapping is fixed: every checkpoint in the
        ``pth/`` registry was trained on it.

        ``out_size`` resamples the SDF to a backbone's native ``img_size`` when
        it differs from the pipeline's ``target_size`` (224). Resampling the
        *normalized* SDF rather than re-running ``_mask_to_sdf`` at the smaller
        size is sound because both divide by their own half-width, so the field
        is resolution-independent to within interpolation error — see
        ``TestInputSpecResize`` for the numeric check. Only the encoder input
        shrinks: the SDF targets the decoder reconstructs stay at 224.
        """
        sdf = sdf_hw.float()
        if self.out_size is not None and sdf.shape[-1] != self.out_size:
            sdf = F.interpolate(
                sdf[None, None], size=(self.out_size, self.out_size),
                mode="bilinear", align_corners=False,
            )[0, 0]
        img = (sdf + 1.0) * 0.5
        if self.mode == "replicate":
            return img.unsqueeze(0).expand(self.n_channels, -1, -1)
        out = img.new_zeros((self.n_channels,) + tuple(img.shape))
        out[0] = img
        return out


# ---------------------------------------------------------------------------
# Cell-DINO registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellDINOSpec:
    """Immutable description of one Cell-DINO variant.

    Args:
        entrypoint: ``hubconf.py`` entrypoint name.
        img_size: Spatial size the variant was pretrained at.
        n_channels: Pinned to the entrypoint's ``in_channels`` default —
            ``_make_cell_dino_model`` calls ``load_state_dict(strict=True)``, so
            a mismatched ``patch_embed.proj.weight`` is a hard failure.
        norm: ``"self"`` for per-image standardization (Meta's own inference
            recipe), ``"fixed"`` for constant mean/std.
        mean, std: Per-channel constants, required when ``norm == "fixed"``.
        channel_mode: :class:`InputSpec` mode for this variant.
    """

    entrypoint: str
    img_size: int
    n_channels: int
    norm: str
    mean: tuple[float, ...] | None
    std: tuple[float, ...] | None
    channel_mode: str

    def __post_init__(self):
        if self.norm not in ("self", "fixed"):
            raise ValueError(f"CellDINOSpec.norm must be 'self' or 'fixed', got {self.norm!r}")
        if self.norm == "fixed":
            if self.mean is None or self.std is None:
                raise ValueError("CellDINOSpec.norm='fixed' requires mean and std")
            if len(self.mean) != self.n_channels or len(self.std) != self.n_channels:
                raise ValueError("CellDINOSpec mean/std must have n_channels entries")

    @property
    def input_spec(self) -> InputSpec:
        return InputSpec(self.n_channels, self.channel_mode, self.img_size)


# Keys must stay disjoint from ModelType so a single `dino_model` string
# unambiguously selects a backend.
#
# norm="self" for all four: Meta's inference notebook
# (dinov2/notebooks/cell_dino/inference.ipynb) standardizes every variant
# per-image, commented "Normalise image as it was done for training". There are
# no published Cell-DINO mean/std constants — dinov2/data/transforms.py carries
# only the 3-channel ImageNet ones and dinov2/data/augmentations.py has no
# Normalize at all — so shipping a fixed (0.5, 0.5) would be an invention.
CELL_MODELS: dict[str, CellDINOSpec] = {
    "CELL_CA_L16":  CellDINOSpec("channel_adaptive_dino_vitl16", 224, 1, "self", None, None, "replicate"),
    "CELL_CP_S8":   CellDINOSpec("cell_dino_cp_vits8",           128, 5, "self", None, None, "replicate"),
    "CELL_HPA_L16": CellDINOSpec("cell_dino_hpa_vitl16",         224, 4, "self", None, None, "replicate"),
    "CELL_HPA_L14": CellDINOSpec("cell_dino_hpa_vitl14",         518, 4, "self", None, None, "replicate"),
}


def _self_normalize(x: torch.Tensor) -> torch.Tensor:
    """Per-image, per-channel standardization — Meta's Cell-DINO inference recipe.

    Their reference divides by 255 first; that is affine and cancels here, so a
    [0, 1] input is equivalent.

    Consequence worth stating plainly: this removes each crop's mean and scale,
    making the resulting embedding strictly shape-only (unlike the DINOv3 one,
    which sees a fixed-offset image). That is acceptable in this pipeline
    because ``_mask_to_sdf`` already scale-normalizes each crop, and the true
    cell scale reaches the GNN separately through ``crop_sides``, which feeds
    its own scale head.
    """
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), unbiased=False, keepdim=True)
    return (x - mean) / (std + 1e-7)


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

class BaseFeatureExtractor(nn.Module):
    """Common interface: ``embed_dim``, ``input_spec``, ``forward -> (cls, patch)``."""

    @property
    def embed_dim(self) -> int:
        raise NotImplementedError

    @property
    def input_spec(self) -> InputSpec:
        raise NotImplementedError

    def forward(self, images):
        raise NotImplementedError


class DINOv3FeatureExtractor(BaseFeatureExtractor):
    """Frozen DINOv3 backbone for feature extraction."""

    def __init__(self, model: ModelType | str = ModelType.H):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel
        from huggingface_hub import login, try_to_load_from_cache

        model_name = model.value if isinstance(model, ModelType) else model
        if try_to_load_from_cache(model_name, "config.json") is None:
            login(os.getenv("HF_TOKEN"))
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        # Number of leading non-patch tokens: CLS + registers. Asserted for the
        # model actually loaded rather than across ModelType — checking every
        # member would mean downloading up to ViT-7B.
        n_reg = getattr(self.model.config, "num_register_tokens", 4)
        self._n_prefix_tokens = 1 + int(n_reg)
        assert self._n_prefix_tokens == 5, (
            f"{model_name} has {n_reg} register tokens (expected 4); the "
            f"pth/ checkpoints were all trained against a 5-token prefix."
        )

        # Cache normalization constants as buffers so they move with .to(device)
        mean = torch.tensor(self.processor.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    @property
    def embed_dim(self) -> int:
        return self.model.config.hidden_size

    @property
    def input_spec(self) -> InputSpec:
        return InputSpec(3, "replicate")

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) float tensor in [0, 1], or list of PIL images
        Returns:
            cls_token: (B, hidden_dim) CLS token
            patch_tokens: (B, num_patches, hidden_dim) patch tokens
        """
        with torch.no_grad():
            if isinstance(images, torch.Tensor):
                pixel_values = (images.to(self._mean.device) - self._mean) / self._std
            else:
                inputs = self.processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self._mean.device)

            outputs = self.model(pixel_values=pixel_values)
            cls_token = outputs.last_hidden_state[:, 0, :]
            patch_tokens = outputs.last_hidden_state[:, self._n_prefix_tokens:, :]

        return cls_token, patch_tokens


class CellDINOFeatureExtractor(BaseFeatureExtractor):
    """Frozen Cell-DINO backbone (DINOv2 architecture, microscopy pretraining)."""

    def __init__(self, model_key: str, repo_dir=None, weights_path=None,
                 weights_url=None, trust_weights: bool = False):
        super().__init__()
        if model_key not in CELL_MODELS:
            raise KeyError(
                f"Unknown Cell-DINO key {model_key!r}. "
                f"Available: {sorted(CELL_MODELS)}"
            )
        self.model_key = model_key
        self.spec = CELL_MODELS[model_key]
        self.model = _load_cell_backbone(
            self.spec, repo_dir=repo_dir, weights_path=weights_path,
            weights_url=weights_url, trust_weights=trust_weights,
        )

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # A parameter-free stub backbone (tests) still needs a device handle.
        self.register_buffer("_device_anchor", torch.zeros(1), persistent=False)
        if self.spec.norm == "fixed":
            c = self.spec.n_channels
            self.register_buffer("_mean", torch.tensor(self.spec.mean).view(1, c, 1, 1))
            self.register_buffer("_std", torch.tensor(self.spec.std).view(1, c, 1, 1))

    @property
    def embed_dim(self) -> int:
        backbone: Any = self.model
        return int(backbone.embed_dim)

    @property
    def input_spec(self) -> InputSpec:
        return self.spec.input_spec

    def forward(self, images):
        """
        Args:
            images: (B, C, H, W) float tensor in [0, 1]
        Returns:
            cls_token: (B, embed_dim)
            patch_tokens: (B, num_patches, embed_dim)
        """
        with torch.no_grad():
            x = images.to(self._device_anchor.device)
            if self.spec.norm == "self":
                x = _self_normalize(x)
            else:
                x = (x - self._mean) / self._std

            backbone: Any = self.model
            out = backbone.forward_features(x)
            # Read the named outputs rather than slicing: register tokens are
            # returned in their own key, so there is nothing to skip.
            return out["x_norm_clstoken"], out["x_norm_patchtokens"]


# ---------------------------------------------------------------------------
# Backbone loading (the one seam tests monkeypatch)
# ---------------------------------------------------------------------------

_GATED_URL = "https://ai.meta.com/resources/models-and-libraries/cell-dino-downloads/"


def _unwrap_state_dict(sd):
    """Peel a training-checkpoint wrapper down to a bare tensor state dict."""
    for key in ("model", "teacher", "student", "state_dict"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    for prefix in ("module.", "backbone."):
        if isinstance(sd, dict) and sd and all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def _load_cell_backbone(spec: CellDINOSpec, repo_dir=None, weights_path=None,
                        weights_url=None, trust_weights: bool = False) -> nn.Module:
    """Build a Cell-DINO backbone from a local dinov2 clone + gated weights.

    Never falls back to ``pretrained=False``: a randomly initialized backbone
    would emit perfectly plausible embeddings and cache them under a valid
    companion suffix, poisoning the cache silently.
    """
    # dinov2's layers probe this at import time and fall back to SDPA.
    os.environ.setdefault("XFORMERS_DISABLED", "1")

    if repo_dir is None or not os.path.exists(os.path.join(str(repo_dir), "hubconf.py")):
        raise FileNotFoundError(
            f"No dinov2 clone at {repo_dir!r} (looked for hubconf.py). Cell-DINO "
            f"lives in facebookresearch/dinov2 and is CC-by-NC, so it is kept "
            f"out-of-tree rather than vendored: "
            f"`git clone https://github.com/facebookresearch/dinov2`, then set "
            f"pth.DINOV2_REPO (or shape.dino_repo_dir) to that directory."
        )
    repo_dir = str(repo_dir)

    if weights_path is None and weights_url is None:
        raise FileNotFoundError(
            f"Cell-DINO weights are gated: accept the licence at {_GATED_URL} "
            f"and the download URLs arrive by e-mail. Set shape.dino_weights_path "
            f"to the downloaded .pth (or shape.dino_weights_url to the e-mailed URL)."
        )
    if weights_path is not None and not os.path.exists(str(weights_path)):
        raise FileNotFoundError(
            f"Cell-DINO weights not found at {weights_path!r}. They are gated: "
            f"accept the licence at {_GATED_URL}, then download the file with "
            f"wget (the e-mailed URLs are not browser-friendly)."
        )

    kwargs: dict[str, Any] = {"pretrained": True}
    if weights_path is not None:
        kwargs["pretrained_path"] = str(weights_path)
    else:
        kwargs["pretrained_url"] = str(weights_url)

    def _build(**extra) -> Any:
        try:
            return torch.hub.load(repo_dir, spec.entrypoint, source="local",
                                  **{**kwargs, **extra})
        except ImportError:
            # hubconf.py also pulls dinotxt / depthers / xray_dino; if any of
            # those has an unmet dependency, go straight at the module.
            import sys
            if repo_dir not in sys.path:
                sys.path.insert(0, repo_dir)
            from dinov2.hub.cell_dino import backbones as cell_backbones
            return getattr(cell_backbones, spec.entrypoint)(**{**kwargs, **extra})

    try:
        return _build()
    except (pickle.UnpicklingError, RuntimeError) as e:
        # torch 2.9 defaults torch.load to weights_only=True, and
        # _make_cell_dino_model calls it bare — a released .pth carrying any
        # non-tensor object fails here rather than at load_state_dict.
        if "weights_only" not in str(e):
            raise
        if not trust_weights:
            raise RuntimeError(
                f"torch.load refused {weights_path!r} under weights_only=True "
                f"({e}). If you trust the file's provenance, re-run with "
                f"shape.dino_trust_weights=true to load it with "
                f"weights_only=False."
            ) from e
        if weights_path is None:
            raise
        model: Any = _build(pretrained=False)
        sd = _unwrap_state_dict(
            torch.load(str(weights_path), map_location="cpu", weights_only=False))
        model.load_state_dict(sd, strict=True)
        return model


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def create_feature_extractor(model_key, backend: str = "hf", repo_dir=None,
                             weights_path=None, weights_url=None,
                             trust_weights: bool = False) -> BaseFeatureExtractor:
    """Build the frozen backbone named by (backend, model_key).

    Args:
        model_key: ``ModelType`` member name / HF repo id for ``hf``;
            a :data:`CELL_MODELS` key for ``cell_dino``.
        backend: ``"hf"`` or ``"cell_dino"``.
        repo_dir: Local dinov2 clone (``cell_dino`` only).
        weights_path: Gated Cell-DINO ``.pth`` (``cell_dino`` only).
        weights_url: URL from the Cell-DINO access e-mail (``cell_dino`` only).
        trust_weights: Opt into ``torch.load(weights_only=False)``.
    """
    if backend not in ("hf", "cell_dino"):
        raise ValueError(f"Unknown dino_backend {backend!r}. Available: 'hf', 'cell_dino'.")

    key_is_cell = isinstance(model_key, str) and model_key in CELL_MODELS
    if key_is_cell != (backend == "cell_dino"):
        raise ValueError(
            f"dino_backend={backend!r} does not match dino_model={model_key!r}: "
            f"{sorted(CELL_MODELS)} require backend='cell_dino', everything else "
            f"requires backend='hf'."
        )

    if backend == "cell_dino":
        return CellDINOFeatureExtractor(
            model_key, repo_dir=repo_dir, weights_path=weights_path,
            weights_url=weights_url, trust_weights=trust_weights,
        )

    if isinstance(model_key, ModelType):
        model_type = model_key
    else:
        model_type = ModelType[model_key] if model_key in ModelType.__members__ else model_key
    return DINOv3FeatureExtractor(model_type)
