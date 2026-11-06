import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_dyno.encoders.base import ShapeEncoder


class ConvEncoder(ShapeEncoder):
    """Convolutional encoder for SDF images.

    Mirrors the ConvDecoder architecture in reverse: Conv2d downsampling
    stages produce a spatial feature map that is flattened and projected
    to a bottleneck vector.
    """

    def __init__(self, bottleneck_dim=64, output_size=224,
                 channels=(256, 128, 64, 32, 16),
                 intermediate_conv_layers=0,
                 checkpoint_path: str | None = None,
                 sdf_workers: int = 32,
                 target_size: int = 224):
        """
        Args:
            bottleneck_dim: Dimension of the output bottleneck vector.
            output_size: Spatial size of input SDF images.
            channels: Channel sizes matching the decoder (used in reverse).
            intermediate_conv_layers: Same-resolution Conv2d layers per stage.
            checkpoint_path: Path to the .pt checkpoint these weights came from.
                Used to derive a content hash for `companion_suffix` so
                precomputed embeddings are invalidated when the encoder
                changes. Optional for in-process training (autoencoder), but
                required to use this encoder for embedding precomputation.
            sdf_workers: CPU workers for mask→SDF conversion in encode_masks.
            target_size: Spatial size for the SDFs fed to forward().
        """
        super().__init__()
        self._bottleneck_dim = bottleneck_dim
        self.output_size = output_size
        self.channels = channels
        self.init_spatial = 7
        self.checkpoint_path = checkpoint_path
        self.sdf_workers = sdf_workers
        self.target_size = target_size
        self._ckpt_hash_cache: str | None = None

        # Reversed channels for encoding path: e.g. [16, 32, 64, 128, 256]
        enc_channels = list(reversed(channels))

        # Initial conv: 1 -> enc_channels[0], stride 2
        layers = [
            nn.Conv2d(1, enc_channels[0], kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(1, enc_channels[0]),
            nn.ReLU(inplace=True),
        ]
        for _ in range(intermediate_conv_layers):
            layers += [
                nn.Conv2d(enc_channels[0], enc_channels[0],
                          kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(1, enc_channels[0]),
                nn.ReLU(inplace=True),
            ]

        # Downsampling stages
        for i in range(len(enc_channels) - 1):
            layers += [
                nn.Conv2d(enc_channels[i], enc_channels[i + 1],
                          kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(1, enc_channels[i + 1]),
                nn.ReLU(inplace=True),
            ]
            for _ in range(intermediate_conv_layers):
                layers += [
                    nn.Conv2d(enc_channels[i + 1], enc_channels[i + 1],
                              kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(1, enc_channels[i + 1]),
                    nn.ReLU(inplace=True),
                ]

        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[0] * self.init_spatial * self.init_spatial,
                            bottleneck_dim)

    @property
    def bottleneck_dim(self) -> int:
        return self._bottleneck_dim

    @property
    def embedding_dim(self) -> int:
        return self._bottleneck_dim

    @property
    def companion_suffix(self) -> str:
        if self.checkpoint_path is None:
            raise ValueError(
                "ConvEncoder.companion_suffix requires checkpoint_path; "
                "construct ConvEncoder with checkpoint_path=… to use it for "
                "embedding precomputation.")
        if self._ckpt_hash_cache is None:
            self._ckpt_hash_cache = _hash_file(self.checkpoint_path)
        return f"conv_{self._bottleneck_dim}_{self._ckpt_hash_cache}"

    def forward(self, x):
        """Encode SDF images to bottleneck vectors.

        Args:
            x: (B, H, W) SDF images in [-1, 1]
        Returns:
            bottleneck: (B, bottleneck_dim)
        """
        B = x.shape[0]
        x = x.unsqueeze(1)  # (B, 1, H, W)

        # Interpolate to the spatial size the conv stack expects
        target_spatial = self.init_spatial * (2 ** len(self.channels))
        if x.shape[-1] != target_spatial or x.shape[-2] != target_spatial:
            x = F.interpolate(x, size=(target_spatial, target_spatial),
                              mode='bilinear', align_corners=False)

        x = self.encoder(x)  # (B, channels[0], 7, 7)
        x = x.view(B, -1)
        return self.fc(x)

    @torch.no_grad()
    def encode_masks(
        self,
        masks: torch.Tensor,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode binary masks to bottleneck vectors via mask→SDF→conv.

        Args:
            masks: (M, H, W) binary masks
            batch_size: Inference batch size
            num_workers: Unused (kept for interface compatibility); SDF
                conversion uses self.sdf_workers.

        Returns:
            embeddings: (M, bottleneck_dim) float32
            crop_sides: (M,) int32
        """
        from graph_dyno.precompute.precompute_dino_embeddings import _compute_sdfs_multiprocess

        device = next(self.parameters()).device

        sdf_masks, crop_sides = _compute_sdfs_multiprocess(
            masks.cpu(), target_size=self.target_size,
            num_workers=self.sdf_workers,
        )

        was_training = self.training
        self.eval()
        emb_chunks = []
        for i in range(0, sdf_masks.shape[0], batch_size):
            batch = sdf_masks[i:i + batch_size].float().to(device)
            emb_chunks.append(self.forward(batch).cpu())
        if was_training:
            self.train()

        return torch.cat(emb_chunks, dim=0), crop_sides


def _hash_file(path: str | Path, n_hex: int = 8) -> str:
    """SHA-256 hash of a file's contents, truncated to n_hex hex chars."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n_hex]
