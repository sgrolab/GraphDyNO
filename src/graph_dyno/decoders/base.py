from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeDecoder(nn.Module, ABC):
    """Abstract base class for shape decoders.

    Shape decoders map precomputed embeddings to a bottleneck space and can
    reconstruct shape images from bottleneck vectors. The output format depends
    on the implementation (e.g. SDF for DINO, binary mask for VAE). They are
    created by a factory and passed into GraphModelDINO.
    """

    @property
    @abstractmethod
    def bottleneck_dim(self) -> int:
        """Dimension of the bottleneck representation."""

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Dimension of the raw embedding input (from encoder)."""

    @abstractmethod
    def get_bottleneck(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project raw embeddings to bottleneck space.

        Args:
            embeddings: (B, input_dim)
        Returns:
            bottleneck: (B, bottleneck_dim)
        """

    @abstractmethod
    def decode_from_bottleneck(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Reconstruct a shape image from a bottleneck vector.

        The output format is implementation-specific (e.g. SDF in [-1, 1],
        binary mask in [0, 1], etc.).

        Args:
            bottleneck: (B, bottleneck_dim)
        Returns:
            image: (B, H, W)
        """

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Convenience: embedding -> bottleneck -> shape image."""
        return self.decode_from_bottleneck(self.get_bottleneck(embeddings))

    @abstractmethod
    def training_step(self, embeddings: torch.Tensor, targets: torch.Tensor) -> dict:
        """Compute loss dict for decoder training/fine-tuning.

        Args:
            embeddings: (B, input_dim) raw embeddings
            targets: (B, H, W) ground truth shape image
        Returns:
            Dictionary of loss terms
        """
