from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ShapeEncoder(nn.Module, ABC):
    """Abstract base class for shape encoders.

    Shape encoders convert binary masks to compact embedding vectors.
    They are used during precomputation only (not loaded at training time).
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of the output embedding vector."""

    @property
    @abstractmethod
    def companion_suffix(self) -> str:
        """Suffix for companion files, e.g. 'dino_B' or 'fourier_32'."""

    @abstractmethod
    def encode_masks(
        self,
        masks: torch.Tensor,
        batch_size: int = 64,
        num_workers: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode binary masks to embeddings and crop side lengths.

        Args:
            masks: (M, H, W) binary masks
            batch_size: Batch size for processing
            num_workers: Number of parallel workers

        Returns:
            embeddings: (M, embedding_dim) float32
            crop_sides: (M,) int32
        """
