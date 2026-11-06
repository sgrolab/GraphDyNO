import cv2
import numpy as np
import torch
import torch.nn.functional as F

from graph_dyno.decoders.base import ShapeDecoder


def _contour_to_mask(contour_points: np.ndarray, output_size: int = 224) -> np.ndarray:
    """Rasterize a centered contour to a binary mask.

    Args:
        contour_points: (P, 2) xy coordinates, centered at origin,
            normalized so that the shape fits in [-0.5, 0.5].
        output_size: Spatial size of the output image.

    Returns:
        mask: (output_size, output_size) float32 in {0, 1}
    """
    # Map from normalized [-0.5, 0.5] to pixel coords [0, output_size)
    pixel_coords = (contour_points + 0.5) * output_size
    pixel_coords = pixel_coords.astype(np.int32)

    mask = np.zeros((output_size, output_size), dtype=np.float32)
    if len(pixel_coords) >= 3:
        cv2.fillPoly(mask, [pixel_coords], 1.0)

    return mask


def coefficients_to_contour(coeffs: np.ndarray, n_coefficients: int) -> np.ndarray:
    """Reconstruct contour points from interleaved Fourier coefficients.

    Args:
        coeffs: (2*n_coefficients,) interleaved [real, imag, ...]
        n_coefficients: Number of Fourier coefficients

    Returns:
        contour: (n_points, 2) xy coordinates in normalized space [-0.5, 0.5]
    """
    real = coeffs[0::2]
    imag = coeffs[1::2]
    Z_trunc = real + 1j * imag

    n_pos = n_coefficients // 2
    n_neg = n_coefficients - n_pos

    n_points = max(n_coefficients * 4, 128)
    Z_full = np.zeros(n_points, dtype=np.complex128)
    Z_full[:n_pos] = Z_trunc[:n_pos]
    Z_full[-n_neg:] = Z_trunc[n_pos:]

    z = np.fft.ifft(Z_full)
    return np.column_stack([z.real, z.imag])


class FourierDecoder(ShapeDecoder):
    """Shape decoder using inverse Fourier transform.

    Reconstructs contours from Fourier coefficients via IFFT,
    then rasterizes to binary masks. No learnable parameters.

    Only used for visualization (never in the training gradient path).
    """

    def __init__(self, n_coefficients: int = 32, output_size: int = 224):
        super().__init__()
        self._n_coefficients = n_coefficients
        self.output_size = output_size

    @property
    def bottleneck_dim(self) -> int:
        return 2 * self._n_coefficients

    @property
    def input_dim(self) -> int:
        return 2 * self._n_coefficients

    def get_bottleneck(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Identity — Fourier coefficients are already the bottleneck."""
        return embeddings

    def decode_to_contour(self, bottleneck: torch.Tensor) -> list[np.ndarray]:
        """Reconstruct contour points from Fourier coefficients.

        Args:
            bottleneck: (B, 2*n_coefficients) interleaved [real, imag, ...]
        Returns:
            List of (n_points, 2) arrays in normalized [-0.5, 0.5] space
        """
        coeffs = bottleneck.detach().cpu().numpy()
        return [
            coefficients_to_contour(coeffs[i], self._n_coefficients)
            for i in range(coeffs.shape[0])
        ]

    def decode_from_bottleneck(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Reconstruct binary mask images from Fourier coefficients.

        Args:
            bottleneck: (B, 2*n_coefficients) interleaved [real, imag, ...]
        Returns:
            masks: (B, output_size, output_size) binary masks
        """
        device = bottleneck.device
        contours = self.decode_to_contour(bottleneck)

        masks = np.empty((len(contours), self.output_size, self.output_size), dtype=np.float32)
        for i, contour in enumerate(contours):
            masks[i] = _contour_to_mask(contour, self.output_size)

        return torch.from_numpy(masks).to(device)

    def training_step(self, embeddings: torch.Tensor, targets: torch.Tensor) -> dict:
        """Compute reconstruction loss (for validation/metrics only)."""
        preds = self.forward(embeddings)
        return {'image_mse': F.mse_loss(preds, targets)}
