from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import torch
from tqdm import tqdm

from graph_dyno.encoders.base import ShapeEncoder


def _mask_to_fourier(mask_np: np.ndarray, n_coefficients: int) -> tuple[np.ndarray, int] | None:
    """Extract Fourier shape descriptors from a binary mask.

    Finds the largest contour, resamples it uniformly by arc length,
    computes the FFT, and keeps the first n_coefficients.

    Args:
        mask_np: (H, W) binary mask (uint8 or float with values 0/1)
        n_coefficients: Number of Fourier coefficients to keep

    Returns:
        (coefficients, crop_side) where coefficients is (2*n_coefficients,) float32
        with interleaved [real_0, imag_0, real_1, imag_1, ...],
        and crop_side is the bounding square side length in original pixels,
        or None if the mask is empty.
    """
    mask_uint8 = (mask_np > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)
    contour = contour.squeeze(1).astype(np.float64)  # (P, 2) xy coordinates

    if len(contour) < 4:
        return None

    # Compute bounding square side (same convention as DINO SDF pipeline)
    ys, xs = np.nonzero(mask_np > 0.5)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop_side = max(y1 - y0, x1 - x0)

    # Center the contour at the origin
    centroid = contour.mean(axis=0)
    contour = contour - centroid

    # Normalize by crop_side so coefficients are scale-invariant
    if crop_side > 0:
        contour = contour / crop_side

    # Resample contour uniformly by arc length
    n_resample = max(n_coefficients * 4, 128)  # oversample for accuracy
    diffs = np.diff(contour, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    # Close the contour
    close_diff = contour[0] - contour[-1]
    seg_lengths = np.append(seg_lengths, np.sqrt((close_diff ** 2).sum()))
    cumulative = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]

    if total_length < 1e-8:
        return None

    # Uniform arc-length parameterization
    target_s = np.linspace(0, total_length, n_resample, endpoint=False)
    # Close the contour for interpolation
    contour_closed = np.vstack([contour, contour[0:1]])
    x_interp = np.interp(target_s, cumulative, contour_closed[:, 0])
    y_interp = np.interp(target_s, cumulative, contour_closed[:, 1])

    # Complex signal and FFT
    z = x_interp + 1j * y_interp
    Z = np.fft.fft(z)

    # Keep low frequencies: first n//2 (positive) and last n//2 (negative)
    n_pos = n_coefficients // 2
    n_neg = n_coefficients - n_pos
    Z_trunc = np.concatenate([Z[:n_pos], Z[-n_neg:]])

    # Interleave real and imaginary parts
    coefficients = np.empty(2 * n_coefficients, dtype=np.float32)
    coefficients[0::2] = Z_trunc.real.astype(np.float32)
    coefficients[1::2] = Z_trunc.imag.astype(np.float32)

    return coefficients, crop_side


def _mask_to_fourier_worker(args):
    """Worker wrapper for ProcessPoolExecutor (top-level for pickling)."""
    mask_np, n_coefficients = args
    result = _mask_to_fourier(mask_np, n_coefficients)
    if result is None:
        return np.zeros(2 * n_coefficients, dtype=np.float32), 0
    return result


class FourierEncoder(ShapeEncoder):
    """Shape encoder using Fourier shape descriptors.

    Extracts contours from binary masks and computes truncated Fourier
    coefficients. No GPU or neural network required.
    """

    def __init__(self, n_coefficients: int = 32):
        super().__init__()
        self._n_coefficients = n_coefficients

    @property
    def embedding_dim(self) -> int:
        return 2 * self._n_coefficients

    @property
    def companion_suffix(self) -> str:
        return f"fourier_{self._n_coefficients}"

    def encode_masks(
        self,
        masks: torch.Tensor,
        batch_size: int = 64,
        num_workers: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode binary masks to Fourier coefficients.

        Args:
            masks: (M, H, W) binary masks
            batch_size: Unused (kept for interface compatibility)
            num_workers: Number of CPU workers for parallel processing

        Returns:
            embeddings: (M, 2*n_coefficients) float32
            crop_sides: (M,) int32
        """
        M = masks.shape[0]
        masks_np = masks.cpu().numpy()

        work_items = [(masks_np[i], self._n_coefficients) for i in range(M)]

        embeddings = np.empty((M, 2 * self._n_coefficients), dtype=np.float32)
        crop_sides = np.empty(M, dtype=np.int32)

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for i, (coeffs, crop_side) in enumerate(
                tqdm(pool.map(_mask_to_fourier_worker, work_items, chunksize=64),
                     total=M, desc="  Fourier encoding", leave=False)
            ):
                embeddings[i] = coeffs
                crop_sides[i] = crop_side

        return torch.from_numpy(embeddings), torch.from_numpy(crop_sides)
