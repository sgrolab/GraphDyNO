"""
Visualize Fourier shape descriptor encode/decode for various mask shapes.

Produces a figure showing:
  - Original binary mask
  - Contour extracted by encoder
  - Reconstructed mask from decoder at various n_coefficients

Usage:
    python tests/visualize_fourier.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask

from graph_dyno.encoders.fourier import FourierEncoder
from graph_dyno.decoders.fourier import FourierDecoder


# -----------------------------------------------------------------------
# Shape generators (reused from visualize_sdf.py)
# -----------------------------------------------------------------------

def _rasterize_polygon(vertices, image_size, r_max):
    scale = (image_size - 1) / (2 * r_max)
    grid_center = (image_size - 1) / 2.0
    pixel_verts = vertices * scale + grid_center
    return polygon2mask((image_size, image_size), pixel_verts).astype(np.float32)


def make_circle(image_size, r_max, r):
    grid = np.linspace(-r_max, r_max, image_size)
    yy, xx = np.meshgrid(grid, grid, indexing='ij')
    return (xx ** 2 + yy ** 2 < r ** 2).astype(np.float32)


def make_ellipse(image_size, r_max, ry, rx):
    grid = np.linspace(-r_max, r_max, image_size)
    yy, xx = np.meshgrid(grid, grid, indexing='ij')
    return ((yy / ry) ** 2 + (xx / rx) ** 2 < 1).astype(np.float32)


def make_star(image_size, r_max, outer_r, inner_r, n_points=5):
    angles = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False) - np.pi / 2
    pts = []
    for i, a in enumerate(angles):
        r = outer_r if i % 2 == 0 else inner_r
        pts.append([r * np.sin(a), r * np.cos(a)])
    return _rasterize_polygon(np.array(pts), image_size, r_max)


def make_crescent(image_size, r_max, outer_r, inner_r, offset):
    grid = np.linspace(-r_max, r_max, image_size)
    yy, xx = np.meshgrid(grid, grid, indexing='ij')
    outer = xx ** 2 + yy ** 2 < outer_r ** 2
    inner = (xx - offset) ** 2 + yy ** 2 < inner_r ** 2
    return (outer & ~inner).astype(np.float32)


def make_blob(image_size, r_max, base_r, seed=42):
    rng = np.random.RandomState(seed)
    n_harmonics = 8
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    radii = np.ones_like(angles) * base_r
    for k in range(2, n_harmonics + 2):
        amp = rng.uniform(0.05, 0.25) * base_r
        phase = rng.uniform(0, 2 * np.pi)
        radii += amp * np.sin(k * angles + phase)
    pts = np.stack([radii * np.sin(angles), radii * np.cos(angles)], axis=-1)
    return _rasterize_polygon(pts, image_size, r_max)


# -----------------------------------------------------------------------
# Build test masks
# -----------------------------------------------------------------------

H, W = 128, 128
r_max = 50.0

masks = {
    "circle": make_circle(H, r_max, 20),
    "ellipse": make_ellipse(H, r_max, 15, 30),
    "star": make_star(H, r_max, 35, 15, n_points=5),
    "crescent": make_crescent(H, r_max, 30, 24, 12),
    "blob seed=0": make_blob(H, r_max, 25, seed=0),
    "blob seed=7": make_blob(H, r_max, 20, seed=7),
}

n_coeff_values = [4, 8, 16, 32, 64]

# -----------------------------------------------------------------------
# Figure 1: Reconstruction at different n_coefficients
# -----------------------------------------------------------------------

n_shapes = len(masks)
n_cols = 1 + len(n_coeff_values)  # original + reconstructions
fig, axes = plt.subplots(n_shapes, n_cols, figsize=(2.5 * n_cols, 2.5 * n_shapes))
fig.suptitle("Fourier Shape Descriptors: Original vs Reconstruction", fontsize=14, y=1.02)

for row, (name, mask) in enumerate(masks.items()):
    # Original
    ax = axes[row, 0]
    ax.imshow(mask, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.set_title("Original" if row == 0 else "", fontsize=9)
    ax.set_ylabel(name, fontsize=9, rotation=0, labelpad=70, va="center")
    ax.set_xticks([]); ax.set_yticks([])

    masks_tensor = torch.from_numpy(mask).unsqueeze(0)

    for col, n_coeff in enumerate(n_coeff_values):
        encoder = FourierEncoder(n_coefficients=n_coeff)
        decoder = FourierDecoder(n_coefficients=n_coeff, output_size=H)

        embeddings, crop_sides = encoder.encode_masks(masks_tensor, num_workers=1)
        reconstruction = decoder.decode_from_bottleneck(embeddings)

        ax = axes[row, col + 1]
        ax.imshow(reconstruction[0].numpy(), cmap="gray", origin="lower", vmin=0, vmax=1)
        if row == 0:
            ax.set_title(f"n={n_coeff}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout(rect=[0, 0, 1, 0.98])
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/fourier_reconstruction.png", dpi=150, bbox_inches="tight")
print("Saved to plots/fourier_reconstruction.png")

# -----------------------------------------------------------------------
# Figure 2: Contour overlay — original mask with reconstructed contour
# -----------------------------------------------------------------------

n_coeff_overlay = 32
encoder_overlay = FourierEncoder(n_coefficients=n_coeff_overlay)
decoder_overlay = FourierDecoder(n_coefficients=n_coeff_overlay, output_size=H)

fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8))
fig2.suptitle(f"Fourier Contour Overlay (n_coefficients={n_coeff_overlay})", fontsize=14)

for idx, (name, mask) in enumerate(masks.items()):
    row, col = divmod(idx, 3)
    ax = axes2[row, col]

    ax.imshow(mask, cmap="gray", origin="lower", vmin=0, vmax=1, alpha=0.5)

    masks_tensor = torch.from_numpy(mask).unsqueeze(0)
    embeddings, _ = encoder_overlay.encode_masks(masks_tensor, num_workers=1)
    contours = decoder_overlay.decode_to_contour(embeddings)

    if len(contours) > 0:
        contour = contours[0]
        # Map from normalized [-0.5, 0.5] back to pixel coords
        contour_px = (contour + 0.5) * H
        # Close the contour for plotting
        contour_px = np.vstack([contour_px, contour_px[0:1]])
        ax.plot(contour_px[:, 0], contour_px[:, 1], 'r-', linewidth=1.5, label="Fourier contour")

    ax.set_title(name, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

axes2[0, 0].legend(fontsize=8, loc="upper right")
plt.tight_layout()
plt.savefig("plots/fourier_contour_overlay.png", dpi=150, bbox_inches="tight")
print("Saved to plots/fourier_contour_overlay.png")

# -----------------------------------------------------------------------
# Figure 3: Coefficient magnitude spectrum
# -----------------------------------------------------------------------

n_coeff_spectrum = 64
encoder_spectrum = FourierEncoder(n_coefficients=n_coeff_spectrum)
fig3, axes3 = plt.subplots(2, 3, figsize=(14, 8))
fig3.suptitle(f"Fourier Coefficient Magnitudes (n={n_coeff_spectrum})", fontsize=14)

for idx, (name, mask) in enumerate(masks.items()):
    row, col = divmod(idx, 3)
    ax = axes3[row, col]

    masks_tensor = torch.from_numpy(mask).unsqueeze(0)
    embeddings, _ = encoder_spectrum.encode_masks(masks_tensor, num_workers=1)
    coeffs = embeddings[0].numpy()
    real = coeffs[0::2]
    imag = coeffs[1::2]
    magnitudes = np.sqrt(real**2 + imag**2)

    ax.bar(range(n_coeff_spectrum), magnitudes, width=0.8, color="steelblue")
    ax.set_xlabel("Coefficient index")
    ax.set_ylabel("|c_k|")
    ax.set_title(name, fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-6)

plt.tight_layout()
plt.savefig("plots/fourier_spectrum.png", dpi=150, bbox_inches="tight")
print("Saved to plots/fourier_spectrum.png")

plt.show()
