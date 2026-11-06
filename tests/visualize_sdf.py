"""
Visualize SDF conversion for various mask shapes and sizes.

Produces a figure showing original binary masks alongside their SDFs
to verify the crop → resize → SDF pipeline matches expectations.

Usage:
    python scripts/visualize_sdf.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from skimage.draw import polygon2mask

from graph_dyno.precompute.precompute_dino_embeddings import _mask_to_sdf


# -----------------------------------------------------------------------
# Shape generation using polyhoop grid convention:
#   grid = linspace(-r_max, r_max, image_size)
#   center at pixel (image_size - 1) / 2  (= 31.5 for image_size=64)
#   continuous coord p -> pixel p * scale + grid_center
# -----------------------------------------------------------------------

def _rasterize_polygon(vertices, image_size, r_max):
    """Rasterize polygon vertices (in continuous coords centered at 0)
    onto an image_size x image_size grid, matching polyhoop convention."""
    scale = (image_size - 1) / (2 * r_max)
    grid_center = (image_size - 1) / 2.0
    pixel_verts = vertices * scale + grid_center  # (N, 2) in pixel coords
    return polygon2mask((image_size, image_size), pixel_verts).astype(np.float32)


def make_circle(image_size, r_max, r):
    """Binary circle on polyhoop-style linspace grid."""
    grid = np.linspace(-r_max, r_max, image_size)
    yy, xx = np.meshgrid(grid, grid, indexing='ij')
    return (xx ** 2 + yy ** 2 < r ** 2).astype(np.float32)


def make_ellipse(image_size, r_max, ry, rx):
    """Binary ellipse on polyhoop-style linspace grid."""
    grid = np.linspace(-r_max, r_max, image_size)
    yy, xx = np.meshgrid(grid, grid, indexing='ij')
    return ((yy / ry) ** 2 + (xx / rx) ** 2 < 1).astype(np.float32)


def make_star(image_size, r_max, outer_r, inner_r, n_points=5):
    """Binary star via polygon2mask on polyhoop grid."""
    angles = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False) - np.pi / 2
    pts = []
    for i, a in enumerate(angles):
        r = outer_r if i % 2 == 0 else inner_r
        pts.append([r * np.sin(a), r * np.cos(a)])  # (row, col) in continuous
    return _rasterize_polygon(np.array(pts), image_size, r_max)


def make_L_shape(image_size, r_max, arm_len, arm_width):
    """Binary L-shape via polygon2mask on polyhoop grid, centered at centroid."""
    hl = arm_len / 2
    # L-shape vertices (continuous coords, then shift to center at centroid)
    verts = np.array([
        [-hl, -hl],
        [-hl, -hl + arm_width],
        [hl - arm_width, -hl + arm_width],
        [hl - arm_width, hl],
        [hl, hl],
        [hl, -hl],
    ], dtype=np.float64)
    # Center at centroid
    centroid = verts.mean(axis=0)
    verts -= centroid
    return _rasterize_polygon(verts, image_size, r_max)


def make_crescent(image_size, r_max, outer_r, inner_r, offset):
    """Binary crescent on polyhoop-style linspace grid."""
    grid = np.linspace(-r_max, r_max, image_size)
    yy, xx = np.meshgrid(grid, grid, indexing='ij')
    outer = xx ** 2 + yy ** 2 < outer_r ** 2
    inner = (xx - offset) ** 2 + yy ** 2 < inner_r ** 2
    return (outer & ~inner).astype(np.float32)


def make_blob(image_size, r_max, base_r, seed=42):
    """Binary blob via polygon2mask on polyhoop grid."""
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

H, W = 64, 64
r_max = 30.0  # grid extent for all shapes

masks = {
    "circle r=5": make_circle(H, r_max, 5),
    "circle r=10": make_circle(H, r_max, 10),
    "circle r=20": make_circle(H, r_max, 20),
    "circle r=28": make_circle(H, r_max, 28),
    "ellipse": make_ellipse(H, r_max, 10, 20),
    "star": make_star(H, r_max, 22, 10, n_points=5),
    "L-shape": make_L_shape(H, r_max, 40, 8),
    "crescent": make_crescent(H, r_max, 20, 16, 8),
    "blob seed=0": make_blob(H, r_max, 18, seed=0),
    "blob seed=7": make_blob(H, r_max, 15, seed=7),
    "blob seed=99": make_blob(H, r_max, 20, seed=99),
}

# -----------------------------------------------------------------------
# Compute SDFs and visualize
# -----------------------------------------------------------------------

n = len(masks)
fig, axes = plt.subplots(n, 3, figsize=(12, 3.2 * n))
fig.suptitle("Binary Mask → Crop to BBox → SDF (cv2 + scipy pipeline)", fontsize=14, y=1.0)

for row, (name, mask) in enumerate(masks.items()):
    result = _mask_to_sdf(mask, target_size=224)

    # --- Column 0: original binary mask ---
    ax0 = axes[row, 0]
    ax0.imshow(mask, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax0.set_title(f"{name}\n({H}x{W})", fontsize=10)
    ax0.set_ylabel(name, fontsize=9, rotation=0, labelpad=80, va="center")
    ax0.set_xticks([])
    ax0.set_yticks([])

    if result is None:
        axes[row, 1].text(0.5, 0.5, "empty mask", ha="center", va="center", transform=axes[row, 1].transAxes)
        axes[row, 2].text(0.5, 0.5, "empty mask", ha="center", va="center", transform=axes[row, 2].transAxes)
        continue

    sdf, crop_side = result

    # --- Column 1: square crop region (centered, for context) ---
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, min(cy - side // 2, H - side))
    x0 = max(0, min(cx - side // 2, W - side))
    crop_end_y = min(y0 + side, H)
    crop_end_x = min(x0 + side, W)
    crop = mask[y0:crop_end_y, x0:crop_end_x]

    ax1 = axes[row, 1]
    ax1.imshow(crop, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax1.set_title(f"crop ({crop_side}x{crop_side})", fontsize=10)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # --- Column 2: SDF (diverging colormap, 0 = boundary) ---
    ax2 = axes[row, 2]
    sdf_f32 = sdf.astype(np.float32)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax2.imshow(sdf_f32, cmap="RdBu", norm=norm, origin="lower")
    ax2.contour(sdf_f32, levels=[0], colors="black", linewidths=0.8)
    ax2.set_title(f"SDF (224x224)\nmin={sdf_f32.min():.2f} max={sdf_f32.max():.2f}", fontsize=10)
    ax2.set_xticks([])
    ax2.set_yticks([])

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
sm = plt.cm.ScalarMappable(cmap="RdBu", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
fig.colorbar(sm, cax=cbar_ax, label="SDF value (+ inside, − outside)")

plt.tight_layout(rect=[0, 0, 0.91, 0.98])
plt.savefig("scripts/sdf_visualization.png", dpi=150, bbox_inches="tight")
print("Saved to scripts/sdf_visualization.png")
plt.show()


# -----------------------------------------------------------------------
# Verify rescale_dino_to_channeled round-trip
#
# Pipeline: original mask → _mask_to_sdf → 224x224 SDF
#           → rescale_dino_to_channeled → recovered mask (image_size x image_size)
# -----------------------------------------------------------------------

import torch
from graph_dyno.plotting.plot import rescale_dino_to_channeled

image_size = H  # same as the masks above

# Collect SDF images and crop_sides for all masks
names = []
original_masks = []
sdf_images = []
crop_sides_list = []

for name, mask in masks.items():
    result = _mask_to_sdf(mask, target_size=224)
    if result is None:
        continue
    sdf, crop_side = result
    names.append(name)
    original_masks.append(mask)
    sdf_images.append(torch.from_numpy(sdf.astype(np.float32)))
    crop_sides_list.append(crop_side)

N = len(names)
sdf_tensor = torch.stack(sdf_images)                    # (N, 224, 224)
crop_sides_tensor = torch.tensor(crop_sides_list, dtype=torch.float32)  # (N,)

recovered = rescale_dino_to_channeled(sdf_tensor, crop_sides_tensor, image_size)  # (N, H, W)

# Plot: original | SDF 224x224 | recovered mask | difference
fig2, axes2 = plt.subplots(N, 4, figsize=(14, 3.0 * N))
fig2.suptitle(
    "Rescale Verification: mask → SDF (224×224) → rescale_dino_to_channeled → recovered mask",
    fontsize=13, y=1.0,
)

for row in range(N):
    orig = original_masks[row]
    rec = recovered[row].numpy()
    sdf_f32 = sdf_images[row].numpy()
    cs = crop_sides_list[row]

    # Col 0: original binary mask
    ax = axes2[row, 0]
    ax.imshow(orig, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.set_title(f"{names[row]}\n({image_size}×{image_size})", fontsize=9)
    if row == 0:
        ax.set_xlabel("Original Mask")
    ax.set_xticks([]); ax.set_yticks([])

    # Col 1: SDF at 224x224
    ax = axes2[row, 1]
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    ax.imshow(sdf_f32, cmap="RdBu", norm=norm, origin="lower")
    ax.contour(sdf_f32, levels=[0], colors="black", linewidths=0.5)
    ax.set_title(f"SDF (224×224)\ncrop_side={cs}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # Col 2: recovered binary mask
    ax = axes2[row, 2]
    ax.imshow(rec, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.set_title(f"Recovered\n({image_size}×{image_size})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # Col 3: difference (original - recovered)
    ax = axes2[row, 3]
    diff = orig - rec
    iou_mask = (orig > 0.5) | (rec > 0.5)
    intersection = ((orig > 0.5) & (rec > 0.5)).sum()
    iou = intersection / iou_mask.sum() if iou_mask.sum() > 0 else 1.0
    ax.imshow(diff, cmap="RdBu", origin="lower", vmin=-1, vmax=1)
    ax.set_title(f"Diff (orig−rec)\nIoU={iou:.3f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("scripts/sdf_rescale_verification.png", dpi=150, bbox_inches="tight")
print("Saved to scripts/sdf_rescale_verification.png")
plt.show()
