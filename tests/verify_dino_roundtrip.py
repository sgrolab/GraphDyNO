"""
Verification: round-trip channeled images through the DINO SDF encode/decode pipeline
and compare with originals to identify pixel shifting errors.

Pipeline:
  Original channeled image (64x64 binary mask)
  -> _mask_to_sdf: crop bounding box, resize to 224x224, compute SDF
  -> (simulate perfect DINO decode: just use the SDF directly)
  -> rescale_dino_to_channeled: threshold, resize back, place in canvas
  -> Compare with original

Usage:
    python tests/verify_dino_roundtrip.py \
        --traj ./data/<cache_dir>/<hash>_0000.pt \
        --dino ./data/<cache_dir>/<hash>_0000_dino_B.pt
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from graph_dyno.precompute.precompute_dino_embeddings import _mask_to_sdf
from graph_dyno.plotting.plot import rescale_dino_to_channeled
from graph_dyno.data.polyhoop_simulation import generate_combined_image

parser = argparse.ArgumentParser()
parser.add_argument("--traj", required=True,
                    help="Path to trajectory cache file (e.g. ./data/cache_<hash>/<hash>_0000.pt)")
parser.add_argument("--dino", required=True,
                    help="Path to companion DINO embedding file")
parser.add_argument("--timestep", type=int, default=500)
args = parser.parse_args()

# Load cached data
print("Loading data...")
traj = torch.load(args.traj, map_location="cpu")
dino = torch.load(args.dino, map_location="cpu")

channeled_images = traj["channeled_images"]  # (2500, 400, 64, 64)
positions = traj["states"][:, :, :2]          # (2500, 400, 2)
crop_sides = dino["crop_sides"]               # (2500, 400) int32
r_max = traj["sim_metadata"]["r_max"].item()  # 3.6742

image_size = 64
domain_size = 100
timestep = 500  # pick a mid-simulation timestep

# Pick a few cells with different crop_sides to test
cs_values = crop_sides[timestep]  # (400,)
unique_cs = torch.unique(cs_values)
print(f"Timestep {timestep}: crop_side range [{cs_values.min()}, {cs_values.max()}], "
      f"{len(unique_cs)} unique values")

# Select cells: one small, one medium, one large, plus some odd-sized
cell_indices = []
for target_cs in [cs_values.min().item(), cs_values.median().item(), cs_values.max().item()]:
    idx = (cs_values - target_cs).abs().argmin().item()
    if idx not in cell_indices:
        cell_indices.append(idx)

# Also find odd-sized crop_sides specifically
odd_mask = cs_values % 2 == 1
if odd_mask.any():
    odd_indices = odd_mask.nonzero(as_tuple=True)[0]
    for idx in odd_indices[:3].tolist():
        if idx not in cell_indices:
            cell_indices.append(idx)

# Limit to 6 cells
cell_indices = cell_indices[:6]
print(f"Testing cells: {cell_indices}")
print(f"  crop_sides: {[cs_values[i].item() for i in cell_indices]}")

# --- Round-trip each cell ---
N = len(cell_indices)
originals = []
roundtrips_sdf = []
crop_sides_selected = []

for cell_idx in cell_indices:
    orig = channeled_images[timestep, cell_idx].numpy()  # (64, 64)
    cs = crop_sides[timestep, cell_idx].item()
    originals.append(orig)
    crop_sides_selected.append(cs)

    # Encode: channeled image -> SDF (same pipeline as precompute_dino_embeddings)
    result = _mask_to_sdf(orig, target_size=224)
    if result is not None:
        sdf, computed_cs = result
        sdf = torch.tensor(sdf.astype(np.float32))
    else:
        sdf = torch.full((224, 224), -1.0)
        computed_cs = 0

    roundtrips_sdf.append(sdf)

# Decode: SDF -> channeled image (same pipeline as plot.py)
sdf_stack = torch.stack(roundtrips_sdf)  # (N, 224, 224)
cs_tensor = torch.tensor(crop_sides_selected, dtype=torch.float32)
decoded = rescale_dino_to_channeled(sdf_stack, cs_tensor, image_size)  # (N, 64, 64)

# --- Plot individual cell comparisons ---
fig, axes = plt.subplots(4, N, figsize=(3 * N, 12), dpi=150)
if N == 1:
    axes = axes.reshape(4, 1)

for j in range(N):
    orig = originals[j]
    dec = decoded[j].numpy()
    cs = crop_sides_selected[j]
    diff = orig - dec

    axes[0, j].imshow(orig, cmap='gray', interpolation='nearest')
    axes[0, j].set_title(f'Original\ncell {cell_indices[j]}, cs={cs}')

    axes[1, j].imshow(dec, cmap='gray', interpolation='nearest')
    axes[1, j].set_title(f'Round-trip')

    # Difference: red=original only, blue=roundtrip only
    diff_rgb = np.zeros((*orig.shape, 3))
    diff_rgb[diff > 0.5, 0] = 1    # red: original has pixel, roundtrip doesn't
    diff_rgb[diff < -0.5, 2] = 1   # blue: roundtrip has pixel, original doesn't
    diff_rgb[(orig > 0.5) & (dec > 0.5), 1] = 0.5  # green: both have pixel
    axes[2, j].imshow(diff_rgb, interpolation='nearest')
    axes[2, j].set_title(f'Diff (red=orig, blue=rt)')

    # Overlay with transparency
    axes[3, j].imshow(orig, cmap='Reds', alpha=0.5, interpolation='nearest')
    axes[3, j].imshow(dec, cmap='Blues', alpha=0.5, interpolation='nearest')
    axes[3, j].set_title('Overlay (red=orig, blue=rt)')

    for row in range(4):
        axes[row, j].set_xticks([])
        axes[row, j].set_yticks([])

    # Print pixel shift stats
    orig_ys, orig_xs = np.where(orig > 0.5)
    dec_ys, dec_xs = np.where(dec > 0.5)
    if len(orig_ys) > 0 and len(dec_ys) > 0:
        orig_cy, orig_cx = orig_ys.mean(), orig_xs.mean()
        dec_cy, dec_cx = dec_ys.mean(), dec_xs.mean()
        print(f"  Cell {cell_indices[j]} (cs={cs}): "
              f"orig center=({orig_cy:.2f}, {orig_cx:.2f}), "
              f"rt center=({dec_cy:.2f}, {dec_cx:.2f}), "
              f"shift=({dec_cy - orig_cy:.2f}, {dec_cx - orig_cx:.2f})")

fig.suptitle(f'DINO Round-trip Verification (timestep={timestep})', fontsize=14)
plt.tight_layout()
plt.savefig('verify_dino_roundtrip_cells.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: verify_dino_roundtrip_cells.png")

# --- Combined image comparison ---
print("\nGenerating combined images...")
pos_t = positions[timestep]  # (400, 2)
ch_imgs_t = channeled_images[timestep]  # (400, 64, 64)

# Original combined image
combined_orig = generate_combined_image(pos_t, ch_imgs_t, domain_size, r_max)

# Round-trip combined image: encode all cells, decode, then combine
print("  Encoding all 400 cells through SDF pipeline...")
all_sdfs = []
all_cs = []
for i in range(400):
    mask = ch_imgs_t[i].numpy()
    result = _mask_to_sdf(mask, target_size=224)
    if result is not None:
        sdf, cs = result
        all_sdfs.append(torch.tensor(sdf.astype(np.float32)))
        all_cs.append(cs)
    else:
        all_sdfs.append(torch.full((224, 224), -1.0))
        all_cs.append(1)

all_sdfs = torch.stack(all_sdfs)
all_cs = torch.tensor(all_cs, dtype=torch.float32)

print("  Decoding through rescale_dino_to_channeled...")
decoded_all = rescale_dino_to_channeled(all_sdfs, all_cs, image_size)

combined_rt = generate_combined_image(pos_t, decoded_all, domain_size, r_max)

# Plot combined comparison
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

axes2[0].imshow(combined_orig.numpy(), cmap='gray', origin='lower',
                extent=[0, domain_size, 0, domain_size])
axes2[0].set_title('Original Combined')

axes2[1].imshow(combined_rt.numpy(), cmap='gray', origin='lower',
                extent=[0, domain_size, 0, domain_size])
axes2[1].set_title('Round-trip Combined')

diff_combined = combined_orig.numpy() - combined_rt.numpy()
im = axes2[2].imshow(diff_combined, cmap='RdBu', origin='lower',
                     extent=[0, domain_size, 0, domain_size],
                     vmin=-1, vmax=1)
axes2[2].set_title('Difference (orig - roundtrip)')
plt.colorbar(im, ax=axes2[2])

fig2.suptitle(f'Combined Image Comparison (timestep={timestep})', fontsize=14)
plt.tight_layout()
plt.savefig('verify_dino_roundtrip_combined.png', dpi=150, bbox_inches='tight')
print(f"Saved: verify_dino_roundtrip_combined.png")

# --- Stats summary ---
print("\n=== Summary Statistics ===")
print(f"Crop sides at timestep {timestep}:")
print(f"  min={cs_values.min().item()}, max={cs_values.max().item()}, "
      f"mean={cs_values.float().mean().item():.1f}")
print(f"  odd count: {(cs_values % 2 == 1).sum().item()}/{len(cs_values)}")
print(f"  even count: {(cs_values % 2 == 0).sum().item()}/{len(cs_values)}")

# Per-cell pixel difference stats
total_diff_pixels = 0
total_orig_pixels = 0
for i in range(400):
    orig_mask = ch_imgs_t[i].numpy()
    rt_mask = decoded_all[i].numpy()
    total_diff_pixels += np.abs(orig_mask - rt_mask).sum()
    total_orig_pixels += orig_mask.sum()
print(f"\nPixel difference across all 400 cells:")
print(f"  Total changed pixels: {total_diff_pixels:.0f}")
print(f"  Total original pixels: {total_orig_pixels:.0f}")
print(f"  Error rate: {total_diff_pixels / total_orig_pixels * 100:.2f}%")

# Check center-of-mass shifts
shifts_y = []
shifts_x = []
for i in range(400):
    orig_mask = ch_imgs_t[i].numpy()
    rt_mask = decoded_all[i].numpy()
    oy, ox = np.where(orig_mask > 0.5)
    ry, rx = np.where(rt_mask > 0.5)
    if len(oy) > 0 and len(ry) > 0:
        shifts_y.append(ry.mean() - oy.mean())
        shifts_x.append(rx.mean() - ox.mean())

shifts_y = np.array(shifts_y)
shifts_x = np.array(shifts_x)
print(f"\nCenter-of-mass shifts (roundtrip - original):")
print(f"  Y: mean={shifts_y.mean():.4f}, std={shifts_y.std():.4f}, "
      f"min={shifts_y.min():.4f}, max={shifts_y.max():.4f}")
print(f"  X: mean={shifts_x.mean():.4f}, std={shifts_x.std():.4f}, "
      f"min={shifts_x.min():.4f}, max={shifts_x.max():.4f}")

plt.show()