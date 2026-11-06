"""
Visualization utilities for DINO autoencoder training.
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Synthetic shape embeddings (precomputed once for PCA plots)
# ============================================================================

def _make_shape_sdfs(image_size=224):
    """Generate SDF images for circles and ellipses that fill the frame.

    Returns list of (H, W) float32 SDF arrays in [-1, 1], plus labels and params.
    """
    sdfs, labels, params = [], [], []
    half = image_size // 2

    def _binary_to_sdf(img):
        mask = np.array(img).astype(bool)
        dist_in = distance_transform_edt(mask)
        dist_out = distance_transform_edt(~mask)
        sdf = dist_in - dist_out
        return np.clip(sdf / (image_size / 2), -1, 1).astype(np.float32)

    # Circles: varying radius from 20 to half
    for r in np.linspace(20, half - 2, 30).astype(int):
        img = Image.new("L", (image_size, image_size), 0)
        ImageDraw.Draw(img).ellipse(
            [half - r, half + r, half + r, half + r], fill=255)
        sdfs.append(_binary_to_sdf(img))
        labels.append("circle")
        params.append({"radius": int(r), "aspect_ratio": 1.0})

    # Ellipses: varying aspect ratio, scaled to fill the frame
    for ar in [0.3, 0.5, 0.7, 1.4, 2.0, 3.0]:
        rx = int(min(half - 2, (half - 2) * min(ar, 1.0) + (half - 2) * max(0, 1 - 1/ar)))
        ry = int(rx / ar)
        scale = (half - 2) / max(rx, ry)
        rx, ry = int(rx * scale), int(ry * scale)
        img = Image.new("L", (image_size, image_size), 0)
        ImageDraw.Draw(img).ellipse(
            [half - rx, half - ry, half + rx, half + ry], fill=255)
        sdfs.append(_binary_to_sdf(img))
        labels.append("ellipse")
        params.append({"radius": max(rx, ry), "aspect_ratio": ar})

    return sdfs, labels, params


@torch.no_grad()
def precompute_shape_embeddings(feature_extractor, use_patches, device):
    """Run DINO on synthetic shape SDFs, return embeddings + metadata."""
    sdfs, labels, params = _make_shape_sdfs()
    input_spec = feature_extractor.input_spec
    embs = []
    for sdf in sdfs:
        t = input_spec(torch.from_numpy(sdf))  # (C, H, W) in [0, 1]
        cls_tok, patch_tok = feature_extractor(t.unsqueeze(0).to(device))
        if use_patches:
            emb = torch.cat([cls_tok, patch_tok.mean(dim=1)], dim=1)
        else:
            emb = cls_tok
        embs.append(emb.cpu())
    return torch.cat(embs), labels, params


# ============================================================================
# Per-epoch visualisation helpers
# ============================================================================

def plot_predictions(decoder, val_embs, val_masks, epoch, plot_dir, device,
                     num_samples=8):
    """Save a grid of GT vs predicted masks."""
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(val_masks), size=num_samples, replace=False)

    _, axes = plt.subplots(num_samples, 2, figsize=(6, 2.5 * num_samples))
    decoder.eval()

    with torch.no_grad():
        for row, idx in enumerate(idxs):
            emb = val_embs[idx].unsqueeze(0).to(device)
            pred_sdf = decoder(emb).squeeze().cpu().numpy()
            gt_sdf = val_masks[idx].numpy().astype(float)

            pred_bin = (pred_sdf > 0).astype(float)
            gt_bin = (gt_sdf > 0).astype(float)
            inter = (pred_bin * gt_bin).sum()
            union = pred_bin.sum() + gt_bin.sum() - inter
            iou = inter / (union + 1e-6)

            axes[row, 0].imshow(gt_sdf, cmap="RdBu", vmin=-1, vmax=1)
            axes[row, 0].set_title("GT SDF")
            axes[row, 0].axis("off")
            axes[row, 1].imshow(pred_sdf, cmap="RdBu", vmin=-1, vmax=1)
            axes[row, 1].set_title(f"Pred SDF  IoU={iou:.3f}")
            axes[row, 1].axis("off")

    plt.suptitle(f"Epoch {epoch}", fontsize=14)
    plt.tight_layout()
    plt.savefig(plot_dir / f"predictions_epoch{epoch:03d}.png", dpi=120)
    plt.close()


def plot_shape_pca(decoder, shape_embs, shape_labels, shape_params, epoch,
                   plot_dir, device):
    """PCA of bottleneck embeddings for synthetic circles / ellipses."""
    decoder.eval()
    with torch.no_grad():
        bottlenecks = decoder.get_bottleneck(
            shape_embs.to(device)).cpu().numpy()

    radii = np.array([p["radius"] for p in shape_params])
    ars = np.array([p["aspect_ratio"] for p in shape_params])
    is_circle = np.array([l == "circle" for l in shape_labels])

    pca = PCA(n_components=2)
    z = pca.fit_transform(bottlenecks)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # left: colored by radius
    sc = ax1.scatter(z[is_circle, 0], z[is_circle, 1],
                     c=radii[is_circle], cmap="viridis", s=60, label="circle")
    ax1.scatter(z[~is_circle, 0], z[~is_circle, 1],
                c="red", marker="x", s=100, linewidths=2, label="ellipse")
    for i in np.where(~is_circle)[0]:
        ax1.annotate(f'{ars[i]:.1f}', (z[i, 0], z[i, 1]), fontsize=8)
    plt.colorbar(sc, ax=ax1, label="radius")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_title("Bottleneck PCA — colored by radius")
    ax1.legend(fontsize=8)

    # right: decoded masks for a few shapes
    pick = np.linspace(0, len(shape_embs) - 1, 8).astype(int)
    with torch.no_grad():
        preds = decoder(shape_embs[pick].to(device)).cpu().numpy()
    for i, idx in enumerate(pick):
        ax2_sub = ax2.inset_axes([i / 8, 0, 1 / 8, 1])
        ax2_sub.imshow(preds[i], cmap="RdBu", vmin=-1, vmax=1)
        ar = ars[idx]
        ax2_sub.set_title(f"ar={ar:.1f}" if ar != 1.0 else f"r={radii[idx]}",
                          fontsize=7)
        ax2_sub.axis("off")
    ax2.axis("off")
    ax2.set_title("Decoded shapes")

    plt.suptitle(f"Epoch {epoch}", fontsize=14)
    plt.tight_layout()
    plt.savefig(plot_dir / f"shape_pca_epoch{epoch:03d}.png", dpi=120)
    plt.close()


def plot_history(history, plot_dir):
    """Plot training/validation loss and IoU curves."""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()
    ax1.set_title("Loss")
    ax2.plot(history["val_iou"], label="Val IoU", color="green")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("IoU"); ax2.legend()
    ax2.set_title("Validation IoU")
    plt.tight_layout()
    plt.savefig(plot_dir / "training_history.png", dpi=150)
    plt.close()
