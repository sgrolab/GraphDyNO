"""Verify that pretrained DINO decoder reconstructions match paired SDFs.

Loads embeddings and SDFs for a given dataset and checkpoint, decodes a sample
of embeddings, and plots them side-by-side with their SDF targets.

Usage:
    python tests/verify_decoder_reconstruction.py \
        --checkpoint ./checkpoints/F_256.pt \
        --embedding-file ./data/oscillating_cell_clusters/dino_F_<hash>_0000.pt \
        --sdf-file ./data/oscillating_cell_clusters/sdf_<hash>_0000.npy
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from graph_dyno.decoders.dino import DINODecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to DINO decoder checkpoint")
    parser.add_argument("--embedding-file", required=True, help="Path to dino_*_*.pt embedding file")
    parser.add_argument("--sdf-file", required=True, help="Path to sdf_*.npy file")
    parser.add_argument("--n-samples", type=int, default=8, help="Number of samples to plot")
    parser.add_argument("--timestep", type=int, default=0, help="Timestep to sample from")
    parser.add_argument("--output", type=str, default="verify_reconstruction.png", help="Output plot path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load decoder
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    decoder = DINODecoder(
        input_dim=ckpt["input_dim"],
        bottleneck_dim=ckpt["bottleneck_dim"],
        channels=tuple(ckpt.get("channels", (256, 128, 64, 32, 16))),
        intermediate_conv_layers=ckpt.get("intermediate_conv_layers", 0),
    )
    decoder.load_state_dict(ckpt["decoder"])
    decoder.to(device).eval()
    print(f"Decoder: model={ckpt.get('model')}, input_dim={ckpt['input_dim']}, "
          f"bottleneck_dim={ckpt['bottleneck_dim']}")

    # Load embeddings
    emb_data = torch.load(args.embedding_file, map_location="cpu", weights_only=True)
    if isinstance(emb_data, dict):
        emb = emb_data["embeddings"]  # (T, N, D)
    else:
        emb = emb_data
    print(f"Embeddings: shape={emb.shape}, dtype={emb.dtype}")

    # Load SDFs (memmap to avoid loading entire file)
    sdf = np.load(args.sdf_file, mmap_mode="r")  # (T, N, S, S)
    print(f"SDFs: shape={sdf.shape}, dtype={sdf.dtype}")
    print(f"SDF range at t={args.timestep}: [{sdf[args.timestep].min():.3f}, {sdf[args.timestep].max():.3f}]")

    # Sample from the given timestep
    t = args.timestep
    n = min(args.n_samples, emb.shape[1])
    sample_emb = emb[t, :n].float().to(device)  # (n, D)
    sample_sdf = torch.from_numpy(np.array(sdf[t, :n])).float()  # (n, S, S)

    # Decode
    with torch.no_grad():
        preds = decoder(sample_emb)  # (n, S, S)
    preds_cpu = preds.cpu()

    # Compute per-sample IoU
    pred_bin = (preds_cpu > 0).float()
    gt_bin = (sample_sdf > 0).float()
    intersection = (pred_bin * gt_bin).sum(dim=(1, 2))
    union = pred_bin.sum(dim=(1, 2)) + gt_bin.sum(dim=(1, 2)) - intersection
    ious = intersection / (union + 1e-6)
    mses = ((preds_cpu - sample_sdf) ** 2).mean(dim=(1, 2))

    print(f"\nPer-sample metrics (timestep={t}):")
    for i in range(n):
        print(f"  Cell {i}: IoU={ious[i]:.4f}, MSE={mses[i]:.6f}")
    print(f"Mean IoU={ious.mean():.4f}, Mean MSE={mses.mean():.6f}")

    # Plot
    fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 7.5))
    if n == 1:
        axes = axes.reshape(3, 1)
    for i in range(n):
        axes[0, i].imshow(sample_sdf[i], cmap="RdBu", vmin=-1, vmax=1)
        axes[0, i].set_title(f"Target {i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(preds_cpu[i], cmap="RdBu", vmin=-1, vmax=1)
        axes[1, i].set_title(f"Pred (IoU={ious[i]:.3f})")
        axes[1, i].axis("off")

        diff = (preds_cpu[i] - sample_sdf[i]).abs()
        axes[2, i].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
        axes[2, i].set_title(f"Diff (MSE={mses[i]:.4f})")
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Target SDF", fontsize=12)
    axes[1, 0].set_ylabel("Prediction", fontsize=12)
    axes[2, 0].set_ylabel("|Diff|", fontsize=12)

    fig.suptitle(f"DINO Decoder Roundtrip (model={ckpt.get('model')}, t={t}, mean IoU={ious.mean():.4f})")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
