"""
Train a convolutional autoencoder on LIVECell SDF masks.

Usage:
    python train_autoencoder.py
    python train_autoencoder.py shape.bottleneck_dim=128 dino.training.lr=5e-4
    python train_autoencoder.py --multirun shape.bottleneck_dim=64,128,256
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import mlflow

from train import _resolve_mlflow_uri
from dino.dataloader import load_all_masks, SDFDataset, CUDAPrefetcher
from graph_dyno.plotting.plot_autoencoder import (
    plot_predictions, plot_shape_pca, plot_history, precompute_shape_sdfs,
)


def _flatten_dict(d, parent_key='', sep='.'):
    """Flatten nested dict for MLflow params logging."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def create_model(model_cfg):
    """Create encoder/decoder pair from config.

    Args:
        model_cfg: Hydra config with shape section

    Returns:
        (encoder, decoder) tuple of nn.Module
    """
    model_type = model_cfg.get("type", "conv")
    channels = tuple(model_cfg.get("channels", [256, 128, 64, 32, 16]))
    icl = model_cfg.get("intermediate_conv_layers", 0)

    if model_type == "conv":
        from graph_dyno.encoders.conv import ConvEncoder
        from graph_dyno.decoders.conv import ConvDecoder

        encoder = ConvEncoder(
            bottleneck_dim=model_cfg.bottleneck_dim,
            output_size=model_cfg.output_size,
            channels=channels,
            intermediate_conv_layers=icl,
        )
        decoder = ConvDecoder(
            bottleneck_dim=model_cfg.bottleneck_dim,
            output_size=model_cfg.output_size,
            channels=channels,
            intermediate_conv_layers=icl,
        )
        return encoder, decoder
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@hydra.main(version_base=None, config_path="configs", config_name="shape_train_config")
def main(cfg: DictConfig) -> None:
    model_cfg = cfg.shape
    data_cfg = cfg.dino.data
    training_cfg = cfg.dino.training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hydra output directory
    hydra_cfg = HydraConfig.get()
    save_path = Path(hydra_cfg.runtime.output_dir)
    plot_dir = save_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # MLflow setup
    mlflow.set_tracking_uri(_resolve_mlflow_uri(
        cfg.get("mlflow_uri", os.environ.get("MLFLOW_URI", "./mlruns"))))
    mlflow.set_experiment(cfg.experiment_name)
    mlflow.start_run()

    flat_cfg = OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_params(_flatten_dict(flat_cfg))
    mlflow.set_tag("hydra_output_dir", str(save_path))
    mlflow.log_artifact(save_path / ".hydra/config.yaml")

    # --- Load masks ---
    data_dir = Path(data_cfg.data_dir)
    print("Loading masks...")
    train_masks = load_all_masks(data_dir / "train")
    val_masks = load_all_masks(data_dir / "val")
    print(f"  train: {train_masks.shape[0]}   val: {val_masks.shape[0]}")

    # --- Precompute synthetic shape SDFs for PCA ---
    print("Precomputing synthetic shape SDFs...")
    shape_sdfs, shape_labels, shape_params = precompute_shape_sdfs()

    # --- Datasets & loaders ---
    train_ds = SDFDataset(train_masks)
    val_ds = SDFDataset(val_masks)

    train_loader = DataLoader(train_ds, batch_size=training_cfg.batch_size,
                              shuffle=True, num_workers=training_cfg.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=training_cfg.batch_size,
                            shuffle=False, num_workers=training_cfg.num_workers,
                            pin_memory=True)

    # --- Model ---
    encoder, decoder = create_model(model_cfg)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    all_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=training_cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=training_cfg.scheduler.step_size,
        gamma=training_cfg.scheduler.gamma,
    )

    loss_weights = OmegaConf.to_container(training_cfg.loss_terms, resolve=True)

    history = {"train_loss": [], "val_loss": [], "val_iou": []}
    best_val_iou = 0.0
    patience = training_cfg.get("early_stopping_patience", 15)
    epochs_without_improvement = 0

    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"Encoder params: {enc_params:,}  Decoder params: {dec_params:,}  "
          f"Total: {enc_params + dec_params:,}")
    print("Starting training...")

    for epoch in range(training_cfg.n_epochs):
        # ---- Train ----
        encoder.train()
        decoder.train()
        train_loss = 0.0

        for images, targets in tqdm(CUDAPrefetcher(train_loader, device),
                                    desc=f"Epoch {epoch+1}/{training_cfg.n_epochs}",
                                    total=len(train_loader)):
            bottleneck = encoder(images)
            preds = decoder(bottleneck)

            loss_dict = {
                'image_mse': F.mse_loss(preds, targets),
                'embedding_l2': bottleneck.pow(2).mean(),
            }
            loss = sum(v * loss_weights[k] for k, v in loss_dict.items()
                       if k in loss_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # ---- Validate ----
        encoder.eval()
        decoder.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, targets in CUDAPrefetcher(val_loader, device):
                bottleneck = encoder(images)
                preds = decoder(bottleneck)

                loss_dict = {
                    'image_mse': F.mse_loss(preds, targets),
                    'embedding_l2': bottleneck.pow(2).mean(),
                }
                loss = sum(v * loss_weights[k] for k, v in loss_dict.items()
                           if k in loss_weights)
                val_loss += loss.item()

                pred_bin = (preds > 0).float()
                gt_bin = (targets > 0).float()
                intersection = (pred_bin * gt_bin).sum(dim=(1, 2))
                union = pred_bin.sum(dim=(1, 2)) + gt_bin.sum(dim=(1, 2)) - intersection
                iou = (intersection / (union + 1e-6)).mean()
                val_iou += iou.item()

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        scheduler.step()
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}  "
              f"Val Loss={val_loss:.4f}  Val IoU={val_iou:.4f}")

        # Log to MLflow
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou,
        }, step=epoch)

        # --- Checkpoint ---
        channels = tuple(model_cfg.get("channels", [256, 128, 64, 32, 16]))
        ckpt = {
            "epoch": epoch + 1,
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "type": model_cfg.get("type", "conv"),
            "bottleneck_dim": model_cfg.bottleneck_dim,
            "output_size": model_cfg.output_size,
            "channels": list(channels),
            "intermediate_conv_layers": model_cfg.get("intermediate_conv_layers", 0),
            "loss_terms": loss_weights,
        }
        torch.save(ckpt, save_path / "latest.pt")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_without_improvement = 0
            torch.save(ckpt, save_path / "best.pt")
            print(f"  New best IoU: {best_val_iou:.4f}")
        else:
            epochs_without_improvement += 1

        # --- Periodic plots ---
        if (epoch + 1) % training_cfg.plot_every == 0 or epoch == 0:
            # Use float32 version of val masks for plotting
            val_sdfs_f32 = val_masks.float()
            plot_predictions(encoder, decoder, val_sdfs_f32, epoch + 1,
                             plot_dir, device)
            plot_shape_pca(encoder, decoder, shape_sdfs, shape_labels,
                           shape_params, epoch + 1, plot_dir, device)
            plot_history(history, plot_dir)
            print(f"  Saved plots to {plot_dir}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break

    # --- Final ---
    print(f"\nTraining complete! Best IoU: {best_val_iou:.4f}")
    print(f"Outputs saved to {save_path}")

    mlflow.log_artifact(str(save_path / "best.pt"))
    mlflow.log_artifact(str(save_path / "latest.pt"))
    mlflow.end_run()


if __name__ == "__main__":
    main()
