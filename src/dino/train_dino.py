"""
Train the DINO bottleneck decoder.

Embeddings are computed automatically on first run and cached to disk.
Subsequent runs with the same model/data config reuse the cached embeddings.

Usage:
    python -m dino.train_dino
    python -m dino.train_dino dino.model.bottleneck_dim=128 dino.training.lr=5e-4
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import mlflow
import os

from dino.decoder import DINODecoder
from dino.feature_extractor import create_feature_extractor
from dino.dataloader import load_all_masks, CachedEmbeddingDataset, CUDAPrefetcher
from dino.precompute_embeddings import load_or_compute_split
from dino.plot_dino import (
    plot_predictions, plot_shape_pca, plot_history, precompute_shape_embeddings,
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


@hydra.main(version_base=None, config_path="../../configs", config_name="dino_config")
def main(cfg: DictConfig) -> None:
    model_cfg = cfg.dino.model
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
    mlflow.set_tracking_uri(os.environ.get(
        "MLFLOW_URI",
        f"file://{(Path(__file__).resolve().parents[2] / 'mlruns').resolve()}",
    ))
    mlflow.set_experiment(cfg.experiment_name)
    mlflow.start_run()

    flat_cfg = OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_params(_flatten_dict(flat_cfg))
    mlflow.set_tag("hydra_output_dir", str(save_path))
    mlflow.log_artifact(save_path / ".hydra/config.yaml")

    # --- Load feature extractor (used for embeddings + shape PCA) ---
    data_dir = Path(data_cfg.data_dir)
    model_key = model_cfg.dino_model
    backend = model_cfg.get("dino_backend", "hf")
    # Safe to key the cache on model_key alone: preprocessing is frozen per key
    # in dino/feature_extractor.py, so the same key always means the same pixels.
    embedding_dir = data_dir / f"features_{model_key}"

    print(f"Loading feature extractor (backend={backend}, model={model_key})...")
    fe = create_feature_extractor(
        model_key,
        backend=backend,
        repo_dir=model_cfg.get("dino_repo_dir", None),
        weights_path=model_cfg.get("dino_weights_path", None),
        weights_url=model_cfg.get("dino_weights_url", None),
        trust_weights=model_cfg.get("dino_trust_weights", False),
    ).to(device)

    # --- Load masks ---
    print("Loading masks...")
    train_masks = load_all_masks(data_dir / "train")
    val_masks = load_all_masks(data_dir / "val")

    # --- Load or compute embeddings ---
    train_cls, train_pm = load_or_compute_split(
        "train", train_masks, fe, embedding_dir,
        batch_size=training_cfg.batch_size, device=device,
        num_workers=training_cfg.num_workers)
    val_cls, val_pm = load_or_compute_split(
        "val", val_masks, fe, embedding_dir,
        batch_size=training_cfg.batch_size, device=device,
        num_workers=training_cfg.num_workers)

    feature_dim = train_cls.shape[-1]
    if model_cfg.use_patches:
        input_dim = feature_dim * 2
        print(f"  Using CLS + mean patches  (input_dim={input_dim})")
    else:
        train_pm, val_pm = None, None
        input_dim = feature_dim
        print(f"  Using CLS only  (input_dim={input_dim})")
    print(f"  train: {train_cls.shape[0]}   val: {val_cls.shape[0]}")

    # Combined val embeddings for visualization
    if val_pm is not None:
        val_embs = torch.cat([val_cls, val_pm], dim=1)
    else:
        val_embs = val_cls

    # --- Precompute synthetic shape embeddings for PCA ---
    print("Precomputing synthetic shape embeddings...")
    shape_embs, shape_labels, shape_params = precompute_shape_embeddings(
        fe, model_cfg.use_patches, device)
    del fe
    torch.cuda.empty_cache()

    # --- Datasets & loaders ---
    train_ds = CachedEmbeddingDataset(train_cls, train_masks, train_pm)
    val_ds = CachedEmbeddingDataset(val_cls, val_masks, val_pm)

    train_loader = DataLoader(train_ds, batch_size=training_cfg.batch_size,
                              shuffle=True, num_workers=training_cfg.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=training_cfg.batch_size,
                            shuffle=False, num_workers=training_cfg.num_workers,
                            pin_memory=True)

    # --- Decoder ---
    channels = tuple(model_cfg.get("channels", [256, 128, 64, 32, 16]))
    intermediate_conv_layers = model_cfg.get("intermediate_conv_layers", 0)
    proj_hidden_layers = model_cfg.get("proj_hidden_layers", 0)

    decoder: DINODecoder = DINODecoder(
        input_dim=input_dim,
        bottleneck_dim=model_cfg.bottleneck_dim,
        output_size=model_cfg.output_size,
        channels=channels,
        intermediate_conv_layers=intermediate_conv_layers,
        proj_hidden_layers=proj_hidden_layers,
    ).to(device)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=training_cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=training_cfg.scheduler.step_size,
        gamma=training_cfg.scheduler.gamma,
    )
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_iou": []}
    best_val_iou = 0.0
    patience = training_cfg.get("early_stopping_patience", 15)
    epochs_without_improvement = 0

    print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
    print("Starting training...")

    for epoch in range(training_cfg.n_epochs):
        # ---- Train ----
        decoder.train()
        train_loss = 0.0

        for emb, targets in tqdm(CUDAPrefetcher(train_loader, device),
                                 desc=f"Epoch {epoch+1}/{training_cfg.n_epochs}",
                                 total=len(train_loader)):
            loss_dict = decoder.training_step(emb, targets)
            # add a regularization term on the embedding to encourage smoothness
            loss = sum([v*training_cfg.loss_terms[k] for k,v in loss_dict.items() if k in training_cfg.loss_terms])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # ---- Validate ----
        decoder.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for emb, targets in CUDAPrefetcher(val_loader, device):
                pred = decoder(emb)

                loss_dict = decoder.training_step(emb, targets)
                loss = sum(v * training_cfg.loss_terms[k] for k, v in loss_dict.items() if k in training_cfg.loss_terms)
                val_loss += loss.item()

                pred_bin = (pred > 0).float()
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
        ckpt = {
            "epoch": epoch + 1,
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "bottleneck_dim": model_cfg.bottleneck_dim,
            "input_dim": input_dim,
            "use_patches": model_cfg.use_patches,
            "output_size": model_cfg.output_size,
            "model": model_key,
            "dino_backend": backend,
            "channels": list(channels),
            "intermediate_conv_layers": intermediate_conv_layers,
            "proj_hidden_layers": proj_hidden_layers,
            "loss_terms": OmegaConf.to_container(training_cfg.loss_terms, resolve=True),
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
            plot_predictions(decoder, val_embs, val_masks, epoch + 1,
                             plot_dir, device)
            plot_shape_pca(decoder, shape_embs, shape_labels, shape_params,
                           epoch + 1, plot_dir, device)
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
