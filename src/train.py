import os
from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset import PCamDataset, download_pcam
from src.models.vit_bp import ViTBP
from src.models.vit_fa import ViTFA


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute top-1 accuracy for a mini-batch."""
    preds = output.argmax(dim=1)
    correct = (preds == target).sum().item()
    return correct / target.size(0)


# -----------------------------------------------------------------------------
# Main training routine
# -----------------------------------------------------------------------------

def train(
    data_root: str = "data/pcam",
    download_dataset: bool = False,
    *,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 3e-4,
    num_workers: int = 4,
    checkpoint_dir: str = "checkpoints",
    device: Optional[str] = None,
    limit_samples: Optional[int] = None,
    visualize: bool = False,
    log_dir: str = "logs",
    model_type: str = "fa",
) -> None:
    """Train ViT with Feedback Alignment (ViTFA) on the PCam dataset.

    Parameters
    ----------
    data_root : str, default "data/pcam"
        Root directory containing the PCam dataset split folders.
    download_dataset : bool, default False
        If True, automatically download PCam via Kaggle CLI into ``data_root``.
    epochs : int, default 10
        Number of training epochs.
    batch_size : int, default 128
        Mini-batch size.
    lr : float, default 3e-4
        Learning rate for Adam optimizer.
    num_workers : int, default 4
        Number of worker processes for ``torch.utils.data.DataLoader``.
    checkpoint_dir : str, default "checkpoints"
        Root directory under which model-specific subfolders will be created.
    model_type : {"fa", "bp"}, default "fa"
        Which model variant to train.
    log_dir : str, default "logs"
        Directory where log files will be stored under a subfolder for each model.
    device : str or None, default None
        Device on which to run the training ("cuda"/"cpu"). If None, chooses
        CUDA if available, else CPU.
    limit_samples : int or None, default None
        If set, use only the first ``limit_samples`` elements of the train and
        validation datasets. Useful for quick debugging (e.g., ``limit_samples=1000``).

    visualize : bool, default False
        If True, plot training/validation loss and accuracy curves using matplotlib.
    """

    # ---------------------------------------------------------------------
    # Setup & configuration
    # ---------------------------------------------------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if download_dataset:
        print("Downloading PCam dataset…")
        download_pcam(data_root)

    # Data augmentations and preprocessing
    train_tfms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]
    )
    val_tfms = transforms.Compose([transforms.ToTensor()])

    # ---------------------------------------------------------------------
    # Dataset & DataLoader objects
    # ---------------------------------------------------------------------
    train_ds_full = PCamDataset(data_root, split="train", transform=train_tfms)
    val_ds_full = PCamDataset(data_root, split="val", transform=val_tfms)

    # Optionally reduce dataset size for faster experiments
    if limit_samples is not None:
        from torch.utils.data import Subset

        train_len = min(limit_samples, len(train_ds_full))
        val_len = min(limit_samples, len(val_ds_full))

        train_ds = Subset(train_ds_full, list(range(train_len)))  # type: ignore[arg-type]
        val_ds = Subset(val_ds_full, list(range(val_len)))        # type: ignore[arg-type]
    else:
        train_ds = train_ds_full
        val_ds = val_ds_full

    train_loader: DataLoader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader: DataLoader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ---------------------------------------------------------------------
    # Model, loss, optimizer, scheduler
    # ---------------------------------------------------------------------
    model_type_lc = model_type.lower()
    if model_type_lc in {"fa", "vitfa"}:
        model = ViTFA(num_classes=2)
    elif model_type_lc in {"bp", "vitbp", "backprop"}:
        model = ViTBP(num_classes=2)
    else:
        raise ValueError("model_type must be 'fa' or 'bp'")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------------------------
    # Training / validation loop
    # ---------------------------------------------------------------------
    model_ckpt_dir = Path(checkpoint_dir) / model_type_lc
    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------
    from pathlib import Path as _Path
    import logging as _logging

    log_path_dir = _Path(log_dir) / model_type_lc
    log_path_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_path_dir / "train.log"

    logger = _logging.getLogger(f"train_{model_type_lc}")
    logger.setLevel(_logging.INFO)
    # Avoid duplicate handlers if train() called multiple times
    if not logger.handlers:
        _fh = _logging.FileHandler(log_path, mode="w")
        _fh.setFormatter(_logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(_fh)

    os.makedirs(model_ckpt_dir, exist_ok=True)
    best_acc = 0.0

    # Lists for visualization
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []

    for epoch in range(1, epochs + 1):
        # --------------------------- TRAINING ---------------------------
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_acc += accuracy(logits, labels) * imgs.size(0)

        train_loss /= len(train_loader.dataset)  # type: ignore[arg-type]
        train_acc /= len(train_loader.dataset)   # type: ignore[arg-type]

        # Record metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ------------------------- VALIDATION ---------------------------
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                val_acc += accuracy(logits, labels) * imgs.size(0)

        val_loss /= len(val_loader.dataset)  # type: ignore[arg-type]
        val_acc /= len(val_loader.dataset)   # type: ignore[arg-type]
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        scheduler.step()

        # Log only validation metrics each epoch (skip per-epoch train metrics)
        logger.info(
            "Epoch %d/%d - Val Loss %.4f - Val Acc %.4f",
            epoch,
            epochs,
            val_loss,
            val_acc,
        )

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = model_ckpt_dir / "best_model.pth"
            torch.save(model.state_dict(), ckpt_path)
            logger.info("New best model saved with Val Acc %.4f -> %s", best_acc, ckpt_path)

    # Final training summary
    logger.info("Training complete. Best Val Acc %.4f", best_acc)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    if visualize:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            epochs_range = range(1, epochs + 1)

            plt.figure(figsize=(10, 4))
            # Loss subplot
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, train_losses, label="Train Loss")
            plt.plot(epochs_range, val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Curves")
            plt.legend()

            # Accuracy subplot
            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, train_accs, label="Train Acc")
            plt.plot(epochs_range, val_accs, label="Val Acc")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Accuracy Curves")
            plt.legend()

            plt.tight_layout()
            fig_path = model_ckpt_dir / "training_curves.png"
            plt.savefig(fig_path)
            plt.show()
            logger.info("Training curves saved to %s", fig_path)
        except ImportError:
            print("matplotlib not installed; skipping visualization.")


# -----------------------------------------------------------------------------
# Stand-alone execution with default parameters
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick sanity-check run on 10,000 samples with visualization.
    train(limit_samples=10000, visualize=True, model_type="bp")
    # train(limit_samples=10000, visualize=True, model_type="fa") 