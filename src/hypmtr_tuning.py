"""Hyper-parameter tuning script for ViTFA.

This script performs an exhaustive grid search over the user-specified
hyper-parameters and trains a separate model for every valid
combination. The best validation accuracy achieved within the training
run is recorded along with the checkpoint and log files.

Search space (can be modified below):
    dropouts   : [0.05, 0.10]
    patch_size : [12, 16]
    depth      : [4, 6, 8]
    mlp_ratio  : [3, 4]
    embed_dim  : [64, 128, 256]
    num_heads  : [4, 8]

Outputs
-------
• tuned_checkpoints/<cfg_name>/best_model.pth — best checkpoint per run
• tuned_logs/<cfg_name>/train.log              — log file per run
• tuned_results.csv                            — CSV summary of all runs

Run with:
    $ python -m src.tune_vit_fa

Tips
----
• Stop the script at any time — completed configurations are detected
  via the CSV and skipped on the next launch.
• Adapt NUM_EPOCHS or BATCH_SIZE for longer or faster sweeps.
"""

from __future__ import annotations

import csv
import itertools
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset import PCamDataset, download_pcam
from src.models.vit_fa import ViTFA
# Remove argparse; switching to function parameter interface
from src.models.vit_bp import ViTBP

# Map string identifiers to model classes for easy selection
MODEL_MAP = {
    "fa": ViTFA,  # Feedback Alignment
    "bp": ViTBP,  # Standard Back-propagation
}

# -----------------------------------------------------------------------------
# Search-space definition
# -----------------------------------------------------------------------------
PARAM_GRID = {
    "p": [0.05, 0.10],  # dropout rates
    "patch_size": [12, 16],
    "depth": [4, 6, 8],
    "mlp_ratio": [3, 4],
    "embed_dim": [64, 128, 256],
    "num_heads": [4, 8],
}

# -----------------------------------------------------------------------------
# Training constants (tweak for your hardware)
# -----------------------------------------------------------------------------
NUM_EPOCHS = 8  # keep modest for sweeping; increase once narrowed down
BATCH_SIZE = 128
LR = 3e-4
NUM_WORKERS = 4
DATA_ROOT = "data/pcam"
DOWNLOAD_DATASET = False  # set True for first-time download
LIMIT_SAMPLES: int | None = 10000  # e.g. 1000 for quick debugging
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output directories
CKPT_ROOT = Path("tuned_checkpoint")
CKPT_ROOT.mkdir(exist_ok=True, parents=True)

RESULTS_PATH = Path("tuned_results.csv")
CSV_HEADER = [
    "cfg_name",
    "p",
    "patch_size",
    "depth",
    "mlp_ratio",
    "embed_dim",
    "num_heads",
    "best_val_acc",
]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def iter_param_grid(grid: Dict[str, Iterable]) -> Iterable[Dict]:
    """Yield dicts for the Cartesian product of a param grid."""
    keys, values = zip(*grid.items())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))  # type: ignore[arg-type]


def is_valid_combo(hp: Dict) -> bool:
    """Check divisibility constraint embed_dim % num_heads == 0."""
    return hp["embed_dim"] % hp["num_heads"] == 0


def cfg2name(hp: Dict, model_type: str | None = None) -> str:
    """Create a unique string identifier for a hyper-parameter set.

    Adding the optional ``model_type`` prefix avoids collisions when tuning
    both Feedback Alignment (``fa``) and Back-propagation (``bp``) variants.
    """
    base = (
        f"p{hp['p']}_ps{hp['patch_size']}_d{hp['depth']}_"
        f"mr{hp['mlp_ratio']}_ed{hp['embed_dim']}_h{hp['num_heads']}"
    )
    return f"{model_type}_{base}" if model_type else base


def prepare_dataloaders() -> Tuple[DataLoader, DataLoader]:
    """Prepare train/val DataLoaders (same setup as train.py)."""
    if DOWNLOAD_DATASET:
        download_pcam(DATA_ROOT)

    train_tfms = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor()]
    )
    val_tfms = transforms.Compose([transforms.ToTensor()])

    train_ds = PCamDataset(DATA_ROOT, split="train", transform=train_tfms)
    val_ds = PCamDataset(DATA_ROOT, split="val", transform=val_tfms)

    # Optionally reduce dataset size for faster debugging, similar to train.py
    if LIMIT_SAMPLES is not None:
        from torch.utils.data import Subset

        train_len = min(LIMIT_SAMPLES, len(train_ds))
        val_len = min(LIMIT_SAMPLES, len(val_ds))

        train_ds = Subset(train_ds, list(range(train_len)))  # type: ignore[arg-type]
        val_ds = Subset(val_ds, list(range(val_len)))        # type: ignore[arg-type]

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one(
    hp: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    ckpt_dir: Path,
    log_path: Path,
    model_type: str,
) -> float:
    """Train a single ViT model (FA or BP) for one hyper-parameter combination.

    Returns the best validation accuracy achieved.
    """
    # ---------------------------- logging setup ----------------------------
    logger = logging.getLogger(cfg2name(hp, model_type))
    logger.setLevel(logging.INFO)
    logger.handlers = []  # reset between runs
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # ----------------------------- model -----------------------------------
    model_cls = MODEL_MAP[model_type]
    model = model_cls(
        img_size=96,  # matches dataset size
        patch_size=hp["patch_size"],
        embed_dim=hp["embed_dim"],
        depth=hp["depth"],
        num_heads=hp["num_heads"],
        mlp_ratio=hp["mlp_ratio"],
        p=hp["p"],
        num_classes=2,
        in_chans=3,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_acc = 0.0
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, NUM_EPOCHS + 1):
        # ----------------- training -----------------
        model.train()
        for imgs, labels in tqdm(train_loader, leave=False, desc=f"E{epoch} train {cfg2name(hp, model_type)}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

        # ----------------- validation ---------------
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs)
                val_correct += (logits.argmax(1) == labels).sum().item()
        val_acc = val_correct / len(val_loader.dataset)  # type: ignore[arg-type]
        best_val_acc = max(best_val_acc, val_acc)
        scheduler.step()

        logger.info("Epoch %d/%d - Val Acc %.4f", epoch, NUM_EPOCHS, val_acc)

    # save best model state dict
    torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
    logger.info("Best Val Acc %.4f", best_val_acc)
    return best_val_acc


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main(model_type: str = "fa") -> None:
    """Hyper-parameter tuning entry point.

    Parameters
    ----------
    model_type : {"fa", "bp"}, default "fa"
        Select Feedback Alignment ("fa") or Back-propagation ("bp") variant.
    """
    model_type = model_type.lower()

    # load existing CSV to skip finished configs
    done_cfgs: set[str] = set()
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                done_cfgs.add(row["cfg_name"])

    # prepare dataloaders once to amortise cost
    train_loader, val_loader = prepare_dataloaders()

    # open CSV in append mode
    csv_fh = RESULTS_PATH.open("a", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=CSV_HEADER)
    if RESULTS_PATH.stat().st_size == 0:
        writer.writeheader()

    # iterate over grid
    for hp in iter_param_grid(PARAM_GRID):
        if not is_valid_combo(hp):
            continue  # skip invalid embed_dim / num_heads combos
        cfg_name = cfg2name(hp, model_type)
        if cfg_name in done_cfgs:
            print(f"Skipping already completed {cfg_name}")
            continue

        print(f"\n=== Running {cfg_name} ===")
        ckpt_dir = CKPT_ROOT / model_type / cfg_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_path = ckpt_dir / "train.log"

        best_acc = train_one(hp, train_loader, val_loader, ckpt_dir, log_path, model_type)

        writer.writerow({
            "cfg_name": cfg_name,
            "p": hp["p"],
            "patch_size": hp["patch_size"],
            "depth": hp["depth"],
            "mlp_ratio": hp["mlp_ratio"],
            "embed_dim": hp["embed_dim"],
            "num_heads": hp["num_heads"],
            "best_val_acc": best_acc,
        })
        csv_fh.flush()
        print(f"Completed {cfg_name} | Best Val Acc: {best_acc:.4f}")

    csv_fh.close()
    print("\nHyper-parameter tuning finished. Results saved to", RESULTS_PATH.resolve())


if __name__ == "__main__":
    # Default run (Feedback Alignment). Importers can call ``main("bp")``.
    main("bp") 