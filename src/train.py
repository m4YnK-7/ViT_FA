import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset import PCamDataset, download_pcam
from src.models.vit_fa import ViTFA


def get_args():
    parser = argparse.ArgumentParser(description="Train ViT with Feedback Alignment on PCam")
    parser.add_argument("--data_root", type=str, default="data/pcam", help="Dataset root dir")
    parser.add_argument("--download_dataset", action="store_true", help="Download PCam via Kaggle API")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def accuracy(output, target):
    preds = output.argmax(dim=1)
    correct = (preds == target).sum().item()
    return correct / target.size(0)


def main():
    args = get_args()

    if args.download_dataset:
        print("Downloading PCam dataset…")
        download_pcam(args.data_root)

    # Transforms
    train_tfms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]
    )
    val_tfms = transforms.Compose([transforms.ToTensor()])

    # Datasets & loaders
    train_ds = PCamDataset(args.data_root, split="train", transform=train_tfms)
    val_ds = PCamDataset(args.data_root, split="val", transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    model = ViTFA(num_classes=2)
    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            imgs, labels = imgs.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_acc += accuracy(logits, labels) * imgs.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                val_acc += accuracy(logits, labels) * imgs.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        scheduler.step()

        print(
            f"Epoch {epoch}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}"
        )

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = Path(args.checkpoint_dir) / "best_model.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved new best model ({best_acc:.4f}) to {ckpt_path}")

    print("Training complete.")


if __name__ == "__main__":
    main() 