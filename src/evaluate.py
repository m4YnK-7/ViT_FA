"""Evaluation & visualization for ViT-FA on PCam test set.

Run as a module:
$ python -m src.evaluate --checkpoint checkpoints/best_model.pth --visualize

Or import and call `evaluate(...)` programmatically.
"""

from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import make_grid

from src.dataset import PCamDataset
from src.models.vit_fa import ViTFA


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:  # noqa: D401
    """Compute top-1 accuracy for a mini-batch."""
    preds = output.argmax(dim=1)
    correct = (preds == target).sum().item()
    return correct / target.size(0)


# -----------------------------------------------------------------------------
# Main evaluation routine
# -----------------------------------------------------------------------------

def evaluate(
    checkpoint: Optional[Union[str, Path]] = None,
    data_root: Union[str, Path] = "data/pcam",
    *,
    batch_size: int = 128,
    num_workers: int = 4,
    device: Optional[str] = None,
    visualize: bool = True,
    num_vis_images: int = 25,
    limit_samples: Optional[int] = None,
    model_type: str = "fa",
) -> None:
    """Evaluate a trained model on the *test* split and optionally visualise.

    Parameters
    ----------
    checkpoint : str or Path or None
        Path to the ``.pth`` checkpoint containing model weights. If ``None``
        (default), will look for
        ``checkpoints/<model_type>/best_model.pth``.
    data_root : str or Path, default "data/pcam"
        Root directory comprising train/val/test sub-folders with HDF5 files.
    batch_size : int, default 128
        Mini-batch size for evaluation.
    num_workers : int, default 4
        ``DataLoader`` workers.
    device : str or None, default None
        Device to use (auto-detected if None).
    visualize : bool, default False
        If True, show confusion matrix + a grid of sample predictions.
    num_vis_images : int, default 25
        Number of sample images to plot in the grid (must be square-number).
    limit_samples : int or None, default None
        Restrict evaluation to first ``limit_samples`` items (debugging).
    model_type : {"fa", "bp"}, default "fa"
        Which model type to evaluate:
        - "fa" for Feedback Alignment (ViTFA)
        - "bp" for standard back-prop (ViTBP)
    """

    # ------------------------ setup ------------------------
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset / DataLoader
    test_tfms = transforms.Compose([transforms.ToTensor()])
    test_ds_full = PCamDataset(str(data_root), split="test", transform=test_tfms)

    if limit_samples is not None:
        test_len = min(limit_samples, len(test_ds_full))
        test_ds = Subset(test_ds_full, list(range(test_len)))  # type: ignore[arg-type]
    else:
        test_ds = test_ds_full

    test_loader: DataLoader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ------------------------ model ------------------------
    # Resolve model class based on 'model_type'
    model_type_lc = model_type.lower()
    if model_type_lc in {"fa", "vitfa"}:
        model_cls = ViTFA
    elif model_type_lc in {"bp", "vitbp", "backprop"}:
        from src.models import ViTBP  # local import to avoid unnecessary deps
        model_cls = ViTBP
    else:
        raise ValueError("model_type must be 'fa' or 'bp'")

    model = model_cls(num_classes=2)

    if checkpoint is None:
        checkpoint = Path("checkpoints") / model_type_lc / "best_model.pth"
    else:
        checkpoint = Path(checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device)
    model.eval()

    # ----------------------- evaluation -----------------------
    total_loss = 0.0
    total_acc = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    all_preds: list[int] = []
    all_labels: list[int] = []
    sample_imgs: list[torch.Tensor] = []
    sample_pred_labels: list[int] = []
    sample_true_labels: list[int] = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            total_acc += accuracy(logits, labels) * imgs.size(0)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            # Stash a few samples for visualisation
            if visualize and len(sample_imgs) < num_vis_images:
                needed = num_vis_images - len(sample_imgs)
                sample_imgs.extend(imgs.cpu()[:needed])
                sample_pred_labels.extend(preds.cpu().tolist()[:needed])
                sample_true_labels.extend(labels.cpu().tolist()[:needed])

    total_loss /= len(test_loader.dataset)  # type: ignore[arg-type]
    total_acc /= len(test_loader.dataset)   # type: ignore[arg-type]

    print(f"Test Loss: {total_loss:.4f} | Test Accuracy: {total_acc:.4f}")

    # ----------------------- visualisation -----------------------
    if visualize:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import seaborn as sns  # type: ignore
            from sklearn.metrics import confusion_matrix  # type: ignore
        except ImportError:
            print("Install matplotlib seaborn scikit-learn for visualization.")
            return

        # --------------------------------------------------------
        # Prepare output directory
        # --------------------------------------------------------
        out_root = Path("evaluation_results")
        out_dir = out_root / model_type_lc  # type: ignore[name-defined]
        out_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix heatmap
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["normal", "tumor"], yticklabels=["normal", "tumor"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()

        # Save confusion matrix image
        cm_path = out_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300)

        # Sample grid
        grid_imgs = torch.stack(sample_imgs)  # shape (N, C, H, W)
        grid = make_grid(grid_imgs, nrow=int(num_vis_images ** 0.5), normalize=True, value_range=(0, 1))
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.title("Sample predictions (green = correct, red = wrong)")

        # Overlay rectangles for correctness
        import matplotlib.patches as patches

        ax = plt.gca()
        h, w = grid_imgs.shape[-2:]
        nrow = int(num_vis_images ** 0.5)
        for idx in range(len(sample_imgs)):
            row = idx // nrow
            col = idx % nrow
            correct = sample_pred_labels[idx] == sample_true_labels[idx]
            color = "lime" if correct else "red"
            rect = patches.Rectangle(
                (col * w, row * h),
                w,
                h,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

        # Add a caption mapping class idx to name
        plt.figtext(0.5, 0.01, "0 = normal, 1 = tumor", ha="center")

        # Save sample grid image
        grid_path = out_dir / "sample_grid.png"
        plt.savefig(grid_path, dpi=300)
        plt.show()

        # --------------------------------------------------------
        # Additional individual image display with labels
        # --------------------------------------------------------
        n_show = min(9, len(sample_imgs))
        if n_show > 0:
            cols = 3
            rows = (n_show + cols - 1) // cols
            plt.figure(figsize=(cols * 3, rows * 3))
            for i in range(n_show):
                plt.subplot(rows, cols, i + 1)
                img = sample_imgs[i]
                plt.imshow(img.permute(1, 2, 0))
                plt.axis("off")
                pred = sample_pred_labels[i]
                true = sample_true_labels[i]
                color = "green" if pred == true else "red"
                plt.title(f"Pred: {pred} | True: {true}", color=color, fontsize=8)
            plt.tight_layout()
            # Save individual grid image
            indiv_path = out_dir / "sample_individuals.png"
            plt.savefig(indiv_path, dpi=300)
            plt.show()


# -----------------------------------------------------------------------------
# Optional stand-alone execution with default parameters
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Directly call evaluate() with default arguments. Adjust here if needed.
    evaluate(model_type="bp") 