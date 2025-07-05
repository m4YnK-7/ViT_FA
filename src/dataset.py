import os
from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class PCamDataset(Dataset):
    """PatchCamelyon (PCam) dataset loader.

    Expected directory tree:
        root/
            train/
                tumor/...
                normal/...
            val/
                tumor/...
                normal/...
            test/
                tumor/...
                normal/...
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        assert split in {"train", "val", "test"}, "split must be train/val/test"
        self.root = Path(root) / split
        if not self.root.exists():
            raise FileNotFoundError(
                f"Split directory {self.root} does not exist. Did you download the dataset?"
            )

        self.transform = transform if transform else transforms.Compose(
            [
                transforms.Resize(96),
                transforms.ToTensor(),
            ]
        )

        # Map class folder names to integer labels
        self.classes = {"normal": 0, "tumor": 1}
        self.samples = self._gather_samples()

    def _gather_samples(self) -> List[Tuple[Path, int]]:
        samples = []
        for class_name, class_idx in self.classes.items():
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.glob("*.png"):
                samples.append((img_path, class_idx))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# Optional helper for automatic Kaggle download

def download_pcam(data_dir: str) -> None:
    """Download and extract PCam dataset using Kaggle CLI."""
    try:
        import kaggle  # noqa: F401
    except ImportError as e:
        raise RuntimeError("kaggle package missing. Install and set up Kaggle API token.") from e

    os.makedirs(data_dir, exist_ok=True)
    os.chdir(data_dir)

    # Download
    os.system("kaggle competitions download -c histopathologic-cancer-detection -p . --force")

    # Extract
    for zip_name in [
        "train.zip",
        "test.zip",
    ]:
        if Path(zip_name).exists():
            os.system(f"unzip -q {zip_name} -d .")
            os.remove(zip_name)

    print("PCam dataset downloaded and extracted.") 