import os
from pathlib import Path
from typing import Optional

try:
    import h5py  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Package 'h5py' is required for HDF5 data loading. Install it via 'pip install h5py'"
    ) from e

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PCamDataset(Dataset):
    """PatchCamelyon (PCam) dataset loader for **HDF5** format.

    Expected directory tree::

        root/
            train/
                train_img.h5
                train_label.h5
            val/
                val_img.h5
                val_label.h5  # or val_labels.h5
            test/
                test_img.h5
                test_label.h5  # or test_labels.h5

    ``*.h5`` files are assumed to each contain **exactly one dataset**:
        - The image file stores an array shaped ``(N, H, W, C)`` **or** ``(N, C, H, W)``.
        - The label file stores an array shaped ``(N,)``.

    The class treats *all* non-zero labels as ``1`` (tumor) and zeros as ``0`` (normal),
    preserving the original binary-classification semantics of PCam.
    """

    IMG_FILENAMES = {
        "train": "train_img.h5",
        "val": "val_img.h5",
        "test": "test_img.h5",
    }

    LABEL_FILENAMES = {
        "train": ["train_label.h5", "train_labels.h5"],  # support both spellings
        "val": ["val_label.h5", "val_labels.h5"],
        "test": ["test_label.h5", "test_labels.h5"],
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        assert split in {"train", "val", "test"}, "split must be train/val/test"

        self.root = Path(root) / split
        if not self.root.exists():
            raise FileNotFoundError(f"Split directory {self.root} does not exist.")

        # Resolve image & label HDF5 paths
        self.img_path = self.root / self.IMG_FILENAMES[split]
        if not self.img_path.exists():
            raise FileNotFoundError(f"Expected image file {self.img_path} not found.")

        label_candidates = [self.root / fname for fname in self.LABEL_FILENAMES[split]]
        self.label_path = next((p for p in label_candidates if p.exists()), None)
        if self.label_path is None:
            raise FileNotFoundError(
                f"None of the expected label files found: {', '.join(str(p) for p in label_candidates)}"
            )

        self.transform = (
            transform
            if transform is not None
            else transforms.Compose(
                [
                    transforms.Resize(96),  # Maintain compatibility with original pipeline
                    transforms.ToTensor(),
                ]
            )
        )

        # File handles are opened lazily in each worker process to avoid pickling issues.
        self._h5_imgs: Optional[h5py.File] = None
        self._h5_labels: Optional[h5py.File] = None
        self._images_ds = None  # type: ignore
        self._labels_ds = None  # type: ignore

        # Determine dataset length without keeping files open
        with h5py.File(self.img_path, "r") as f_imgs:
            # Assume first dataset is images if name unknown
            img_key = list(f_imgs.keys())[0]
            # Check if we have a Dataset (not a Group/Datatype) before accessing shape
            h5_obj = f_imgs[img_key]
            if isinstance(h5_obj, h5py.Dataset):
                # Cast for static type checkers before using .shape
                from typing import cast

                ds = cast("h5py.Dataset", h5_obj)
                self._length = ds.shape[0]
            elif isinstance(f_imgs[img_key], h5py.Group):
                raise ValueError(f"HDF5 object at key '{img_key}' is a Group, not a Dataset")
            elif isinstance(f_imgs[img_key], h5py.Datatype):
                raise ValueError(f"HDF5 object at key '{img_key}' is a Datatype, not a Dataset") 
            else:
                raise ValueError(f"HDF5 object at key '{img_key}' is not a Dataset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_open(self):
        """Open HDF5 files on first access in each worker process."""
        if self._h5_imgs is None or self._h5_labels is None:
            # Open in read-only, SWMR-safe mode
            self._h5_imgs = h5py.File(self.img_path, "r", swmr=True)
            self._h5_labels = h5py.File(self.label_path, "r", swmr=True)

            self._images_ds = self._h5_imgs[list(self._h5_imgs.keys())[0]]  # type: ignore[index]
            self._labels_ds = self._h5_labels[list(self._h5_labels.keys())[0]]  # type: ignore[index]

    # ------------------------------------------------------------------
    # Standard Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:  # noqa: D401 – simple form is fine
        return self._length

    def __getitem__(self, idx: int):  # noqa: D401 – simple form is fine
        self._ensure_open()

        # Fetch data from HDF5
        img_np: np.ndarray = self._images_ds[idx]  # type: ignore[index]
        label_np: np.ndarray = self._labels_ds[idx]  # type: ignore[index]

        # Convert image to PIL.Image for torchvision transforms
        if img_np.ndim == 3 and img_np.shape[-1] in {1, 3, 4}:  # (H, W, C)
            img_pil = Image.fromarray(img_np.astype(np.uint8))
        elif img_np.ndim == 3:  # (C, H, W)
            img_pil = Image.fromarray(np.transpose(img_np, (1, 2, 0)).astype(np.uint8))
        else:
            raise ValueError("Unexpected image shape in HDF5 dataset: {img_np.shape}")

        # Apply transforms
        img_tensor = self.transform(img_pil) if self.transform else torch.tensor(img_np)

        # Normalize label to {0,1}
        label = int(label_np)  # Ensure Python int
        label = 1 if label != 0 else 0

        return img_tensor, torch.tensor(label, dtype=torch.long)


# -----------------------------------------------------------------------------
# (Optional) Dataset download helper remains here for completeness.
# -----------------------------------------------------------------------------


def download_pcam(data_dir: str) -> None:
    """Download PCam PNG dataset using Kaggle CLI (left unchanged).

    NOTE: This helper remains for backward-compatibility but is **not** required
    when using the new HDF5-based dataset layout described above.
    """

    try:
        import kaggle  # noqa: F401 – imported for side-effects only
    except ImportError as e:
        raise RuntimeError(
            "kaggle package missing. Install and set up Kaggle API token before calling download_pcam()."
        ) from e

    os.makedirs(data_dir, exist_ok=True)
    os.chdir(data_dir)

    # Download competition files via Kaggle CLI
    os.system("kaggle competitions download -c histopathologic-cancer-detection -p . --force")

    # Extract zips if they exist
    for zip_name in ["train.zip", "test.zip"]:
        if Path(zip_name).exists():
            os.system(f"unzip -q {zip_name} -d .")
            os.remove(zip_name)

    print("PCam dataset downloaded and extracted.") 