# Vision Transformer with Feedback Alignment on PCam

This project implements a lightweight Vision Transformer (ViT) trained with **Feedback Alignment (FA)** on the [PatchCamelyon (PCam)](https://www.kaggle.com/c/histopathologic-cancer-detection) histopathology dataset.

## Project Structure

```
cursor_project/
├── src/
│   ├── dataset.py         # PCam dataset wrapper
│   ├── models/
│   │   └── vit_fa.py      # ViT + FA layers
│   └── train.py           # Training script
├── requirements.txt       # Python dependencies
└── README.md              # You are here
```

## Installation

```bash
# 1. Clone or download this repo
# 2. (Optional but recommended) create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install deps
pip install -r requirements.txt
```

## Dataset Preparation

1. Create a Kaggle API token from your Kaggle account.
2. Place the downloaded `kaggle.json` in `~/.kaggle/` (Linux/macOS) or `C:\Users\<USER>\.kaggle` (Windows).
3. Run the helper inside `train.py` once with `--download_dataset` to automatically fetch and extract PCam to `data/pcam/`.

The expected folder structure after download is:

```
data/
└── pcam/
    ├── train/
    │   ├── tumor/...
    │   └── normal/...
    ├── val/
    │   ├── tumor/...
    │   └── normal/...
    └── test/...
```

## Quick Start

```bash
python -m src.train \
  --data_root data/pcam \
  --epochs 10 \
  --batch_size 128 \
  --lr 3e-4
```

Checkpoints are saved to `checkpoints/` and TensorBoard logs to `runs/`.

## Feedback Alignment

Gradient back-propagation in standard neural networks uses the transpose of forward weights. **Feedback Alignment (FA)** replaces these feedback paths with fixed, random matrices. This drastically reduces biological implausibility while still enabling learning.

In this codebase, every `nn.Linear` inside the ViT is replaced by a `LinearFA` layer whose backward pass employs a fixed random matrix **B** rather than `Wᵀ`.

## References

* Lillicrap, Timothy P., et al. "Random synaptic feedback weights support error backpropagation for deep learning." Nature Communications (2016).
* Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." ICLR 2021. 