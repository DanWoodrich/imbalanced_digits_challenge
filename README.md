# Digit Classification

A neural network application for recognizing handwritten digits from an imbalanced subset of the MNIST dataset (digits `0`, `5`, and `8`).

## Overview

- **Curated Dataset**: 5,000 total MNIST images intentionally imbalanced:
  - 3,500 images with label `8`
  - 1,200 images with label `0`
  - 300 images with label `5`
- **Data Partitions**: Reproducible split into 60% training (3,000 images), 20% validation (1,000 images), and 20% evaluation/test (1,000 images).
- **Imbalance Handling**: Optional `WeightedRandomSampler` that dynamically inverts class frequencies to mitigate minority class underperformance.
- **Model Architecture**: Custom 3-block Convolutional Neural Network built with PyTorch Lightning (`LightningModule`) with 3 output classes.
- **CLI**: Standardized CLI powered by Typer supporting data downloading, CPU training (max 20 epochs), evaluation reports, and single-image inference.

## Installation

```bash
# Create and activate a virtual environment (Python >= 3.11)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package in editable mode with test dependencies
pip install -e ".[test]"
```

## CLI Usage

### 1. Download MNIST Data
```bash
digit-classification download-data --data-dir ./data
```

### 2. Train Model
```bash
digit-classification train \
  --data-dir ./data \
  --output-dir ./runs \
  --epochs 20 \
  --weighted
```

### 3. Evaluate Model
```bash
digit-classification evaluate \
  --checkpoint-path ./runs/checkpoints/digit-classifier-epoch=XX-val_loss=X.XXXX.ckpt \
  --data-dir ./data
```

### 4. Run Inference on an Image
```bash
digit-classification predict \
  --checkpoint-path ./runs/checkpoints/digit-classifier-epoch=XX-val_loss=X.XXXX.ckpt \
  --input-path ./sample_image.png
```

## Running Tests

```bash
pytest
```
