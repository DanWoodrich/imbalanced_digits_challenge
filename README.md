# Digit Classification

A Typer application for a custom CNN implementation of handwritten digit classification on an imbalanced dataset. 

## Overview

- **Curated Dataset**: 5,000 total MNIST images intentionally imbalanced:
  - 3,500 images with label `8`
  - 1,200 images with label `0`
  - 300 images with label `5`

 Project requirements dicate an 80:20 train/val : test split among this imbalanced sample. In total, we designate a 60% training (3,000 images), 20% validation (1,000 images), and 20% evaluation/test (1,000 images) split. 
- **Model Architecture**: Custom 3-block Convolutional Neural Network built with PyTorch Lightning (`LightningModule`) with 3 output classes and per-class loss tracking.
- **Design decisions around imbalance**: Two primary mechanisms used: 

1. using `WeightedRandomSampler` to counter minority class underperformance.
2. using frame-shift augmentation (pytorch "RandomAffine" transform) to boost the variance in the underrepresented class. 

- **CLI**: Typer-powered interface supporting download of the curated dataset, model training, test set evaluation with confusion matrix, and single-image inference.

## Installation

```bash
# Create and activate a virtual environment (Python >= 3.11)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1 or .venv\Scripts\activate.bat

# Install package in editable mode with test dependencies
pip install -e ".[test]"
```

## CLI Usage

At most minimal (using provided defaults)

See CLI tooltip (ie; digit-classification download-data --help) for full options. 

### 1. Download & Curate Dataset
```bash
digit-classification download-data
```

### 2. Train Model
```bash
digit-classification train 
```

### 3. Evaluate Model
```bash
digit-classification evaluate
```

### 4. Run Inference on an Image
```bash
digit-classification predict --input-path ./sample_image.png
```

## Running Tests

```bash
pytest -v
```

## Generative AI used for this project:

In general, I tried to stay light on the GenAI for the actual ML ideation and implementation. I used Google search built-in AI and free tiers of basic chatbots, in most cases asking about the most normal implementation patterns in the pytorch/lightning framework. 

I used AI more heavily (Antigravity free tier) to develop the project skeleton, CLI and tests, and some initial back and forth around missed assumptions. This allowed me to get past the initial gruntwork and boilerplate and start digging into the tool behaviors and bugs, and start adding features.  
