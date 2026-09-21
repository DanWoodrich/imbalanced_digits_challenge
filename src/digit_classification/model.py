"""Model implementation for imbalanced digit classification using PyTorch Lightning."""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from lightning.pytorch import LightningModule
import matplotlib.pyplot as plt
from digit_classification.data import DIGIT_TO_IDX, IDX_TO_DIGIT, NUM_CLASSES, TARGET_DIGITS

#define certain augmentation packages
AUGMENTATION_MODIFIER = {
    1: ((0.2, 0.2), (0.9, 1.1)),
    2: ((0.35, 0.35), (0.75, 1.25)),
}

def preprocess_image(image: Union[Image.Image, torch.Tensor, Path, str]) -> torch.Tensor:
    """Preprocesses an input image (PIL Image, file path, or torch.Tensor) to shape (1, 1, 28, 28).

    Resizes (squishing larger images and blowing up smaller images) directly to 28x28
    without cropping, and normalizes pixel values to [0, 1].
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("L")
    elif isinstance(image, Image.Image):
        img = image.convert("L")
    elif isinstance(image, torch.Tensor):
        t = image.float()
        if t.dim() == 2:  # (H, W)
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 3:
            if t.shape[0] in (1, 3, 4):  # (C, H, W)
                if t.shape[0] > 1:
                    t = t.mean(dim=0, keepdim=True)
                t = t.unsqueeze(0)
            else:  # (B, H, W)
                t = t.unsqueeze(1)
        elif t.dim() == 4 and t.shape[1] > 1:
            t = t.mean(dim=1, keepdim=True)

        # Squish larger or blow up smaller to (28, 28)
        if t.shape[-2:] != (28, 28):
            t = torch.nn.functional.interpolate(t, size=(28, 28), mode="bilinear", align_corners=False)
        return t
    else:
        raise TypeError(f"Unsupported image input type: {type(image)}")

    transform = transforms.Compose([
        transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)

class FrameShiftAugmentation(torch.nn.Module):
    """Applies a custom frame shift augmentation implementation.
    Accepts a mode argument that defines two different packages, of low (1) and medium (2) degree
    """
    def __init__(self, mode: int):
        super().__init__()
    
        translate,scale = AUGMENTATION_MODIFIER[mode]

        self.transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=0,
                translate=translate,
                scale=scale
            ),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            self.transform(image)
            for image in x
        ])
    
#by definition, this is the 'high' augmentation, so hardcode parameters here. 
class FrameShiftAugmentationBlur(torch.nn.Module):
    """Applies a custom frame shift augmentation implementation, along with a blur to alleviate some of the pixelated look from the degree shift.
    """
    def __init__(self,
                translate: tuple[float, float] = (0.5, 0.5),
                degrees: float = 45,
                scale: tuple[float, float] = (0.5, 1.5),
                blur_kernel: tuple[int, int] = (3, 3),
                blur_sigma: tuple[float, float] = (0.05, 0.65)):
        super().__init__()

        self.transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=degrees,
                translate=translate,
                scale=scale
            ),
            transforms.GaussianBlur(
                kernel_size=blur_kernel,
                sigma=blur_sigma
            )
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            self.transform(image)
            for image in x
        ])

class DigitClassifier(LightningModule):
    """Custom Convolutional Neural Network for recognizing digits (0, 5, and 8) from MNIST.

    Tracks overall and per-class train/val loss history for diagnostic visualization.
    """

    def __init__(self, learning_rate: float = 0.001, num_classes: int = NUM_CLASSES,frameshift_augmentation: int = 1):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        if frameshift_augmentation in AUGMENTATION_MODIFIER:
            self.augmentation = FrameShiftAugmentation(
                mode=frameshift_augmentation
            )
        elif frameshift_augmentation == 3:
            self.augmentation = FrameShiftAugmentationBlur()
        elif frameshift_augmentation == 0:
            self.augmentation = torch.nn.Identity()
        else:
            raise ValueError(
                f"Invalid frameshift_augmentation: {frameshift_augmentation}. "
                "Expected 0, 1, 2, or 3."
            )
        # 3 Convolutional blocks
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Max pooling (28x28 -> 14x14 -> 7x7 -> 3x3)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Dense classification head
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Loss functions: reduction='none' for per-class tracking, mean for backprop
        self.loss_fn = nn.CrossEntropyLoss()
        self.unreduced_loss_fn = nn.CrossEntropyLoss(reduction="none")

        # In-epoch accumulators for per-class loss tracking
        self.training_step_outputs: List[Dict[str, torch.Tensor]] = []
        self.validation_step_outputs: List[Dict[str, torch.Tensor]] = []

        # Epoch-level history dictionary for loss curves
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_loss_8": [],
            "train_loss_0": [],
            "train_loss_5": [],
            "val_loss_8": [],
            "val_loss_0": [],
            "val_loss_5": [],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN architecture."""
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        logits = self.fc2(x)

        return logits

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize Adam optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute training loss and accumulate batch outputs for per-class metrics."""
        x, y = batch

        x = self.augmentation(x)

        logits = self(x)
        loss = self.loss_fn(logits, y)
        raw_losses = self.unreduced_loss_fn(logits, y)

        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y).float().mean()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        self.training_step_outputs.append({
            "loss": loss.detach(),
            "raw_losses": raw_losses.detach(),
            "targets": y.detach(),
            "accuracy": accuracy.detach(),
        })
        return loss

    def on_train_epoch_end(self) -> None:
        """Aggregate per-class training loss and overall metrics at the end of each epoch."""
        if not self.training_step_outputs:
            return

        all_raw_losses = torch.cat([out["raw_losses"] for out in self.training_step_outputs])
        all_targets = torch.cat([out["targets"] for out in self.training_step_outputs])
        epoch_loss = torch.stack([out["loss"] for out in self.training_step_outputs]).mean().item()
        epoch_acc = torch.stack([out["accuracy"] for out in self.training_step_outputs]).mean().item()

        self.history["train_loss"].append(epoch_loss)
        self.history["train_acc"].append(epoch_acc)

        for digit in [8, 0, 5]:
            mapped_idx = DIGIT_TO_IDX[digit]
            mask = all_targets == mapped_idx
            if mask.any():
                digit_loss = all_raw_losses[mask].mean().item()
            else:
                digit_loss = float("nan")
            self.history[f"train_loss_{digit}"].append(digit_loss)
            self.log(f"train_loss_digit_{digit}", digit_loss, on_step=False, on_epoch=True)

        self.training_step_outputs.clear()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute validation loss and accumulate outputs for per-class tracking."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Skip recording metrics during Lightning's initial sanity check
        if getattr(self, "_trainer", None) is not None and getattr(self._trainer, "sanity_checking", False):
            return loss

        raw_losses = self.unreduced_loss_fn(logits, y)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y).float().mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        self.validation_step_outputs.append({
            "loss": loss.detach(),
            "raw_losses": raw_losses.detach(),
            "targets": y.detach(),
            "accuracy": accuracy.detach(),
        })
        return loss

    def on_validation_epoch_end(self) -> None:
        """Aggregate per-class validation loss at epoch end."""
        # Skip recording metrics during Lightning's initial sanity check
        if getattr(self, "_trainer", None) is not None and getattr(self._trainer, "sanity_checking", False):
            self.validation_step_outputs.clear()
            return

        if not self.validation_step_outputs:
            return

        all_raw_losses = torch.cat([out["raw_losses"] for out in self.validation_step_outputs])
        all_targets = torch.cat([out["targets"] for out in self.validation_step_outputs])
        epoch_loss = torch.stack([out["loss"] for out in self.validation_step_outputs]).mean().item()
        epoch_acc = torch.stack([out["accuracy"] for out in self.validation_step_outputs]).mean().item()

        self.history["val_loss"].append(epoch_loss)
        self.history["val_acc"].append(epoch_acc)

        for digit in [8, 0, 5]:
            mapped_idx = DIGIT_TO_IDX[digit]
            mask = all_targets == mapped_idx
            if mask.any():
                digit_loss = all_raw_losses[mask].mean().item()
            else:
                digit_loss = float("nan")
            self.history[f"val_loss_{digit}"].append(digit_loss)
            self.log(f"val_loss_digit_{digit}", digit_loss, on_step=False, on_epoch=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute test loss and test accuracy."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y).float().mean()
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", accuracy, on_step=False, on_epoch=True)
        return loss

    def predict_step(
        self,
        batch: Union[torch.Tensor, Tuple[torch.Tensor, Any]],
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ) -> Dict[str, Any]:
        """Predicts the probability of each digit label (0, 5, and 8) for given image(s)."""
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        x = preprocess_image(x)
        logits = self(x)
        probabilities = torch.softmax(logits, dim=-1)

        return {
            "probabilities": probabilities,
            "target_digits": TARGET_DIGITS,
            "predictions": [IDX_TO_DIGIT[i] for i in torch.argmax(probabilities, dim=-1).tolist()],
        }

    def plot_loss_curves(self, save_path: Union[str, Path]) -> Path:
        """Plots overall and per-class train/val loss curves over training epochs and saves as PNG."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        train_epochs = list(range(1, len(self.history["train_loss"]) + 1))
        val_epochs = list(range(1, len(self.history["val_loss"]) + 1))

        # Plot 1: Overall Train & Val Loss
        if train_epochs:
            axes[0].plot(train_epochs, self.history["train_loss"], label="Train Loss", color="#1f77b4", lw=2, marker="o")
        if val_epochs:
            axes[0].plot(val_epochs, self.history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, marker="s")
        axes[0].set_title("Overall Loss Progression", fontsize=12)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Per-Class Validation Loss
        colors = {"8": "#2ca02c", "0": "#9467bd", "5": "#d62728"}
        for digit in [8, 0, 5]:
            val_digit_loss = self.history.get(f"val_loss_{digit}", [])
            if val_digit_loss:
                digit_epochs = list(range(1, len(val_digit_loss) + 1))
                axes[1].plot(
                    digit_epochs,
                    val_digit_loss,
                    label=f"Val Loss Digit {digit}",
                    color=colors[str(digit)],
                    lw=1.8,
                    marker="^",
                )
        axes[1].set_title("Per-Class Validation Loss (Digits 8, 0, 5)", fontsize=12)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        h1, l1 = axes[1].get_legend_handles_labels()
        if l1:
            axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        return save_path
