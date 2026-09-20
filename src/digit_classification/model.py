"""Model implementation for imbalanced digit classification using PyTorch Lightning."""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from digit_classification.data import IDX_TO_DIGIT, NUM_CLASSES, TARGET_DIGITS


class DigitClassifier(LightningModule):
    """Custom Convolutional Neural Network for recognizing digits (0, 5, and 8) from MNIST.

    Built without pre-trained backbones, matching the author's architecture.
    """

    def __init__(self, learning_rate: float = 0.001, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # 3 Convolutional blocks
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Max pooling (28x28 -> 14x14 -> 7x7 -> 3x3)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Dense classification head
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN architecture."""
        # Conv block 1: (B, 1, 28, 28) -> (B, 32, 14, 14)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Conv block 2: (B, 32, 14, 14) -> (B, 64, 7, 7)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Conv block 3: (B, 64, 7, 7) -> (B, 128, 3, 3)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Dense layers
        x = self.fc1(x)
        x = torch.relu(x)
        logits = self.fc2(x)

        return logits

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize Adam optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute training loss and log metrics."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y).float().mean()

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Compute validation loss and validation accuracy."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y).float().mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

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
        """Predicts the probability of each digit label (0, 5, and 8) for the given images.

        Returns a dictionary containing the probability tensor and the ordered digit labels.
        """
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        logits = self(x)
        probabilities = torch.softmax(logits, dim=-1)

        return {
            "probabilities": probabilities,
            "target_digits": TARGET_DIGITS,
            "predictions": [IDX_TO_DIGIT[i] for i in torch.argmax(probabilities, dim=-1).tolist()],
        }

