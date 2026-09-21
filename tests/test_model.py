"""Tests for DigitClassifier architecture, forward pass, predict_step, loss plotting, and non-28x28 resizing."""

from pathlib import Path
from PIL import Image
import pytest
import torch
from digit_classification.model import DigitClassifier, preprocess_image
from digit_classification.data import TARGET_DIGITS, NUM_CLASSES


def test_model_initialization():
    """Verify model initialization and architecture parameters."""
    model = DigitClassifier(learning_rate=0.001)
    assert model.fc2.out_features == NUM_CLASSES  # 3 classes
    assert model.learning_rate == 0.001


def test_model_forward_shape():
    """Verify forward pass output shape is (batch_size, 3)."""
    model = DigitClassifier()
    dummy_input = torch.randn(4, 1, 28, 28)
    logits = model(dummy_input)

    assert logits.shape == (4, NUM_CLASSES)
    assert logits.dtype == torch.float32


def test_training_and_validation_steps():
    """Verify training_step, validation_step, and epoch end hooks execute cleanly."""
    model = DigitClassifier()
    x = torch.randn(4, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    loss_train = model.training_step((x, y), batch_idx=0)
    assert loss_train is not None and not torch.isnan(loss_train)
    model.on_train_epoch_end()
    assert len(model.history["train_loss"]) == 1

    loss_val = model.validation_step((x, y), batch_idx=0)
    assert loss_val is not None and not torch.isnan(loss_val)
    model.on_validation_epoch_end()
    assert len(model.history["val_loss"]) == 1


def test_predict_step_standard_shape():
    """Verify predict_step returns probabilities for digits (0, 5, 8) with standard 28x28 inputs."""
    model = DigitClassifier()
    dummy_input = torch.randn(2, 1, 28, 28)
    output = model.predict_step(dummy_input)

    assert "probabilities" in output
    assert "target_digits" in output
    assert "predictions" in output

    probs = output["probabilities"]
    assert probs.shape == (2, NUM_CLASSES)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert output["target_digits"] == TARGET_DIGITS  # [0, 5, 8]
    assert len(output["predictions"]) == 2
    for pred in output["predictions"]:
        assert pred in TARGET_DIGITS


def test_predict_step_with_non_28x28_inputs(tmp_path: Path):
    """Verify predict_step and preprocess_image gracefully handle non-28x28 images

    by squishing larger images and blowing up smaller images without cropping.
    """
    model = DigitClassifier()

    # 1. Larger image squished: 120 x 90 PIL image
    large_img = Image.new("L", (120, 90), color=128)
    large_path = tmp_path / "large_digit.png"
    large_img.save(large_path)

    large_tensor = preprocess_image(large_path)
    assert large_tensor.shape == (1, 1, 28, 28)
    out_large = model.predict_step(large_tensor)
    assert out_large["probabilities"].shape == (1, 3)
    assert torch.allclose(out_large["probabilities"].sum(dim=-1), torch.ones(1), atol=1e-5)

    # 2. Smaller image blown up: 14 x 10 PIL image
    small_img = Image.new("L", (10, 14), color=200)
    small_path = tmp_path / "small_digit.png"
    small_img.save(small_path)

    small_tensor = preprocess_image(small_path)
    assert small_tensor.shape == (1, 1, 28, 28)
    out_small = model.predict_step(small_tensor)
    assert out_small["probabilities"].shape == (1, 3)
    assert torch.allclose(out_small["probabilities"].sum(dim=-1), torch.ones(1), atol=1e-5)

    # 3. Direct tensor inputs: larger (1, 1, 64, 64) and smaller (1, 1, 12, 12)
    large_direct = torch.randn(1, 1, 64, 64)
    small_direct = torch.randn(1, 1, 12, 12)
    out_direct_large = model.predict_step(large_direct)
    out_direct_small = model.predict_step(small_direct)
    assert out_direct_large["probabilities"].shape == (1, 3)
    assert out_direct_small["probabilities"].shape == (1, 3)


def test_plot_loss_curves(tmp_path: Path):
    """Verify plot_loss_curves saves a valid PNG file."""
    model = DigitClassifier()
    model.history["train_loss"] = [0.5, 0.3, 0.1]
    model.history["val_loss"] = [0.6, 0.4, 0.2]
    model.history["val_loss_8"] = [0.5, 0.3, 0.1]
    model.history["val_loss_0"] = [0.6, 0.4, 0.2]
    model.history["val_loss_5"] = [0.8, 0.5, 0.3]

    out_file = tmp_path / "loss_curves.png"
    saved_path = model.plot_loss_curves(out_file)

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0
