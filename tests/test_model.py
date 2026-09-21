"""Tests for DigitClassifier architecture, forward pass, predict_step, loss plotting, and non-28x28 resizing."""

from pathlib import Path
from PIL import Image
import pytest
import torch
from digit_classification.model import DigitClassifier, preprocess_image, AUGMENTATION_MODIFIER
from digit_classification.data import TARGET_DIGITS, NUM_CLASSES


def test_predict_step_image_formats(tmp_path: Path):
    """Verify preprocess_image and predict_step work correctly with JPEG and TIFF file inputs.

    PIL's Image.open supports both formats; this test ensures the file-path branch of
    preprocess_image handles each without error and produces valid predictions.
    """
    model = DigitClassifier()

    base_img = Image.new("L", (28, 28), color=128)

    for fmt, filename in [("JPEG", "digit.jpg"), ("TIFF", "digit.tiff")]:
        img_path = tmp_path / filename
        base_img.save(img_path, format=fmt)

        tensor = preprocess_image(img_path)
        assert tensor.shape == (1, 1, 28, 28), f"Wrong shape for {fmt}"

        out = model.predict_step(tensor)
        assert out["probabilities"].shape == (1, NUM_CLASSES), f"Wrong prob shape for {fmt}"
        assert torch.allclose(out["probabilities"].sum(dim=-1), torch.ones(1), atol=1e-5), (
            f"Probabilities do not sum to 1 for {fmt}"
        )
        assert out["predictions"][0] in TARGET_DIGITS, f"Prediction not a valid digit for {fmt}"


def test_training_step_augmentation_modes():
    """Verify DigitClassifier.training_step runs without error for every augmentation mode.

    Modes: 0 (none/Identity), 1 (low FrameShift), 2 (moderate FrameShift), 3 (high + blur).
    Each mode is exercised with a small synthetic batch to confirm the augmentation pipeline
    is wired correctly end-to-end and produces a finite scalar loss.
    """
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    # Use all three mapped label indices to avoid zero_division warnings inside the loss
    y = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    batch = (x, y)

    all_modes = [0, 1, 2, 3]
    for mode in all_modes:
        model = DigitClassifier(frameshift_augmentation=mode)
        model.train()

        loss = model.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor), f"Mode {mode}: loss is not a Tensor"
        assert loss.ndim == 0, f"Mode {mode}: loss should be a scalar"
        assert torch.isfinite(loss), f"Mode {mode}: loss is not finite ({loss.item()})"

    # Verify invalid mode raises ValueError
    with pytest.raises(ValueError, match="Invalid frameshift_augmentation"):
        DigitClassifier(frameshift_augmentation=99)


def test_model_initialization():
    """Verify model initialization and architecture parameters."""
    model = DigitClassifier(learning_rate=0.001)
    assert model.fc2.out_features == NUM_CLASSES  # 3 classes
    assert model.learning_rate == 0.001

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
    # Populate mock history
    model.history["train_loss"] = [0.5, 0.3, 0.1]
    model.history["val_loss"] = [0.6, 0.4, 0.2]
    model.history["val_loss_8"] = [0.5, 0.3, 0.1]
    model.history["val_loss_0"] = [0.6, 0.4, 0.2]
    model.history["val_loss_5"] = [0.8, 0.5, 0.3]

    out_file = tmp_path / "loss_curves.png"
    saved_path = model.plot_loss_curves(out_file)

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0
