"""Tests for DigitClassifier architecture, forward pass, and predict_step."""

import pytest
import torch
from digit_classification.model import DigitClassifier
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


def test_training_step():
    """Verify training_step returns scalar loss."""
    model = DigitClassifier()
    x = torch.randn(4, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    loss = model.training_step((x, y), batch_idx=0)

    assert loss is not None
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)


def test_validation_step():
    """Verify validation_step returns scalar loss."""
    model = DigitClassifier()
    x = torch.randn(4, 1, 28, 28)
    y = torch.tensor([1, 2, 0, 1], dtype=torch.long)
    loss = model.validation_step((x, y), batch_idx=0)

    assert loss is not None
    assert loss.dim() == 0
    assert not torch.isnan(loss)


def test_predict_step():
    """Verify predict_step returns probabilities for digits (0, 5, 8)."""
    model = DigitClassifier()
    dummy_input = torch.randn(2, 1, 28, 28)
    output = model.predict_step(dummy_input)

    assert "probabilities" in output
    assert "target_digits" in output
    assert "predictions" in output

    probs = output["probabilities"]
    assert probs.shape == (2, NUM_CLASSES)
    # Probabilities should sum to 1 across classes
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert output["target_digits"] == TARGET_DIGITS  # [0, 5, 8]
    assert len(output["predictions"]) == 2
    for pred in output["predictions"]:
        assert pred in TARGET_DIGITS

