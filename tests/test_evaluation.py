"""Tests for evaluation metrics and classification report generation."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from digit_classification.evaluation import evaluate_model
from digit_classification.model import DigitClassifier
from digit_classification.data import NUM_CLASSES


def test_evaluate_model():
    """Verify evaluation routine generates classification report and structured dict."""
    model = DigitClassifier()
    model.eval()

    # Create synthetic dataset: 12 images and labels in {0, 1, 2}
    x = torch.randn(12, 1, 28, 28)
    y = torch.tensor([0, 1, 2] * 4, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    report_str, report_dict = evaluate_model(model, loader)

    assert isinstance(report_str, str)
    assert "Digit 0" in report_str
    assert "Digit 5" in report_str
    assert "Digit 8" in report_str
    assert "accuracy" in report_str

    assert isinstance(report_dict, dict)
    assert "Digit 0" in report_dict
    assert "Digit 5" in report_dict
    assert "Digit 8" in report_dict
    assert "accuracy" in report_dict

