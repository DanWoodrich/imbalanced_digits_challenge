"""Tests for evaluation metrics, formatted reporting table, and confusion matrix."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from digit_classification.evaluation import (
    compute_metrics_and_matrix,
    format_evaluation_report,
    evaluate_model,
)
from digit_classification.model import DigitClassifier


def test_compute_metrics_and_matrix():
    """Verify compute_metrics_and_matrix calculates metrics with requested keys and matrix shape."""
    # Synthetic targets: 6 samples (mapped labels: 0=Digit 0, 1=Digit 5, 2=Digit 8)
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 2, 2, 2]  # One mistake: one '1' predicted as '2'

    metrics_table, cm = compute_metrics_and_matrix(y_true, y_pred)

    assert "All" in metrics_table
    assert "Digit 8" in metrics_table
    assert "Digit 0" in metrics_table
    assert "Digit 5" in metrics_table

    assert metrics_table["All"]["n"] == 6
    assert metrics_table["Digit 8"]["n"] == 2
    assert metrics_table["Digit 0"]["n"] == 2
    assert metrics_table["Digit 5"]["n"] == 2

    # Confusion matrix shape must be 3x3 for classes [8, 0, 5]
    assert len(cm) == 3
    assert len(cm[0]) == 3


def test_format_evaluation_report():
    """Verify formatted string contains requested CSV header, n column, and confusion matrix."""
    metrics_table = {
        "All": {"n": 1000, "accuracy": 0.992, "precision": 0.978, "recall": 0.969, "f1": 0.974},
        "Digit 8": {"n": 700, "accuracy": 0.997, "precision": 0.995, "recall": 0.997, "f1": 0.996},
        "Digit 0": {"n": 240, "accuracy": 0.995, "precision": 0.991, "recall": 0.995, "f1": 0.993},
        "Digit 5": {"n": 60, "accuracy": 0.916, "precision": 0.948, "recall": 0.916, "f1": 0.932},
    }
    cm = [
        [698, 1, 1],
        [1, 239, 0],
        [2, 3, 55],
    ]

    report = format_evaluation_report(metrics_table, cm)

    assert "class,n,accuracy,precision,recall,f1" in report
    assert "All,1000,0.9920,0.9780,0.9690,0.9740" in report
    assert "Digit 8,700,0.9970,0.9950,0.9970,0.9960" in report
    assert "Confusion Matrix:" in report
    assert "Pred 8" in report
    assert "Actual 8" in report


def test_evaluate_model():
    """Verify evaluate_model runs over a DataLoader and returns report string and dictionary."""
    model = DigitClassifier()
    model.eval()

    x = torch.randn(12, 1, 28, 28)
    y = torch.tensor([0, 1, 2] * 4, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    report_str, report_dict = evaluate_model(model, loader)

    assert isinstance(report_str, str)
    assert "class,n,accuracy,precision,recall,f1" in report_str
    assert "metrics" in report_dict
    assert "confusion_matrix" in report_dict
