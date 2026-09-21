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


def test_compute_metrics_and_matrix_one_class_absent():
    """Verify compute_metrics_and_matrix handles a class that has no true samples gracefully.

    When one class is completely absent from y_true, precision_recall_fscore_support with
    zero_division=0 should return zeros for that class without raising an error.
    """
    # Only Digit_0 (mapped 0) and Digit_8 (mapped 2) are present; Digit_5 (mapped 1) is absent.
    y_true = [0, 0, 2, 2]
    y_pred = [0, 2, 2, 2]

    metrics_table, cm = compute_metrics_and_matrix(y_true, y_pred)

    # All class keys must still be present
    assert "All" in metrics_table
    assert "Digit_0" in metrics_table
    assert "Digit_5" in metrics_table
    assert "Digit_8" in metrics_table

    # Absent class: n=0 and all scores zeroed out (zero_division=0 policy)
    assert metrics_table["Digit_5"]["n"] == 0
    assert metrics_table["Digit_5"]["recall"] == pytest.approx(0.0)
    assert metrics_table["Digit_5"]["precision"] == pytest.approx(0.0)
    assert metrics_table["Digit_5"]["f1"] == pytest.approx(0.0)

    # Overall count reflects only the present samples
    assert metrics_table["All"]["n"] == 4

    # CM is still 3×3 even with an absent class
    assert len(cm) == 3
    assert all(len(row) == 3 for row in cm)

    # The absent Digit_5 row is all zeros
    digit5_row_index = 2  # ordered [8, 0, 5] → row 2 = Digit_5
    assert cm[digit5_row_index] == [0, 0, 0]


def test_compute_metrics_and_matrix_all_classes_absent():
    """Verify compute_metrics_and_matrix handles predictions where only one class appears in y_true.

    Simulates the degenerate case where the dataset contains only one digit class,
    so the other two are entirely absent. All per-class scores for missing classes
    should be 0.0 and the function must not raise.
    """
    # Only Digit_0 (mapped 0) samples exist
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]  

    metrics_table, cm = compute_metrics_and_matrix(y_true, y_pred)

    # Structure is intact
    for key in ("All", "Digit_0", "Digit_5", "Digit_8"):
        assert key in metrics_table

    # Both absent classes report n=0 and zeroed metrics
    for absent_digit in ("Digit_5", "Digit_8"):
        assert metrics_table[absent_digit]["n"] == 0
        assert metrics_table[absent_digit]["recall"] == pytest.approx(0.0)
        assert metrics_table[absent_digit]["f1"] == pytest.approx(0.0)

    # Digit_0 has 4 samples, 3 correct
    assert metrics_table["Digit_0"]["n"] == 4
    assert metrics_table["Digit_0"]["recall"] == pytest.approx(1.0)

    # CM is 3×3; rows for absent classes are all zeros
    assert len(cm) == 3
    assert cm[0] == [0, 0, 0]  # Actual Digit_8 row: no samples
    assert cm[2] == [0, 0, 0]  # Actual Digit_5 row: no samples

def test_compute_metrics_and_matrix():
    """Verify compute_metrics_and_matrix calculates metrics with requested keys and matrix shape."""
    # Synthetic targets: 6 samples (mapped labels: 0=Digit 0, 1=Digit 5, 2=Digit 8)
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 2, 2, 2]  # One mistake: one '1' predicted as '2'

    metrics_table, cm = compute_metrics_and_matrix(y_true, y_pred)

    assert "All" in metrics_table
    assert "Digit_8" in metrics_table
    assert "Digit_0" in metrics_table
    assert "Digit_5" in metrics_table

    assert metrics_table["All"]["n"] == 6
    assert metrics_table["Digit_8"]["n"] == 2
    assert metrics_table["Digit_0"]["n"] == 2
    assert metrics_table["Digit_5"]["n"] == 2

    # Confusion matrix shape must be 3x3 for classes [8, 0, 5]
    assert len(cm) == 3
    assert len(cm[0]) == 3



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
