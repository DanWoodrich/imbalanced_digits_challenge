"""Evaluation utilities for digit classification model."""

from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from digit_classification.data import IDX_TO_DIGIT, TARGET_DIGITS
from digit_classification.model import DigitClassifier


def evaluate_model(
    model: DigitClassifier,
    test_loader: DataLoader,
) -> Tuple[str, Dict]:
    """Runs evaluation on the test DataLoader and computes a comprehensive classification report."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())

    # Map mapped class indices (0, 1, 2) back to human-readable digit names
    target_names = [f"Digit {IDX_TO_DIGIT[i]}" for i in range(len(TARGET_DIGITS))]

    report_str = classification_report(
        all_targets,
        all_preds,
        labels=list(range(len(TARGET_DIGITS))),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        all_targets,
        all_preds,
        labels=list(range(len(TARGET_DIGITS))),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    return report_str, report_dict

