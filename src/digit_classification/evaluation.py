"""Evaluation utilities and custom reporting for digit classification."""

from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from digit_classification.data import DIGIT_TO_IDX, IDX_TO_DIGIT, TARGET_DIGITS
from digit_classification.model import DigitClassifier


def compute_metrics_and_matrix(
    y_true: List[int],
    y_pred: List[int],
) -> Tuple[Dict[str, Dict[str, float]], List[List[int]]]:
    """Computes per-class and overall metrics (n, accuracy, precision, recall, f1)

    along with the confusion matrix ordered by [8, 0, 5].
    """
    ordered_digits = [8, 0, 5]
    ordered_mapped_indices = [DIGIT_TO_IDX[d] for d in ordered_digits]

    total_n = len(y_true)
    overall_acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=ordered_mapped_indices, average="macro", zero_division=0
    )

    metrics_table: Dict[str, Dict[str, float]] = {
        "All": {
            "n": total_n,
            "accuracy": overall_acc,
            "precision": macro_p,
            "recall": macro_r,
            "f1": macro_f1,
        }
    }

    # Per-class metrics
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=ordered_mapped_indices, average=None, zero_division=0
    )

    for idx, digit in enumerate(ordered_digits):
        mapped_idx = ordered_mapped_indices[idx]
        class_n = int(supports[idx])
        # Per-class accuracy matching the notebook evaluation logic (class recall)
        class_acc = recalls[idx]
        metrics_table[f"Digit {digit}"] = {
            "n": class_n,
            "accuracy": class_acc,
            "precision": precisions[idx],
            "recall": recalls[idx],
            "f1": f1s[idx],
        }

    # Confusion matrix with row order [8, 0, 5] and column order [8, 0, 5]
    cm = confusion_matrix(y_true, y_pred, labels=ordered_mapped_indices).tolist()

    return metrics_table, cm


def format_evaluation_report(
    metrics_table: Dict[str, Dict[str, float]],
    confusion_mat: List[List[int]],
) -> str:
    """Formats the metrics into the requested table and text confusion matrix."""
    lines = []
    lines.append("class,n,accuracy,precision,recall,f1")
    for class_name, metrics in metrics_table.items():
        n = int(metrics["n"])
        acc = f"{metrics['accuracy']:.4f}"
        p = f"{metrics['precision']:.4f}"
        r = f"{metrics['recall']:.4f}"
        f1 = f"{metrics['f1']:.4f}"
        lines.append(f"{class_name},{n},{acc},{p},{r},{f1}")

    lines.append("")
    lines.append("Confusion Matrix:")
    header = f"{'':12} {'Pred 8':>10} {'Pred 0':>10} {'Pred 5':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    ordered_labels = ["Actual 8", "Actual 0", "Actual 5"]
    for row_name, row in zip(ordered_labels, confusion_mat):
        lines.append(f"{row_name:12} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

    return "\n".join(lines)


def evaluate_model(
    model: DigitClassifier,
    test_loader: DataLoader,
) -> Tuple[str, Dict]:
    """Runs evaluation on the test DataLoader and computes custom classification report and matrix."""
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

    metrics_table, cm = compute_metrics_and_matrix(all_targets, all_preds)
    report_str = format_evaluation_report(metrics_table, cm)

    results_dict = {
        "metrics": metrics_table,
        "confusion_matrix": cm,
    }

    return report_str, results_dict
