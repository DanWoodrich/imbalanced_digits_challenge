"""Tests for data curation, quotas, reproducibility, and weighted sampler."""

import pytest
import torch
from torch.utils.data import Dataset
from digit_classification.data import (
    curate_imbalanced_splits,
    create_weighted_sampler,
    LabelMappedDataset,
    DIGIT_TO_IDX,
    IDX_TO_DIGIT,
    TARGET_QUOTAS,
    TOTAL_SAMPLES,
)


class MockMNISTDataset(Dataset):
    """Synthetic dataset simulating MNIST training data with abundant samples per class."""

    def __init__(self, count_per_class: int = 4000):
        self.samples = []
        for digit in range(10):
            for _ in range(count_per_class):
                # (1, 28, 28) dummy image tensor, label
                self.samples.append((torch.zeros((1, 28, 28)), digit))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def test_digit_mapping():
    """Verify bidirectional digit mapping."""
    for digit, idx in DIGIT_TO_IDX.items():
        assert IDX_TO_DIGIT[idx] == digit
    assert set(DIGIT_TO_IDX.keys()) == {0, 5, 8}
    assert set(DIGIT_TO_IDX.values()) == {0, 1, 2}


def test_label_mapped_dataset():
    """Verify LabelMappedDataset maps raw labels to contiguous class indices."""
    mock_data = [(torch.zeros(1), 0), (torch.zeros(1), 5), (torch.zeros(1), 8)]
    mapped_ds = LabelMappedDataset(mock_data, DIGIT_TO_IDX)

    assert len(mapped_ds) == 3
    assert mapped_ds[0][1] == 0  # 0 -> 0
    assert mapped_ds[1][1] == 1  # 5 -> 1
    assert mapped_ds[2][1] == 2  # 8 -> 2


def test_curate_imbalanced_splits_quotas():
    """Verify curated dataset adheres to 3500 '8's, 1200 '0's, 300 '5's (5000 total)."""
    dataset = MockMNISTDataset(count_per_class=4000)
    shuffled_dataset, splits = curate_imbalanced_splits(
        dataset,
        seed=42,
        train_count=3000,
        val_count=1000,
        test_count=1000,
    )

    # Check total counts across splits for each digit
    count_8 = sum(len(splits[s][8]) for s in ["train", "val", "test"])
    count_0 = sum(len(splits[s][0]) for s in ["train", "val", "test"])
    count_5 = sum(len(splits[s][5]) for s in ["train", "val", "test"])

    assert count_8 == TARGET_QUOTAS[8]  # 3500
    assert count_0 == TARGET_QUOTAS[0]  # 1200
    assert count_5 == TARGET_QUOTAS[5]  # 300

    total_train = sum(len(indices) for indices in splits["train"].values())
    total_val = sum(len(indices) for indices in splits["val"].values())
    total_test = sum(len(indices) for indices in splits["test"].values())

    assert total_train == 3000
    assert total_val == 1000
    assert total_test == 1000  # 20% test split of 5000
    assert total_train + total_val + total_test == TOTAL_SAMPLES


def test_curate_splits_reproducibility():
    """Verify split generation is fully reproducible with seed."""
    dataset = MockMNISTDataset(count_per_class=4000)

    _, splits_1 = curate_imbalanced_splits(dataset, seed=42)
    _, splits_2 = curate_imbalanced_splits(dataset, seed=42)

    assert splits_1["train"] == splits_2["train"]
    assert splits_1["val"] == splits_2["val"]
    assert splits_1["test"] == splits_2["test"]


def test_create_weighted_sampler():
    """Verify WeightedRandomSampler assigns higher weights to the minority class (5)."""
    dataset = MockMNISTDataset(count_per_class=4000)
    shuffled_dataset, splits = curate_imbalanced_splits(dataset, seed=42)
    train_indices = [idx for list_ in splits["train"].values() for idx in list_]

    sampler = create_weighted_sampler(shuffled_dataset, train_indices, splits)
    weights = sampler.weights

    # Find sample indices corresponding to label 5 and label 8
    idx_5 = next(i for i, train_idx in enumerate(train_indices) if shuffled_dataset[train_idx][1] == 5)
    idx_8 = next(i for i, train_idx in enumerate(train_indices) if shuffled_dataset[train_idx][1] == 8)

    # Minority class 5 must have significantly higher sample weight than majority class 8
    assert weights[idx_5] > weights[idx_8]

