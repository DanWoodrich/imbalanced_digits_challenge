"""Tests for data curation, quotas, reproducibility against stored pregenerated reference, and weighted sampling."""

from pathlib import Path
import pytest
import torch
from torch.utils.data import Dataset
from digit_classification.data import (
    curate_imbalanced_splits,
    create_weighted_sampler,
    download_and_curate_data,
    TARGET_QUOTAS,
    TOTAL_SAMPLES,
)

REFERENCE_DATASET_PATH = Path(__file__).parent / "reference_curated_seed42.pt"

class MockMNISTDataset(Dataset):
    """Synthetic dataset simulating MNIST training data with abundant samples per class."""

    def __init__(self, count_per_class: int = 4000):
        self.samples = []
        for digit in range(10):
            for _ in range(count_per_class):
                self.samples.append((torch.zeros((1, 28, 28)), digit))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def test_curate_imbalanced_splits_quotas():
    """Verify curated dataset adheres to 3,500 '8's, 1,200 '0's, 300 '5's (5,000 total)."""
    dataset = MockMNISTDataset(count_per_class=4000)
    _, splits = curate_imbalanced_splits(
        dataset,
        seed=42,
        train_count=3000,
        val_count=1000,
        test_count=1000,
    )

    count_8 = sum(len(splits[s][8]) for s in ["train", "val", "test"])
    count_0 = sum(len(splits[s][0]) for s in ["train", "val", "test"])
    count_5 = sum(len(splits[s][5]) for s in ["train", "val", "test"])

    assert count_8 == TARGET_QUOTAS[8]  # 3,500
    assert count_0 == TARGET_QUOTAS[0]  # 1,200
    assert count_5 == TARGET_QUOTAS[5]  # 300

    total_train = sum(len(indices) for indices in splits["train"].values())
    total_val = sum(len(indices) for indices in splits["val"].values())
    total_test = sum(len(indices) for indices in splits["test"].values())

    assert total_train == 3000
    assert total_val == 1000
    assert total_test == 1000  # 20% test partition
    assert total_train + total_val + total_test == TOTAL_SAMPLES

def test_curate_imbalanced_splits_no_contamination():
    """Verify that no index appears in more than one split (train/val/test are disjoint)."""
    dataset = MockMNISTDataset(count_per_class=4000)
    _, splits = curate_imbalanced_splits(
        dataset,
        seed=42,
        train_count=3000,
        val_count=1000,
        test_count=1000,
    )

    all_train_indices = {idx for indices in splits["train"].values() for idx in indices}
    all_val_indices = {idx for indices in splits["val"].values() for idx in indices}
    all_test_indices = {idx for indices in splits["test"].values() for idx in indices}

    # Each split must be independent — no sample appears in more than one
    assert all_train_indices.isdisjoint(all_val_indices)
    assert all_val_indices.isdisjoint(all_test_indices)
    assert all_train_indices.isdisjoint(all_test_indices)

def test_matches_pregenerated_reference_dataset(tmp_path):
    """Verify that curating a new dataset with seed=42 matches the pregenerated project reference dataset."""
    assert REFERENCE_DATASET_PATH.exists(), f"Missing reference dataset: {REFERENCE_DATASET_PATH}"
    ref_data = torch.load(REFERENCE_DATASET_PATH, map_location="cpu")
    
    #download a fresh dataset and curate
    new_file = download_and_curate_data(data_dir=tmp_path, seed=42)
    new_data = torch.load(new_file, map_location="cpu")
    
    # Assert exact match across all designated splits and tensors
    for key in ["train_images", "train_labels", "val_images", "val_labels", "test_images", "test_labels"]:
        assert torch.equal(new_data[key], ref_data[key]), f"Mismatch in tensor '{key}' against pregenerated reference"

def test_create_weighted_sampler():
    """Verify WeightedRandomSampler assigns higher sampling probability to minority class (5)."""
    # Sample labels: 70 '8's (mapped to 2), 20 '0's (mapped to 0), 10 '5's (mapped to 1)
    mapped_labels = torch.tensor([2] * 70 + [0] * 20 + [1] * 10, dtype=torch.long)
    sampler = create_weighted_sampler(mapped_labels)
    weights = sampler.weights

    # Index 0 is class '8' (mapped 2), index 95 is class '5' (mapped 1)
    weight_8 = weights[0].item()
    weight_5 = weights[95].item()

    assert weight_5 > weight_8
    assert abs(weight_5 / weight_8 - 7.0) < 1e-4
