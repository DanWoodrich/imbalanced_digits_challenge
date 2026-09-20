"""Data curation and loading for imbalanced digit classification."""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets as tv_datasets
from torchvision import transforms

# Target digit labels and mapping to 0-indexed contiguous classes (0, 1, 2)
TARGET_DIGITS: List[int] = [0, 5, 8]
DIGIT_TO_IDX: Dict[int, int] = {0: 0, 5: 1, 8: 2}
IDX_TO_DIGIT: Dict[int, int] = {0: 0, 1: 5, 2: 8}
NUM_CLASSES: int = len(TARGET_DIGITS)

# Prescribed quotas matching challenge specification
TARGET_QUOTAS: Dict[int, int] = {8: 3500, 0: 1200, 5: 300}
TOTAL_SAMPLES: int = sum(TARGET_QUOTAS.values())  # 5,000


class LabelMappedDataset(Dataset):
    """Wraps a PyTorch Dataset/Subset to remap target labels (e.g. 0, 5, 8 -> 0, 1, 2)."""

    def __init__(self, dataset: Dataset, mapping: Dict[int, int]):
        self.dataset = dataset
        self.mapping = mapping

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        return image, self.mapping[label]


def download_mnist(data_dir: str) -> tv_datasets.MNIST:
    """Downloads the MNIST training dataset if not present and returns it with ToTensor transform."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return tv_datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transforms.ToTensor(),
        download=True,
    )


def curate_imbalanced_splits(
    mnist_dataset: Dataset,
    seed: int = 42,
    train_count: int = 3000,
    val_count: int = 1000,
    test_count: int = 1000,
) -> Tuple[Dataset, Dict[str, Dict[int, List[int]]]]:
    """Curates an imbalanced subset of 5,000 images (3500 '8's, 1200 '0's, 300 '5's)

    and partitions them into reproducible train, val, and test (evaluation) splits.
    Evaluation is 20% (1,000 / 5,000).
    """
    assert train_count + val_count + test_count == TOTAL_SAMPLES, (
        f"Splits must sum to total samples ({TOTAL_SAMPLES})"
    )

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(len(mnist_dataset), generator=generator).tolist()
    shuffled_dataset = Subset(mnist_dataset, shuffled_indices)

    label_quota = TARGET_QUOTAS.copy()
    split_quota = {"train": train_count, "val": val_count, "test": test_count}

    split_label_indices: Dict[str, Dict[int, List[int]]] = {
        "train": {8: [], 0: [], 5: []},
        "val": {8: [], 0: [], 5: []},
        "test": {8: [], 0: [], 5: []},
    }

    # Deterministic assignment using PyTorch Generator with manual seed
    rng = torch.Generator().manual_seed(seed)

    for idx in range(len(shuffled_dataset)):
        _, label = shuffled_dataset[idx]
        if label in label_quota:
            active_splits = list(split_quota.keys())
            rand_idx = int(torch.randint(high=len(active_splits), size=(1,), generator=rng).item())
            chosen_split = active_splits[rand_idx]

            split_label_indices[chosen_split][label].append(idx)
            label_quota[label] -= 1
            split_quota[chosen_split] -= 1

            if split_quota[chosen_split] == 0:
                del split_quota[chosen_split]
            if label_quota[label] == 0:
                del label_quota[label]
                if not label_quota:
                    break

    return shuffled_dataset, split_label_indices


def create_weighted_sampler(
    shuffled_dataset: Dataset,
    train_indices: List[int],
    split_label_indices: Dict[str, Dict[int, List[int]]],
) -> WeightedRandomSampler:
    """Builds a WeightedRandomSampler that inverts class frequencies to balance training batches."""
    class_counts = {
        label: len(indices)
        for label, indices in split_label_indices["train"].items()
    }
    class_weights = {label: 1.0 / count for label, count in class_counts.items()}

    # Compute per-sample weight based on raw label
    sample_weights = [
        class_weights[shuffled_dataset[idx][1]]
        for idx in train_indices
    ]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_indices),
        replacement=True,
    )


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    use_weighted_sampler: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Downloads dataset, curates splits, and returns mapped DataLoader instances for (train, val, test)."""
    mnist_dataset = download_mnist(data_dir)
    shuffled_dataset, split_label_indices = curate_imbalanced_splits(mnist_dataset, seed=seed)

    train_indices = [idx for list_ in split_label_indices["train"].values() for idx in list_]
    val_indices = [idx for list_ in split_label_indices["val"].values() for idx in list_]
    test_indices = [idx for list_ in split_label_indices["test"].values() for idx in list_]

    train_subset = LabelMappedDataset(Subset(shuffled_dataset, train_indices), DIGIT_TO_IDX)
    val_subset = LabelMappedDataset(Subset(shuffled_dataset, val_indices), DIGIT_TO_IDX)
    test_subset = LabelMappedDataset(Subset(shuffled_dataset, test_indices), DIGIT_TO_IDX)

    if use_weighted_sampler:
        sampler = create_weighted_sampler(shuffled_dataset, train_indices, split_label_indices)
        train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

