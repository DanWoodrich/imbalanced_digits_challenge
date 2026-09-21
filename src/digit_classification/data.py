"""Data curation, loading, and persistence for imbalanced digit classification."""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
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

CURATED_FILENAME = "curated_dataset.pt"


def curate_imbalanced_splits(
    mnist_dataset: tv_datasets.MNIST,
    seed: int = 42,
    train_count: int = 3000,
    val_count: int = 1000,
    test_count: int = 1000,
) -> Tuple[List[Tuple[torch.Tensor, int]], Dict[str, Dict[int, List[int]]]]:
    """Curates an imbalanced subset of 5,000 images (3500 '8's, 1200 '0's, 300 '5's)

    and partitions them into reproducible train, val, and test splits using a fixed random seed.
    """
    assert train_count + val_count + test_count == TOTAL_SAMPLES

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(len(mnist_dataset), generator=generator).tolist()

    shuffled_samples: List[Tuple[torch.Tensor, int]] = [
        mnist_dataset[i] for i in shuffled_indices
    ]

    label_quota = TARGET_QUOTAS.copy()
    split_quota = {"train": train_count, "val": val_count, "test": test_count}

    split_label_indices: Dict[str, Dict[int, List[int]]] = {
        "train": {8: [], 0: [], 5: []},
        "val": {8: [], 0: [], 5: []},
        "test": {8: [], 0: [], 5: []},
    }

    rng = torch.Generator().manual_seed(seed)

    for idx, (_, label) in enumerate(shuffled_samples):
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

    return shuffled_samples, split_label_indices


def download_and_curate_data(
    data_dir: str = "./data",
    seed: int = 42,
) -> Path:
    """Downloads MNIST into a temporary workspace, curates the imbalanced dataset splits

    using seed (default: 42), and saves curated_dataset.pt to data_dir.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        mnist_dataset = tv_datasets.MNIST(
            root=tmp_dir,
            train=True,
            transform=transforms.ToTensor(),
            download=True,
        )

        shuffled_samples, splits = curate_imbalanced_splits(mnist_dataset, seed=seed)

        curated_data: Dict[str, Any] = {"metadata": {"seed": seed}}

        for split_name in ["train", "val", "test"]:
            imgs = []
            labels = []
            raw_labels = []
            for digit in [8, 0, 5]:
                for idx in splits[split_name][digit]:
                    img, raw_lbl = shuffled_samples[idx]
                    imgs.append(img)
                    labels.append(DIGIT_TO_IDX[raw_lbl])
                    raw_labels.append(raw_lbl)

            curated_data[f"{split_name}_images"] = torch.stack(imgs)
            curated_data[f"{split_name}_labels"] = torch.tensor(labels, dtype=torch.long)
            curated_data[f"{split_name}_raw_labels"] = torch.tensor(raw_labels, dtype=torch.long)

        curated_file = data_path / CURATED_FILENAME
        torch.save(curated_data, curated_file)

    return curated_file


def create_weighted_sampler(mapped_train_labels: torch.Tensor) -> WeightedRandomSampler:
    """Builds a WeightedRandomSampler that inverts class frequencies on the training split."""
    class_counts = torch.bincount(mapped_train_labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[mapped_train_labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(mapped_train_labels),
        replacement=True,
    )


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    use_weighted_sampler: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Loads datasets and returns (train_loader, val_loader, test_loader).

    Automatically creates the curated dataset if not yet present in data_dir.
    """
    curated_file = Path(data_dir) / CURATED_FILENAME
    if not curated_file.exists():
        download_and_curate_data(data_dir=data_dir, seed=seed)

    data = torch.load(curated_file, map_location="cpu")

    train_ds = TensorDataset(data["train_images"], data["train_labels"])
    val_ds = TensorDataset(data["val_images"], data["val_labels"])
    test_ds = TensorDataset(data["test_images"], data["test_labels"])

    if use_weighted_sampler:
        sampler = create_weighted_sampler(data["train_labels"])
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
