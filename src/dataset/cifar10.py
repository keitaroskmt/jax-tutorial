"""CIFAR10 dataset loading and preprocessing using PyTorch and JAX."""

import jax
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset, default_collate


def get_datasets() -> tuple[Dataset, Dataset]:
    """Download and return the CIFAR10 train and test datasets."""
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x * 2.0 - 1.0),  # Scale to [-1, 1]
            torchvision.transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # CHW to HWC
        ],
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root="~/pytorch_datasets",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="~/pytorch_datasets",
        train=False,
        download=True,
        transform=transform,
    )
    return train_dataset, test_dataset


def revert_transform(x: jax.Array) -> jax.Array:
    """Revert the normalization transform to get pixel values in [0, 1]."""
    return (x + 1.0) / 2.0


def collate_fn(batch: list):  # noqa: ANN201
    """Convert batch data to JAX arrays."""
    return jax.tree_util.tree_map(np.asarray, default_collate(batch))


def get_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Create DataLoader for CIFAR10 train and test datasets."""
    train_dataset, test_dataset = get_datasets()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, test_loader
