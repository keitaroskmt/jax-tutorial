"""MNIST dataset loading and preprocessing using PyTorch and JAX."""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import torchvision
from torch.utils.data import DataLoader, Dataset, default_collate

DATA_DIR = Path(os.environ.get("DATA_DIR", "~/pytorch_datasets")).expanduser()


def get_datasets() -> tuple[Dataset, Dataset]:
    """Download and return the MNIST train and test datasets."""
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ],
    )
    train_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = torchvision.datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )
    return train_dataset, test_dataset


def collate_fn(batch: list):  # noqa: ANN201
    """Convert batch data to JAX arrays."""
    return jax.tree_util.tree_map(jnp.asarray, default_collate(batch))


def get_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Create DataLoader for MNIST train and test datasets."""
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
