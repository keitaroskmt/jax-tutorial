"""CelebA dataset loading and preprocessing using PyTorch and JAX."""

import os
from pathlib import Path

import jax
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset, default_collate

DATA_DIR = Path(os.environ.get("DATA_DIR", "~/pytorch_datasets")).expanduser()


def _make_transforms(image_size: int = 128) -> torchvision.transforms.Compose:
    # CelebA images are typically (H, W) = (218, 178) in torchvision.
    # We center-crop to a square of side 178, then resize to image_size.
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop(178),  # 178x178
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.ToTensor(),  # [0,1], CHW
            torchvision.transforms.Lambda(lambda x: x * 2.0 - 1.0),  # [-1, 1]
            torchvision.transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # CHW -> HWC
        ],
    )


def _random_subset(dataset: Dataset, subset_size: int, seed: int = 0) -> Dataset:
    """Return a deterministic random subset of the dataset."""
    n = len(dataset)
    if subset_size <= 0:
        msg = f"subset_size must be > 0, got {subset_size}"
        raise ValueError(msg)
    if subset_size > n:
        msg = f"subset_size={subset_size} exceeds dataset size={n}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)[:subset_size].tolist()
    return Subset(dataset, indices)


def get_datasets(
    image_size: int = 128,
    train_subset_size: int | None = None,
    test_subset_size: int | None = None,
    subset_seed: int = 0,
) -> tuple[Dataset, Dataset]:
    """Return CelebA train and test datasets."""
    transform = _make_transforms(image_size=image_size)

    train_dataset = torchvision.datasets.CelebA(
        root=DATA_DIR,
        split="train",
        download=True,
        transform=transform,
        target_type="attr",
    )
    test_dataset = torchvision.datasets.CelebA(
        root=DATA_DIR,
        split="test",
        download=True,
        transform=transform,
        target_type="attr",
    )

    if train_subset_size is not None:
        train_dataset = _random_subset(
            train_dataset,
            train_subset_size,
            seed=subset_seed,
        )

    if test_subset_size is not None:
        # use a different seed stream to avoid overlapping indices by accident
        test_dataset = _random_subset(
            test_dataset,
            test_subset_size,
            seed=subset_seed + 1,
        )

    return train_dataset, test_dataset


def revert_transform(x: jax.Array) -> jax.Array:
    """Revert the normalization transform to get pixel values in [0, 1]."""
    return (x + 1.0) / 2.0


def collate_fn(batch: list):  # noqa: ANN201
    """Convert batch data to JAX arrays."""
    return jax.tree_util.tree_map(np.asarray, default_collate(batch))


def get_loaders(
    batch_size: int,
    image_size: int = 128,
    train_subset_size: int | None = 50000,
    test_subset_size: int | None = 2000,
    subset_seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create DataLoader for CelebA train and test datasets."""
    train_dataset, test_dataset = get_datasets(
        image_size=image_size,
        train_subset_size=train_subset_size,
        test_subset_size=test_subset_size,
        subset_seed=subset_seed,
    )

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
