"""Training script for a simple MLP on the MNIST dataset using JAX."""

import argparse
import functools
import logging
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from src.dataset.mnist import get_loaders


# Hyperparameters
@dataclass
class Config:
    """Configuration for training."""

    batch_size: int
    num_epochs: int
    learning_rate: float
    num_classes: int = 10


def parse_args() -> Config:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer",
    )
    args = parser.parse_args()
    return Config(**vars(args))


# Model definition
def init_linear_params(
    in_size: int,
    out_size: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Initialize weights and biases for a linear layer."""
    k1, _ = jax.random.split(key)
    weights = jax.random.normal(k1, (in_size, out_size)) * jnp.sqrt(2.0 / in_size)
    biases = jnp.zeros(shape=(out_size,))
    return weights, biases


def init_mlp_params(
    layer_sizes: list[int],
    key: jax.Array,
) -> list[tuple[jax.Array, jax.Array]]:
    """Initialize parameters for an MLP given layer sizes."""
    keys = jax.random.split(key, len(layer_sizes))
    return [
        init_linear_params(in_size, out_size, k)
        for in_size, out_size, k in zip(
            layer_sizes[:-1],
            layer_sizes[1:],
            keys,
            strict=False,
        )
    ]


def mlp_forward(params: list[tuple[jax.Array, jax.Array]], x: jax.Array) -> jax.Array:
    """Forward pass through the MLP."""
    x = x.reshape((x.shape[0], -1))  # Flatten input
    for i, (w, b) in enumerate(params):
        x = jnp.dot(x, w) + b
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    return x


# Loss and accuracy functions
def cross_entropy_loss(
    logits: jax.Array,
    labels: jax.Array,
    num_classes: int,
) -> jax.Array:
    """Compute the cross-entropy loss."""
    one_hot_labels = jax.nn.one_hot(labels, num_classes=num_classes)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(one_hot_labels * log_probs, axis=1))


def compute_accuracy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Compute the accuracy."""
    predictions = jnp.argmax(logits, axis=1)
    return jnp.mean(predictions == labels)


# Training and evaluation step
@functools.partial(jax.jit, static_argnames=("num_classes",))
def train_step(
    params: list[tuple[jax.Array, jax.Array]],
    x: jax.Array,
    y: jax.Array,
    num_classes: int,
    learning_rate: float,
) -> list[tuple[jax.Array, jax.Array]]:
    """Perform a single training step."""
    grads = jax.grad(
        lambda p, x, y: cross_entropy_loss(
            mlp_forward(p, x),
            y,
            num_classes=num_classes,
        ),
    )(
        params,
        x,
        y,
    )
    return [
        (w - learning_rate * dw, b - learning_rate * db)
        for (w, b), (dw, db) in zip(params, grads, strict=False)
    ]


@functools.partial(jax.jit, static_argnames=("num_classes",))
def eval_step(
    params: list[tuple[jax.Array, jax.Array]],
    x: jax.Array,
    y: jax.Array,
    num_classes: int,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate the model on a batch."""
    logits = mlp_forward(params, x)
    loss = cross_entropy_loss(logits, y, num_classes=num_classes)
    accuracy = compute_accuracy(logits, y)
    return loss, accuracy


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    config = parse_args()
    train_loader, test_loader = get_loaders(config.batch_size)
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    params = init_mlp_params([784, 512, 512, config.num_classes], subkey)

    for epoch in range(config.num_epochs):
        start_time = time.time()
        for data, target in train_loader:
            params = train_step(
                params,
                data,
                target,
                num_classes=config.num_classes,
                learning_rate=config.learning_rate,
            )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), params)
        epoch_time = time.time() - start_time
        logger.info("Epoch %d completed in %.2f seconds", epoch + 1, epoch_time)

        # Evaluation
        test_loss = 0.0
        test_accuracy = 0.0
        num_batches = 0
        for data, labels in test_loader:
            loss, accuracy = eval_step(
                params, data, labels, num_classes=config.num_classes
            )
            test_loss += loss
            test_accuracy += accuracy
            num_batches += 1
        test_loss /= num_batches
        test_accuracy /= num_batches
        logger.info(
            "Epoch %d, Test Loss: %.4f, Test Accuracy: %.2f%%",
            epoch + 1,
            test_loss,
            test_accuracy * 100,
        )
