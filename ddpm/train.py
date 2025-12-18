"""Training script for denoising diffusion probabilistic models (DDPM) using JAX."""

import logging
import time
from pathlib import Path

import absl
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState

from ddpm.diffusion import Diffusion
from src.dataset.cifar10 import get_loaders
from src.model.unet import UNet

# Hyperparameters
batch_size = 128
num_epochs = 20
learning_rate = 1e-3


def create_train_state(key: jax.Array) -> TrainState:
    """Create initial TrainState for training."""
    model = UNet()
    params = model.init(
        key,
        jnp.ones((1, 32, 32, 3)),
        jnp.ones((1,)),
        train=False,
    )["params"]
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(
    state: TrainState,
    diffusion: Diffusion,
    key: jax.Array,
    x: jax.Array,
) -> tuple[TrainState, jax.Array]:
    """Perform a single training step.

    Args:
        state: Current TrainState.
        diffusion: Instance of the Diffusion class.
        key: JAX random key.
        x: Batch of input images.

    Returns:
        Updated TrainState and computed loss.

    """

    def loss_fn(params: dict) -> jax.Array:
        key_data, key_dropout = jax.random.split(key)

        def model_apply(x_t: jax.Array, t: jax.Array) -> jax.Array:
            return state.apply_fn(
                {"params": params},
                x_t,
                t,
                train=True,
                rngs={"dropout": key_dropout},
            )

        return diffusion.calc_loss(key_data, x, model_apply)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def eval_step(
    state: TrainState,
    diffusion: Diffusion,
    key: jax.Array,
    x: jax.Array,
) -> jax.Array:
    """Evaluate the model on a batch."""
    key_data, _ = jax.random.split(key)

    def model_apply(x_t: jax.Array, t: jax.Array) -> jax.Array:
        return state.apply_fn(
            {"params": state.params},
            x_t,
            t,
            train=False,
        )

    return diffusion.calc_loss(key_data, x, model_apply)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    logger = logging.getLogger(__name__)
    absl.logging.set_verbosity(absl.logging.WARNING)

    train_loader, test_loader = get_loaders(batch_size)
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    state = create_train_state(subkey)

    diffusion = Diffusion.create()

    for epoch in range(num_epochs):
        start_time = time.time()
        for data, _ in train_loader:
            key, subkey = jax.random.split(key)
            state, loss = train_step(state, diffusion, subkey, data)
        epoch_time = time.time() - start_time
        logger.info("Epoch %d, Loss: %.4f", epoch + 1, loss)
        logger.info("Epoch %d completed in %.2f seconds", epoch + 1, epoch_time)

        # Evaluation
        test_loss = 0.0
        num_samples = 0
        for data, _ in test_loader:
            key, subkey = jax.random.split(key)
            loss = eval_step(state, diffusion, subkey, data)
            test_loss += jnp.sum(loss)
            num_samples += data.shape[0]
        test_loss /= num_samples
        logger.info("Epoch %d, Test Loss: %.4f", epoch + 1, test_loss)

    # Save the trained model
    save_dir = Path("./ddpm/saved_model")
    save_dir.mkdir(parents=True, exist_ok=True)

    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    with ocp.CheckpointManager(save_dir.absolute(), options=options) as mngr:
        mngr.save(step=state.step, args=ocp.args.StandardSave(state))
        mngr.wait_until_finished()
    logger.info("Model saved to %s", save_dir)
