"""Generate samples from a trained diffusion model."""

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from PIL import Image

from ddpm.diffusion import Diffusion
from src.dataset.cifar10 import revert_transform
from src.model.unet import UNet


# Hyperparameters
@dataclass
class Config:
    """Configuration for sampling."""

    num_samples: int
    num_intermediate: int
    checkpoint_dir: Path


def parse_args() -> Config:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--num-intermediate",
        type=int,
        default=100,
        help="Number of intermediate steps to generate",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./ddpm/saved_model/unet_cifar10_lr_0.001"),
        help="Directory of the model checkpoint",
    )
    args = parser.parse_args()
    return Config(**vars(args))


def create_empty_state(key: jax.Array) -> TrainState:
    """Create an empty TrainState for loading checkpoints."""
    model = UNet()
    params = model.init(
        key,
        jnp.ones((1, 32, 32, 3)),
        jnp.ones((1,)),
        train=False,
    )["params"]
    tx = optax.adam(1e-3)  # Dummy optimizer
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def restore_state(
    checkpoint_dir: Path,
    state: TrainState,
) -> TrainState:
    """Restore TrainState from checkpoint."""
    with ocp.CheckpointManager(checkpoint_dir) as mngr:
        step = mngr.latest_step()
        return mngr.restore(step=step, args=ocp.args.StandardRestore(state))


@partial(jax.jit, static_argnames=("num_samples",))
def sample_final(
    state: TrainState,
    diffusion: Diffusion,
    key: jax.Array,
    num_samples: int,
) -> jax.Array:
    """Generate final samples from the diffusion model."""

    def model_apply(x_t: jax.Array, t: jax.Array) -> jax.Array:
        return state.apply_fn({"params": state.params}, x_t, t, train=False)

    return diffusion.reverse_diffusion(
        key=key,
        model_apply=model_apply,
        shape=(num_samples, 32, 32, 3),
    )


@partial(jax.jit, static_argnames=("num_samples", "num_intermediate"))
def sample_with_intermediate(
    state: TrainState,
    diffusion: Diffusion,
    key: jax.Array,
    num_samples: int,
    num_intermediate: int = 10,
) -> tuple[jax.Array, jax.Array]:
    """Generate samples with intermediate steps from the diffusion model.

    Returns:
        Tuple of:
        - Final reconstructed images with shape (num_samples, 32, 32, 3).
        - Intermediate reconstructed images
            with shape (num_intermediate, num_samples, 32, 32, 3).

    """

    def model_apply(x_t: jax.Array, t: jax.Array) -> jax.Array:
        return state.apply_fn({"params": state.params}, x_t, t, train=False)

    return diffusion.reverse_diffusion_with_intermediate(
        key=key,
        model_apply=model_apply,
        shape=(num_samples, 32, 32, 3),
        num_intermediate=num_intermediate,
    )


if __name__ == "__main__":
    diffusion = Diffusion.create()

    key = jax.random.key(0)
    key, subkey = jax.random.split(key)

    config = parse_args()

    state = create_empty_state(subkey)
    state = restore_state(
        config.checkpoint_dir.absolute(),
        state,
    )

    samples, intermediate = sample_with_intermediate(
        state,
        diffusion,
        key,
        config.num_samples,
        num_intermediate=config.num_intermediate,
    )

    for i in range(config.num_samples):
        for j in range(intermediate.shape[0]):
            img = revert_transform(intermediate[j, i])
            img = jnp.clip(img, 0.0, 1.0)
            img_np = np.array(img * 255.0, dtype=np.uint8)
            img_path = (
                Path("./ddpm/samples")
                / config.checkpoint_dir.name
                / f"sample_{i}_step_{j}.png"
            )
            img_path.parent.mkdir(parents=True, exist_ok=True)
            with img_path.open("wb") as f:
                Image.fromarray(img_np).save(f)
        final_img = revert_transform(samples[i])
        final_img = jnp.clip(final_img, 0.0, 1.0)
        final_img_np = np.array(final_img * 255.0, dtype=np.uint8)
        final_img_path = (
            Path("./ddpm/samples")
            / config.checkpoint_dir.name
            / f"sample_{i}_final.png"
        )
        with final_img_path.open("wb") as f:
            Image.fromarray(final_img_np).save(f)
