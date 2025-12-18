"""Diffusion process implementation."""

from collections.abc import Callable
from typing import Self

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct


@struct.dataclass
class Diffusion:
    """Diffusion model for generative modeling."""

    diffusion_steps: int
    beta: jax.Array
    alpha: jax.Array
    alpha_bar: jax.Array

    @classmethod
    def create(
        cls,
        diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> Self:
        """Initialize the Diffusion model."""
        beta = jnp.linspace(beta_start, beta_end, diffusion_steps)
        alpha = 1.0 - beta
        alpha_bar = jnp.cumprod(alpha, axis=0)
        return cls(
            diffusion_steps=diffusion_steps,
            beta=beta,
            alpha=alpha,
            alpha_bar=alpha_bar,
        )

    def calc_loss(
        self,
        key: jax.Array,
        x_0: jax.Array,
        model_apply: Callable,
    ) -> jax.Array:
        """Calculate the loss for the diffusion model.

        Args:
            key: JAX random key.
            x_0: Input image.
            model_apply: Model apply function, which accepts (x_t, t) as input.

        """
        key1, key2 = jax.random.split(key)
        noise = jax.random.normal(key1, x_0.shape)
        random_timesteps = jax.random.randint(
            key2,
            (x_0.shape[0],),
            0,
            self.diffusion_steps,
        )
        x_t = self.forward_diffusion(x_0, random_timesteps, noise)
        predicted_noise = model_apply(x_t, random_timesteps)
        return jnp.mean((noise - predicted_noise) ** 2)

    def forward_diffusion(
        self,
        x_0: jax.Array,
        timesteps: int | jax.Array,
        noise: jax.Array,
    ) -> jax.Array:
        """Create samples following the forward diffusion process, i.e., q(x_t | x_0).

        The samples are created from the original inputs x_0 and sampled noises.
        """
        if isinstance(timesteps, int):
            timesteps = jnp.full((x_0.shape[0],), timesteps)
        return (
            jnp.sqrt(self.alpha_bar[timesteps])[:, None, None, None] * x_0
            + jnp.sqrt(1 - self.alpha_bar[timesteps])[:, None, None, None] * noise
        )

    def reverse_diffusion_one_step(
        self,
        key: jax.Array,
        x_t: jax.Array,
        t: int,
        model: nn.Module,
        params: dict,
    ) -> jax.Array:
        """Sample x_{t-1} from p(x_{t-1} | x_t).

        Args:
            key: JAX random key.
            x_t: Image being restored at time step t.
            t: Time step.
            model: Noise estimator, which accepts (x_t, t) as input.
            params: Model parameters.

        """
        _, subkey = jax.random.split(key)
        beta_t = self.beta[t][:, None, None, None]
        alpha_bar_t = self.alpha_bar[t][:, None, None, None]
        noise = jax.random.normal(subkey, x_t.shape) if t > 0 else 0
        predicted_noise = model.apply(params, x_t, jnp.array([t]))
        if x_t.shape != predicted_noise.shape:
            msg = f"Shape mismatch: {x_t.shape} != {predicted_noise.shape}"
            raise ValueError(msg)

        # Equation (11) in the paper
        return (x_t - beta_t / jnp.sqrt(1 - alpha_bar_t) * predicted_noise) / jnp.sqrt(
            alpha_bar_t,
        ) + jnp.sqrt(self.beta[t]) * noise

    def reverse_diffusion(
        self,
        key: jax.Array,
        x_t: jax.Array,
        model: nn.Module,
        params: dict,
    ) -> jax.Array:
        """Restore the image x_0 from noisy samples x_t using p(x_{t-1} | x_t).

        Args:
            key: JAX random key.
            x_t: Noisy samples following the normal distribution: N(0, I), where t is
                the number of steps.
            model: Noise estimator, which accepts (x_t, t) as input.
            params: Model parameters.

        """
        for step in reversed(range(self.diffusion_steps)):
            x_t = self.reverse_diffusion_one_step(key, x_t, step, model, params)
        return x_t


def sample(
    key: jax.Array,
    diffusion_model: Diffusion,
    noise_estimator: nn.Module,
    shape: tuple,
) -> jax.Array:
    """Sample from the diffusion model.

    Args:
        key: JAX random key.
        diffusion_model: Diffusion model.
        noise_estimator: Noise estimator.
        shape: Shape of the output image.

    Returns:
        Reconstructed image with the given shape.

    """
    x_t = jax.random.normal(key, shape)
    return diffusion_model.reverse_diffusion(
        key,
        x_t,
        noise_estimator,
    )
