"""Diffusion process implementation."""

from collections.abc import Callable
from typing import Self

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Diffusion:
    """Diffusion model for generative modeling."""

    diffusion_steps: int = struct.field(pytree_node=False)
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
        model_apply: Callable,
    ) -> jax.Array:
        """Sample x_{t-1} from p(x_{t-1} | x_t).

        Args:
            key: JAX random key.
            x_t: Image being restored at time step t.
            t: Time step.
            model_apply: Noise estimator, which accepts (x_t, t) as input.

        """
        beta_t = self.beta[t]
        alpha_bar_t = self.alpha_bar[t]
        noise = jax.lax.cond(
            t > 0,
            lambda k: jax.random.normal(k, x_t.shape),
            lambda _: jnp.zeros(x_t.shape),
            key,
        )
        predicted_noise = model_apply(x_t, jnp.full((x_t.shape[0],), t))

        # Equation (11) in the paper
        return (x_t - beta_t / jnp.sqrt(1 - alpha_bar_t) * predicted_noise) / jnp.sqrt(
            alpha_bar_t,
        ) + jnp.sqrt(self.beta[t]) * noise

    def reverse_diffusion(
        self,
        key: jax.Array,
        model_apply: Callable,
        shape: tuple,
    ) -> jax.Array:
        """Restore the image x_0 from noisy samples x_t using p(x_{t-1} | x_t).

        Args:
            key: JAX random key.
            diffusion_model: Diffusion model.
            model_apply: Noise estimator, which accepts (x_t, t) as input.
            shape: Shape of the output image.

        Returns:
            Reconstructed image with the given shape.

        """
        key, subkey = jax.random.split(key)

        def body_fun(
            i: int,
            val: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array]:
            key, x = val
            key, subkey = jax.random.split(key)
            x_prev = self.reverse_diffusion_one_step(
                key=subkey,
                x_t=x,
                t=self.diffusion_steps - 1 - i,  # Time step in reverse order
                model_apply=model_apply,
            )
            return key, x_prev

        return jax.lax.fori_loop(
            lower=0,
            upper=self.diffusion_steps,
            body_fun=body_fun,
            init_val=(key, jax.random.normal(subkey, shape)),
        )

    def reverse_diffusion_with_intermediate(
        self,
        key: jax.Array,
        model_apply: Callable,
        shape: tuple,
        num_intermediate: int = 10,
    ) -> tuple[jax.Array, jax.Array]:
        """Restore the image x_0 from noisy samples x_t, returning intermediate steps.

        Args:
            key: JAX random key.
            diffusion_model: Diffusion model.
            model_apply: Noise estimator, which accepts (x_t, t) as input.
            shape: Shape of the output image.
            num_intermediate: Number of intermediate steps to return.

        Returns:
            Tuple of:
            - Reconstructed image with the given shape.
            - List of reconstructed images with `num_intermediate` length.

        """
        key, subkey = jax.random.split(key)
        buf = jnp.zeros((num_intermediate, *shape))

        def body_fun(
            carry: tuple[jax.Array, jax.Array, jax.Array, int],
            t: int,
        ) -> tuple[jax.Array, jax.Array, jax.Array, int]:
            key, x, buf, k = carry
            key, subkey = jax.random.split(key)
            x_prev = self.reverse_diffusion_one_step(
                key=subkey,
                x_t=x,
                t=t,  # Time step in reverse order
                model_apply=model_apply,
            )
            update_buf = t % (self.diffusion_steps // num_intermediate) == 0
            buf = jax.lax.cond(
                update_buf,
                lambda b: b.at[k].set(x_prev),
                lambda b: b,
                buf,
            )
            k = k + update_buf.astype(jnp.int32)

            return (key, x_prev, buf, k), None

        (_, x_0, intermediate, _), _ = jax.lax.scan(
            f=body_fun,
            init=(key, jax.random.normal(subkey, shape), buf, 0),
            xs=jnp.arange(self.diffusion_steps - 1, -1, -1),
        )
        return x_0, intermediate
