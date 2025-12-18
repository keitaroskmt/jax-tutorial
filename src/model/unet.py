"""U-Net architecture for image-to-image translation using JAX and Flax."""

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn


def sinusoidal_positional_encoding(timestep: jax.Array, dim: int) -> jax.Array:
    """Create sinusoidal positional encoding for the given time step.

    Args:
        timestep: Time step array, with shape (batch_size,).
        dim: Dimension of the positional encoding.

    Returns:
        Positional encoding array, with shape (batch_size, dim).

    """
    if timestep.ndim == 1:
        timestep = timestep[:, None]
    half_dim = dim // 2
    div_term = jnp.exp(
        jnp.arange(half_dim) * jnp.log(10000.0) / dim,
    )
    pos = timestep / div_term[None, :]
    return jnp.concatenate([jnp.sin(pos), jnp.cos(pos)], axis=-1)


class ResNetBlock(nn.Module):
    """Residual block with time step embedding."""

    out_channels: int
    group_norm_num_groups: int = 32
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jax.Array, t: jax.Array, *, train: bool) -> jax.Array:
        """Forward pass of the ResNet block."""
        h = x
        h = nn.silu(nn.GroupNorm(num_groups=self.group_norm_num_groups)(h))
        h = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            padding="SAME",
        )(h)

        time_emb = nn.Dense(features=self.out_channels)(t)  # [B, out_channels]
        time_emb = time_emb[:, None, None, :]  # [B, 1, 1, out_channels]
        h = h + time_emb

        h = nn.silu(nn.GroupNorm(num_groups=self.group_norm_num_groups)(h))
        h = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(h)
        h = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            padding="SAME",
        )(h)
        if x.shape[-1] != self.out_channels:
            x = nn.Conv(
                features=self.out_channels,
                kernel_size=(1, 1),
                padding="SAME",
            )(x)
        if h.shape != x.shape:
            msg = f"Shape mismatch: {h.shape} != {x.shape}"
            raise ValueError(msg)
        return h + x


class AttentionBlockWithSkipConnection(nn.Module):
    """Attention block with skip connection."""

    out_channels: int
    group_norm_num_groups: int = 32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass of the attention block with skip connection."""
        h = nn.GroupNorm(num_groups=self.group_norm_num_groups)(x)
        qkv = nn.Conv(
            features=self.out_channels * 3,
            kernel_size=(1, 1),
            padding="SAME",
        )(h)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        qk = jnp.einsum("bhwc,bHWc->bhwHW", q, k) / jnp.sqrt(q.shape[-1])
        qk = rearrange(qk, "b h w H W -> b h w (H W)")
        qk = nn.softmax(qk, axis=-1)
        qk = rearrange(qk, "b h w (H W) -> b h w H W", H=qk.shape[1])
        qkv = jnp.einsum("bhwHW,bHWc->bhwc", qk, v)
        h = nn.Conv(
            features=self.out_channels,
            kernel_size=(1, 1),
            padding="SAME",
        )(qkv)
        if h.shape != x.shape:
            msg = f"Shape mismatch: {h.shape} != {x.shape}"
            raise ValueError(msg)
        return h + x


class UNet(nn.Module):
    """U-Net architecture for image-to-image translation."""

    out_channels: int = 3
    ch_mult: tuple = (1, 2, 2, 2)
    dropout_rate: float = 0.1
    group_norm_num_groups: int = 32
    dim: int = 128
    timestep_dim: int = 512

    @nn.compact
    def __call__(self, x: jax.Array, t: jax.Array, *, train: bool) -> jax.Array:
        """Forward pass of the U-Net.

        Args:
            x: Input image tensor, with shape (batch_size, height, width, in_channels).
            t: Time step tensor, with shape (batch_size,).
            train: Boolean indicating whether in training mode.

        """
        # Time step encoding
        t = sinusoidal_positional_encoding(t, self.dim)
        t = nn.Dense(features=self.timestep_dim)(t)
        t = nn.silu(t)
        t = nn.Dense(features=self.timestep_dim)(t)

        # Down sampling
        # [32, 32] -> [16, 16] -> [8, 8] -> [4, 4] for the data with input size 32.
        # Use attention blocks at 16x16 resolution.
        x = nn.Conv(
            features=self.dim,
            kernel_size=(1, 1),
            padding="SAME",
        )(x)
        channels = [self.dim] + [self.dim * m for m in self.ch_mult]
        feature_stack = []
        for i, out_ch in enumerate(channels[1:]):
            x = ResNetBlock(
                out_channels=out_ch,
                dropout_rate=self.dropout_rate,
            )(x, t, train=train)
            feature_stack.append(x)
            if i == 1:
                x = AttentionBlockWithSkipConnection(
                    out_channels=out_ch,
                )(x)
            if i < len(self.ch_mult) - 1:
                x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")

        # Middle block
        x = ResNetBlock(
            out_channels=channels[-1],
            dropout_rate=self.dropout_rate,
        )(x, t, train=train)
        x = AttentionBlockWithSkipConnection(
            out_channels=channels[-1],
        )(x)
        x = ResNetBlock(
            out_channels=channels[-1],
            dropout_rate=self.dropout_rate,
        )(x, t, train=train)

        # Up sampling
        # [4, 4] -> [8, 8] -> [16, 16] -> [32, 32] for the data with input size 32.
        for i, in_ch in enumerate(reversed(channels[:-1])):
            x = ResNetBlock(
                out_channels=in_ch,
                dropout_rate=self.dropout_rate,
            )(jnp.concatenate([x, feature_stack.pop()], axis=-1), t, train=train)
            if i == len(self.ch_mult) - 2:
                x = AttentionBlockWithSkipConnection(
                    out_channels=in_ch,
                )(x)
            if i < len(self.ch_mult) - 1:
                x = jax.image.resize(
                    x,
                    shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]),
                    method="nearest",
                )
                x = nn.Conv(
                    features=in_ch,
                    kernel_size=(3, 3),
                    padding="SAME",
                )(x)

        if len(feature_stack) != 0:
            msg = "Feature stack is not empty after up sampling."
            raise ValueError(msg)

        x = nn.GroupNorm(num_groups=self.group_norm_num_groups)(x)
        x = nn.silu(x)
        return nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            padding="SAME",
        )(x)
