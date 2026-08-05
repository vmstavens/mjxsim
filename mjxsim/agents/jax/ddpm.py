"""Small JAX-native DDPM scheduler matching Diffusers' leading-step semantics."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def squaredcos_cap_v2_betas(
    num_train_timesteps: int,
    *,
    max_beta: float = 0.999,
    s: float = 0.008,
) -> jax.Array:
    """Create the Glide/Diffusers squared-cosine beta schedule."""
    if num_train_timesteps < 1:
        raise ValueError("num_train_timesteps must be positive")
    steps = jnp.arange(num_train_timesteps + 1, dtype=jnp.float32)
    steps = steps / num_train_timesteps
    alpha_bar = jnp.cos((steps + s) / (1 + s) * jnp.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return jnp.clip(betas, 0, max_beta).astype(jnp.float32)


def inference_timesteps(
    num_train_timesteps: int,
    num_inference_steps: int,
) -> jax.Array:
    """Return Diffusers-compatible ``leading`` inference timesteps."""
    if not 1 <= num_inference_steps <= num_train_timesteps:
        raise ValueError(
            "num_inference_steps must be between 1 and num_train_timesteps"
        )
    step_ratio = num_train_timesteps // num_inference_steps
    return jnp.arange(num_inference_steps - 1, -1, -1, dtype=jnp.int32) * step_ratio


def add_noise(
    original_samples: jax.Array,
    noise: jax.Array,
    timesteps: jax.Array,
    alphas_cumprod: jax.Array,
) -> jax.Array:
    """Apply the DDPM forward process to a batch of samples."""
    alpha = alphas_cumprod[timesteps]
    broadcast_shape = alpha.shape + (1,) * (original_samples.ndim - timesteps.ndim)
    alpha = jnp.reshape(alpha, broadcast_shape)
    return jnp.sqrt(alpha) * original_samples + jnp.sqrt(1 - alpha) * noise


def step(
    model_output: jax.Array,
    timestep: jax.Array | int,
    sample: jax.Array,
    alphas_cumprod: jax.Array,
    noise: jax.Array,
    *,
    previous_timestep: jax.Array | int | None = None,
    prediction_type: str = "epsilon",
    clip_sample: bool = True,
    clip_sample_range: float = 1.0,
    variance_type: str = "fixed_small",
) -> jax.Array:
    """Predict ``x_(t-1)`` using the DDPM posterior.

    The formulas and defaults mirror ``diffusers.DDPMScheduler`` for epsilon
    prediction, sample clipping, and fixed-small variance.
    """
    if prediction_type != "epsilon":
        raise ValueError("Only epsilon prediction is supported")
    if variance_type not in {"fixed_small", "fixed_large"}:
        raise ValueError("variance_type must be 'fixed_small' or 'fixed_large'")

    timestep = jnp.asarray(timestep, dtype=jnp.int32)
    if previous_timestep is None:
        previous_timestep = timestep - 1
    previous_timestep = jnp.asarray(previous_timestep, dtype=jnp.int32)

    alpha_prod_t = alphas_cumprod[timestep]
    alpha_prod_t_prev = jnp.where(
        previous_timestep >= 0,
        alphas_cumprod[jnp.maximum(previous_timestep, 0)],
        jnp.asarray(1, dtype=alphas_cumprod.dtype),
    )
    current_alpha = alpha_prod_t / alpha_prod_t_prev
    current_beta = 1 - current_alpha

    predicted_original = (
        sample - jnp.sqrt(1 - alpha_prod_t) * model_output
    ) / jnp.sqrt(alpha_prod_t)
    if clip_sample:
        predicted_original = jnp.clip(
            predicted_original,
            -clip_sample_range,
            clip_sample_range,
        )

    original_coefficient = (
        jnp.sqrt(alpha_prod_t_prev) * current_beta / (1 - alpha_prod_t)
    )
    sample_coefficient = (
        jnp.sqrt(current_alpha) * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
    )
    previous_sample = (
        original_coefficient * predicted_original + sample_coefficient * sample
    )

    posterior_variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta
    variance = (
        current_beta
        if variance_type == "fixed_large"
        else jnp.clip(posterior_variance, min=1e-20)
    )
    return previous_sample + jnp.where(
        timestep > 0,
        jnp.sqrt(variance) * noise,
        jnp.zeros_like(noise),
    )


__all__ = [
    "add_noise",
    "inference_timesteps",
    "squaredcos_cap_v2_betas",
    "step",
]
