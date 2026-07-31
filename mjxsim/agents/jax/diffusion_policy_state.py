"""JAX/Flax state diffusion policy components."""

from __future__ import annotations

import copy
import dataclasses
import math
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jp
import optax


@dataclasses.dataclass(kw_only=True)
class LRSchedulerCfg:
    """Configuration for diffusion policy learning-rate schedules."""

    num_warmup_steps: int = 500
    num_training_steps: int = 10_000


@dataclasses.dataclass(kw_only=True)
class DP_CFG:
    """Configuration for the JAX state diffusion policy."""

    diffusion_step_embed_dim: int = 256
    down_dims: list[int] = dataclasses.field(default_factory=lambda: [256, 512, 1024])
    kernel_size: int = 5
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    num_diffusion_iters: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    ema_power: float = 0.75
    num_epochs: int = 100
    max_steps: int = 200
    eval_frequency: int = 10
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    prediction_type: str = "epsilon"
    lr_scheduler_cfg: LRSchedulerCfg = dataclasses.field(default_factory=LRSchedulerCfg)
    checkpoint_path: str = ""

    def expand(self) -> None:
        if self.pred_horizon < 1:
            raise ValueError("pred_horizon must be positive")
        if self.obs_horizon < 1:
            raise ValueError("obs_horizon must be positive")
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if self.num_diffusion_iters < 1:
            raise ValueError("num_diffusion_iters must be positive")
        if self.prediction_type != "epsilon":
            raise ValueError("Only epsilon prediction is currently supported")

    def to_dict(self) -> dict[str, Any]:
        self.expand()
        return dataclasses.asdict(self)

    def copy(self) -> dict[str, Any]:
        return copy.deepcopy(self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


DIFFUSION_POLICY_STATE_DEFAULT_CONFIG = DP_CFG()


def _coerce_cfg(cfg: DP_CFG | Mapping[str, Any] | None) -> DP_CFG:
    if cfg is None:
        result = DP_CFG()
    elif isinstance(cfg, DP_CFG):
        result = copy.deepcopy(cfg)
    elif isinstance(cfg, Mapping):
        result = DP_CFG()
        for key, value in cfg.items():
            if not hasattr(result, key):
                raise ValueError(f"Invalid diffusion policy config key: {key}")
            setattr(result, key, copy.deepcopy(value))
    else:
        raise TypeError(
            f"cfg must be a DP_CFG, mapping, or None (got {type(cfg).__name__})"
        )
    result.expand()
    return result


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding."""

    dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        emb = jp.exp(jp.arange(half_dim, dtype=jp.float32) * -scale)
        emb = x[:, None].astype(jp.float32) * emb[None, :]
        emb = jp.concatenate([jp.sin(emb), jp.cos(emb)], axis=-1)
        if self.dim % 2:
            emb = jp.pad(emb, ((0, 0), (0, 1)))
        return emb


class Mlp(nn.Module):
    """Simple MLP block."""

    hidden_sizes: Sequence[int]
    output_size: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for width in self.hidden_sizes:
            x = nn.Dense(width)(x)
            x = x * jp.tanh(nn.softplus(x))
        return nn.Dense(self.output_size)(x)


class ConditionalUnet1D(nn.Module):
    """Compact JAX denoiser with the same call surface as a state DP U-Net.

    The first JAX implementation favors a stable supervised-training contract
    over exact architectural parity with the Torch U-Net. It consumes noisy
    action sequences, diffusion timesteps, and flattened state conditioning,
    then predicts per-action noise with shape ``(B, pred_horizon, a_dim)``.
    """

    a_dim: int
    o_dim: int
    config: DP_CFG = dataclasses.field(default_factory=DP_CFG)

    @nn.compact
    def __call__(
        self,
        sample: jax.Array,
        timestep: jax.Array,
        global_cond: jax.Array,
    ) -> jax.Array:
        if timestep.ndim == 0:
            timestep = jp.full((sample.shape[0],), timestep)
        if timestep.ndim == 1:
            timestep = timestep.astype(jp.float32)

        time_emb = SinusoidalPosEmb(self.config.diffusion_step_embed_dim)(timestep)
        cond = jp.reshape(global_cond, (global_cond.shape[0], -1))
        cond = jp.concatenate([cond, time_emb], axis=-1)
        cond = Mlp(
            (self.config.diffusion_step_embed_dim,),
            self.config.diffusion_step_embed_dim,
            name="cond_encoder",
        )(cond)
        cond = jp.repeat(cond[:, None, :], sample.shape[1], axis=1)
        x = jp.concatenate([sample, cond], axis=-1)
        return Mlp(
            tuple(self.config.down_dims),
            self.a_dim,
            name="denoiser",
        )(x)


class EMAModel:
    """Exponential moving average for JAX parameter pytrees."""

    def __init__(self, params: Any, power: float = 0.75):
        self.power = power
        self.shadow_params = copy.deepcopy(jax.device_get(params))

    def update(self, params: Any) -> None:
        self.shadow_params = jax.tree_util.tree_map(
            lambda shadow, current: self.power * shadow + (1.0 - self.power) * current,
            self.shadow_params,
            jax.device_get(params),
        )

    def copy_to(self) -> Any:
        return copy.deepcopy(self.shadow_params)


class DiffusionPolicy:
    """JAX state diffusion policy wrapper with loss and sampling helpers."""

    input_keys = ("obs", "observations", "states", "state")
    action_keys = ("action", "actions")

    def __init__(
        self,
        *,
        a_dim: int,
        o_dim: int,
        config: DP_CFG | Mapping[str, Any] | None = None,
        rng: jax.Array | None = None,
        stats: Mapping[str, Any] | None = None,
    ):
        self.config = _coerce_cfg(config)
        self.a_dim = a_dim
        self.o_dim = o_dim
        self.rng = jax.random.PRNGKey(0) if rng is None else rng
        self.stats = None
        if stats is not None:
            self.set_stats(stats)
        self.model = ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=self.config)
        variables = self.model.init(
            self.rng,
            jp.zeros((1, self.config.pred_horizon, a_dim), dtype=jp.float32),
            jp.zeros((1,), dtype=jp.int32),
            jp.zeros((1, self.config.obs_horizon, o_dim), dtype=jp.float32),
        )
        self.params = variables["params"]
        self.optimizer = optax.adamw(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.ema = EMAModel(self.params, power=self.config.ema_power)
        self._sample_actions_jit = jax.jit(
            self._sample_actions,
            static_argnames=("num_steps",),
        )

    @property
    def betas(self) -> jax.Array:
        return jp.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.config.num_diffusion_iters,
            dtype=jp.float32,
        )

    @property
    def alphas_cumprod(self) -> jax.Array:
        return jp.cumprod(1.0 - self.betas)

    def loss(self, params: Any, batch: Any, rng: jax.Array) -> jax.Array:
        obs, actions = self._get_batch_arrays(batch)
        if self.stats is not None:
            obs = self._minmax_scale(obs, self.stats["obs"], inverse=False)
            actions = self._minmax_scale(actions, self.stats["action"], inverse=False)
        batch_size = actions.shape[0]
        rng_t, rng_noise = jax.random.split(rng)
        timesteps = jax.random.randint(
            rng_t,
            (batch_size,),
            minval=0,
            maxval=self.config.num_diffusion_iters,
        )
        noise = jax.random.normal(rng_noise, actions.shape)
        alpha = self.alphas_cumprod[timesteps]
        noisy_actions = (
            jp.sqrt(alpha)[:, None, None] * actions
            + jp.sqrt(1.0 - alpha)[:, None, None] * noise
        )
        pred = self.model.apply({"params": params}, noisy_actions, timesteps, obs)
        return jp.mean(jp.square(pred - noise))

    def _sample_actions(
        self,
        params: Any,
        obs: jax.Array,
        rng: jax.Array,
        *,
        num_steps: int,
    ) -> jax.Array:
        """Sample an action plan in one compiled denoising program."""

        batch_size = obs.shape[0]
        actions = jax.random.normal(
            rng,
            (batch_size, self.config.pred_horizon, self.a_dim),
        )
        alphas_cumprod = self.alphas_cumprod

        def denoise(index: int, current_actions: jax.Array) -> jax.Array:
            timestep = num_steps - index - 1
            timesteps = jp.full((batch_size,), timestep, dtype=jp.int32)
            pred_noise = self.model.apply(
                {"params": params},
                current_actions,
                timesteps,
                obs,
            )
            alpha = alphas_cumprod[timestep]
            return (current_actions - jp.sqrt(1.0 - alpha) * pred_noise) / jp.sqrt(
                alpha
            )

        return jax.lax.fori_loop(0, num_steps, denoise, actions)

    def act(
        self,
        observations: Any,
        *,
        params: Any | None = None,
        rng: jax.Array | None = None,
        num_steps: int | None = None,
        normalize_obs: bool | None = None,
        unnormalize_act: bool | None = None,
    ) -> jax.Array:
        rng = self.rng if rng is None else rng
        variables = {"params": self.params if params is None else params}
        obs = self._as_obs_array(observations)
        if self.stats is not None and normalize_obs is not False:
            obs = self._minmax_scale(obs, self.stats["obs"], inverse=False)
        steps = self.config.num_diffusion_iters if num_steps is None else num_steps
        if not 0 <= steps <= self.config.num_diffusion_iters:
            raise ValueError(
                "num_steps must be between 0 and configured num_diffusion_iters"
            )
        actions = self._sample_actions_jit(
            variables["params"],
            obs,
            rng,
            num_steps=steps,
        )
        if self.stats is not None and unnormalize_act is not False:
            actions = self._minmax_scale(actions, self.stats["action"], inverse=True)
        return actions

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "a_dim": self.a_dim,
            "o_dim": self.o_dim,
            "params": jax.device_get(self.params),
            "ema_params": self.ema.copy_to(),
            "stats": self.stats,
        }

    def save(self, path: str | Path | None = None) -> None:
        output_value = path or self.config.checkpoint_path
        if not output_value:
            raise ValueError("No checkpoint path provided")
        output = Path(output_value)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as file:
            pickle.dump(self.state_dict(), file)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        rng: jax.Array | None = None,
    ) -> DiffusionPolicy:
        with Path(path).open("rb") as file:
            checkpoint = pickle.load(file)
        policy = cls(
            a_dim=checkpoint["a_dim"],
            o_dim=checkpoint["o_dim"],
            config=checkpoint["config"],
            rng=rng,
            stats=checkpoint.get("stats"),
        )
        policy.params = checkpoint["params"]
        policy.ema.shadow_params = checkpoint.get("ema_params", policy.params)
        return policy

    def set_stats(self, stats: Mapping[str, Any]) -> None:
        """Attach validated training-split observation and action statistics."""
        result = {}
        for name, expected_dim in (("obs", self.o_dim), ("action", self.a_dim)):
            if name not in stats:
                raise KeyError(f"Missing diffusion statistics: {name}")
            item = stats[name]
            if "min" not in item or "max" not in item:
                raise KeyError(f"{name} statistics require 'min' and 'max'")
            minimum = jp.asarray(item["min"], dtype=jp.float32)
            maximum = jp.asarray(item["max"], dtype=jp.float32)
            if minimum.shape[-1] != expected_dim or maximum.shape != minimum.shape:
                raise ValueError(f"Invalid {name} statistics shape")
            if not bool(jp.all(jp.isfinite(minimum)) and jp.all(jp.isfinite(maximum))):
                raise ValueError(f"{name} statistics contain non-finite values")
            result[name] = {"min": minimum, "max": maximum}
        self.stats = result

    @staticmethod
    def compute_stats(
        observations: Any, actions: Any
    ) -> dict[str, dict[str, jax.Array]]:
        """Compute min/max normalization statistics from a training split."""
        observations = jp.asarray(observations, dtype=jp.float32)
        actions = jp.asarray(actions, dtype=jp.float32)
        obs_axes = tuple(range(observations.ndim - 1))
        action_axes = tuple(range(actions.ndim - 1))
        return {
            "obs": {
                "min": jp.min(observations, axis=obs_axes),
                "max": jp.max(observations, axis=obs_axes),
            },
            "action": {
                "min": jp.min(actions, axis=action_axes),
                "max": jp.max(actions, axis=action_axes),
            },
        }

    @staticmethod
    def _minmax_scale(value: jax.Array, stats: Mapping[str, Any], *, inverse: bool):
        minimum = jp.asarray(stats["min"], dtype=value.dtype)
        maximum = jp.asarray(stats["max"], dtype=value.dtype)
        scale = jp.where(maximum > minimum, maximum - minimum, 1)
        if inverse:
            return 0.5 * (value + 1) * scale + minimum
        return 2 * (value - minimum) / scale - 1

    def _get_batch_arrays(self, batch: Any) -> tuple[jax.Array, jax.Array]:
        if not isinstance(batch, Mapping):
            obs, actions = batch
            return self._as_obs_array(obs), self._as_action_array(actions)
        obs = self._first_present(batch, self.input_keys)
        actions = self._first_present(batch, self.action_keys)
        return self._as_obs_array(obs), self._as_action_array(actions)

    @staticmethod
    def _first_present(batch: Mapping[str, Any], keys: Sequence[str]) -> Any:
        for key in keys:
            if key in batch:
                return batch[key]
        joined = ", ".join(keys)
        raise KeyError(f"Expected one of [{joined}] in diffusion-policy batch")

    def _as_obs_array(self, value: Any) -> jax.Array:
        obs = jp.asarray(value, dtype=jp.float32)
        if obs.ndim == 2:
            obs = obs[:, None, :]
        if obs.shape[-1] != self.o_dim:
            raise ValueError(f"Expected obs dim {self.o_dim}, got {obs.shape[-1]}")
        return obs

    def _as_action_array(self, value: Any) -> jax.Array:
        actions = jp.asarray(value, dtype=jp.float32)
        if actions.ndim == 2:
            actions = actions[:, None, :]
        if actions.shape[-1] != self.a_dim:
            raise ValueError(
                f"Expected action dim {self.a_dim}, got {actions.shape[-1]}"
            )
        return actions
