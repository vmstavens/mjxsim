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
import numpy as np
import optax

from mjxsim.agents.action_normalization import (
    ActionNormalization,
    action_normalization_from_legacy_stats,
)
from mjxsim.agents.jax.action_transform import ActionTransform
from mjxsim.agents.jax.ddpm import (
    add_noise,
    inference_timesteps,
    squaredcos_cap_v2_betas,
    step as ddpm_step,
)


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
    n_groups: int = 8
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
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    variance_type: str = "fixed_small"
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
        if self.action_horizon > self.pred_horizon:
            raise ValueError("action_horizon cannot exceed pred_horizon")
        if self.num_diffusion_iters < 1:
            raise ValueError("num_diffusion_iters must be positive")
        if not self.down_dims or any(width < 1 for width in self.down_dims):
            raise ValueError("down_dims must contain positive channel widths")
        if self.n_groups < 1 or any(
            width % self.n_groups != 0 for width in self.down_dims
        ):
            raise ValueError("Every down_dims value must be divisible by n_groups")
        divisor = 2 ** (len(self.down_dims) - 1)
        if self.pred_horizon % divisor:
            raise ValueError(
                f"pred_horizon must be divisible by {divisor} for this U-Net"
            )
        if self.beta_schedule != "squaredcos_cap_v2":
            raise ValueError("Only beta_schedule='squaredcos_cap_v2' is supported")
        if self.prediction_type != "epsilon":
            raise ValueError("Only epsilon prediction is currently supported")
        if self.variance_type not in {"fixed_small", "fixed_large"}:
            raise ValueError("Unsupported DDPM variance_type")

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
    if isinstance(result.lr_scheduler_cfg, Mapping):
        result.lr_scheduler_cfg = LRSchedulerCfg(**result.lr_scheduler_cfg)
    result.expand()
    return result


def _learning_rate_schedule(config: DP_CFG) -> optax.Schedule:
    """Match the Torch backend's linear warmup followed by cosine decay."""
    warmup_steps = config.lr_scheduler_cfg.num_warmup_steps
    total_steps = config.lr_scheduler_cfg.num_training_steps
    if total_steps < 1:
        raise ValueError("num_training_steps must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("num_warmup_steps must be in [0, num_training_steps)")
    if warmup_steps == 0:
        return optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=total_steps,
        )
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0,
    )


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


def mish(x: jax.Array) -> jax.Array:
    """Mish activation used by the reference Diffusion Policy U-Net."""
    return x * jp.tanh(nn.softplus(x))


class Downsample1d(nn.Module):
    """Stride-two temporal convolution."""

    features: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Conv(
            self.features,
            kernel_size=(3,),
            strides=(2,),
            padding=((1, 1),),
            name="conv",
        )(x)


class Upsample1d(nn.Module):
    """Stride-two temporal transposed convolution."""

    features: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.ConvTranspose(
            self.features,
            kernel_size=(4,),
            strides=(2,),
            padding="SAME",
            name="conv",
        )(x)


class Conv1dBlock(nn.Module):
    """Temporal convolution followed by GroupNorm and Mish."""

    features: int
    kernel_size: int
    n_groups: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Conv(
            self.features,
            kernel_size=(self.kernel_size,),
            padding=((self.kernel_size // 2, self.kernel_size // 2),),
            name="conv",
        )(x)
        x = nn.GroupNorm(
            num_groups=self.n_groups,
            epsilon=1e-5,
            name="norm",
        )(x)
        return mish(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two convolution blocks with FiLM scale/bias conditioning."""

    features: int
    kernel_size: int
    n_groups: int

    @nn.compact
    def __call__(self, x: jax.Array, cond: jax.Array) -> jax.Array:
        residual = x
        out = Conv1dBlock(
            self.features,
            self.kernel_size,
            self.n_groups,
            name="block_0",
        )(x)
        embedding = nn.Dense(2 * self.features, name="cond_encoder")(mish(cond))
        scale, bias = jp.split(embedding, 2, axis=-1)
        out = scale[:, None, :] * out + bias[:, None, :]
        out = Conv1dBlock(
            self.features,
            self.kernel_size,
            self.n_groups,
            name="block_1",
        )(out)
        if residual.shape[-1] != self.features:
            residual = nn.Conv(
                self.features,
                kernel_size=(1,),
                padding="VALID",
                name="residual_conv",
            )(residual)
        return out + residual


class ConditionalUnet1D(nn.Module):
    """Flax port of the reference temporal conditional Diffusion Policy U-Net.

    This implementation follows the layer graph used by the Torch backend while
    keeping JAX's channels-last convention: ``(batch, horizon, channels)``.
    The block structure is adapted from the MIT-licensed Octo implementation.
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
        timestep = jp.asarray(timestep)
        if timestep.ndim == 0:
            timestep = jp.full((sample.shape[0],), timestep)
        timestep = timestep.astype(jp.float32)

        time_embedding = SinusoidalPosEmb(
            self.config.diffusion_step_embed_dim,
            name="diffusion_pos_emb",
        )(timestep)
        time_embedding = nn.Dense(
            4 * self.config.diffusion_step_embed_dim,
            name="diffusion_dense_0",
        )(time_embedding)
        time_embedding = mish(time_embedding)
        time_embedding = nn.Dense(
            self.config.diffusion_step_embed_dim,
            name="diffusion_dense_1",
        )(time_embedding)
        observation_condition = jp.reshape(
            global_cond,
            (global_cond.shape[0], -1),
        )
        condition = jp.concatenate(
            [time_embedding, observation_condition],
            axis=-1,
        )

        x = sample
        hidden: list[jax.Array] = []
        for index, features in enumerate(self.config.down_dims):
            x = ConditionalResidualBlock1D(
                features,
                self.config.kernel_size,
                self.config.n_groups,
                name=f"down_{index}_res_0",
            )(x, condition)
            x = ConditionalResidualBlock1D(
                features,
                self.config.kernel_size,
                self.config.n_groups,
                name=f"down_{index}_res_1",
            )(x, condition)
            hidden.append(x)
            if index < len(self.config.down_dims) - 1:
                x = Downsample1d(
                    features,
                    name=f"down_{index}_sample",
                )(x)

        for index in range(2):
            x = ConditionalResidualBlock1D(
                self.config.down_dims[-1],
                self.config.kernel_size,
                self.config.n_groups,
                name=f"mid_{index}",
            )(x, condition)

        for index, features in enumerate(reversed(self.config.down_dims[:-1])):
            x = jp.concatenate([x, hidden.pop()], axis=-1)
            x = ConditionalResidualBlock1D(
                features,
                self.config.kernel_size,
                self.config.n_groups,
                name=f"up_{index}_res_0",
            )(x, condition)
            x = ConditionalResidualBlock1D(
                features,
                self.config.kernel_size,
                self.config.n_groups,
                name=f"up_{index}_res_1",
            )(x, condition)
            x = Upsample1d(features, name=f"up_{index}_sample")(x)

        x = Conv1dBlock(
            self.config.down_dims[0],
            self.config.kernel_size,
            self.config.n_groups,
            name="final_block",
        )(x)
        return nn.Conv(
            self.a_dim,
            kernel_size=(1,),
            padding="VALID",
            name="final_conv",
        )(x)


class EMAModel:
    """Exponential moving average for JAX parameter pytrees."""

    def __init__(self, params: Any, power: float = 0.75):
        self.power = power
        self.shadow_params = jax.tree.map(jp.asarray, params)

    def update(self, params: Any) -> None:
        self.shadow_params = jax.tree.map(
            lambda shadow, current: self.power * shadow + (1.0 - self.power) * current,
            self.shadow_params,
            params,
        )

    def copy_to(self) -> Any:
        return copy.deepcopy(jax.device_get(self.shadow_params))


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
        action_normalization: ActionNormalization | Mapping[str, Any] | None = None,
    ):
        self.config = _coerce_cfg(config)
        self.a_dim = a_dim
        self.o_dim = o_dim
        self.rng = jax.random.PRNGKey(0) if rng is None else rng
        self.stats = None
        self.action_normalization = (
            None
            if action_normalization is None
            else ActionNormalization.coerce(action_normalization)
        )
        if self.action_normalization is not None:
            if self.action_normalization.size != a_dim:
                raise ValueError("Action normalization dimension does not match a_dim")
            self.action_transform = ActionTransform.from_normalization(
                self.action_normalization
            )
        else:
            self.action_transform = None
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
        schedule = _learning_rate_schedule(self.config)
        self.optimizer = optax.adamw(
            learning_rate=schedule,
            weight_decay=self.config.weight_decay,
        )
        self.ema = EMAModel(self.params, power=self.config.ema_power)
        self._sample_actions_jit = jax.jit(
            self._sample_actions,
            static_argnames=("num_steps",),
        )

    @property
    def betas(self) -> jax.Array:
        return squaredcos_cap_v2_betas(
            self.config.num_diffusion_iters,
        )

    @property
    def alphas_cumprod(self) -> jax.Array:
        return jp.cumprod(1.0 - self.betas)

    def loss(self, params: Any, batch: Any, rng: jax.Array) -> jax.Array:
        obs, actions = self._get_batch_arrays(batch)
        if self.action_transform is None:
            raise ValueError(
                "Diffusion Policy training requires action_normalization. Use "
                "ActionNormalization.from_bounds(...) or explicitly "
                "ActionNormalization.from_dataset(...)."
            )
        if self.stats is not None:
            obs = self._minmax_scale(obs, self.stats["obs"], inverse=False)
        actions = self.action_transform.normalize(actions)
        batch_size = actions.shape[0]
        rng_t, rng_noise = jax.random.split(rng)
        timesteps = jax.random.randint(
            rng_t,
            (batch_size,),
            minval=0,
            maxval=self.config.num_diffusion_iters,
        )
        noise = jax.random.normal(rng_noise, actions.shape)
        noisy_actions = add_noise(
            actions,
            noise,
            timesteps,
            self.alphas_cumprod,
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
        rng, initial_noise_key = jax.random.split(rng)
        actions = jax.random.normal(
            initial_noise_key,
            (batch_size, self.config.pred_horizon, self.a_dim),
        )
        alphas_cumprod = self.alphas_cumprod
        timesteps = inference_timesteps(
            self.config.num_diffusion_iters,
            num_steps,
        )
        step_ratio = self.config.num_diffusion_iters // num_steps

        def denoise(
            carry: tuple[jax.Array, jax.Array],
            timestep: jax.Array,
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            current_actions, current_rng = carry
            timesteps = jp.full((batch_size,), timestep, dtype=jp.int32)
            pred_noise = self.model.apply(
                {"params": params},
                current_actions,
                timesteps,
                obs,
            )
            current_rng, noise_key = jax.random.split(current_rng)
            noise = jax.random.normal(
                noise_key,
                current_actions.shape,
                dtype=current_actions.dtype,
            )
            previous_actions = ddpm_step(
                pred_noise,
                timestep,
                current_actions,
                alphas_cumprod,
                noise,
                previous_timestep=timestep - step_ratio,
                prediction_type=self.config.prediction_type,
                clip_sample=self.config.clip_sample,
                clip_sample_range=self.config.clip_sample_range,
                variance_type=self.config.variance_type,
            )
            return (previous_actions, current_rng), None

        (actions, _), _ = jax.lax.scan(denoise, (actions, rng), timesteps)
        return actions

    def act(
        self,
        observations: Any,
        *,
        params: Any | None = None,
        rng: jax.Array | None = None,
        num_steps: int | None = None,
        normalize_obs: bool | None = None,
        unnormalize_act: bool | None = None,
        output_domain: str | None = None,
    ) -> jax.Array:
        rng = self.rng if rng is None else rng
        sampling_params = self.ema.shadow_params if params is None else params
        obs = self._as_obs_array(observations)
        if self.stats is not None and normalize_obs is not False:
            obs = self._minmax_scale(obs, self.stats["obs"], inverse=False)
        steps = self.config.num_diffusion_iters if num_steps is None else num_steps
        if not 1 <= steps <= self.config.num_diffusion_iters:
            raise ValueError(
                "num_steps must be between 1 and configured num_diffusion_iters"
            )
        actions = jp.clip(
            self._sample_actions_jit(
                sampling_params,
                obs,
                rng,
                num_steps=steps,
            ),
            -1.0,
            1.0,
        )
        if output_domain is not None and output_domain not in (
            "normalized",
            "physical",
        ):
            raise ValueError("output_domain must be 'normalized' or 'physical'")
        if output_domain is None:
            output_domain = (
                "physical"
                if unnormalize_act is True
                or (unnormalize_act is None and self.action_transform is not None)
                else "normalized"
            )
        if output_domain == "physical":
            if self.action_transform is None:
                raise ValueError("Physical DP output requires action_normalization")
            actions = self.action_transform.denormalize(actions)
        return actions

    def state_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_version": 2,
            "architecture": "conditional_unet1d",
            "config": self.config.to_dict(),
            "a_dim": self.a_dim,
            "o_dim": self.o_dim,
            "params": jax.device_get(self.params),
            "ema_params": self.ema.copy_to(),
            "stats": self.stats,
            "normalization": {
                "schema_version": 1,
                "observation": self.stats["obs"] if self.stats is not None else None,
                "action": None
                if self.action_normalization is None
                else self.action_normalization.to_dict(),
            },
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
        if checkpoint.get("architecture") != "conditional_unet1d":
            raise ValueError(
                "This checkpoint predates the temporal JAX U-Net and is not "
                "architecture-compatible; retrain it with the current implementation."
            )
        normalization = checkpoint.get("normalization")
        policy = cls(
            a_dim=checkpoint["a_dim"],
            o_dim=checkpoint["o_dim"],
            config=checkpoint["config"],
            rng=rng,
            stats=checkpoint.get("stats"),
            action_normalization=None
            if normalization is None
            else normalization.get("action"),
        )
        policy.params = checkpoint["params"]
        policy.ema.shadow_params = checkpoint.get("ema_params", policy.params)
        return policy

    def set_training_result(
        self,
        params: Any,
        *,
        ema_params: Any | None = None,
    ) -> None:
        """Install online and EMA parameters returned by a trainer."""
        self.params = params
        self.ema.shadow_params = params if ema_params is None else ema_params

    def set_stats(self, stats: Mapping[str, Any]) -> None:
        """Attach validated training-split observation statistics.

        Old ``stats["action"]`` values are accepted as explicitly
        dataset-derived bounds for checkpoint compatibility.
        """
        if self.action_normalization is None and "action" in stats:
            self.action_normalization = action_normalization_from_legacy_stats(stats)
            self.action_transform = ActionTransform.from_normalization(
                self.action_normalization
            )
        result = {}
        for name, expected_dim in (("obs", self.o_dim),):
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

    def set_action_normalization(
        self, normalization: ActionNormalization | Mapping[str, Any]
    ) -> None:
        value = ActionNormalization.coerce(normalization)
        if value.size != self.a_dim:
            raise ValueError("Action normalization dimension does not match a_dim")
        self.action_normalization = value
        self.action_transform = ActionTransform.from_normalization(value)

    @staticmethod
    def compute_stats(
        observations: Any,
        actions: Any | None = None,
        *,
        action_mode: str | None = None,
    ) -> dict[str, dict[str, jax.Array]]:
        """Compute training-split statistics without implicit action bounds.

        Passing actions requires the explicit legacy/exploratory
        ``action_mode="dataset_minmax"`` opt-in.
        """
        observations = jp.asarray(observations, dtype=jp.float32)
        obs_axes = tuple(range(observations.ndim - 1))
        result = {
            "obs": {
                "min": jp.min(observations, axis=obs_axes),
                "max": jp.max(observations, axis=obs_axes),
            }
        }
        if actions is not None:
            if action_mode != "dataset_minmax":
                raise ValueError(
                    "Dataset-derived action bounds require action_mode='dataset_minmax'"
                )
            actions = jp.asarray(actions, dtype=jp.float32)
            action_axes = tuple(range(actions.ndim - 1))
            result["action"] = {
                "min": jp.min(actions, axis=action_axes),
                "max": jp.max(actions, axis=action_axes),
            }
        return result

    @staticmethod
    def compute_normalization(
        observations: Any,
        *,
        action_normalization: ActionNormalization | Mapping[str, Any] | None = None,
        actions: Any | None = None,
        action_mode: str | None = None,
    ) -> dict[str, Any]:
        """Build an explicit normalization contract from physical training data."""

        observations = jp.asarray(observations, dtype=jp.float32)
        obs_axes = tuple(range(observations.ndim - 1))
        if action_normalization is None:
            if action_mode != "dataset_minmax" or actions is None:
                raise ValueError(
                    "Provide fixed action_normalization or explicitly set "
                    "action_mode='dataset_minmax' with actions"
                )
            action_normalization = ActionNormalization.from_dataset(np.asarray(actions))
        else:
            action_normalization = ActionNormalization.coerce(action_normalization)
        return {
            "schema_version": 1,
            "observation": {
                "kind": "dataset_minmax",
                "min": jp.min(observations, axis=obs_axes),
                "max": jp.max(observations, axis=obs_axes),
            },
            "action": action_normalization.to_dict(),
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
