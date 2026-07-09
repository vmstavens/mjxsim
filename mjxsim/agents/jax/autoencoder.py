"""JAX/Flax state autoencoder agents compatible with JAX supervised training."""

from __future__ import annotations

import copy
import dataclasses
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jp
import optax


@dataclasses.dataclass(kw_only=True)
class AE_CFG:
    """Configuration for deterministic state autoencoding."""

    state_dim: int = 16
    latent_dim: int = 8
    hidden_dims: list[int] = dataclasses.field(default_factory=lambda: [128, 64])
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    loss: str = "mse"
    checkpoint_path: str = ""

    def expand(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        self.expand()
        return dataclasses.asdict(self)

    def copy(self) -> dict[str, Any]:
        return copy.deepcopy(self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


AE_DEFAULT_CONFIG = AE_CFG()


def _coerce_cfg(cfg: AE_CFG | Mapping[str, Any] | None) -> AE_CFG:
    if cfg is None:
        result = AE_CFG()
    elif isinstance(cfg, AE_CFG):
        result = copy.deepcopy(cfg)
    elif isinstance(cfg, Mapping):
        result = AE_CFG()
        for key, value in cfg.items():
            if not hasattr(result, key):
                raise ValueError(f"Invalid autoencoder config key: {key}")
            setattr(result, key, copy.deepcopy(value))
    else:
        raise TypeError(
            "cfg must be an AE_CFG, mapping, or None "
            f"(got {type(cfg).__name__})"
        )
    result.expand()
    return result


class Mlp(nn.Module):
    """Simple MLP block."""

    sizes: Sequence[int]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        for width in self.sizes[:-1]:
            x = nn.relu(nn.Dense(width)(x))
        return nn.Dense(self.sizes[-1])(x)


class Autoencoder(nn.Module):
    """MLP autoencoder for compact vector-state embeddings."""

    state_dim: int
    latent_dim: int
    hidden_dims: Sequence[int] = (128, 64)

    @nn.compact
    def encode(self, states: jax.Array) -> jax.Array:
        return Mlp([*self.hidden_dims, self.latent_dim], name="encoder")(states)

    @nn.compact
    def decode(self, latent: jax.Array) -> jax.Array:
        return Mlp([*reversed(self.hidden_dims), self.state_dim], name="decoder")(
            latent
        )

    def __call__(self, states: jax.Array) -> tuple[jax.Array, jax.Array]:
        latent = self.encode(states)
        reconstruction = self.decode(latent)
        return reconstruction, latent


class AutoencoderAgent:
    """Small JAX autoencoder wrapper exposing params and loss functions."""

    input_keys = ("states", "state", "obs", "observations", "x")

    def __init__(
        self,
        cfg: AE_CFG | Mapping[str, Any] | None = None,
        *,
        rng: jax.Array | None = None,
    ):
        self.cfg = _coerce_cfg(cfg)
        self.rng = jax.random.PRNGKey(0) if rng is None else rng
        self.model = Autoencoder(
            state_dim=self.cfg.state_dim,
            latent_dim=self.cfg.latent_dim,
            hidden_dims=tuple(self.cfg.hidden_dims),
        )
        variables = self.model.init(
            self.rng,
            jp.zeros((1, self.cfg.state_dim), dtype=jp.float32),
        )
        self.params = variables["params"]
        self.optimizer = optax.adamw(
            learning_rate=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

    def loss(self, params: Any, batch: Any, rng: jax.Array | None = None) -> jax.Array:
        del rng
        states = self._get_state_array(batch)
        reconstruction, _ = self.model.apply({"params": params}, states)
        if self.cfg.loss == "mse":
            return jp.mean(jp.square(reconstruction - states))
        if self.cfg.loss in {"l1", "mae"}:
            return jp.mean(jp.abs(reconstruction - states))
        if self.cfg.loss == "smooth_l1":
            error = jp.abs(reconstruction - states)
            return jp.mean(jp.where(error < 1.0, 0.5 * error**2, error - 0.5))
        raise ValueError(f"Unsupported loss: {self.cfg.loss}")

    def encode(self, states: Any, params: Any | None = None) -> jax.Array:
        states = self._as_2d_float_array(states)
        variables = {"params": self.params if params is None else params}
        return self.model.apply(variables, states, method=Autoencoder.encode)

    def reconstruct(self, states: Any, params: Any | None = None) -> jax.Array:
        states = self._as_2d_float_array(states)
        variables = {"params": self.params if params is None else params}
        reconstruction, _ = self.model.apply(variables, states)
        return reconstruction

    def state_dict(self) -> dict[str, Any]:
        return {
            "cfg": self.cfg.to_dict(),
            "params": jax.device_get(self.params),
        }

    def save(self, path: str | Path | None = None) -> None:
        output_value = path or self.cfg.checkpoint_path
        if not output_value:
            raise ValueError("No checkpoint path provided")
        output = Path(output_value)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as file:
            pickle.dump(self.state_dict(), file)

    def load(self, path: str | Path) -> None:
        with Path(path).open("rb") as file:
            checkpoint = pickle.load(file)
        self.params = checkpoint["params"]

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        rng: jax.Array | None = None,
    ) -> AutoencoderAgent:
        with Path(path).open("rb") as file:
            checkpoint = pickle.load(file)
        agent = cls(checkpoint["cfg"], rng=rng)
        agent.params = checkpoint["params"]
        return agent

    def _get_state_array(self, batch: Any) -> jax.Array:
        if isinstance(batch, Mapping):
            value = None
            for key in self.input_keys:
                if key in batch:
                    value = batch[key]
                    break
            if value is None:
                joined = ", ".join(self.input_keys)
                raise KeyError(f"Expected one of [{joined}] in autoencoder batch")
            return self._as_2d_float_array(value)
        if isinstance(batch, tuple):
            value = batch[1] if len(batch) > 1 and batch[1] is not None else batch[0]
            return self._as_2d_float_array(value)
        return self._as_2d_float_array(batch)

    @staticmethod
    def _as_2d_float_array(value: Any) -> jax.Array:
        array = jp.asarray(value, dtype=jp.float32)
        if array.ndim == 1:
            array = array[None, :]
        return array
