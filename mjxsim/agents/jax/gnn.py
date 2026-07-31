"""JAX/Flax graph neural network agent."""

from __future__ import annotations

import copy
import dataclasses
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jp
import optax


@dataclasses.dataclass(kw_only=True)
class GNN_CFG:
    """Configuration for supervised graph-level GNN regression."""

    in_features: int = 2
    out_features: int = 1
    hidden_features: int = 64
    num_nodes: int = 20
    num_layers: int = 4
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    loss: str = "mse"
    checkpoint_path: str = ""

    def expand(self) -> None:
        if self.in_features < 1:
            raise ValueError("in_features must be positive")
        if self.out_features < 1:
            raise ValueError("out_features must be positive")
        if self.hidden_features < 1:
            raise ValueError("hidden_features must be positive")
        if self.num_nodes < 1:
            raise ValueError("num_nodes must be positive")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        self.expand()
        return dataclasses.asdict(self)

    def copy(self) -> dict[str, Any]:
        return copy.deepcopy(self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


GNN_DEFAULT_CONFIG = GNN_CFG()


def _coerce_cfg(cfg: GNN_CFG | Mapping[str, Any] | None) -> GNN_CFG:
    if cfg is None:
        result = GNN_CFG()
    elif isinstance(cfg, GNN_CFG):
        result = copy.deepcopy(cfg)
    elif isinstance(cfg, Mapping):
        result = GNN_CFG()
        for key, value in cfg.items():
            if not hasattr(result, key):
                raise ValueError(f"Invalid GNN config key: {key}")
            setattr(result, key, copy.deepcopy(value))
    else:
        raise TypeError(
            f"cfg must be a GNN_CFG, mapping, or None (got {type(cfg).__name__})"
        )
    result.expand()
    return result


def normalized_chain_adjacency(num_nodes: int) -> jax.Array:
    """A_hat = D^-1/2 (A + I) D^-1/2 for an ordered-chain graph."""
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    adjacency = jp.eye(num_nodes, dtype=jp.float32)
    if num_nodes > 1:
        idx = jp.arange(num_nodes - 1)
        adjacency = adjacency.at[idx, idx + 1].set(1.0)
        adjacency = adjacency.at[idx + 1, idx].set(1.0)
    degree = jp.sum(adjacency, axis=1)
    inv_sqrt_degree = jp.power(jp.maximum(degree, 1.0), -0.5)
    return inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]


class GraphConvolution(nn.Module):
    """Graph convolution layer with pre-normalized adjacency."""

    out_features: int

    @nn.compact
    def __call__(self, x: jax.Array, adj: jax.Array) -> jax.Array:
        support = nn.Dense(self.out_features, use_bias=False)(x)
        return jp.einsum("bij,bjf->bif", adj, support)


class GraphRegressionGCN(nn.Module):
    """GCN with mean graph pooling and graph-level regression head."""

    out_features: int = 1
    hidden_features: int = 64
    num_layers: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        adj: jax.Array,
        *,
        train: bool = False,
    ) -> jax.Array:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, num_nodes, features]")
        if adj.ndim == 2:
            adj = jp.broadcast_to(adj[None, :, :], (x.shape[0], *adj.shape))
        elif adj.ndim != 3:
            raise ValueError("adj must have shape [num_nodes, num_nodes] or batched")

        h = x
        for index in range(self.num_layers):
            h = GraphConvolution(self.hidden_features, name=f"gcn_{index}")(h, adj)
            h = nn.relu(h)
            h = nn.Dropout(rate=self.dropout)(h, deterministic=not train)
        pooled = jp.mean(h, axis=1)
        return nn.Dense(self.out_features, name="head")(pooled)


class GNNAgent:
    """JAX GNN wrapper exposing params and a supervised loss."""

    def __init__(
        self,
        cfg: GNN_CFG | Mapping[str, Any] | None = None,
        *,
        rng: jax.Array | None = None,
    ):
        self.cfg = _coerce_cfg(cfg)
        self.rng = jax.random.PRNGKey(0) if rng is None else rng
        self.model = GraphRegressionGCN(
            out_features=self.cfg.out_features,
            hidden_features=self.cfg.hidden_features,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
        )
        self.adj = normalized_chain_adjacency(self.cfg.num_nodes)
        variables = self.model.init(
            self.rng,
            jp.zeros(
                (1, self.cfg.num_nodes, self.cfg.in_features),
                dtype=jp.float32,
            ),
            self.adj,
            train=False,
        )
        self.params = variables["params"]
        self.optimizer = optax.adamw(
            learning_rate=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

    def act(self, inputs: Any, params: Any | None = None) -> jax.Array:
        variables = {"params": self.params if params is None else params}
        x = jp.asarray(inputs, dtype=jp.float32)
        return self.model.apply(variables, x, self.adj, train=False)

    def loss(self, params: Any, batch: Any, rng: jax.Array | None = None) -> jax.Array:
        inputs, targets = self._get_batch_arrays(batch)
        apply_kwargs: dict[str, Any] = {}
        train = self.cfg.dropout > 0.0
        if train:
            if rng is None:
                raise ValueError("rng is required when dropout is enabled")
            apply_kwargs["rngs"] = {"dropout": rng}
        predictions = self.model.apply(
            {"params": params},
            inputs,
            self.adj,
            train=train,
            **apply_kwargs,
        )
        if targets.ndim == 1:
            targets = targets[:, None]
        if self.cfg.loss == "mse":
            return jp.mean(jp.square(predictions - targets))
        if self.cfg.loss in {"l1", "mae"}:
            return jp.mean(jp.abs(predictions - targets))
        if self.cfg.loss == "smooth_l1":
            error = jp.abs(predictions - targets)
            return jp.mean(jp.where(error < 1.0, 0.5 * error**2, error - 0.5))
        raise ValueError(f"Unsupported loss: {self.cfg.loss}")

    def state_dict(self) -> dict[str, Any]:
        return {
            "cfg": self.cfg.to_dict(),
            "params": jax.device_get(self.params),
            "adjacency": jax.device_get(self.adj),
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

    def _get_batch_arrays(self, batch: Any) -> tuple[jax.Array, jax.Array]:
        if isinstance(batch, Mapping):
            inputs = _first_present(batch, ("x", "inputs", "nodes", "states"))
            targets = _first_present(batch, ("y", "targets", "target"))
        else:
            inputs, targets = batch
        return (
            jp.asarray(inputs, dtype=jp.float32),
            jp.asarray(targets, dtype=jp.float32),
        )


def _first_present(batch: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in batch:
            return batch[key]
    joined = ", ".join(keys)
    raise KeyError(f"Expected one of [{joined}] in GNN batch")
