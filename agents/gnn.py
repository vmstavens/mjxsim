"""SupervisedTrainer-compatible generic GNN agent for mjxsim."""

from __future__ import annotations

import copy
import dataclasses
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from skrl.agents.torch.base import ExperimentCfg
from torch import nn


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
    experiment: ExperimentCfg | dict[str, Any] = dataclasses.field(
        default_factory=lambda: ExperimentCfg(
            directory="",
            experiment_name="",
            write_interval=1,
            checkpoint_interval=0,
            wandb=False,
            wandb_kwargs={},
        )
    )

    def expand(self) -> None:
        if isinstance(self.experiment, dict):
            self.experiment = ExperimentCfg(**self.experiment)

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
            "cfg must be a GNN_CFG, mapping, or None "
            f"(got {type(cfg).__name__})"
        )
    result.expand()
    return result


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def normalized_chain_adjacency(num_nodes: int) -> torch.Tensor:
    """A_hat = D^-1/2 (A + I) D^-1/2 for an ordered-chain graph."""
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    adjacency = torch.eye(num_nodes, dtype=torch.float32)
    if num_nodes > 1:
        idx = torch.arange(num_nodes - 1)
        adjacency[idx, idx + 1] = 1.0
        adjacency[idx + 1, idx] = 1.0
    degree = adjacency.sum(dim=1)
    inv_sqrt_degree = torch.pow(degree.clamp_min(1.0), -0.5)
    return inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return adj @ self.linear(x)


class GraphRegressionGCN(nn.Module):
    """GCN with mean graph pooling and a graph-level regression head."""

    def __init__(
        self,
        *,
        in_features: int = 2,
        out_features: int = 1,
        hidden_features: int = 64,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        layers = [GraphConvolution(in_features, hidden_features)]
        layers.extend(
            GraphConvolution(hidden_features, hidden_features)
            for _ in range(num_layers - 1)
        )
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, num_nodes, features]")
        if adj.ndim == 2:
            adj = adj.to(x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
        elif adj.ndim == 3:
            adj = adj.to(x.device)
        else:
            raise ValueError("adj must have shape [num_nodes, num_nodes] or batched")

        h = x
        for layer in self.layers:
            h = torch.relu(layer(h, adj))
            h = self.dropout(h)
        return self.head(h.mean(dim=1))


class GNNAgent:
    """Agent adapter for ``mjxsim.SupervisedTrainer``.

    The trainer calls ``_update(inputs, targets)`` for both training and
    validation. This agent only performs optimizer steps while in train mode,
    so validation under ``torch.no_grad()`` is safe.
    """

    def __init__(
        self,
        cfg: GNN_CFG | Mapping[str, Any] | None = None,
        *,
        device: str | torch.device | None = None,
    ):
        self.cfg = _coerce_cfg(cfg)
        self.device = _resolve_device(device)
        self.model = GraphRegressionGCN(
            in_features=self.cfg.in_features,
            out_features=self.cfg.out_features,
            hidden_features=self.cfg.hidden_features,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
        ).to(self.device)
        self.adj = normalized_chain_adjacency(self.cfg.num_nodes).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        self.loss_fn = self._make_loss(self.cfg.loss)
        self.training = False
        self.tracking_data: dict[str, list[float]] = {}
        self.experiment_dir = ""
        self.write_interval = 0

    @staticmethod
    def _make_loss(name: str) -> nn.Module:
        if name == "mse":
            return nn.MSELoss()
        if name in {"l1", "mae"}:
            return nn.L1Loss()
        if name == "smooth_l1":
            return nn.SmoothL1Loss()
        raise ValueError(f"Unsupported loss: {name}")

    def init(self, *, trainer_cfg: Any | None = None) -> None:
        del trainer_cfg
        experiment = self.cfg.experiment
        directory = experiment.directory or os.path.join(os.getcwd(), "runs")
        name = experiment.experiment_name or self.__class__.__name__
        self.experiment_dir = os.path.join(directory, name)
        self.write_interval = 0

    def set_mode(self, mode: str) -> None:
        if mode == "train":
            self.train()
        elif mode == "eval":
            self.eval()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def train(self) -> None:
        self.training = True
        self.model.train()

    def eval(self) -> None:
        self.training = False
        self.model.eval()

    def act(self, inputs: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(inputs.to(self.device), self.adj)

    def _update(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(self.device, dtype=torch.float32)
        targets = targets.to(self.device, dtype=torch.float32)
        if targets.ndim == 1:
            targets = targets[:, None]

        predictions = self.model(inputs, self.adj)
        loss = self.loss_fn(predictions, targets)

        if self.training and torch.is_grad_enabled():
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "cfg": self.cfg.to_dict(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "adjacency": self.adj.detach().cpu(),
        }

    def save(self, path: str | Path | None = None) -> None:
        output_value = path or self.cfg.checkpoint_path
        if not output_value:
            raise ValueError("No checkpoint path provided")
        output = Path(output_value)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def track_data(self, tag: str, value: float) -> None:
        self.tracking_data.setdefault(tag, []).append(float(value))

    def write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        del timestep, timesteps
        self.tracking_data.clear()
