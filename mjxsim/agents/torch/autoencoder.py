"""State autoencoder agents compatible with ``SupervisedTrainer``."""

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
class AE_CFG:
    """Configuration for deterministic state autoencoding."""

    state_dim: int = 16
    latent_dim: int = 8
    hidden_dims: list[int] = dataclasses.field(default_factory=lambda: [128, 64])
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
            f"cfg must be an AE_CFG, mapping, or None (got {type(cfg).__name__})"
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


def _make_mlp(sizes: list[int], *, final_activation: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (in_features, out_features) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(in_features, out_features))
        if final_activation or index < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    """MLP autoencoder for compact vector-state embeddings."""

    def __init__(
        self,
        *,
        state_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (128, 64),
    ):
        super().__init__()
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        hidden = list(hidden_dims)
        self.encoder = _make_mlp([state_dim, *hidden, latent_dim])
        self.decoder = _make_mlp([latent_dim, *reversed(hidden), state_dim])

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder(states)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(states)
        reconstruction = self.decode(latent)
        return reconstruction, latent


class AutoencoderAgent:
    """SupervisedTrainer adapter for deterministic state autoencoding."""

    input_keys = ("states", "state", "obs", "observations", "x")

    def __init__(
        self,
        cfg: AE_CFG | Mapping[str, Any] | None = None,
        *,
        device: str | torch.device | None = None,
    ):
        self.cfg = _coerce_cfg(cfg)
        self.device = _resolve_device(device)
        self.model = Autoencoder(
            state_dim=self.cfg.state_dim,
            latent_dim=self.cfg.latent_dim,
            hidden_dims=self.cfg.hidden_dims,
        ).to(self.device)
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

    @torch.no_grad()
    def encode(self, states: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        latent = self.model.encode(self._get_state_tensor(states, None))
        if was_training:
            self.model.train()
        return latent

    @torch.no_grad()
    def reconstruct(
        self, states: torch.Tensor | Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        reconstruction, _ = self.model(self._get_state_tensor(states, None))
        if was_training:
            self.model.train()
        return reconstruction

    def act(self, states: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.encode(states)

    def _update(
        self,
        inputs: torch.Tensor | Mapping[str, torch.Tensor],
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        states = self._get_state_tensor(inputs, targets)
        reconstruction, _ = self.model(states)
        loss = self.loss_fn(reconstruction, states)

        if self.training and torch.is_grad_enabled():
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        return loss

    def _get_state_tensor(
        self,
        inputs: torch.Tensor | Mapping[str, torch.Tensor],
        targets: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(inputs, Mapping):
            value = None
            for key in self.input_keys:
                if key in inputs:
                    value = inputs[key]
                    break
            if value is None:
                joined = ", ".join(self.input_keys)
                raise KeyError(f"Expected one of [{joined}] in autoencoder batch")
            states = value
        else:
            states = targets if targets is not None else inputs

        states = states.to(self.device, dtype=torch.float32)
        if states.ndim == 1:
            states = states.unsqueeze(0)
        return states

    def state_dict(self) -> dict[str, Any]:
        return {
            "cfg": self.cfg.to_dict(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
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

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> AutoencoderAgent:
        checkpoint = torch.load(path, map_location=_resolve_device(device))
        agent = cls(checkpoint["cfg"], device=device)
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.eval()
        return agent

    def track_data(self, tag: str, value: float) -> None:
        self.tracking_data.setdefault(tag, []).append(float(value))

    def write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        del timestep, timesteps
        self.tracking_data.clear()
