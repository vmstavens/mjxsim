"""Privileged-state autoencoder compatible with ``SupervisedTrainer``."""

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
class PAE_CFG:
    """Configuration for privileged-state autoencoding."""

    privileged_dim: int = 16
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

    def to_dict(self) -> dict[str, Any]:
        self.expand()
        return dataclasses.asdict(self)

    def copy(self) -> dict[str, Any]:
        return copy.deepcopy(self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


PAE_DEFAULT_CONFIG = PAE_CFG()


def _coerce_cfg(cfg: PAE_CFG | Mapping[str, Any] | None) -> PAE_CFG:
    if cfg is None:
        result = PAE_CFG()
    elif isinstance(cfg, PAE_CFG):
        result = copy.deepcopy(cfg)
    elif isinstance(cfg, Mapping):
        result = PAE_CFG()
        for key, value in cfg.items():
            if not hasattr(result, key):
                raise ValueError(f"Invalid privileged autoencoder config key: {key}")
            setattr(result, key, copy.deepcopy(value))
    else:
        raise TypeError(
            "cfg must be a PAE_CFG, mapping, or None "
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


def _make_mlp(sizes: list[int], *, final_activation: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (in_features, out_features) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(in_features, out_features))
        if final_activation or index < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class PrivilegedAutoencoder(nn.Module):
    """MLP autoencoder for compact privileged-state embeddings."""

    def __init__(
        self,
        *,
        privileged_dim: int,
        latent_dim: int,
        hidden_dims: list[int] | tuple[int, ...] = (128, 64),
    ):
        super().__init__()
        if privileged_dim < 1:
            raise ValueError("privileged_dim must be positive")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        hidden = list(hidden_dims)
        self.encoder = _make_mlp([privileged_dim, *hidden, latent_dim])
        self.decoder = _make_mlp([latent_dim, *reversed(hidden), privileged_dim])

    def encode(self, privileged: torch.Tensor) -> torch.Tensor:
        return self.encoder(privileged)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, privileged: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(privileged)
        reconstruction = self.decode(latent)
        return reconstruction, latent


class PrivilegedAutoencoderAgent:
    """SupervisedTrainer adapter for privileged-state autoencoding."""

    def __init__(
        self,
        cfg: PAE_CFG | Mapping[str, Any] | None = None,
        *,
        device: str | torch.device | None = None,
    ):
        self.cfg = _coerce_cfg(cfg)
        self.device = _resolve_device(device)
        self.model = PrivilegedAutoencoder(
            privileged_dim=self.cfg.privileged_dim,
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
    def encode(self, privileged: torch.Tensor) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        latent = self.model.encode(privileged.to(self.device, dtype=torch.float32))
        if was_training:
            self.model.train()
        return latent

    @torch.no_grad()
    def reconstruct(self, privileged: torch.Tensor) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        reconstruction, _ = self.model(privileged.to(self.device, dtype=torch.float32))
        if was_training:
            self.model.train()
        return reconstruction

    def act(self, privileged: torch.Tensor) -> torch.Tensor:
        return self.encode(privileged)

    def _update(
        self,
        inputs: torch.Tensor | Mapping[str, torch.Tensor],
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        privileged = self._get_privileged_tensor(inputs, targets)
        reconstruction, _ = self.model(privileged)
        loss = self.loss_fn(reconstruction, privileged)

        if self.training and torch.is_grad_enabled():
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        return loss

    def _get_privileged_tensor(
        self,
        inputs: torch.Tensor | Mapping[str, torch.Tensor],
        targets: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(inputs, Mapping):
            value = None
            for key in (
                "privileged",
                "privileged_states",
                "privileged_obs",
                "states",
                "obs",
            ):
                if key in inputs:
                    value = inputs[key]
                    break
            if value is None:
                raise KeyError(
                    "Expected one of privileged, privileged_states, "
                    "privileged_obs, states, or obs in batch"
                )
            privileged = value
        else:
            privileged = targets if targets is not None else inputs

        privileged = privileged.to(self.device, dtype=torch.float32)
        if privileged.ndim == 1:
            privileged = privileged.unsqueeze(0)
        return privileged

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
    ) -> PrivilegedAutoencoderAgent:
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
