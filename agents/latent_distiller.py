"""Privileged latent distillation agent compatible with ``SupervisedTrainer``."""

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
class LATENT_DISTILLER_CFG:
    """Configuration for privileged latent distillation."""

    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    loss: str = "mse"
    freeze_privileged_encoder: bool = True
    checkpoint_path: str = ""
    sensor_keys: tuple[str, ...] = ("sensor", "sensors", "states", "obs", "x")
    privileged_keys: tuple[str, ...] = (
        "privileged",
        "privileged_states",
        "privileged_obs",
    )
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


LATENT_DISTILLER_DEFAULT_CONFIG = LATENT_DISTILLER_CFG()


def _coerce_cfg(
    cfg: LATENT_DISTILLER_CFG | Mapping[str, Any] | None,
) -> LATENT_DISTILLER_CFG:
    if cfg is None:
        result = LATENT_DISTILLER_CFG()
    elif isinstance(cfg, LATENT_DISTILLER_CFG):
        result = copy.deepcopy(cfg)
    elif isinstance(cfg, Mapping):
        result = LATENT_DISTILLER_CFG()
        for key, value in cfg.items():
            if not hasattr(result, key):
                raise ValueError(f"Invalid latent distiller config key: {key}")
            setattr(result, key, copy.deepcopy(value))
    else:
        raise TypeError(
            "cfg must be a LATENT_DISTILLER_CFG, mapping, or None "
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


def _make_loss(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name in {"l1", "mae"}:
        return nn.L1Loss()
    if name == "smooth_l1":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss: {name}")


def _unwrap_encoder(encoder: Any) -> nn.Module:
    if isinstance(encoder, nn.Module):
        return encoder
    if hasattr(encoder, "model"):
        model = encoder.model
        if hasattr(model, "_unwrapped_module"):
            return model._unwrapped_module
        if isinstance(model, nn.Module):
            return model
    raise TypeError(
        "encoder must be an nn.Module or an agent exposing a torch module as .model"
    )


def _encode_latent(module: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if not hasattr(module, "encode"):
        raise TypeError(f"{module.__class__.__name__} does not expose encode()")

    encoded = module.encode(inputs)
    if isinstance(encoded, tuple):
        return encoded[0]
    return encoded


class LatentDistillerAgent:
    """Distill privileged encoder latents into a deployable encoder.

    The trainable ``encoder`` receives deployable/sensor observations. The
    ``privileged_encoder`` receives privileged observations and provides the
    stop-gradient latent target, matching the PTLD latent loss.
    """

    def __init__(
        self,
        encoder: Any,
        privileged_encoder: Any,
        cfg: LATENT_DISTILLER_CFG | Mapping[str, Any] | None = None,
        *,
        device: str | torch.device | None = None,
    ):
        self.cfg = _coerce_cfg(cfg)
        self.device = _resolve_device(device)
        self.encoder = _unwrap_encoder(encoder).to(self.device)
        self.privileged_encoder = _unwrap_encoder(privileged_encoder).to(self.device)
        if self.cfg.freeze_privileged_encoder:
            for parameter in self.privileged_encoder.parameters():
                parameter.requires_grad_(False)
        trainable_parameters = (
            parameter
            for parameter in self.encoder.parameters()
            if parameter.requires_grad
        )
        self.optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        self.loss_fn = _make_loss(self.cfg.loss)
        self.training = False
        self.tracking_data: dict[str, list[float]] = {}
        self.experiment_dir = ""
        self.write_interval = 0
        self._last_loss_info: dict[str, torch.Tensor] = {}

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
        self.encoder.train()
        self.privileged_encoder.eval()

    def eval(self) -> None:
        self.training = False
        self.encoder.eval()
        self.privileged_encoder.eval()

    @torch.no_grad()
    def encode(
        self, observations: torch.Tensor | Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        was_training = self.encoder.training
        self.encoder.eval()
        latent = _encode_latent(
            self.encoder,
            self._get_tensor(observations, self.cfg.sensor_keys),
        )
        if was_training:
            self.encoder.train()
        return latent

    def act(
        self, observations: torch.Tensor | Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        return self.encode(observations)

    def _update(
        self,
        inputs: torch.Tensor | Mapping[str, torch.Tensor],
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sensor, privileged = self._get_distillation_tensors(inputs, targets)
        sensor_latent = _encode_latent(self.encoder, sensor)
        with torch.no_grad():
            privileged_latent = _encode_latent(self.privileged_encoder, privileged)

        loss = self.loss_fn(sensor_latent, privileged_latent.detach())
        if self.training and torch.is_grad_enabled():
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        self._last_loss_info = {
            "latent_loss": loss.detach(),
            "sensor_latent_norm": sensor_latent.detach().norm(dim=-1).mean(),
            "privileged_latent_norm": privileged_latent.detach().norm(dim=-1).mean(),
        }
        return loss

    def _get_distillation_tensors(
        self,
        inputs: torch.Tensor | Mapping[str, torch.Tensor],
        targets: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(inputs, Mapping):
            sensor = self._get_tensor(inputs, self.cfg.sensor_keys)
            privileged = self._get_tensor(inputs, self.cfg.privileged_keys)
        else:
            if targets is None:
                raise ValueError(
                    "LatentDistillerAgent requires privileged targets when inputs "
                    "is not a mapping"
                )
            sensor = self._coerce_tensor(inputs)
            privileged = self._coerce_tensor(targets)
        return sensor, privileged

    def _get_tensor(
        self,
        batch: torch.Tensor | Mapping[str, torch.Tensor],
        keys: tuple[str, ...],
    ) -> torch.Tensor:
        if not isinstance(batch, Mapping):
            return self._coerce_tensor(batch)
        for key in keys:
            if key in batch:
                return self._coerce_tensor(batch[key])
        joined = ", ".join(keys)
        raise KeyError(f"Expected one of [{joined}] in latent distiller batch")

    def _coerce_tensor(self, value: torch.Tensor) -> torch.Tensor:
        tensor = value.to(self.device, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def to(self, device: str | torch.device = "cuda") -> LatentDistillerAgent:
        self.device = _resolve_device(device)
        self.encoder.to(self.device)
        self.privileged_encoder.to(self.device)
        return self

    def state_dict(self) -> dict[str, Any]:
        return {
            "cfg": self.cfg.to_dict(),
            "encoder_state_dict": self.encoder.state_dict(),
            "privileged_encoder_state_dict": self.privileged_encoder.state_dict(),
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
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.privileged_encoder.load_state_dict(
            checkpoint["privileged_encoder_state_dict"]
        )
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def track_data(self, tag: str, value: float) -> None:
        self.tracking_data.setdefault(tag, []).append(float(value))

    def write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        del timestep, timesteps
        self.tracking_data.clear()


__all__ = [
    "LATENT_DISTILLER_CFG",
    "LATENT_DISTILLER_DEFAULT_CONFIG",
    "LatentDistillerAgent",
]
