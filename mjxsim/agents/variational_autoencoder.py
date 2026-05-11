"""Image and state variational autoencoder agents compatible with skrl."""

from __future__ import annotations

import copy
import dataclasses
import logging
import os
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl.agents.torch.base import Agent, AgentCfg, ExperimentCfg, Model
from skrl.models.torch import DeterministicMixin

logging.basicConfig(level=logging.INFO)
relative_path = os.path.relpath(__file__)
logger = logging.getLogger(relative_path)
logger.setLevel(logging.WARNING)
logger.propagate = False


def _resolve_device(device: str | torch.device | None = None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return device


def _batch_value(batch: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in batch:
            return batch[key]
    joined = ", ".join(keys)
    raise KeyError(f"Expected one of [{joined}] in VAE batch")


@dataclasses.dataclass(kw_only=True)
class VAEBaseCfg(AgentCfg):
    """Shared configuration for skrl-compatible VAE agents."""

    latent_dim: int = 16
    """Dimension of the latent vector."""

    beta: float = 1.0
    """Weight of the KL-divergence term in the VAE objective."""

    reconstruction_loss: str = "mse"
    """Reconstruction loss: ``"bce"`` or ``"mse"``."""

    reduction: str = "sum"
    """Loss reduction used for reconstruction and KL terms."""

    learning_rate: float = 1e-3
    """Learning rate for the VAE optimizer."""

    weight_decay: float = 0.0
    """Weight decay for the VAE optimizer."""

    batch_size: int = 128
    """Default supervised dataloader batch size."""

    num_workers: int = 0
    """Default number of supervised dataloader workers."""

    num_epochs: int = 100
    """Default number of supervised training epochs."""

    experiment: ExperimentCfg = dataclasses.field(
        default_factory=lambda: ExperimentCfg(
            write_interval=500,
            checkpoint_interval=1000,
        )
    )
    """Experiment settings."""

    def expand(self) -> None:
        super().expand()
        if self.reconstruction_loss not in {"bce", "mse"}:
            raise ValueError("reconstruction_loss must be 'bce' or 'mse'")
        if self.reduction not in {"sum", "mean"}:
            raise ValueError("reduction must be 'sum' or 'mean'")


@dataclasses.dataclass(kw_only=True)
class VAE_VISION_CFG(VAEBaseCfg):
    """Configuration for an image-based variational autoencoder agent."""

    image_shape: tuple[int, int, int] = (1, 28, 28)
    """Input image shape as ``(channels, height, width)``."""

    latent_dim: int = 2
    """Dimension of the latent vector."""

    capacity: int = 64
    """Base number of convolutional channels."""

    reconstruction_loss: str = "bce"
    """Image VAEs default to binary cross-entropy reconstruction loss."""


@dataclasses.dataclass(kw_only=True)
class VAE_STATE_CFG(VAEBaseCfg):
    """Configuration for a state/vector variational autoencoder agent."""

    state_dim: int = 0
    """Input state dimension."""

    hidden_dims: list[int] = dataclasses.field(default_factory=lambda: [256, 256])
    """Hidden layer sizes for the MLP encoder and decoder."""

    reconstruction_loss: str = "mse"
    """State VAEs default to mean-squared reconstruction loss."""

    def expand(self) -> None:
        super().expand()
        if self.state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size")


VAE_CFG = VAE_VISION_CFG
VARIATIONAL_AUTOENCODER_DEFAULT_CONFIG = VAE_VISION_CFG()
VARIATIONAL_AUTOENCODER_VISION_DEFAULT_CONFIG = VAE_VISION_CFG()
VARIATIONAL_AUTOENCODER_STATE_DEFAULT_CONFIG = VAE_STATE_CFG(state_dim=1)


class ModuleWrapper(DeterministicMixin, Model):
    """Wrap a PyTorch module as a skrl deterministic model."""

    def __init__(
        self,
        module: nn.Module,
        device: str | torch.device | None = None,
        observation_space=None,
        action_space=None,
        clip_actions: bool = False,
    ) -> None:
        self.device = _resolve_device(device)
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=self.device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self._unwrapped_module = module
        self.to(self.device)

    def compute(self, inputs: dict[str, Any], role: str):
        if "states" in inputs and "x" not in inputs:
            inputs = {**inputs, "x": inputs["states"]}
        return self._unwrapped_module.forward(**inputs), {}

    def act(self, inputs: dict[str, Any], role: str = ""):
        outputs, extras = self.compute(inputs, role=role)
        return outputs, None, extras


class ImageEncoder(nn.Module):
    """Convolutional encoder producing latent Gaussian parameters."""

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (1, 28, 28),
        latent_dim: int = 2,
        capacity: int = 64,
    ) -> None:
        super().__init__()
        channels, _, _ = image_shape
        self.conv1 = nn.Conv2d(channels, capacity, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(
            capacity, capacity * 2, kernel_size=4, stride=2, padding=1
        )

        with torch.no_grad():
            conv_out = self._forward_conv(torch.zeros(1, *image_shape))
        self.conv_shape = tuple(conv_out.shape[1:])
        self.flat_dim = int(conv_out.flatten(start_dim=1).shape[1])
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._forward_conv(x).flatten(start_dim=1)
        return self.fc_mu(x), self.fc_logvar(x)


class ImageDecoder(nn.Module):
    """Convolutional decoder mapping latent vectors back to image space."""

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (1, 28, 28),
        latent_dim: int = 2,
        capacity: int = 64,
        conv_shape: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        if conv_shape is None:
            conv_shape = (capacity * 2, image_shape[1] // 4, image_shape[2] // 4)
        channels, _, _ = image_shape
        self.conv_shape = conv_shape
        self.flat_dim = int(torch.tensor(conv_shape).prod().item())
        self.fc = nn.Linear(latent_dim, self.flat_dim)
        self.conv2 = nn.ConvTranspose2d(
            capacity * 2, capacity, kernel_size=4, stride=2, padding=1
        )
        self.conv1 = nn.ConvTranspose2d(
            capacity, channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.fc(z).view(z.size(0), *self.conv_shape)
        z = F.relu(self.conv2(z))
        return torch.sigmoid(self.conv1(z))


class MLPEncoder(nn.Module):
    """MLP encoder for vector observations."""

    def __init__(
        self, state_dim: int, latent_dim: int, hidden_dims: list[int]
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(x)
        return self.fc_mu(x), self.fc_logvar(x)


class MLPDecoder(nn.Module):
    """MLP decoder for vector observations."""

    def __init__(
        self, state_dim: int, latent_dim: int, hidden_dims: list[int]
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _BaseVariationalAutoencoder(nn.Module):
    """Base VAE module with shared reparameterization behavior."""

    config: VAEBaseCfg

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def latent_sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu


class VariationalAutoencoderVision(_BaseVariationalAutoencoder):
    """Convolutional variational autoencoder for image observations."""

    def __init__(self, config: VAE_VISION_CFG | None = None) -> None:
        super().__init__()
        self.config = VAE_VISION_CFG() if config is None else copy.deepcopy(config)
        self.config.expand()
        self.encoder = ImageEncoder(
            image_shape=self.config.image_shape,
            latent_dim=self.config.latent_dim,
            capacity=self.config.capacity,
        )
        self.decoder = ImageDecoder(
            image_shape=self.config.image_shape,
            latent_dim=self.config.latent_dim,
            capacity=self.config.capacity,
            conv_shape=self.encoder.conv_shape,
        )
        self.output_shape = self.config.image_shape


class VariationalAutoencoderState(_BaseVariationalAutoencoder):
    """MLP variational autoencoder for flat state observations."""

    def __init__(self, config: VAE_STATE_CFG) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.config.expand()
        self.encoder = MLPEncoder(
            state_dim=self.config.state_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims,
        )
        self.decoder = MLPDecoder(
            state_dim=self.config.state_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims,
        )
        self.output_shape = (self.config.state_dim,)


VariationalAutoencoder = VariationalAutoencoderVision
Encoder = ImageEncoder
Decoder = ImageDecoder


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 1.0,
    reconstruction_loss: str = "mse",
    reduction: str = "sum",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the VAE objective and detached loss components."""
    if reconstruction_loss == "bce":
        recon = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    elif reconstruction_loss == "mse":
        recon = F.mse_loss(recon_x, x, reduction=reduction)
    else:
        raise ValueError("reconstruction_loss must be 'bce' or 'mse'")

    kl_per_sample = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp(),
        dim=1,
    )
    kl = kl_per_sample.mean() if reduction == "mean" else kl_per_sample.sum()
    loss = recon + beta * kl
    return loss, {
        "reconstruction_loss": recon.detach(),
        "kl_divergence": kl.detach(),
    }


class _VAEAgent(Agent):
    """Shared skrl-compatible supervised agent for VAE modules."""

    model_cls: type[_BaseVariationalAutoencoder]
    input_keys: tuple[str, ...]

    def __init__(
        self,
        models: dict[str, Model | nn.Module] | None = None,
        device: str | torch.device | None = "cuda",
        observation_space=None,
        action_space=None,
        memory=None,
        config: VAEBaseCfg | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        if config is None:
            config = self.default_config()
        _config = copy.deepcopy(config)
        _config.expand()
        self.config = _config
        self.stats = stats
        self._learning_rate = _config.learning_rate
        self._weight_decay = _config.weight_decay
        self._beta = _config.beta
        self._reconstruction_loss = _config.reconstruction_loss
        self._reduction = _config.reduction

        if models is None:
            models = {"model": self.model_cls(_config)}
        self.models = {
            key: value
            if isinstance(value, Model)
            else ModuleWrapper(
                value,
                device=self.device,
                observation_space=observation_space,
                action_space=action_space,
            )
            for key, value in models.items()
        }
        self.model = self.models["model"]

        super().__init__(
            cfg=self.config,
            models=self.models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=self.device,
        )

        self.optimizer: torch.optim.Optimizer | None = None
        self.configure_optimizers()
        self.checkpoint_modules = {
            "vae": self.model,
            "optimizer": self.optimizer,
        }
        self.is_trained = False
        self.enable_training_mode(True)

    @classmethod
    def default_config(cls) -> VAEBaseCfg:
        raise NotImplementedError

    def init(self, trainer_cfg: dict[str, Any] | None = None) -> None:
        self.enable_training_mode(True)
        super().init(trainer_cfg=trainer_cfg)

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def update(self, *, timestep: int, timesteps: int) -> None:
        pass

    def configure_optimizers(self) -> None:
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self._learning_rate,
                weight_decay=self._weight_decay,
            )

    def _coerce_input(
        self, inputs: torch.Tensor | Mapping[str, Any], _targets: torch.Tensor | None
    ) -> torch.Tensor:
        if isinstance(inputs, Mapping):
            x = _batch_value(inputs, *self.input_keys, "x")
        else:
            x = inputs
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)

    def _input_stats(self) -> Mapping[str, Any] | None:
        if self.stats is None:
            return None
        if "obs" in self.stats:
            return self.stats["obs"]
        for key in self.input_keys:
            if key in self.stats:
                return self.stats[key]
        if "min" in self.stats and "max" in self.stats:
            return self.stats
        return None

    def _normalize_input(
        self, x: torch.Tensor, normalize_obs: bool | None = None
    ) -> torch.Tensor:
        stats = self._input_stats()
        if stats is None:
            do_normalize = False
        else:
            do_normalize = normalize_obs if normalize_obs is not None else True

        if not do_normalize:
            return x

        min_val = torch.as_tensor(stats["min"], device=x.device, dtype=x.dtype)
        max_val = torch.as_tensor(stats["max"], device=x.device, dtype=x.dtype)
        range_val = max_val - min_val
        range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
        return 2.0 * (x - min_val) / range_val - 1.0

    def _unnormalize_input(
        self, x: torch.Tensor, unnormalize: bool | None = None
    ) -> torch.Tensor:
        stats = self._input_stats()
        if stats is None:
            do_unnormalize = False
        else:
            do_unnormalize = unnormalize if unnormalize is not None else True

        if not do_unnormalize:
            return x

        min_val = torch.as_tensor(stats["min"], device=x.device, dtype=x.dtype)
        max_val = torch.as_tensor(stats["max"], device=x.device, dtype=x.dtype)
        range_val = max_val - min_val
        range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
        return 0.5 * (x + 1.0) * range_val + min_val

    def _update(
        self,
        inputs: torch.Tensor | Mapping[str, Any],
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform one supervised VAE training step."""
        x = self._coerce_input(inputs, targets)
        recon_x, mu, logvar = self.model.act({"x": x})[0]
        loss, loss_info = vae_loss(
            recon_x,
            x,
            mu,
            logvar,
            beta=self._beta,
            reconstruction_loss=self._reconstruction_loss,
            reduction=self._reduction,
        )

        if self.model.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self._last_loss_info = loss_info
        return loss.detach()

    def train_step(self, batch: Mapping[str, Any] | torch.Tensor) -> float:
        self.enable_training_mode(True)
        return float(self._update(batch).item())

    @torch.no_grad()
    def act(
        self,
        states: torch.Tensor | Mapping[str, Any],
        timestep: int = 0,
        timesteps: int = 0,
        role: str = "policy",
        normalize_obs: bool | None = None,
        unnormalize_recon: bool | None = None,
    ) -> tuple[torch.Tensor, None, dict[str, torch.Tensor]]:
        x = self._coerce_input(states, None)
        x = self._normalize_input(x, normalize_obs=normalize_obs)
        recon_x, mu, logvar = self.model.act({"x": x})[0]
        recon_x = self._unnormalize_input(recon_x, unnormalize=unnormalize_recon)
        return recon_x, None, {"mu": mu, "logvar": logvar}

    @torch.no_grad()
    def encode(
        self,
        states: torch.Tensor | Mapping[str, Any],
        normalize_obs: bool | None = None,
    ) -> torch.Tensor:
        x = self._coerce_input(states, None)
        x = self._normalize_input(x, normalize_obs=normalize_obs)
        module = self.model._unwrapped_module
        mu, _ = module.encode(x)
        return mu

    @torch.no_grad()
    def reconstruct(
        self,
        states: torch.Tensor | Mapping[str, Any],
        normalize_obs: bool | None = None,
        unnormalize_recon: bool | None = None,
    ) -> torch.Tensor:
        recon, _, _ = self.act(
            states,
            normalize_obs=normalize_obs,
            unnormalize_recon=unnormalize_recon,
        )
        return recon

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        unnormalize_sample: bool | None = None,
    ) -> torch.Tensor:
        module = self.model._unwrapped_module
        z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
        return self._unnormalize_input(
            module.decode(z),
            unnormalize=unnormalize_sample,
        )

    def eval(self) -> None:
        self.model.eval()

    def train(self) -> None:
        self.model.train()

    def set_mode(self, mode: str) -> None:
        if mode == "train":
            self.train()
        elif mode == "eval":
            self.eval()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def enable_training_mode(
        self, enabled: bool = True, *, apply_to_models: bool = False
    ) -> None:
        super().enable_training_mode(enabled, apply_to_models=False)
        if enabled:
            self.train()
        else:
            self.eval()

    def to(self, device: str | torch.device = "cuda"):
        self.device = _resolve_device(device)
        for model in self.models.values():
            model.to(self.device)
        return self

    def save(self, path: str) -> None:
        """Save model weights and configuration."""
        checkpoint = {
            "agent_class": self.__class__.__name__,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
            if self.optimizer
            else None,
            "config": self.config,
            "is_trained": self.is_trained,
            "stats": self.stats,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls,
        path: str,
        device: str | torch.device | None = None,
    ) -> "_VAEAgent":
        """Load a VAE agent from a checkpoint."""
        device = _resolve_device(device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        agent_cls = _agent_class_from_config(config)
        if cls not in {_VAEAgent, VariationalAutoencoderAgent, agent_cls}:
            raise TypeError(
                f"Checkpoint config requires {agent_cls.__name__}, "
                f"but load was called on {cls.__name__}"
            )
        models = {"model": agent_cls.model_cls(config)}
        agent = agent_cls(
            models=models,
            device=device,
            config=config,
            stats=checkpoint.get("stats"),
        )
        agent.model.load_state_dict(checkpoint["model_state_dict"])

        if checkpoint["optimizer_state_dict"]:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.is_trained = checkpoint.get("is_trained", False)
        return agent


class VariationalAutoencoderVisionAgent(_VAEAgent):
    """skrl-compatible supervised agent for image VAE training."""

    model_cls = VariationalAutoencoderVision
    input_keys = ("image", "images", "pixels", "states")

    @classmethod
    def default_config(cls) -> VAE_VISION_CFG:
        return VAE_VISION_CFG()


class VariationalAutoencoderStateAgent(_VAEAgent):
    """skrl-compatible supervised agent for state/vector VAE training."""

    model_cls = VariationalAutoencoderState
    input_keys = ("states", "state", "obs", "observations")

    @classmethod
    def default_config(cls) -> VAE_STATE_CFG:
        return VAE_STATE_CFG(state_dim=1)


def _agent_class_from_config(config: VAEBaseCfg) -> type[_VAEAgent]:
    if isinstance(config, VAE_STATE_CFG):
        return VariationalAutoencoderStateAgent
    return VariationalAutoencoderVisionAgent


VariationalAutoencoderAgent = VariationalAutoencoderVisionAgent
