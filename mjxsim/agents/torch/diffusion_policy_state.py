import copy
import dataclasses
import logging
import os
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from skrl.agents.torch.base import Agent, AgentCfg, ExperimentCfg, Model
from skrl.models.torch import DeterministicMixin

from mjxsim.diffusion.torch import (
    ConditionalUnet1D,  # noqa: F401 - backwards-compatible public import
    DiffusionSampler,
    SchedulerConfig,
    build_backbone,
    build_noise_scheduler,
)

logging.basicConfig(level=logging.INFO)  # This adds a default handler
relative_path = os.path.relpath(__file__)  # Relative to current working directory
logger = logging.getLogger(relative_path)
logger.setLevel(logging.WARNING)
# Prevent propagation to root logger to avoid duplicate handling
logger.propagate = False


def _resolve_device(device: str | torch.device | None = None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return device


@dataclasses.dataclass(kw_only=True)
class LRSchedulerCfg:
    """Configuration for the diffusion policy learning rate scheduler."""

    num_warmup_steps: int = 500
    """Number of warmup steps used by the scheduler."""

    num_training_steps: int = 10_000
    """Total number of training steps used by the scheduler."""


@dataclasses.dataclass(kw_only=True)
class DP_CFG(AgentCfg):
    """Configuration for the diffusion policy state agent."""

    diffusion_step_embed_dim: int = 256
    """Dimension of the sinusoidal diffusion timestep embedding."""

    down_dims: list[int] = dataclasses.field(default_factory=lambda: [256, 512, 1024])
    """Channel dimensions for the downsampling path of the conditional U-Net."""

    kernel_size: int = 5
    """Kernel size used by the 1D convolution blocks."""

    n_groups: int = 8
    """Number of groups used by group normalization."""

    pred_horizon: int = 16
    """Number of action steps predicted by the diffusion model."""

    obs_horizon: int = 2
    """Number of observation steps used for conditioning."""

    action_horizon: int = 8
    """Number of generated action steps intended for execution."""

    num_diffusion_iters: int = 100
    """Number of diffusion training timesteps."""

    num_inference_steps: int | None = None
    """Default sampling steps. ``None`` preserves the training-timestep default."""

    scheduler_type: str = "ddpm"
    """Inference/training scheduler: ``"ddpm"`` or few-step ``"ddim"``."""

    backbone: str = "unet"
    """Denoising backbone: ``"unet"`` or the optional ``"mamba"`` backend."""

    warm_start_std: float = 0.1
    """Standard deviation around an explicitly supplied historical action chunk."""

    mamba_model_dim: int = 128
    """Token dimension used by the optional Mamba denoiser."""

    mamba_state_dim: int = 16
    """State dimension used by each Mamba block."""

    mamba_conv_dim: int = 4
    """Local convolution width internal to each Mamba block."""

    mamba_expand: int = 2
    """Expansion factor used by each Mamba block."""

    mamba_layers: int = 4
    """Number of residual Mamba blocks."""

    batch_size: int = 256
    """Batch size for supervised diffusion policy training."""

    learning_rate: float = 1e-4
    """Learning rate for the diffusion policy optimizer."""

    weight_decay: float = 1e-6
    """Weight decay for the diffusion policy optimizer."""

    ema_power: float = 0.75
    """Exponential moving average coefficient for the EMA model."""

    num_workers: int = 1
    """Number of dataloader workers used during supervised training."""

    num_epochs: int = 100
    """Default number of supervised training epochs."""

    max_steps: int = 200
    """Default maximum number of environment steps during evaluation."""

    eval_frequency: int = 10
    """Number of epochs between evaluations."""

    beta_schedule: str = "squaredcos_cap_v2"
    """Beta schedule passed to the DDPMScheduler."""

    clip_sample: bool = True
    """Whether the DDPMScheduler clips predicted samples."""

    prediction_type: str = "epsilon"
    """Prediction target used by the DDPMScheduler."""

    lr_scheduler_cfg: LRSchedulerCfg = dataclasses.field(default_factory=LRSchedulerCfg)
    """Learning rate scheduler settings."""

    experiment: ExperimentCfg = dataclasses.field(
        default_factory=lambda: ExperimentCfg(
            write_interval=500,
            checkpoint_interval=1000,
        )
    )
    """Experiment settings."""

    def expand(self) -> None:
        """Expand the base agent configuration."""
        super().expand()


DIFFUSION_POLICY_STATE_DEFAULT_CONFIG = DP_CFG()


class ModuleWrapper(DeterministicMixin, Model):
    def __init__(
        self,
        module: nn.Module,
        device: Optional[str] = None,
        observation_space=None,
        action_space=None,
        clip_actions=False,
    ):
        self.device = _resolve_device(device)
        # init base classes
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=self.device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self._unwrapped_module = module
        self.to(self.device)

    def compute(self, inputs: dict, role):
        # pick states and actions if available
        return self._unwrapped_module.forward(**inputs), {}

    def act(self, inputs: dict[str, Any], role: str = ""):
        actions, outputs = DeterministicMixin.act(self, inputs, role=role)
        return actions, outputs


class EMAModel:
    """Exponential Moving Average model wrapper."""

    def __init__(self, parameters: list[torch.Tensor], power: float = 0.75):
        self.parameters = list(parameters)
        self.power = power
        self.shadow_params = [p.clone().detach() for p in self.parameters]

    def step(self, parameters: list[torch.Tensor]):
        parameters = list(parameters)
        for i, (s_param, param) in enumerate(zip(self.shadow_params, parameters)):
            if param.requires_grad:
                if s_param.device != param.device:
                    s_param = s_param.to(param.device)
                    self.shadow_params[i] = s_param
                s_param.data = s_param.data * self.power + param.data * (1 - self.power)

    def copy_to(self, parameters: list[torch.Tensor]):
        parameters = list(parameters)
        for i, (s_param, param) in enumerate(zip(self.shadow_params, parameters)):
            if s_param.device != param.device:
                s_param = s_param.to(param.device)
                self.shadow_params[i] = s_param
            param.data.copy_(s_param.data)

    def to(self, device: str | torch.device):
        device = _resolve_device(device)
        self.shadow_params = [p.to(device) for p in self.shadow_params]
        return self


class DiffusionPolicy(Agent):
    def __init__(
        self,
        a_dim: int,
        o_dim: int,
        models: dict[str, Model],
        ema: EMAModel,
        device: str = "cuda",
        observation_space=None,
        dataloader=None,
        action_space=None,
        memory=None,
        config: DP_CFG | None = None,
        stats: Optional[dict[str, Any]] = None,
    ):
        self.device = _resolve_device(device)
        _config = DP_CFG() if config is None else copy.deepcopy(config)
        _config.expand()
        self.config: DP_CFG = _config
        self.stats = stats

        self._a_dim: int = a_dim
        self._o_dim: int = o_dim

        # Load config values
        self._num_diffusion_iters: int = _config.num_diffusion_iters
        self._num_inference_steps: int | None = getattr(
            _config, "num_inference_steps", None
        )
        self._beta_schedule: str = _config.beta_schedule
        self._clip_sample: bool = _config.clip_sample
        self._prediction_type: str = _config.prediction_type
        self._ema_power: float = _config.ema_power
        self._learning_rate: float = _config.learning_rate
        self._weight_decay: float = _config.weight_decay
        self._num_warmup_steps: int = _config.lr_scheduler_cfg.num_warmup_steps
        self._num_training_steps: int = _config.lr_scheduler_cfg.num_training_steps
        self._obs_horizon: int = _config.obs_horizon
        self._pred_horizon: int = _config.pred_horizon
        self._act_horizon: int = _config.action_horizon

        # Save models
        self.models = {
            k: ModuleWrapper(v, device=self.device) for k, v in models.items()
        }
        self.model = self.models["model"]
        self.ema_model = self.models["ema_model"]
        self.ema = ema.to(self.device)

        super().__init__(
            cfg=self.config,
            models=self.models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=self.device,
        )

        # Noise scheduler and framework-neutral sampling loop
        scheduler_config = SchedulerConfig(
            kind=getattr(_config, "scheduler_type", "ddpm"),
            num_train_timesteps=self._num_diffusion_iters,
            beta_schedule=self._beta_schedule,
            clip_sample=self._clip_sample,
            prediction_type=self._prediction_type,
        )
        self.noise_scheduler = build_noise_scheduler(scheduler_config)
        self.sampler = DiffusionSampler(self.noise_scheduler)

        # Optimizer + LR scheduler
        self.optimizer = None
        self.lr_scheduler = None
        self.configure_optimizers(dataloader=dataloader, num_epochs=_config.num_epochs)

        # Register for checkpointing
        self.checkpoint_modules = {
            "policy": self.model,
            "ema_model": self.ema_model,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
        }

        self.is_trained = False
        self.enable_training_mode(True)

    @classmethod
    def from_config(
        cls,
        *,
        a_dim: int,
        o_dim: int,
        config: DP_CFG | None = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> "DiffusionPolicy":
        """Construct the configured backbone pair and skrl policy adapter."""

        config = DP_CFG() if config is None else config
        backbone = getattr(config, "backbone", "unet")
        model = build_backbone(backbone, a_dim=a_dim, o_dim=o_dim, config=config)
        ema_model = build_backbone(backbone, a_dim=a_dim, o_dim=o_dim, config=config)
        return cls(
            a_dim=a_dim,
            o_dim=o_dim,
            models={"model": model, "ema_model": ema_model},
            ema=EMAModel(model.parameters(), power=config.ema_power),
            device=device,
            config=config,
            **kwargs,
        )

    def init(self, trainer_cfg: dict[str, Any] | None = None):
        self.enable_training_mode(True)
        super().init(trainer_cfg=trainer_cfg)

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        pass

    def update(self, *, timestep: int, timesteps: int) -> None:
        pass

    def configure_optimizers(self, dataloader=None, num_epochs=None):
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self._learning_rate,
                weight_decay=self._weight_decay,
            )
        if self.lr_scheduler is None:
            num_training_steps = (
                len(dataloader) * num_epochs
                if dataloader and num_epochs
                else self._num_training_steps
            )
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.optimizer,
                num_warmup_steps=self._num_warmup_steps,
                num_training_steps=num_training_steps,
            )

    def prepare_observation_condition(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, : self._obs_horizon, :].flatten(start_dim=1)

    def _update(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Perform one supervised training step"""

        if len(states.shape) == 2:
            states = states.unsqueeze(1)
        if len(actions.shape) == 2:
            actions = actions.unsqueeze(1)

        # Conditioning
        obs_cond = self.prepare_observation_condition(states)

        # Sample noise
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0, self._num_diffusion_iters, (actions.shape[0],), device=self.device
        ).long()

        # Add noise to actions
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

        inputs = {
            "actions": noisy_actions,
            "timestep": timesteps,
            "global_cond": obs_cond,
        }

        noise_pred, _ = self.model.act(inputs=inputs)

        # Loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        if self.model.training:  # only train if in training mode
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.ema.step(self.model.parameters())

        return loss.detach()

    def act(
        self,
        observations: Union[torch.Tensor, dict] | None = None,
        states: Union[torch.Tensor, dict] | None = None,
        *,
        timestep: int = 0,
        timesteps: int = 0,
        role: str = "policy",
        num_inference_steps: int | None = None,
        initial_action_chunk: torch.Tensor | None = None,
        warm_start_mask: torch.Tensor | None = None,
        warm_start_std: float | None = None,
        generator: torch.Generator | None = None,
        normalize_obs: Optional[bool] = None,
        unnormalize_act: Optional[bool] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Generate action chunks using the configured diffusion scheduler.

        Observations have dimensions ``[num_envs, obs_horizon, o_dim]``. An
        ``initial_action_chunk`` is interpreted in external (unnormalized) action
        units and only applies to chronological online deployment.

        """

        states = observations if states is None else states
        if isinstance(states, dict):
            states = states.get("observations", states.get("states"))
        if states is None:
            raise ValueError("observations or states must be provided")

        if not torch.is_tensor(states):
            states = torch.as_tensor(states, device=self.device, dtype=torch.float32)

        if self.stats is not None:
            do_normalize = normalize_obs if normalize_obs is not None else True
        else:
            do_normalize = False

        if do_normalize:
            stats = self.stats["obs"]
            min_val = torch.as_tensor(
                stats["min"], device=states.device, dtype=states.dtype
            )
            max_val = torch.as_tensor(
                stats["max"], device=states.device, dtype=states.dtype
            )
            range_val = max_val - min_val
            range_val = torch.where(
                range_val == 0, torch.ones_like(range_val), range_val
            )
            states = 2.0 * (states - min_val) / range_val - 1.0

        self.ema_model.eval()
        if num_inference_steps is None:
            num_inference_steps = self._num_inference_steps
        if num_inference_steps is None:
            num_inference_steps = self._num_diffusion_iters

        obs_cond = self.prepare_observation_condition(states)

        shape = (
            states.shape[0],
            self._pred_horizon,
            self.model._unwrapped_module._a_dim,
        )
        initial_sample = None
        if initial_action_chunk is not None:
            prior = torch.as_tensor(
                initial_action_chunk, device=self.device, dtype=obs_cond.dtype
            )
            if tuple(prior.shape) != shape:
                raise ValueError(
                    f"initial_action_chunk has shape {tuple(prior.shape)}, "
                    f"expected {shape}"
                )
            if self.stats is not None:
                stats = self.stats["action"]
                min_val = torch.as_tensor(
                    stats["min"], device=prior.device, dtype=prior.dtype
                )
                max_val = torch.as_tensor(
                    stats["max"], device=prior.device, dtype=prior.dtype
                )
                range_val = torch.where(
                    max_val == min_val,
                    torch.ones_like(max_val),
                    max_val - min_val,
                )
                prior = 2.0 * (prior - min_val) / range_val - 1.0

            std = (
                getattr(self.config, "warm_start_std", 0.1)
                if warm_start_std is None
                else warm_start_std
            )
            if std < 0:
                raise ValueError("warm_start_std must be non-negative")
            noise = torch.randn(
                shape,
                device=self.device,
                dtype=obs_cond.dtype,
                generator=generator,
            )
            warm_sample = prior + std * noise
            if warm_start_mask is None:
                initial_sample = warm_sample
            else:
                mask = torch.as_tensor(
                    warm_start_mask, device=self.device, dtype=torch.bool
                )
                if mask.shape != (shape[0],):
                    raise ValueError(
                        f"warm_start_mask has shape {tuple(mask.shape)}, "
                        f"expected {(shape[0],)}"
                    )
                initial_sample = torch.where(mask[:, None, None], warm_sample, noise)

        noisy_actions = self.sampler.sample(
            self.ema_model._unwrapped_module,
            shape=shape,
            global_cond=obs_cond,
            num_inference_steps=num_inference_steps,
            initial_sample=initial_sample,
            device=self.device,
            dtype=obs_cond.dtype,
            generator=generator,
        )

        if self.stats is not None:
            do_unnormalize = unnormalize_act if unnormalize_act is not None else True
        else:
            do_unnormalize = False

        if do_unnormalize:
            stats = self.stats["action"]
            min_val = torch.as_tensor(
                stats["min"], device=noisy_actions.device, dtype=noisy_actions.dtype
            )
            max_val = torch.as_tensor(
                stats["max"], device=noisy_actions.device, dtype=noisy_actions.dtype
            )
            range_val = max_val - min_val
            range_val = torch.where(
                range_val == 0, torch.ones_like(range_val), range_val
            )
            noisy_actions = 0.5 * (noisy_actions + 1.0) * range_val + min_val

        return noisy_actions, {}

    def eval(self) -> None:
        self.model.eval()
        self.ema_model.eval()

    def train(self) -> None:
        self.model.train()
        self.ema_model.train()

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

    def to(self, device: str = "cuda"):
        self.device = _resolve_device(device)
        for _, v in self.models.items():
            v.to(self.device)
        self.ema.to(self.device)
        return self

    def save(self, path: str):
        """Save model weights and configuration."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "ema_model_state_dict": self.ema_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
            if self.optimizer
            else None,
            "scheduler_state_dict": self.lr_scheduler.state_dict()
            if self.lr_scheduler
            else None,
            "config": self.config,
            "is_trained": self.is_trained,
            "a_dim": self._a_dim,
            "o_dim": self._o_dim,
            "stats": self.stats,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, a_dim: int = None, o_dim: int = None, device: str = None):
        """Load model from checkpoint."""
        device = _resolve_device(device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        a_dim = a_dim if a_dim is not None else checkpoint.get("a_dim")
        o_dim = o_dim if o_dim is not None else checkpoint.get("o_dim")
        if a_dim is None or o_dim is None:
            raise ValueError("Both action and observation dims must be provided.")

        backbone = getattr(config, "backbone", "unet")
        dp_models = {}
        dp_models["model"] = build_backbone(
            backbone, a_dim=a_dim, o_dim=o_dim, config=config
        )
        ema = EMAModel(dp_models["model"].parameters(), power=config.ema_power)
        dp_models["ema_model"] = build_backbone(
            backbone, a_dim=a_dim, o_dim=o_dim, config=config
        )

        policy = cls(
            a_dim=a_dim,
            o_dim=o_dim,
            models=dp_models,
            ema=ema,
            device=device,
            config=config,
            stats=checkpoint.get("stats"),
        )
        policy.model.load_state_dict(checkpoint["model_state_dict"])
        policy.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        if checkpoint["optimizer_state_dict"]:
            policy.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"]:
            policy.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        policy.is_trained = checkpoint["is_trained"]

        return policy
