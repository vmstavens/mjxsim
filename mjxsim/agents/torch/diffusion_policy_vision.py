import copy
import dataclasses
import logging
import math
import os
from collections.abc import Mapping
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import randn_tensor
from skrl.agents.torch.base import Agent, AgentCfg, ExperimentCfg, Model
from skrl.models.torch import DeterministicMixin

from mjxsim.agents.action_normalization import (
    ActionNormalization,
    action_normalization_from_legacy_stats,
)
from mjxsim.agents.torch.action_transform import ActionTransform

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


@dataclasses.dataclass(kw_only=True)
class VisionLRSchedulerCfg:
    """Configuration for the vision diffusion policy learning rate scheduler."""

    num_warmup_steps: int = 500
    """Number of warmup steps used by the scheduler."""

    num_training_steps: int = 10_000
    """Total number of training steps used by the scheduler."""


@dataclasses.dataclass(kw_only=True)
class VISION_DP_CFG(AgentCfg):
    """Configuration for the vision diffusion policy agent."""

    input_dim: int = 2
    """Action dimension predicted by the diffusion model."""

    global_cond_dim: int = 1028
    """Flattened conditioning dimension: obs_horizon * (vision_feature_dim + lowdim_obs_dim)."""

    diffusion_step_embed_dim: int = 256
    """Dimension of the sinusoidal diffusion timestep embedding."""

    down_dims: list[int] = dataclasses.field(default_factory=lambda: [256, 512, 1024])
    """Channel dimensions for the downsampling path of the conditional U-Net."""

    kernel_size: int = 5
    """Kernel size used by the 1D convolution blocks."""

    n_groups: int = 8
    """Number of groups used by group normalization."""

    vision_encoder: str = "resnet18"
    """Vision encoder architecture."""

    vision_feature_dim: int = 512
    """Feature dimension produced by the vision encoder."""

    lowdim_obs_dim: int = 2
    """Low-dimensional observation dimension concatenated with vision features."""

    pred_horizon: int = 16
    """Number of action steps predicted by the diffusion model."""

    obs_horizon: int = 2
    """Number of image/low-dimensional observation steps used for conditioning."""

    action_horizon: int = 8
    """Number of generated action steps intended for execution."""

    num_diffusion_iters: int = 100
    """Number of diffusion training timesteps."""

    batch_size: int = 64
    """Batch size for supervised vision diffusion policy training."""

    learning_rate: float = 1e-4
    """Learning rate for the policy optimizer."""

    weight_decay: float = 1e-6
    """Weight decay for the policy optimizer."""

    ema_power: float = 0.75
    """Exponential moving average coefficient for the EMA model."""

    num_workers: int = 4
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

    lr_scheduler_cfg: VisionLRSchedulerCfg = dataclasses.field(
        default_factory=VisionLRSchedulerCfg
    )
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

    def validate(self) -> bool:
        """Validate dimensions that are coupled by the vision conditioning path."""
        expected_cond_dim = self.obs_horizon * (
            self.vision_feature_dim + self.lowdim_obs_dim
        )
        return self.global_cond_dim == expected_cond_dim


DIFFUSION_POLICY_VISION_DEFAULT_CONFIG = VISION_DP_CFG()


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
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=self.device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self._unwrapped_module = module
        self.to(self.device)

    def compute(self, inputs: dict[str, Any], role):
        return self._unwrapped_module.forward(**inputs), {}

    def act(self, inputs: dict[str, Any], role: str = ""):
        actions, outputs = DeterministicMixin.act(self, inputs, role=role)
        return actions, outputs


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(_num_groups(out_channels), out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(_num_groups(out_channels), out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNetEncoder(nn.Module):
    """Small ResNet encoder for vision observations."""

    def __init__(self, layers: tuple[int, int, int, int] = (2, 2, 2, 2)) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(_num_groups(64), 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = 512

    def _make_layer(
        self, out_channels: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.max() > 2:
            x = x / 255.0
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def _num_groups(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def get_resnet(name: str) -> ResNetEncoder:
    if name != "resnet18":
        raise ValueError(f"Unsupported vision encoder: {name}")
    return ResNetEncoder()


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(
        self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1))
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, config: VISION_DP_CFG):
        super().__init__()
        self.config: VISION_DP_CFG = config
        self._input_dim: int = config.input_dim
        self._down_dims: list[int] = config.down_dims
        self._diffusion_step_embed_dim: int = config.diffusion_step_embed_dim
        self._global_cond_dim: int = config.global_cond_dim
        self._kernel_size: int = config.kernel_size
        self._n_groups: int = config.n_groups

        all_dims = [self._input_dim] + list(self._down_dims)
        start_dim = self._down_dims[0]
        dsed = self._diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + self._global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=self._kernel_size,
                    n_groups=self._n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=self._kernel_size,
                    n_groups=self._n_groups,
                ),
            ]
        )

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=self._kernel_size,
                            n_groups=self._n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=self._kernel_size,
                            n_groups=self._n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=self._kernel_size,
                            n_groups=self._n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=self._kernel_size,
                            n_groups=self._n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=self._kernel_size),
            nn.Conv1d(start_dim, self._input_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sample = sample.moveaxis(-1, -2)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)

        timestep = timestep.expand(sample.shape[0])
        global_feature = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            global_cond = global_cond.to(global_feature.device)
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)


class VisionDiffusionModel(nn.Module):
    """Vision encoder plus conditional U-Net noise predictor."""

    def __init__(self, config: VISION_DP_CFG) -> None:
        super().__init__()
        self.config: VISION_DP_CFG = config
        self.vision_encoder = get_resnet(config.vision_encoder)
        self.noise_pred_net = ConditionalUnet1D(config)

    def encode_obs(
        self, images: torch.Tensor, lowdim_obs: torch.Tensor
    ) -> torch.Tensor:
        images = _format_images(images)
        batch_size = images.shape[0]
        seq_len = images.shape[1]
        image_features = self.vision_encoder(images.flatten(0, 1))
        image_features = image_features.reshape(batch_size, seq_len, -1)
        obs_features = torch.cat([image_features, lowdim_obs], dim=-1)
        return obs_features.flatten(start_dim=1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.noise_pred_net(sample, timestep, global_cond=global_cond)


class EMAModel:
    """Exponential Moving Average model wrapper."""

    def __init__(self, parameters: list[torch.Tensor], power: float = 0.75):
        self.parameters = list(parameters)
        self.power = power
        self.shadow_params = [p.clone().detach() for p in self.parameters]

    def step(self, parameters: list[torch.Tensor]) -> None:
        parameters = list(parameters)
        for i, (s_param, param) in enumerate(zip(self.shadow_params, parameters)):
            if param.requires_grad:
                if s_param.device != param.device:
                    s_param = s_param.to(param.device)
                    self.shadow_params[i] = s_param
                s_param.data = s_param.data * self.power + param.data * (1 - self.power)

    def copy_to(self, parameters: list[torch.Tensor]) -> None:
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


def _format_images(images: torch.Tensor) -> torch.Tensor:
    """Return images as [B, T, C, H, W]."""
    if images.ndim == 4:
        images = images.unsqueeze(1)
    if images.ndim != 5:
        raise ValueError(
            "Expected images with shape [B, T, C, H, W] or [B, T, H, W, C] "
            f"(got {tuple(images.shape)})"
        )
    if images.shape[-1] in {1, 3}:
        images = images.permute(0, 1, 4, 2, 3)
    if images.shape[2] == 1:
        images = images.repeat(1, 1, 3, 1, 1)
    if images.shape[2] != 3:
        raise ValueError(f"Expected 3 image channels (got shape {tuple(images.shape)})")
    return images.contiguous()


def _batch_value(batch: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in batch:
            return batch[key]
    raise KeyError(f"Expected one of keys: {', '.join(keys)}")


class DiffusionPolicyVision(Agent):
    def __init__(
        self,
        models: dict[str, nn.Module] | None = None,
        ema: EMAModel | None = None,
        device: str = "cuda",
        observation_space=None,
        dataloader=None,
        action_space=None,
        memory=None,
        config: VISION_DP_CFG | None = None,
        stats: Optional[dict[str, Any]] = None,
        action_normalization: ActionNormalization | dict[str, Any] | None = None,
        training_data_domain: str | None = None,
    ):
        self.device = _resolve_device(device)
        _config = VISION_DP_CFG() if config is None else copy.deepcopy(config)
        _config.expand()
        if not _config.validate():
            expected_cond_dim = _config.obs_horizon * (
                _config.vision_feature_dim + _config.lowdim_obs_dim
            )
            raise ValueError(
                "Invalid global_cond_dim: "
                f"got {_config.global_cond_dim}, expected {expected_cond_dim}"
            )
        self.config: VISION_DP_CFG = _config
        self.stats = stats
        legacy_normalized_data = (
            action_normalization is None and stats is not None and "action" in stats
        )
        if action_normalization is None and legacy_normalized_data:
            action_normalization = action_normalization_from_legacy_stats(stats)
        self.action_normalization = (
            None
            if action_normalization is None
            else ActionNormalization.coerce(action_normalization)
        )
        if self.action_normalization is not None:
            if self.action_normalization.size != _config.input_dim:
                raise ValueError(
                    "Action normalization dimension does not match input_dim"
                )
            self.action_transform = ActionTransform.from_normalization(
                self.action_normalization, device=self.device
            )
        else:
            self.action_transform = None
        if training_data_domain is None:
            training_data_domain = (
                "normalized" if legacy_normalized_data else "physical"
            )
        if training_data_domain not in ("physical", "normalized"):
            raise ValueError("training_data_domain must be 'physical' or 'normalized'")
        self.training_data_domain = training_data_domain

        self._num_diffusion_iters: int = _config.num_diffusion_iters
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
        self._input_dim: int = _config.input_dim

        if models is None:
            models = {
                "model": VisionDiffusionModel(_config),
                "ema_model": VisionDiffusionModel(_config),
            }
        if ema is None:
            ema = EMAModel(models["model"].parameters(), power=self._ema_power)

        self.models = {
            key: ModuleWrapper(module, device=self.device)
            for key, module in models.items()
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

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self._num_diffusion_iters,
            beta_schedule=self._beta_schedule,
            clip_sample=self._clip_sample,
            prediction_type=self._prediction_type,
        )

        self.optimizer = None
        self.lr_scheduler = None
        self.configure_optimizers(dataloader=dataloader, num_epochs=_config.num_epochs)

        self.checkpoint_modules = {
            "policy": self.model,
            "ema_model": self.ema_model,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
        }

        self.is_trained = False
        self.enable_training_mode(True)

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

    def prepare_observation_condition(
        self, images: torch.Tensor, lowdim_obs: torch.Tensor, *, use_ema: bool = False
    ) -> torch.Tensor:
        model = self.ema_model if use_ema else self.model
        unwrapped = model._unwrapped_module
        images = images[:, : self._obs_horizon].to(self.device)
        lowdim_obs = lowdim_obs[:, : self._obs_horizon].to(self.device)
        return unwrapped.encode_obs(images, lowdim_obs)

    def _update(self, batch: Mapping[str, Any], actions: torch.Tensor | None = None):
        """Perform one supervised training step from a vision batch."""
        if actions is None:
            actions = _batch_value(batch, "action", "actions")
        images = _batch_value(batch, "pixels", "image", "images")
        lowdim_obs = _batch_value(batch, "agent_pos", "lowdim_obs", "states")

        images = torch.as_tensor(images, device=self.device)
        lowdim_obs = torch.as_tensor(
            lowdim_obs, device=self.device, dtype=torch.float32
        )
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if self.action_transform is None:
            raise ValueError(
                "Diffusion Policy training requires action_normalization. Use "
                "ActionNormalization.from_bounds(...) or explicitly "
                "ActionNormalization.from_dataset(...)."
            )
        if self.training_data_domain == "physical":
            actions = self.action_transform.normalize(actions)
        else:
            self.action_transform.assert_normalized(actions)

        obs_cond = self.prepare_observation_condition(images, lowdim_obs)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0, self._num_diffusion_iters, (actions.shape[0],), device=self.device
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

        noise_pred, _ = self.model.act(
            inputs={
                "sample": noisy_actions,
                "timestep": timesteps,
                "global_cond": obs_cond,
            }
        )
        loss = nn.functional.mse_loss(noise_pred, noise)

        if self.model.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.ema.step(self.model.parameters())

        return loss.detach()

    def train_step(self, batch: Mapping[str, Any]) -> float:
        self.enable_training_mode(True)
        return float(self._update(batch).item())

    @torch.no_grad()
    def act(
        self,
        observations: Mapping[str, Any] | None = None,
        states: Mapping[str, Any] | None = None,
        *,
        timestep: int = 0,
        timesteps: int = 0,
        role: str = "policy",
        num_inference_steps=None,
        unnormalize_act: Optional[bool] = None,
        output_domain: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        states = observations if states is None else states
        if states is None:
            raise ValueError("observations or states must be provided")
        images = _batch_value(states, "pixels", "image", "images")
        lowdim_obs = _batch_value(states, "agent_pos", "lowdim_obs", "states")
        images = torch.as_tensor(images, device=self.device)
        lowdim_obs = torch.as_tensor(
            lowdim_obs, device=self.device, dtype=torch.float32
        )

        self.ema_model.eval()
        num_inference_steps = num_inference_steps or self._num_diffusion_iters
        obs_cond = self.prepare_observation_condition(images, lowdim_obs, use_ema=True)

        shape = (images.shape[0], self._pred_horizon, self._input_dim)
        noisy_actions = randn_tensor(shape, device=self.device)
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)

        for t in self.noise_scheduler.timesteps:
            noise_pred, _ = self.ema_model.act(
                inputs={
                    "sample": noisy_actions,
                    "timestep": t,
                    "global_cond": obs_cond,
                }
            )
            noisy_actions = self.noise_scheduler.step(
                noise_pred, t, noisy_actions
            ).prev_sample

        noisy_actions = noisy_actions.clamp(-1.0, 1.0)
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
            noisy_actions = self.action_transform.denormalize(noisy_actions)

        return noisy_actions, {}

    @torch.no_grad()
    def predict(
        self, batch: Mapping[str, Any], num_inference_steps: int | None = None
    ) -> torch.Tensor:
        actions, _ = self.act(batch, num_inference_steps=num_inference_steps)
        return actions

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
        for _, value in self.models.items():
            value.to(self.device)
        self.ema.to(self.device)
        return self

    def save(self, path: str):
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
            "stats": self.stats,
            "normalization": {
                "schema_version": 1,
                "observation": None,
                "action": None
                if self.action_normalization is None
                else self.action_normalization.to_dict(),
            },
            "training_data_domain": self.training_data_domain,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: str = None):
        device = _resolve_device(device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = VisionDiffusionModel(config)
        ema_model = VisionDiffusionModel(config)
        ema = EMAModel(model.parameters(), power=config.ema_power)
        normalization = checkpoint.get("normalization")
        policy = cls(
            models={"model": model, "ema_model": ema_model},
            ema=ema,
            device=device,
            config=config,
            stats=checkpoint.get("stats"),
            action_normalization=None
            if normalization is None
            else normalization.get("action"),
            training_data_domain=checkpoint.get("training_data_domain"),
        )
        policy.model.load_state_dict(checkpoint["model_state_dict"])
        policy.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        if checkpoint["optimizer_state_dict"]:
            policy.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"]:
            policy.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        policy.is_trained = checkpoint["is_trained"]
        return policy


VisionDiffusionPolicy = DiffusionPolicyVision
