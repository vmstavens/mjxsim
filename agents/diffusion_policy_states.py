import logging
import math
import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler

# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.training_utils import EMAModel
from diffusers.utils.torch_utils import randn_tensor

# env import
from matplotlib import pyplot as plt
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

# from utils.dpy.dpy.diffusion_policy_state import DiffusionPolicyState
# from utils.dpy.models import state as mod
# from utils.dpy.models.state import EMAModel
# from utils.dpy.utils import cfg
# from utils.model import ModuleWrapper
from skrl.skrl.agents.torch.base import Agent, Model
from skrl.skrl.models.torch import DeterministicMixin

logging.basicConfig(level=logging.INFO)  # This adds a default handler
relative_path = os.path.relpath(__file__)  # Relative to current working directory
logger = logging.getLogger(relative_path)
logger.setLevel(logging.WARNING)
# Prevent propagation to root logger to avoid duplicate handling
logger.propagate = False

DIFFUSION_POLICY_STATE_DEFAULT_CONFIG = {
    # Model architecture
    # "input_dim": 2,
    # "global_cond_dim": 10,  # obs_horizon * obs_dim
    "diffusion_step_embed_dim": 256,
    "down_dims": [256, 512, 1024],
    "kernel_size": 5,
    "n_groups": 8,
    # Training
    "pred_horizon": 16,
    "obs_horizon": 2,
    "action_horizon": 8,
    "num_diffusion_iters": 100,
    "batch_size": 256,
    "learning_rate": 1e-4,
    "weight_decay": 1e-6,
    "ema_power": 0.75,
    # "num_workers": 0,
    "num_workers": 1,
    "num_epochs": 100,
    "max_steps": 200,
    # "max_steps": 200,
    "eval_frequency": 10,
    # Scheduler
    "beta_schedule": "squaredcos_cap_v2",
    "clip_sample": True,
    "prediction_type": "epsilon",
    "lr_scheduler_cfg": {"num_warmup_steps": 500, "num_training_steps": 10_000},
    "experiment": {
        "directory": "",  # experiment's parent directory
        "experiment_name": "",  # experiment name
        "write_interval": 500,  # TensorBoard writing interval (timesteps)
        "checkpoint_interval": 1000,  # interval for checkpoints (timesteps)
        "store_separately": False,  # whether to store checkpoints separately
        "wandb": False,  # whether to use Weights & Biases
        "wandb_kwargs": {},  # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    },
}


class ModuleWrapper(DeterministicMixin, Model):
    def __init__(
        self,
        module: nn.Module,
        device: Optional[str] = None,
        observation_space=None,
        action_space=None,
        clip_actions=False,
    ):
        if device is None:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        # init base classes
        Model.__init__(self, observation_space, action_space, self.device)
        DeterministicMixin.__init__(self, clip_actions)

        self._unwrapped_module = module
        # self._unwrapped_module = module.to(self.device)

    def compute(self, inputs: dict, role):
        # pick states and actions if available
        return self._unwrapped_module.forward(**inputs), {}


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
    def __init__(self, a_dim: int, o_dim: int, config: dict):
        super().__init__()
        self.config = config

        self._a_dim: int = a_dim
        self._o_dim: int = o_dim
        self._down_dims: list[int] = config["down_dims"]
        self._diffusion_step_embed_dim: int = config["diffusion_step_embed_dim"]
        self._global_cond_dim: int = self._o_dim * self.config["obs_horizon"]
        # print(f"{self._a_dim=}")
        # print(f"{self._o_dim=}")
        # print(f"{self._global_cond_dim=}")
        # print(f"{self.config["obs_horizon"]=}")
        # self._global_cond_dim: int = config["global_cond_dim"]
        self._kernel_size: int = config["kernel_size"]
        self._n_groups: int = config["n_groups"]

        all_dims = [self._a_dim] + list(self._down_dims)
        start_dim = self._down_dims[0]

        dsed = self._diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
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

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
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

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
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

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=self._kernel_size),
            nn.Conv1d(start_dim, self._a_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.down_modules = down_modules
        self.up_modules = up_modules
        self.final_conv = final_conv

    def forward(
        self,
        actions: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond=Optional[torch.Tensor],
    ):
        actions = actions.moveaxis(-1, -2)
        # print(f"{actions.shape=}")

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=actions.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(actions.device)

        timestep = timestep.expand(actions.shape[0])
        global_feature = self.diffusion_step_encoder(timestep)
        # print(f"{global_feature.shape=}")

        if global_cond is not None:
            # print(
            #     f"in global_cond is not None, we got {global_feature.shape=} {global_cond.shape=}"
            # )
            if not global_cond.is_cuda:
                global_cond = global_cond.to(global_feature.device)
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        # print(f"{global_feature.shape=}")
        # print(f"{global_cond.shape=}")
        x = actions
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            # print(f"{x.shape=}")
            # print(f"{global_feature.shape=}")
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
        x = x.moveaxis(-1, -2)
        return x


class EMAModel:
    """Exponential Moving Average model wrapper."""

    def __init__(self, parameters: list[torch.Tensor], power: float = 0.75):
        self.parameters = list(parameters)
        self.power = power
        self.shadow_params = [p.clone().detach() for p in self.parameters]

    def step(self, parameters: list[torch.Tensor]):
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.data = s_param.data * self.power + param.data * (1 - self.power)

    def copy_to(self, parameters: list[torch.Tensor]):
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)


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
        config=None,
        stats: Optional[dict[str, Any]] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = {**DIFFUSION_POLICY_STATE_DEFAULT_CONFIG, **(config or {})}
        self.stats = stats

        self._a_dim: int = a_dim
        self._o_dim: int = o_dim

        # Load config values
        self._num_diffusion_iters: int = self.config["num_diffusion_iters"]
        self._beta_schedule: str = self.config["beta_schedule"]
        self._clip_sample: bool = self.config["clip_sample"]
        self._prediction_type: str = self.config["prediction_type"]
        self._ema_power: float = self.config["ema_power"]
        self._learning_rate: float = self.config["learning_rate"]
        self._weight_decay: float = self.config["weight_decay"]
        self._num_warmup_steps: int = self.config["lr_scheduler_cfg"][
            "num_warmup_steps"
        ]
        self._num_training_steps: int = self.config["lr_scheduler_cfg"][
            "num_training_steps"
        ]
        self._obs_horizon: int = self.config["obs_horizon"]
        self._pred_horizon: int = self.config["pred_horizon"]
        self._act_horizon: int = self.config["action_horizon"]

        # Save models
        self.models = {k: ModuleWrapper(v) for k, v in models.items()}
        # self.models = {k: ModuleWrapper(v).to(device) for k, v in models.items()}
        self.model = self.models["model"]
        self.ema_model = self.models["ema_model"]
        self.ema = ema

        super().__init__(
            self.models, memory, observation_space, action_space, device, config
        )

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self._num_diffusion_iters,
            beta_schedule=self._beta_schedule,
            clip_sample=self._clip_sample,
            prediction_type=self._prediction_type,
        )

        # Optimizer + LR scheduler
        self.optimizer = None
        self.lr_scheduler = None
        self.configure_optimizers(
            dataloader=dataloader, num_epochs=self.config["num_epochs"]
        )

        # Register for checkpointing
        self.checkpoint_modules = {
            "policy": self.model,
            "ema_model": self.ema_model,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
        }

        self.is_trained = False
        self.set_mode("train")

    def init(self, trainer_cfg=None):
        self.set_mode("train")
        super().init(trainer_cfg)

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
        # self.model.train()

        if len(states.shape) == 2:
            states = states.unsqueeze(1)
        if len(actions.shape) == 2:
            actions = actions.unsqueeze(1)

        # Conditioning
        obs_cond = self.prepare_observation_condition(states)
        # print(f"{obs_cond.shape=}")

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

        noise_pred, _, _ = self.model.act(inputs=inputs)

        out, _, _ = self.model.act(inputs)

        # Loss
        loss = nn.functional.mse_loss(noise_pred, noise)

        if self.model.training:  # only train if in training mode
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.ema.step(self.model.parameters())

        # Optimize
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # # EMA update
        # self.ema.step(self.model.parameters())

        return loss.detach()

    def act(
        self,
        states: Union[torch.Tensor, dict],
        timestep: int = 0,
        timesteps: int = 0,
        role: str = "policy",
        num_inference_steps=None,
        normalize_obs: Optional[bool] = None,
        unnormalize_act: Optional[bool] = None,
    ) -> tuple[torch.Tensor, None, dict]:
        """Generate actions using DDIM sampling

        expected dimensions [num_envs, obs_horizon, o_dim]

        """
        # print("----------------- act --------------------")

        if isinstance(states, dict):
            states = states["states"]

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
        num_inference_steps = num_inference_steps or self._num_diffusion_iters
        # print(f"\t{states.shape=}")

        # print(f"{states.shape=} this should be 2,1,6")
        obs_cond = self.prepare_observation_condition(states)
        # print(f"{obs_cond.shape=} this should be 2,1,12")
        # logger.info(f"\t{obs_cond.shape=}")
        # print(f"{obs_cond.shape=}")
        #  i want a obs cond that is (batch, hor, o_dim)

        shape = (
            states.shape[0],
            self._pred_horizon,
            self.model._unwrapped_module._a_dim,
        )
        # print(f"{shape=}")
        # logger.info(f"{shape=}")
        noisy_actions = randn_tensor(shape, device=self.device)
        # print(f"{noisy_actions.shape=}")

        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)

        for t in self.noise_scheduler.timesteps:
            inputs = {
                "actions": noisy_actions,
                "timestep": t,
                "global_cond": obs_cond,
            }

            noise_pred, _, _ = self.ema_model.act(inputs=inputs)
            # print(f"{noise_pred.shape=}")
            # print(f"{noise_pred.shape=}")
            # noise_pred = self.ema_model.act(noisy_actions, t, global_cond=obs_cond)
            noisy_actions = self.noise_scheduler.step(
                noise_pred, t, noisy_actions
            ).prev_sample
            # print(f"\t{noisy_actions.shape=}")

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

        # print(f"{noisy_actions.shape=}")
        # print(f"{noisy_actions[:,:self._act_horizon,:]=}")
        # print(f"{noisy_actions[:,:self._act_horizon,:].shape=}")
        return noisy_actions, None, {}

    def _predict_action(self, o: torch.Tensor) -> torch.Tensor:
        pass

    def set_mode(self, mode: str):
        # print(f"in set_mode of DP with mode '{mode}'")

        if mode == "eval":
            # print("setting to 'eval'")
            self.eval()
        elif mode == "train":
            self.train()
        else:
            raise ValueError(
                f"Wrong Mode: choose either 'train' or 'eval', but got '{mode}'"
            )

    def eval(self) -> None:
        # print(f"In DP eval() {self.stats=}")

        # self.ema.copy_to(self.ema_model.parameters())
        self.model.eval()
        self.ema_model.eval()

    def train(self) -> None:
        # self.ema.copy_to(self.ema_model.parameters())
        self.model.train()
        self.ema_model.train()

    def to(self, device: str = "cuda"):
        for _, v in self.models.items():
            v.to(device)

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
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        a_dim = a_dim if a_dim is not None else checkpoint.get("a_dim")
        o_dim = o_dim if o_dim is not None else checkpoint.get("o_dim")
        if a_dim is None or o_dim is None:
            raise ValueError("Both action and observation dims must be provided.")

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dp_models = {}
        dp_models["model"] = ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=config)
        ema = EMAModel(dp_models["model"].parameters(), power=config["ema_power"])
        dp_models["ema_model"] = ConditionalUnet1D(
            a_dim=a_dim, o_dim=o_dim, config=config
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
