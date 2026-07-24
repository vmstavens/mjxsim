"""Denoising backbones for action diffusion."""

from __future__ import annotations

import math
from typing import Any, Protocol, Union, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class DiffusionBackbone(Protocol):
    """Structural interface implemented by action denoisers."""

    def forward(
        self,
        actions: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("Sinusoidal embedding dimension must be even and >= 4")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=x.device, dtype=torch.float32) * -scale
        )
        embedding = x[:, None].float() * frequencies[None, :]
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)


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
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
            nn.Unflatten(-1, (-1, 1)),
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
        scale, bias = embed[:, 0, ...], embed[:, 1, ...]
        out = scale * out + bias
        return self.blocks[1](out) + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """Conditional 1D UNet used by the original mjxsim diffusion policy."""

    def __init__(self, a_dim: int, o_dim: int, config: Any):
        super().__init__()
        self.config = config
        self._a_dim = a_dim
        self._o_dim = o_dim
        self._down_dims = config.down_dims
        self._diffusion_step_embed_dim = config.diffusion_step_embed_dim
        self._global_cond_dim = o_dim * config.obs_horizon
        self._kernel_size = config.kernel_size
        self._n_groups = config.n_groups

        all_dims = [a_dim] + list(self._down_dims)
        start_dim = self._down_dims[0]
        embed_dim = self._diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        cond_dim = embed_dim + self._global_cond_dim
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
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index >= len(in_out) - 1
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
        for index, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = index >= len(in_out) - 1
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
            nn.Conv1d(start_dim, a_dim, 1),
        )

    def forward(
        self,
        actions: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actions = actions.moveaxis(-1, -2)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=actions.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(actions.device)
        timestep = timestep.expand(actions.shape[0])

        global_feature = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            global_cond = global_cond.to(global_feature.device)
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        x = actions
        residuals = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet2(resnet(x, global_feature), global_feature)
            residuals.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, residuals.pop()), dim=1)
            x = upsample(resnet2(resnet(x, global_feature), global_feature))
        return self.final_conv(x).moveaxis(-1, -2)


class _MambaResidualBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        state_dim: int,
        conv_dim: int,
        expand: int,
    ) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as error:
            raise ImportError(
                "The Mamba diffusion backbone requires the optional 'mamba-ssm' "
                "package. Install mjxsim with the 'fast-diffusion' extra."
            ) from error
        self.norm = nn.LayerNorm(model_dim)
        self.mamba = Mamba(
            d_model=model_dim,
            d_state=state_dim,
            d_conv=conv_dim,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class MambaDenoiser1D(nn.Module):
    """FastDP-style conditional Mamba action denoiser.

    ``mamba-ssm`` is imported lazily, keeping it an optional CUDA-oriented
    acceleration backend instead of a core dependency.
    """

    def __init__(self, a_dim: int, o_dim: int, config: Any) -> None:
        super().__init__()
        self.config = config
        self._a_dim = a_dim
        self._o_dim = o_dim
        model_dim = getattr(config, "mamba_model_dim", 128)
        embed_dim = config.diffusion_step_embed_dim
        cond_dim = o_dim * config.obs_horizon + embed_dim

        self.action_encoder = nn.Linear(a_dim, model_dim)
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.film = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, model_dim * 2),
        )
        self.blocks = nn.ModuleList(
            [
                _MambaResidualBlock(
                    model_dim=model_dim,
                    state_dim=getattr(config, "mamba_state_dim", 16),
                    conv_dim=getattr(config, "mamba_conv_dim", 4),
                    expand=getattr(config, "mamba_expand", 2),
                )
                for _ in range(getattr(config, "mamba_layers", 4))
            ]
        )
        kernel_size = config.kernel_size
        self.local_conv = nn.Conv1d(
            model_dim,
            model_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.action_decoder = nn.Linear(model_dim, a_dim)

    def forward(
        self,
        actions: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if global_cond is None:
            raise ValueError("MambaDenoiser1D requires global conditioning")
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=actions.device)
        elif timestep.ndim == 0:
            timestep = timestep[None].to(actions.device)
        timestep = timestep.expand(actions.shape[0])

        condition = torch.cat(
            [self.diffusion_step_encoder(timestep), global_cond.to(actions.device)],
            dim=-1,
        )
        scale, bias = self.film(condition).chunk(2, dim=-1)
        x = self.action_encoder(actions)
        x = x * (1 + scale[:, None, :]) + bias[:, None, :]
        for block in self.blocks:
            x = block(x)
        x = x + self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        return self.action_decoder(self.output_norm(x))


def build_backbone(name: str, *, a_dim: int, o_dim: int, config: Any) -> nn.Module:
    """Build a diffusion denoiser by stable public name."""

    normalized = name.lower().replace("-", "_")
    if normalized in {"unet", "unet1d", "conditional_unet1d"}:
        return ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=config)
    if normalized in {"mamba", "mamba1d", "fastdp"}:
        return MambaDenoiser1D(a_dim=a_dim, o_dim=o_dim, config=config)
    raise ValueError(f"Unknown diffusion backbone: {name!r}")
