"""Noise schedulers and sampling loops for action diffusion."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import torch
from diffusers import DDIMScheduler, DDPMScheduler


@dataclasses.dataclass(frozen=True, kw_only=True)
class SchedulerConfig:
    """Framework-neutral configuration for a Diffusers noise scheduler."""

    kind: str = "ddpm"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    clip_sample: bool = True
    prediction_type: str = "epsilon"


def build_noise_scheduler(config: SchedulerConfig) -> DDIMScheduler | DDPMScheduler:
    """Create a DDPM or few-step DDIM scheduler."""

    kwargs = {
        "num_train_timesteps": config.num_train_timesteps,
        "beta_schedule": config.beta_schedule,
        "clip_sample": config.clip_sample,
        "prediction_type": config.prediction_type,
    }
    kind = config.kind.lower()
    if kind == "ddpm":
        return DDPMScheduler(**kwargs)
    if kind == "ddim":
        return DDIMScheduler(**kwargs)
    raise ValueError(f"Unknown diffusion scheduler: {config.kind!r}")


class DiffusionSampler:
    """Reusable diffusion sampling loop independent of skrl."""

    def __init__(self, scheduler: DDIMScheduler | DDPMScheduler) -> None:
        self.scheduler = scheduler

    @torch.no_grad()
    def sample(
        self,
        denoiser: Callable[..., torch.Tensor],
        *,
        shape: Sequence[int],
        global_cond: torch.Tensor,
        num_inference_steps: int,
        initial_sample: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
        step_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Generate an action sequence, optionally from a warm-start sample."""

        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        device = torch.device(device or global_cond.device)
        dtype = dtype or global_cond.dtype
        expected_shape = tuple(shape)

        if initial_sample is None:
            sample = torch.randn(
                expected_shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        else:
            if tuple(initial_sample.shape) != expected_shape:
                raise ValueError(
                    "initial_sample has shape "
                    f"{tuple(initial_sample.shape)}, expected {expected_shape}"
                )
            sample = initial_sample.to(device=device, dtype=dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        kwargs = {} if step_kwargs is None else step_kwargs
        for timestep in self.scheduler.timesteps:
            noise_prediction = denoiser(
                actions=sample,
                timestep=timestep,
                global_cond=global_cond,
            )
            sample = self.scheduler.step(
                noise_prediction,
                timestep,
                sample,
                generator=generator,
                **kwargs,
            ).prev_sample
        return sample
