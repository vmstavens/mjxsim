from __future__ import annotations

import torch
from diffusers import DDIMScheduler, DDPMScheduler
from pathlib import Path

from mjxsim.agents.torch.diffusion_policy_state import (
    DP_CFG,
    DiffusionPolicy,
)
from mjxsim.diffusion import (
    ActionChunkHistory,
    ConditionalUnet1D,
    DiffusionSampler,
    SchedulerConfig,
    WarmStartDeployment,
    build_backbone,
    build_noise_scheduler,
)


def _small_config(**overrides) -> DP_CFG:
    values = {
        "diffusion_step_embed_dim": 8,
        "down_dims": [8, 16],
        "kernel_size": 3,
        "n_groups": 4,
        "pred_horizon": 4,
        "obs_horizon": 2,
        "action_horizon": 2,
        "num_diffusion_iters": 10,
        "num_inference_steps": 2,
        "scheduler_type": "ddim",
    }
    values.update(overrides)
    return DP_CFG(**values)


def test_root_exposes_diffusion_namespace() -> None:
    import mjxsim

    assert "diffusion" in mjxsim.__all__
    assert mjxsim.diffusion.ActionChunkHistory is ActionChunkHistory


def test_scheduler_factory_supports_ddpm_and_ddim() -> None:
    assert isinstance(
        build_noise_scheduler(SchedulerConfig(kind="ddpm")), DDPMScheduler
    )
    assert isinstance(
        build_noise_scheduler(SchedulerConfig(kind="ddim")), DDIMScheduler
    )


def test_diffusion_sampler_accepts_explicit_initial_sample() -> None:
    scheduler = build_noise_scheduler(
        SchedulerConfig(kind="ddim", num_train_timesteps=10)
    )
    sampler = DiffusionSampler(scheduler)
    initial = torch.zeros(2, 4, 3)
    condition = torch.zeros(2, 6)

    def zero_denoiser(**inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(inputs["actions"])

    result = sampler.sample(
        zero_denoiser,
        shape=initial.shape,
        global_cond=condition,
        num_inference_steps=2,
        initial_sample=initial,
    )

    assert result.shape == initial.shape
    assert torch.isfinite(result).all()


def test_action_chunk_history_aligns_and_resets_vector_envs() -> None:
    history = ActionChunkHistory(
        num_envs=2,
        pred_horizon=4,
        action_dim=1,
        executed_steps=2,
    )
    chunks = torch.tensor(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0], [7.0]],
        ]
    )
    history.update(chunks)

    prior, valid = history.prior()
    assert torch.equal(prior[0, :, 0], torch.tensor([2.0, 3.0, 3.0, 3.0]))
    assert torch.equal(prior[1, :, 0], torch.tensor([6.0, 7.0, 7.0, 7.0]))
    assert valid.tolist() == [True, True]

    history.reset(torch.tensor([True, False]))
    _, valid = history.prior()
    assert valid.tolist() == [False, True]

    generator = torch.Generator().manual_seed(7)
    initial = history.initial_sample(std=0.0, generator=generator)
    assert torch.equal(initial[1], prior[1])
    assert not torch.equal(initial[0], prior[0])


def test_unet_factory_preserves_legacy_public_class() -> None:
    cfg = _small_config()
    model = build_backbone("unet", a_dim=2, o_dim=3, config=cfg)
    assert isinstance(model, ConditionalUnet1D)
    result = model(
        actions=torch.zeros(2, cfg.pred_horizon, 2),
        timestep=torch.tensor([1, 2]),
        global_cond=torch.zeros(2, cfg.obs_horizon * 3),
    )
    assert result.shape == (2, cfg.pred_horizon, 2)


def test_state_agent_uses_ddim_and_explicit_warm_start() -> None:
    cfg = _small_config()
    policy = DiffusionPolicy.from_config(
        a_dim=2,
        o_dim=3,
        device="cpu",
        config=cfg,
    )
    observations = torch.zeros(2, cfg.obs_horizon, 3)
    prior = torch.zeros(2, cfg.pred_horizon, 2)
    actions, _ = policy.act(
        observations=observations,
        initial_action_chunk=prior,
        warm_start_mask=torch.tensor([False, True]),
        generator=torch.Generator().manual_seed(11),
    )

    assert isinstance(policy.noise_scheduler, DDIMScheduler)
    assert actions.shape == prior.shape
    assert torch.isfinite(actions).all()


def test_state_agent_checkpoint_rebuilds_configured_backbone(tmp_path: Path) -> None:
    cfg = _small_config()
    policy = DiffusionPolicy.from_config(a_dim=2, o_dim=3, config=cfg, device="cpu")
    checkpoint = tmp_path / "diffusion.pt"
    policy.save(str(checkpoint))

    loaded = DiffusionPolicy.load(str(checkpoint), device="cpu")

    assert isinstance(loaded.model._unwrapped_module, ConditionalUnet1D)
    assert isinstance(loaded.noise_scheduler, DDIMScheduler)
    assert loaded.config.num_inference_steps == 2
    assert loaded.model.state_dict().keys() == policy.model.state_dict().keys()


class _IdentitySampler:
    def sample(self, _denoiser, **kwargs):
        return kwargs["initial_sample"]


def test_state_agent_warm_prior_uses_external_action_units() -> None:
    cfg = _small_config(warm_start_std=0.0)
    stats = {
        "obs": {"min": [-1.0] * 3, "max": [1.0] * 3},
        "action": {"min": [-2.0] * 2, "max": [2.0] * 2},
    }
    policy = DiffusionPolicy.from_config(
        a_dim=2,
        o_dim=3,
        config=cfg,
        device="cpu",
        stats=stats,
    )
    policy.sampler = _IdentitySampler()
    prior = torch.full((1, cfg.pred_horizon, 2), 1.0)

    actions, _ = policy.act(
        observations=torch.zeros(1, cfg.obs_horizon, 3),
        initial_action_chunk=prior,
    )

    assert torch.allclose(actions, prior)


class _RecordingPolicy:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def act(self, observations=None, states=None, **kwargs):
        self.calls.append(kwargs)
        prior = kwargs["initial_action_chunk"]
        return torch.ones_like(prior), {}


def test_warm_start_deployment_owns_history_and_forwards_reset_mask() -> None:
    policy = _RecordingPolicy()
    deployment = WarmStartDeployment(
        policy,
        num_envs=2,
        pred_horizon=4,
        action_dim=1,
        executed_steps=2,
    )

    _, first_info = deployment.act(observations=torch.zeros(2, 1))
    _, second_info = deployment.act(
        observations=torch.zeros(2, 1),
        reset_mask=torch.tensor([True, False]),
    )

    assert first_info["warm_start_mask"].tolist() == [False, False]
    assert second_info["warm_start_mask"].tolist() == [False, True]
