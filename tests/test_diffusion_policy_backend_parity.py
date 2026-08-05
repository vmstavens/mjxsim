from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from diffusers import DDPMScheduler
from flax.core import freeze, unfreeze

from mjxsim.agents.jax.ddpm import (
    add_noise,
    inference_timesteps,
    squaredcos_cap_v2_betas,
    step,
)
from mjxsim.agents.jax.diffusion_policy_state import (
    ConditionalUnet1D as JaxConditionalUnet1D,
)
from mjxsim.agents.jax.diffusion_policy_state import (
    ConditionalResidualBlock1D as JaxConditionalResidualBlock1D,
)
from mjxsim.agents.jax.diffusion_policy_state import DP_CFG as JaxDPConfig
from mjxsim.agents.torch.diffusion_policy_state import (
    ConditionalUnet1D as TorchConditionalUnet1D,
)
from mjxsim.agents.torch.diffusion_policy_state import (
    ConditionalResidualBlock1D as TorchConditionalResidualBlock1D,
)
from mjxsim.agents.torch.diffusion_policy_state import DP_CFG as TorchDPConfig
from mjxsim.trainers.jax.supervised_trainer import SupervisedTrainer


def test_jax_ddpm_schedule_and_forward_noise_match_diffusers() -> None:
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    betas = squaredcos_cap_v2_betas(10)
    np.testing.assert_allclose(
        np.asarray(betas),
        scheduler.betas.numpy(),
        rtol=1e-6,
        atol=3e-7,
    )

    samples = np.arange(12, dtype=np.float32).reshape(2, 3, 2) / 10
    noise = np.full_like(samples, 0.2)
    timesteps = np.asarray([0, 7], dtype=np.int64)
    actual = add_noise(
        jnp.asarray(samples),
        jnp.asarray(noise),
        jnp.asarray(timesteps),
        jnp.cumprod(1 - betas),
    )
    expected = scheduler.add_noise(
        torch.from_numpy(samples),
        torch.from_numpy(noise),
        torch.from_numpy(timesteps),
    )
    np.testing.assert_allclose(
        np.asarray(actual),
        expected.numpy(),
        rtol=1e-6,
        atol=2e-7,
    )


def test_jax_ddpm_final_step_matches_diffusers() -> None:
    scheduler = DDPMScheduler(
        num_train_timesteps=10,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    sample = np.linspace(-1.2, 1.2, 12, dtype=np.float32).reshape(2, 3, 2)
    predicted_noise = np.full_like(sample, 0.1)
    betas = squaredcos_cap_v2_betas(10)
    actual = step(
        jnp.asarray(predicted_noise),
        0,
        jnp.asarray(sample),
        jnp.cumprod(1 - betas),
        jnp.zeros_like(jnp.asarray(sample)),
    )
    expected = scheduler.step(
        torch.from_numpy(predicted_noise),
        0,
        torch.from_numpy(sample),
    ).prev_sample
    np.testing.assert_allclose(np.asarray(actual), expected.numpy(), atol=1e-6)
    np.testing.assert_array_equal(
        np.asarray(inference_timesteps(10, 5)),
        np.asarray([8, 6, 4, 2, 0]),
    )


def test_jax_and_torch_unets_have_identical_parameter_counts() -> None:
    common = {
        "pred_horizon": 4,
        "obs_horizon": 2,
        "action_horizon": 2,
        "num_diffusion_iters": 4,
        "down_dims": [8, 16],
        "diffusion_step_embed_dim": 8,
        "kernel_size": 5,
        "n_groups": 8,
    }
    jax_config = JaxDPConfig(**common)
    jax_model = JaxConditionalUnet1D(a_dim=2, o_dim=3, config=jax_config)
    jax_params = jax_model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, 4, 2), dtype=jnp.float32),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((1, 2, 3), dtype=jnp.float32),
    )["params"]

    torch_model = TorchConditionalUnet1D(
        a_dim=2,
        o_dim=3,
        config=TorchDPConfig(**common),
    )
    jax_count = sum(value.size for value in jax.tree.leaves(jax_params))
    torch_count = sum(value.numel() for value in torch_model.parameters())
    assert jax_count == torch_count


def test_jax_residual_block_is_numerically_equivalent_to_torch() -> None:
    torch.manual_seed(4)
    torch_block = TorchConditionalResidualBlock1D(
        in_channels=4,
        out_channels=8,
        cond_dim=6,
        kernel_size=5,
        n_groups=4,
    ).eval()
    jax_block = JaxConditionalResidualBlock1D(
        features=8,
        kernel_size=5,
        n_groups=4,
    )
    inputs = np.random.default_rng(3).normal(size=(2, 8, 4)).astype(np.float32)
    condition = np.random.default_rng(5).normal(size=(2, 6)).astype(np.float32)
    variables = jax_block.init(
        jax.random.PRNGKey(0),
        jnp.asarray(inputs),
        jnp.asarray(condition),
    )
    params = unfreeze(variables["params"])

    for index in range(2):
        torch_conv = torch_block.blocks[index].block[0]
        torch_norm = torch_block.blocks[index].block[1]
        jax_params = params[f"block_{index}"]
        jax_params["conv"]["kernel"] = jnp.asarray(
            torch_conv.weight.detach().numpy().transpose(2, 1, 0)
        )
        jax_params["conv"]["bias"] = jnp.asarray(torch_conv.bias.detach().numpy())
        jax_params["norm"]["scale"] = jnp.asarray(torch_norm.weight.detach().numpy())
        jax_params["norm"]["bias"] = jnp.asarray(torch_norm.bias.detach().numpy())

    torch_condition = torch_block.cond_encoder[1]
    params["cond_encoder"]["kernel"] = jnp.asarray(
        torch_condition.weight.detach().numpy().T
    )
    params["cond_encoder"]["bias"] = jnp.asarray(torch_condition.bias.detach().numpy())
    params["residual_conv"]["kernel"] = jnp.asarray(
        torch_block.residual_conv.weight.detach().numpy().transpose(2, 1, 0)
    )
    params["residual_conv"]["bias"] = jnp.asarray(
        torch_block.residual_conv.bias.detach().numpy()
    )

    actual = jax_block.apply(
        {"params": freeze(params)},
        jnp.asarray(inputs),
        jnp.asarray(condition),
    )
    with torch.no_grad():
        expected = torch_block(
            torch.from_numpy(inputs).movedim(-1, 1),
            torch.from_numpy(condition),
        ).movedim(1, -1)
    np.testing.assert_allclose(
        np.asarray(actual),
        expected.numpy(),
        rtol=2e-5,
        atol=2e-5,
    )


def test_jax_supervised_trainer_updates_ema_parameters() -> None:
    params = {"weight": jnp.asarray([0.0], dtype=jnp.float32)}

    def loss_fn(current_params, batch):
        del batch
        return jnp.square(current_params["weight"][0] - 1)

    trainer = SupervisedTrainer(
        params=params,
        loss_fn=loss_fn,
        optimizer=optax.sgd(learning_rate=0.5),
        trainer_config={
            "num_epochs": 1,
            "progressbar": False,
        },
        train_loader=[{}],
        ema_decay=0.75,
    )
    final_params = trainer.train()
    np.testing.assert_allclose(final_params["weight"], np.asarray([1.0]))
    np.testing.assert_allclose(trainer.ema_params["weight"], np.asarray([0.25]))
