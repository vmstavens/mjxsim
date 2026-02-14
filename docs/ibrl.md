# IBRL overview

This repository packages two related agents built on SKRL and MuJoCo:

- **IBRL (SAC)**: a soft actor-critic agent with optional imitation guidance. If no imitation policy is provided, it behaves like SAC.
- **IBRL + Diffusion Policy (`ibrl_sac_o_o2.py`)**: the SAC agent is conditioned by a diffusion-policy prior trained on demonstrations (PushT by default).

Both agents share the same policy/critic backbones (`agents.models.ibrl_sac`) and logging/checkpointing API.

## Training flows

- **Baseline IBRL**: `mjx-sim-train-ibrl --env-id Ant-v4 --timesteps 100000`
  - Uses Gymnasium MuJoCo tasks, wraps them with SKRL, and trains SAC with replay memory.
  - Checkpoints and TensorBoard logs are stored under `runs/ibrl/<env>/`.
  - Evaluate with `mjx-sim-eval-ibrl --env-id Ant-v4 --checkpoint <path> --render`.

- **Diffusion-conditioned IBRL**: `mjx-sim-train-ibrl-diffusion --timesteps 50000`
  - Downloads the PushT demonstration dataset (or use `--dataset <zip>`).
  - Trains a diffusion policy (state-only) via supervised learning.
  - Fine-tunes the SAC agent while using the diffusion policy as an imitation prior.
  - Evaluate with `mjx-sim-eval-ibrl-diffusion --checkpoint <path> --render` (optionally add `--diffusion-checkpoint` to load the IL prior).

## Key configuration knobs

- `IBRL_SAC_DEFAULT_CONFIG` and `SAC_DEFAULT_CONFIG` in `agents.ibrl_sac(_o_o2)`:
  - `batch_size`, `learning_rate`s, `polyak`, `discount_factor`
  - `experiment` block controls logging/checkpoint cadence and output directory
  - `offline` and `BC` flags enable loading expert demonstrations into replay buffers
- Diffusion policy config (`DIFFUSION_POLICY_STATE_DEFAULT_CONFIG`):
  - `pred_horizon`, `obs_horizon`, `action_horizon` control the denoising window
  - `ema_power`, `num_diffusion_iters`, `learning_rate`, `num_workers` for training

## Working with your own tasks

1. Provide a Gymnasium-compatible environment (observation and action spaces must be Box).
2. Use `utils.envs.mk_env` if you need to wrap MuJoCo/Brax tasks consistently.
3. Swap datasets by creating a `torch.utils.data.Dataset` that yields `(obs, action)` sequences and pointing the diffusion CLI to it.

## Checkpoints and packaging

- Installing the repo (`pip install -e .`) exposes the CLI entrypoints listed above.
- Checkpoints are written under the `experiment.directory` configured in each script (defaults to `runs/`).
- To reuse the agents in another project: `from agents.ibrl_sac import IBRL, SAC_DEFAULT_CONFIG` and build models with `agents.models.ibrl_sac`.
