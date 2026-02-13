# mjx-sim

MuJoCo/Brax simulation helpers, RL agents built on SKRL + PyTorch, and dataset utilities for robot learning experiments.

## Installation
- With uv from this repo: `uv add /path/to/mjxsim`
- From a Git URL: `uv add "git+ssh://git@your-host/mjxsim.git"`
- Enable MuJoCo/Brax support: `uv add "mjx-sim[mujoco]"` (installs jax, brax, mujoco, dm-control, etc.)
- Enable diffusion-policy helpers: `uv add "mjx-sim[diffusion]"`

## Quick start
- Create an environment lazily (Gym or Brax):
  ```python
  from mjx_sim.utils import mk_env
  from mjx_sim.utils.envs import ENV_TYPE

  env = mk_env("Ant-v4", num_envs=1, env_type=ENV_TYPE.GYM)
  ```
- Train or evaluate the IBRL agent from the CLI:
  - `mjx-sim-train-ibrl --env-id Ant-v4 --timesteps 100000`
  - `mjx-sim-eval-ibrl --env-id Ant-v4 --checkpoint path/to/checkpoint`
  - Diffusion variant: `mjx-sim-train-ibrl-diffusion ...` / `mjx-sim-eval-ibrl-diffusion ...`
- Load datasets (e.g., PushT) and split:
  ```python
  from mjx_sim.datasets.pushert import PushTStateDataset
  from mjx_sim.utils import split_dataset

  dataset = PushTStateDataset(cache_dir="~/.cache/mjx-sim")
  train_ds, val_ds = split_dataset(dataset, val_ratio=0.1)
  ```

## Package contents
- `mjx_sim.envs`: MuJoCo/Brax and Gym wrappers (MuJoCo/Brax require the `mujoco` extra).
- `mjx_sim.agents`: IBRL, PPO, BC, diffusion-policy agents built on SKRL.
- `mjx_sim.datasets`: Dataset utilities (PushT, attractor, state-only datasets).
- `mjx_sim.utils`: Helpers for env creation, dataset splitting, and data handling.
- `mjx_sim.cli`: Console entrypoints for training/evaluation.

## Notes
- Optional modules such as `mujoco_playground`, `robots`, or `ctrl` are not bundled; they are used only by experimental files and can be installed separately if needed.
- Package data (e.g., `sim/scene/empty.xml` and stored stats) are included in wheels for direct use.
