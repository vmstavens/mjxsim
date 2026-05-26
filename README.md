# mjxsim

MuJoCo/Brax simulation helpers, RL agents built on SKRL + PyTorch, and dataset utilities for robot learning experiments.

## Installation
- With uv from this repo: `uv add /path/to/mjxsim`
- From a Git URL: `uv add "git+ssh://git@your-host/mjxsim.git"`
- The default install includes the dataset, environment, RL, diffusion-policy, MuJoCo/Brax, and example dependencies.
- CUDA JAX support: `uv add "/path/to/mjxsim[cuda]"` only when you need CUDA-backed JAX.

## Imports
Installers expose a real `mjxsim` package, with the main building blocks
available from the package root:

```python
import mjxsim as ms

policy_cls = ms.DiffusionPolicy
dataset_cls = ms.PushTStateDataset
```

The module layout is also available under the `mjxsim` namespace:

```python
from mjxsim.agents.diffusion_policy_state import DiffusionPolicy
from mjxsim.datasets.pushert import PushTStateDataset
from mjxsim.trainers.supervised_trainer import SupervisedTrainer
from mjxsim.utils.datasets import split_dataset
```

## Quick start
- Create an environment lazily (Gym or Brax):
  ```python
  import mjxsim as ms

  env = ms.PushTEnv()
  ```
- Train or evaluate the IBRL agent from the CLI:
  - `mjx-sim-train-ibrl --env-id Ant-v4 --timesteps 100000`
  - `mjx-sim-eval-ibrl --env-id Ant-v4 --checkpoint path/to/checkpoint`
  - Diffusion variant: `mjx-sim-train-ibrl-diffusion ...` / `mjx-sim-eval-ibrl-diffusion ...`
- Load datasets (e.g., PushT) and split:
  ```python
  from mjxsim.datasets.pushert import PushTStateDataset
  from mjxsim.utils.datasets import split_dataset

  dataset = PushTStateDataset(cache_dir="~/.cache/mjx-sim")
  train_ds, val_ds = split_dataset(dataset, val_ratio=0.1)
  ```

## Package contents
- `envs`: MuJoCo/Brax and Gym wrappers (MuJoCo/Brax require the `mujoco` extra).
- `agents`: IBRL, PPO, BC, diffusion-policy, representation-learning, and
  latent-distillation agents built on SKRL/PyTorch.
- `datasets`: Dataset utilities (PushT, attractor, state-only datasets).
- `utils`: Helpers for env creation, dataset splitting, and data handling.
- `cli`: Console entrypoints for training/evaluation.

## Representation agents
- `AutoencoderAgent` trains a deterministic MLP encoder/decoder on vector
  states with `SupervisedTrainer`.
- `VariationalAutoencoderStateAgent` and `VariationalAutoencoderVisionAgent`
  train VAE representations for vector states and images.
- `LatentDistillerAgent` implements privileged latent distillation: a trainable
  deployment encoder receives sensor observations while a frozen privileged
  encoder receives privileged observations, and the supervised loss matches
  their latent representations. Either encoder can be deterministic or
  variational; VAE encoders use their latent mean as the distillation target.

```python
from mjxsim.agents import AutoencoderAgent, LatentDistillerAgent

student = AutoencoderAgent({"state_dim": 16, "latent_dim": 8})
teacher = AutoencoderAgent({"state_dim": 24, "latent_dim": 8})

distiller = LatentDistillerAgent(
    encoder=student.model,
    privileged_encoder=teacher.model,
)
```

## Notes
- Optional modules such as `mujoco_playground`, `robots`, or `ctrl` are not bundled; they are used only by experimental files and can be installed separately if needed.
- Package data (e.g., `sim/scene/empty.xml` and stored stats) are included in wheels for direct use.
