# mjxsim

Accelerator-first MuJoCo/Brax utilities, datasets, reinforcement-learning
agents, and Rapid Motor Adaptation (RMA) components for robot learning.

## Requirements

- Python 3.11
- Linux with a CUDA 13-compatible NVIDIA setup
- [`uv`](https://docs.astral.sh/uv/) is recommended

CUDA-enabled JAX and PyTorch packages are installed by default. This package is
intentionally aimed at the accelerated systems used by its maintainers rather
than at CPU-only or platform-neutral environments.

## Installation

Install from Git:

```bash
uv add "mjxsim @ git+https://github.com/vmstavens/mjxsim.git"
```

For development:

```bash
git clone https://github.com/vmstavens/mjxsim.git
cd mjxsim
uv sync --dev
```

## Public API

The package root provides lazy access to commonly used components:

```python
import mjxsim

dataset_cls = mjxsim.PushTStateDataset
environment_cls = mjxsim.PushTEnv
```

Stable namespaced imports are also available:

```python
from mjxsim.agents import (
    DRLR,
    DRLRTD3,
    IBRL,
    IBRLTD3,
    AutoencoderAgent,
    DiffusionPolicy,
)
from mjxsim.datasets import PushTStateDataset
from mjxsim.rma import RmaSpec
from mjxsim.rma.jax import make_ppo_rma_models
from mjxsim.rma.torch import make_sac_rma_models
from mjxsim.trainers.jax import JaxSequentialTrainer, JaxSequentialTrainerCfg
from mjxsim.utils.datasets import split_dataset
```

`IBRL` and `DRLR` use SAC; `IBRLTD3` and `DRLRTD3` provide the corresponding
TD3 implementations. The TD3 agents combine a trainable deterministic policy
with a frozen imitation policy, mix online and demonstration replay, and use
twin critics to choose behavior and bootstrap actions.

## RMA

`mjxsim.rma` contains the promoted, reusable RMA implementation:

- framework-neutral observation/history specifications;
- Torch modules and SKRL SAC adapters;
- JAX/Flax modules and SKRL PPO adapters;
- Phase 2 latent distillation;
- actor-only deployment helpers.

See [`mjxsim/rma/README.md`](mjxsim/rma/README.md) for the model contract and
examples.

## JAX trainer

`JaxSequentialTrainer` has two modes:

- `compatibility` uses existing SKRL agent hooks and reports vector steps and
  environment transitions separately;
- `compiled` accepts a functional `CompiledTrainingKernel` so rollout and
  update work can be staged with JAX without changing an existing agent's
  semantics implicitly.

## Development

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv build
```

The wheel contains the `mjxsim` package, including agents, datasets,
environments, trainers, utilities, and both RMA backends. Research experiments
and project-specific training scripts are intentionally not part of the public
package.

## License

See [`LICENSE`](LICENSE).
