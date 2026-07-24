# Diffusion toolkit

`mjxsim.diffusion` contains diffusion-policy components that do not depend on an
skrl agent lifecycle. The existing state diffusion agent remains available from
`mjxsim.agents` and uses these components internally.

For a small executable example that builds a policy, trains it on synthetic
tensors, samples action chunks, saves a checkpoint, and exercises warm-start
deployment, run:

```console
uv run python examples/diffusion_policy_synthetic.py
```

## FastDP-style state policy

Configure DDIM and the optional Mamba backbone when constructing the state policy:

```python
from mjxsim.agents import DP_CFG, DiffusionPolicy

cfg = DP_CFG(
    backbone="mamba",
    scheduler_type="ddim",
    num_diffusion_iters=100,
    num_inference_steps=5,
    warm_start_std=0.1,
)
policy = DiffusionPolicy.from_config(
    a_dim=action_dim,
    o_dim=observation_dim,
    config=cfg,
)
```

Install the Mamba implementation separately with:

```console
uv sync --extra fast-diffusion
```

Mamba is optional because its optimized implementation has CUDA and platform
constraints. Use `backbone="unet"` with the same DDIM and warm-start deployment
features on machines where `mamba-ssm` is unavailable.

## Reset-safe online warm starts

Keep temporal state outside the policy. This is important for vector-environment
resets and prevents live rollout state from entering replay or demonstration
batches:

```python
from mjxsim.diffusion import WarmStartDeployment

deployment = WarmStartDeployment(
    policy,
    num_envs=num_envs,
    pred_horizon=cfg.pred_horizon,
    action_dim=action_dim,
    executed_steps=cfg.action_horizon,
    warm_start_std=cfg.warm_start_std,
    device=policy.device,
)

action_chunk, info = deployment.act(observations=observation_history)
actions_to_execute = action_chunk[:, : cfg.action_horizon]

# Before predicting after one or more individual environments reset:
action_chunk, info = deployment.act(
    observations=next_observation_history,
    reset_mask=terminated | truncated,
)
```

The first prediction for an environment starts from standard Gaussian noise.
Subsequent predictions start from a narrow Gaussian around the time-aligned
remainder of its previous chunk. Tail positions are padded with the last predicted
action.

For offline IBRL/DRLR expert queries, call `policy.act` directly without an
`initial_action_chunk`; historical warm starts only have meaning in chronological
online rollout.

## Latency benchmark

The repository includes a model-only benchmark with 60D observations and 6D
actions by default:

```console
uv run python examples/benchmark_fast_diffusion.py
```

It reports parameters, latency, and action-chunk frequency for 100-step DDPM,
10-step DDIM, 5-step warm DDIM, and the available Mamba variants. It does not
measure task success; evaluate trained checkpoints in the target environment before
selecting a deployment profile.
