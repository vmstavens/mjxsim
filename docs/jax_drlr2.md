# JAX DRLR2

Import the JAX backend explicitly:

```python
from mjxsim.agents.jax import (
    ActionNormalization,
    DRLR2,
    DRLR2_SAC_CFG,
    make_sac_models,
)
from mjxsim.utils.jax_replay import create_drlr2_memory, load_expert_memory
```

Do not use the backend-neutral `mjxsim.DRLR2` name: it remains a compatibility
alias for the Torch implementation.

## Action-domain contract

DRLR2 uses normalized actions internally:

- SAC and diffusion actors emit actions in `[-1, 1]`.
- Both critics consume actions in `[-1, 1]`.
- Online and expert replay store actions in `[-1, 1]`.
- `DRLR2.act` and `act_deterministic` return physical environment actions.
- `DRLR2.record_transition` converts executed physical actions to `[-1, 1]`.

`make_sac_models` creates normalized actor/critic action spaces while the agent
itself receives the physical environment action space. `load_expert_memory`
normalizes physical dataset actions by default.

## Construction

```python
models = make_sac_models(env.observation_space, env.action_space, device)
memory = create_drlr2_memory(
    memory_size=100_000,
    num_envs=env.num_envs,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
)
expert_memory = load_expert_memory(
    transitions,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
)
agent = DRLR2(
    models=models,
    models_il={"policy": diffusion_policy},
    memory=memory,
    expert_memory=expert_memory,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
    cfg=DRLR2_SAC_CFG(),
)
```

See `examples/jax_drlr2_playground.py` for the trainer entrypoint.

## Sequential actor handoff

Save and reload only the RL actor with `save_actor` and `load_actor`. Wrap a
loaded previous-stage actor using `FrozenActorPolicyAdapter` to use it as the
next stage's frozen IL policy.

## Diffusion normalization

Observation statistics come from the training split. Action normalization is a
separate, explicit task contract using the bounds that were available to the
expert controller:

```python
action_normalization = ActionNormalization.from_action_space(
    env.action_space,
    contract_id="grasped_end_delta_pose_v1",
)
normalization = DiffusionPolicy.compute_normalization(
    train_observations,
    action_normalization=action_normalization,
)
policy.set_stats({"obs": normalization["observation"]})
policy.set_action_normalization(action_normalization)
```

For a legacy dataset whose expert-controller limits are unknown, opt in with
`action_mode="dataset_minmax"` and provide the training actions. Such a
checkpoint is suitable for standalone evaluation but is rejected by DRLR2:
RL integration requires stable fixed bounds matching the environment.
