# Rapid Motor Adaptation

`mjxsim.rma` provides reusable RMA model composition and two-phase training
components. It intentionally does not define an RL agent or an environment.

The consuming project owns:

- the deployable observation;
- the privileged factor vector and its normalization;
- exact coupling between sampled factors and applied domain randomization;
- state-action history collection;
- the RL algorithm and experiment configuration.

## Public API

Framework-neutral dimensions and layout:

```python
from mjxsim.rma import RmaObservationLayout, RmaSpec

spec = RmaSpec(
    observation_dim=60,
    action_dim=6,
    factor_dim=4,
    latent_dim=8,
    history_len=100,
)
```

Torch/skrl components:

```python
from mjxsim.rma.torch import (
    ActorOnlyRmaController,
    AdaptationEncoder,
    LatentDistillationTrainer,
    load_phase1_policy,
    make_sac_rma_models,
    save_phase1_policy,
)
```

JAX/Flax components:

```python
from mjxsim.rma.jax import (
    AdaptationEncoder,
    ConditionedActor,
    PrivilegedEncoder,
    make_networks,
    make_ppo_rma_models,
)
```

## Phase 1 with a SAC-family agent

The environment observation supplied to replay must use:

```text
[deployable observation, previous action, normalized privileged factors]
```

Create models and pass them to skrl SAC, DRLR, DRLR2, or another compatible
SAC-family agent:

```python
models = make_sac_rma_models(
    env.observation_space,
    env.action_space,
    env.device,
    spec,
)

agent = SomeSacFamilyAgent(models=models, ...)
```

The actor owns `mu`; actor optimization therefore trains the privileged
encoder. The asymmetric critics receive the raw Phase-1 observation and are
not part of deployment.

Save a portable policy for Phase 2:

```python
save_phase1_policy(
    agent.policy,
    "phase1_rma.pt",
    action_low=env.action_space.low,
    action_high=env.action_space.high,
)
```

## Phase 2

Histories are shaped `[batch, history_len, observation_dim + action_dim]`.
Keep them in trajectory storage rather than duplicating them in every
off-policy replay transition.

```python
phase1_policy, details = load_phase1_policy("phase1_rma.pt", device="cuda")
distiller = LatentDistillationTrainer(phase1_policy)

metrics = distiller.update(
    history=history_batch,
    factors=factor_batch,
)

deployment = distiller.deployment_policy()
deployment.save("actor_only_rma.pt")
```

## Actor-only deployment

```python
from mjxsim.rma.torch import ActorOnlyRmaController, ActorOnlyRmaPolicy

policy, metadata = ActorOnlyRmaPolicy.load(
    "actor_only_rma.pt",
    map_location="cuda",
)
controller = ActorOnlyRmaController(
    policy,
    num_envs=1,
    device="cuda",
    action_low=action_low,
    action_high=action_high,
)

action = controller.act(observation)
controller.reset(done_mask)
```

The deployed checkpoint contains only the conditioned actor and adaptation
encoder. It does not contain critics, the privileged encoder, replay memory,
or the training agent.

## JAX/Flax PPO models

The JAX integration uses a flat PPO observation containing the compact Phase-1
observation followed by flattened state-action history:

```text
[observation, previous action, normalized privileged factors, history]
```

Construct initialized skrl models for each RMA phase or ablation:

```python
from mjxsim.rma.jax import RmaPpoObservationLayout, make_ppo_rma_models

layout = RmaPpoObservationLayout(spec)
models = make_ppo_rma_models(
    observation_space,
    action_space,
    device,
    spec=spec,
    mode="privileged",  # "adaptation" or "no_adapt"
)
```

The underlying `PrivilegedEncoder`, `AdaptationEncoder`, `ConditionedActor`,
and `ValueFunction` are ordinary Flax modules parameterized by `RmaSpec`, so
consuming projects can use them without skrl.

JAX Phase 2 and deployment are also environment-independent:

```python
from mjxsim.rma.jax import (
    ActorOnlyRmaPolicy,
    LatentDistillationTrainer,
    act,
    initialize_controller_state,
)

distiller = LatentDistillationTrainer(
    spec,
    privileged_encoder,
    privileged_variables,
    key=key,
)
metrics = distiller.update(history_batch, factor_batch)

policy = ActorOnlyRmaPolicy(
    spec,
    actor,
    actor_variables,
    distiller.adaptation_encoder,
    distiller.adaptation_variables,
)
controller_state = initialize_controller_state(spec, num_envs=1)
action, controller_state = act(policy, controller_state, observation)
```

The JAX controller is functional: callers retain the returned controller state
and use `reset_controller_state` with environment done masks.

## Using from another project

For a sibling checkout, install this project as a dependency:

```bash
uv add --editable ../mjxsim
```

or build and install the wheel:

```bash
uv build
uv add ../mjxsim/dist/mjxsim-0.1.0-py3-none-any.whl
```

Environment-specific adapters should stay in the consuming project. For a DLO
task, define its own `RmaSpec`, factor extraction, normalization, and history
collection; import the generic encoders, SAC models, distiller, and deployment
controller from `mjxsim.rma`.
