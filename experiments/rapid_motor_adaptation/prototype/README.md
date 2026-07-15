# RMA prototype for `LatentPipeInsert` and DRLR2 SAC

This package is an integration prototype, not a new RL agent. It keeps the RMA
encoders and deployment policy independent of DRLR2 while supplying the skrl
models and environment layout that DRLR2 needs.

## Data contracts

`RmaLatentPipeInsert` uses the 60-D `LatentPipeInsert` observation and the 6-D
action space. The Phase-1 observation stored in SAC replay is:

```text
[observation(60), previous_action(6), normalized_factors(4)]  -> 70D
```

The normalized factors are derived from the applied MJX model:

```text
[pipe_inner_radius, cable_length, cable_thickness, effective_joint_stiffness]
```

The 100 x 66 temporal history is maintained in `state.info["rma_history"]`.
It is intentionally excluded from ordinary SAC replay.

## Phase 1: DRLR2 SAC

Register `RmaLatentPipeInsert` using the existing Playground helper and the
existing pipe domain randomizer:

```python
from skrl.envs.loaders.torch import load_playground_env
from skrl.envs.wrappers.torch import wrap_env

from experiments.pipe_insert.env import default_config, domain_randomize
from experiments.rapid_motor_adaptation.prototype.expert import (
    augment_expert_observations,
)
from experiments.rapid_motor_adaptation.prototype.pipe_env import RmaLatentPipeInsert
from experiments.rapid_motor_adaptation.prototype.spec import PIPE_INSERT_RMA_SPEC
from experiments.rapid_motor_adaptation.prototype.torch_models import (
    BaseObservationPolicyAdapter,
    make_drlr2_rma_models,
)
from mjxsim.utils import load

load.register(
    env_name="RmaLatentPipeInsertPrototype",
    env_type=RmaLatentPipeInsert,
    config=default_config,
    domain_randomize_fn=domain_randomize,
    overwrite=True,
)
env = wrap_env(
    load_playground_env(
        task_name="RmaLatentPipeInsertPrototype",
        num_envs=512,
        randomization=True,
    ),
    wrapper="playground",
)

models = make_drlr2_rma_models(
    env.observation_space,
    env.action_space,
    env.device,
    PIPE_INSERT_RMA_SPEC,
)

# Keep an existing 60-D imitation policy unchanged.
models_il = {
    "policy": BaseObservationPolicyAdapter(existing_il_policy, PIPE_INSERT_RMA_SPEC)
}
```

The ordinary DRLR2 constructor can then receive `models`. Its actor optimizer
owns `policy.privileged_encoder` automatically because that encoder is a child
module of the policy.

### Runnable Phase-1 trainer

Run a short end-to-end integration smoke test on CPU:

```bash
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl \
  uv run --no-sync python \
  -m experiments.rapid_motor_adaptation.prototype.train_pipe_drlr2_phase1 \
  --smoke --cpu
```

Run a real Warp/GPU training job:

```bash
uv run --no-sync python \
  -m experiments.rapid_motor_adaptation.prototype.train_pipe_drlr2_phase1 \
  --expert-npz /path/to/pipe_insert_transitions.npz \
  --impl warp \
  --num-envs 256 \
  --timesteps 100000 \
  --memory-size 100000 \
  --batch-size 256
```

The NPZ file must contain:

```text
states or observations             [N, 60]
next_states or next_observations   [N, 60]
actions                            [N, 6] in environment units
```

It may additionally contain:

```text
rewards                            [N] or [N, 1]
terminated or dones                [N] or [N, 1]
factors                            [N, 4], normalized to [-1, 1]
next_factors                       [N, 4], normalized to [-1, 1]
```

Before DRLR2 starts, the script behavior-clones a small one-step IL policy from
the demonstrations. The script saves both normal DRLR2 checkpoints and a
`phase1_rma_policy.pt` containing the privileged encoder and conditioned actor
needed by Phase 2.

Expert replay states must use the same 70-D layout as online replay. Use
`augment_expert_observations` to add previous actions and factors. If factors
are unavailable for demonstrations, the helper uses zero, the midpoint of the
normalized range. The IL adapter still presents only the original 60 features
to the imitation policy.

## Phase 2: adaptation

Freeze the Phase-1 policy and train `phi` from sequential rollout batches:

```python
from experiments.rapid_motor_adaptation.prototype import LatentDistillationTrainer

distiller = LatentDistillationTrainer(trained_models["policy"])
metrics = distiller.update(
    history=batch_history,   # [batch, 100, 66]
    factors=batch_factors,   # [batch, 4]
)
deployment = distiller.deployment_policy()
deployment.save("actor_only_rma.pt")
```

Initial Phase-2 data can come from the privileged actor. Later datasets should
include rollouts driven by the adapted actor to reduce covariate shift.

## Actor-only deployment

`ActorOnlyRmaController` owns online temporal state and resets individual
environments without retaining stale history:

```python
from experiments.rapid_motor_adaptation.prototype import (
    ActorOnlyRmaController,
    ActorOnlyRmaPolicy,
)

policy, metadata = ActorOnlyRmaPolicy.load("actor_only_rma.pt", map_location="cuda")
controller = ActorOnlyRmaController(
    policy,
    num_envs=1,
    device="cuda",
    action_low=torch.as_tensor(env.action_space.low),
    action_high=torch.as_tensor(env.action_space.high),
)

action = controller.act(observation_60d)
controller.reset(done_mask)
```

Supplying the action bounds converts the actor's normalized SAC output to the
environment action before returning it and before appending it to history. If
the deployment environment already uses `[-1, 1]` actions, omit the bounds.

The deployment artifact contains no critic, target critic, imitation policy,
privileged encoder, replay memory, or DRLR2 selector.

## Integration cautions

- DRLR2's expert memory and online memory must have matching 70-D state spaces.
- The existing DRLR2 state-OOD statistic will include the appended fields. A
  future DRLR2 observation-routing hook should restrict that statistic to the
  common 60-D base observation.
- The current DRLR2 online selector evaluates the IL and RL candidates on
  different state sources. Actor-only deployment intentionally avoids carrying
  that selector into the deployed controller.
- A real deployment must be able to produce the ten cable-point positions used
  by the 60-D `LatentPipeInsert` observation, for example from perception.
