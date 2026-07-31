# IBRL and DRLR agents

`mjxsim.agents` exposes imitation-bootstrapped agents with SAC and TD3
backbones:

| Algorithm | SAC | TD3 |
| --- | --- | --- |
| Per-sample RL/IL selection | `IBRL` | `IBRLTD3` |
| Batch-level RL/IL selection | `DRLR` | `DRLRTD3` |

All variants use a trainable RL policy, a frozen imitation policy, twin critics,
online replay, and optional demonstration replay. The critics compare RL and
imitation candidate actions. IBRL selects per sample, while DRLR can make one
decision for the whole vector-environment batch.

## Imports

```python
from mjxsim.agents import (
    DRLR,
    DRLRTD3,
    DRLR_SAC_CFG,
    DRLR_TD3_CFG,
    IBRL,
    IBRLTD3,
    IBRL_SAC_CFG,
    IBRL_TD3_CFG,
)
```

The agents build on SKRL. Applications provide the SKRL policy/critic model
dictionaries, online and expert memories, spaces, and an imitation policy under
the `models_il["policy"]` key.

For TD3, `models` must contain:

- `policy` and `target_policy`;
- `critic_1`, `critic_2`, `target_critic_1`, and `target_critic_2`.

The imitation policy receives a two-observation history with shape
`[batch, 2, observation_dim]`. It may return either one action per sample or an
action plan; for a plan, the agent executes the first action. Imitation actions
must already use the environment action coordinates.

Important TD3 options include `warmup_timesteps`, `actor`,
`expert_batch_ratio`, and `action_ema_alpha`. `DRLR_TD3_CFG` additionally
provides `decision_block`, `il_ctrl_scale`, and `rl_ctrl_scale`.

This repository intentionally provides reusable package components rather than
task-specific training command-line programs. Environment construction,
demonstration loading, and training entry points belong in the consuming
project.
