# Algorithm contract

## Selected reference

Lab entry point: `workflows.train_drlr_target_align_geometry_only_dp_rma_sparse`.
Config: `configs/train_drlr_target_align_geometry_only_dp_rma_sparse.yaml`.

The from-scratch job builder prepends DR0 to DR5, DR10, ..., DR100. With a null
initial parent, DR0 is included despite `start_dr_level: 5`. DR is fixed within
each stage. The DP teaches the initial stage; accepted RL policies become the
references for subsequent stages.

The selected recipe enables `full_stage_continuation` and persistent reference
replay, but disables online replay inheritance. Preserve the accepted actor,
normalizer, matching learner state, and handoff replay. The older discrete
curriculum note describes fresh critics and optimizers at each stage; that is
a different contract and must not override the current config.

## Learner semantics

- Compiled SAC-based DRLR compares actor and reference candidates at the same
  state. Both critics must strictly favor the actor; ties or disagreement keep
  the reference. Batch-mean candidate selection is disabled.
- Reference replay fraction is 0.15. Warmup learns critics but not the actor.
- RMA Phase 1 conditions the actor on encoded privileged factors. Actor-only
  evaluation removes teacher selection, not privileged information. Sensor-only
  deployment requires separate Phase 2 adaptation training and evaluation.
- Observation normalization is frozen with a 0.001 standard-deviation floor.
  Previous-action context is enabled while action EMA is disabled.
- Entropy temperature is fixed at 0.001; selection-conditioned entropy backup
  is enabled. Preserve these choices when comparing implementations.

The exported `learning/reinforcement/compiled_drlr.py` is the executable reference
for exact losses, action selection, and replay handling.

## Task and schedule

The base observation has 48 values; actions have six components in metres and
radians. Control interval is 2 ms, stride 3, with SE(3) accumulation then clipping.
DP horizons: observation 2, prediction 8, execution 4; RL teacher inference uses
four diffusion steps. Preserve the dataset's episode boundaries and action units.

Sparse reward settings are success scale 100, step reward -0.01, terminal penalty
0. This is distinct from the historical strictly binary proof. Curvature expands
from straight to a 180-degree maximum, alongside cable length and pipe angle.
Other physical parameters are fixed, including Young's/shear moduli at 10 MPa;
sensor noise is disabled. Success tolerances tighten from 2 cm / 10 degrees to
1 mm / 1 degree. The composed stage config defines the domain, including
recipe overrides of the base randomization groups.

| Setting | Reference value |
| --- | --- |
| Environments | 128 |
| Maximum budget per stage | 100 million environment transitions |
| Warmup | 4,000 vector steps |
| Replay capacity / batch size | 1 million / 512 |
| Gradient steps | 4 |
| Discount / Polyak | 0.9995 / 0.005 |
| Actor / critic learning rates | 0.000025 / 0.0003 |
| Actor hidden sizes | 256, 256 |
| Critic hidden sizes | 512, 512, 256, 256, 128 |
| Action std min / initial / max | 0.005 / 0.02 / 0.04 |
| Bellman target bounds | -15 to 100 |

Budgets are per stage. Reducing environment count changes the transition count
of vector-step warmup; it is not merely a memory optimization.

## Acceptance

The gate uses 128 actor-only episodes in batches of 32, a 0.90 success threshold,
two consecutive passes, and a two-million-transition interval/minimum.
`accept_warmup_actor` separately permits acceptance of an inherited actor before
its first SAC update. Handoff collects 256 episodes, keeps at most 128 successful
episodes, and requires at least 64. Seeds are in the YAML.

Report actor-only success, episode length, teacher-selection fraction, twin-Q
disagreement, parent hashes, and evaluation seeds. Evaluate preceding levels for
forgetting and independent held-out starts for final reporting. Gate scores are
selection evidence, not an independent final benchmark.

## Other variants

The legacy sparse sweep requires an external RL0. The continuous variant keeps
one optimizer/replay trajectory and a fixed DP teacher. DR100 polish presets
require historical actor/training/replay artifacts. Their configurations are
included for context, not as interchangeable launch recipes.
