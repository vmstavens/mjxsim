# Promotion map

Paths below refer to the exported lab tree. Proposed responsibilities are not
public APIs that already exist.

| Source | Promote into mjxsim | Keep in task adapter |
| --- | --- | --- |
| learning/reinforcement/compiled_drlr.py | Learner state, updates, reference selection, replay, compiled rollout | DLO observation transplants, task selection, environment creation |
| learning/reinforcement/brax_networks.py | Configurable conditioned actors and privileged critics | Concrete RMA specs and observation widths |
| learning/reinforcement/rma.py | Generic policy and previous-action adapters | Cable factors, target-frame transforms, constants |
| learning/reinforcement/sweep.py | Ordered stages, acceptance, artifact handoff | Named experiment families and physical schedules |
| learning/reinforcement/crash_recovery.py | Versioned state archive and integrity checks | Simulator reconstruction and task identity |
| learning/reinforcement/phase1_checkpoint.py | Policy metadata and contract validation | Task metadata values |
| learning/reinforcement/ppo_rma_phase2.py | History adaptation and deployment state | Dataset/environment factories and task evaluation |
| learning/domain_randomization.py | Generic schedule primitives where useful | Physical ranges and geometry transforms |

Extend existing mjxsim.agents.jax, mjxsim.rma, and mjxsim.trainers.jax where
appropriate. Avoid creating another public DRLR implementation by copying the
entire task-coupled compiled module.

Extract explicit environment, teacher, network, replay, and evaluation interfaces.
The core should not import lab learning/envs modules or name cable/pipe objects.
Keep the lab implementation as a reference until it delegates to the new core.

For each extraction compare fixed-input/RNG candidate selection (including ties),
one learner update, action normalization/history resets, replay proportions,
checkpoint round-trip, and stage handoff. Carry relevant lab tests. Then exercise
a short GPU rollout and a second small environment before claiming portability.
The exporter preserves all source now without presenting task-specific research
code as a finished library API.
