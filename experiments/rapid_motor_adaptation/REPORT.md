# Rapid Motor Adaptation Verification Report

This report explains how to train and test the current RMA implementation in
`experiments/rapid_motor_adaptation`.

## Current state

The implemented code now includes a compact end-to-end RMA training path. The
default command-line budgets are intentionally small so the scripts can be
smoke-tested quickly; increase the iteration/env/rollout counts for meaningful
locomotion performance.

Implemented:

- `env.py`: local `RmaSpotJoystick` environment based on MuJoCo Playground's
  `SpotFlatTerrainJoystick`.
- `terrain.py`: deterministic rough hfield generation and local rough-terrain
  Spot XML.
- `networks.py`: Flax modules for the RMA encoder `mu`, base policy `pi`, value
  function, and adaptation module `phi`.
- `smoke.py`: end-to-end environment and network smoke runner.
- `train_phase1.py`: PPO training for the privileged encoder `mu`, base policy
  `pi`, and value function.
- `train_phase2.py`: supervised adaptation-module training for `phi`.
- `evaluate.py`: rollout evaluation for privileged, RMA, and no-adaptation
  modes.
- `test_rma_components.py`: focused component tests.

Still intentionally minimal:

- PPO uses a local compact implementation rather than Brax's full trainer.
- Defaults are smoke-scale, not paper-scale.
- Checkpoints are simple pickle files under `.runs/`.

## Environment setup

Use the repository root as the working directory:

```bash
cd /home/vims/git/mjxsim
```

If JAX can see the NVIDIA 5080 correctly, run commands without
`JAX_PLATFORMS=cpu`. If CUDA initialization fails, force CPU for component
verification:

```bash
UV_CACHE_DIR=/tmp/uv-cache JAX_PLATFORMS=cpu uv run --no-sync python -m experiments.rapid_motor_adaptation.smoke --steps 1
```

`UV_CACHE_DIR=/tmp/uv-cache` avoids sandbox/cache permission issues.

## Component smoke tests

Run rough-terrain Spot RMA smoke test:

```bash
UV_CACHE_DIR=/tmp/uv-cache JAX_PLATFORMS=cpu uv run --no-sync python -m experiments.rapid_motor_adaptation.smoke --steps 1
```

Run flat-terrain Spot RMA smoke test:

```bash
UV_CACHE_DIR=/tmp/uv-cache JAX_PLATFORMS=cpu uv run --no-sync python -m experiments.rapid_motor_adaptation.smoke --flat --steps 1
```

Expected output includes:

```text
observation_size: {'env_factors': (17,), 'privileged_state': (167,), 'rma_history': (25, 42), 'rma_state': (30,), 'state': (81,)}
action_size: 12
rma_state_shape: (30,)
rma_history_shape: (25, 42)
env_factors_shape: (17,)
z_shape: (8,)
z_hat_shape: (8,)
action_shape: (12,)
value_shape: ()
```

This verifies:

- The local Spot environment resets and steps.
- Rough terrain loads.
- RMA observation keys are present.
- The 25-step history buffer has the expected shape.
- The 17-D privileged environment-factor vector is present.
- `mu`, `pi`, `phi`, and value networks initialize and run forward passes.

## Manual component tests

`pytest` is not currently installed in the environment, but the test functions
can be run directly:

```bash
UV_CACHE_DIR=/tmp/uv-cache JAX_PLATFORMS=cpu uv run --no-sync python -c "from experiments.rapid_motor_adaptation.test_rma_components import test_network_shapes, test_flat_env_reset_step_shapes, test_rough_env_reset_step_shapes, test_domain_randomizer_shapes; test_network_shapes(); test_flat_env_reset_step_shapes(); test_rough_env_reset_step_shapes(); test_domain_randomizer_shapes(); print('manual component tests passed')"
```

Expected output:

```text
manual component tests passed
```

This verifies:

- Network output shapes.
- Flat environment reset/step shapes.
- Rough environment reset/step shapes.
- Domain randomizer batch shapes and `in_axes`.

If `pytest` is later installed, run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync pytest experiments/rapid_motor_adaptation/test_rma_components.py
```

## Syntax check

Compile the experiment package:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m compileall experiments/rapid_motor_adaptation
```

Expected result: compile completes without syntax errors.

## Phase 1 training

Phase 1 trains `mu`, `pi`, and the value function with PPO. It uses privileged
`env_factors` during training:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m experiments.rapid_motor_adaptation.train_phase1 \
  --iterations 2 \
  --num-envs 4 \
  --unroll-length 8 \
  --epochs 2 \
  --output experiments/rapid_motor_adaptation/.runs
```

Expected output: per-iteration reward/loss logs and:

```text
saved experiments/rapid_motor_adaptation/.runs/phase1.pkl
progress experiments/rapid_motor_adaptation/.runs/phase1_progress.csv
```

For a longer GPU run, increase `--iterations`, `--num-envs`, and
`--unroll-length`.

For the configured Warp backend, the convenience full-run launcher is:

```bash
./experiments/rapid_motor_adaptation/run_phase1_full.sh
```

Default full-run Phase 1 settings:

- `TIMESTEPS=1000000`
- `NUM_ENVS=512`
- `UNROLL_LENGTH=32`
- `ITERATIONS=ceil(TIMESTEPS / (NUM_ENVS * UNROLL_LENGTH))`
- `EPOCHS=4`
- `NCONMAX=131072`
- `NJMAX=256`
- `NACONMAX=8192`
- `NACCDMAX=8192`
- `CCD_ITERATIONS=200`
- output: `experiments/rapid_motor_adaptation/.runs/full_warp/phase1.pkl`

Override from the shell without editing the script:

```bash
TIMESTEPS=1000000 NUM_ENVS=1024 ./experiments/rapid_motor_adaptation/run_phase1_full.sh
```

If Warp reports buffer overflows such as `nefc overflow` or `CCD overflow`,
increase the buffers:

```bash
NACCDMAX=8192 NACONMAX=8192 NJMAX=256 NCONMAX=131072 TIMESTEPS=1000000 NUM_ENVS=1024 ./experiments/rapid_motor_adaptation/run_phase1_full.sh
```

`NACONMAX` and `NACCDMAX` are separate Warp pools. `NACONMAX` controls general
contacts, while `NACCDMAX` controls the CCD-specific pool used by convex and
hfield collision. The full launchers print both values and ensure
`NACONMAX >= NACCDMAX`.

If Warp reports `Warning: opt.ccd_iterations, currently set to ..., needs to be
increased`, increase `CCD_ITERATIONS`. This is a CCD convergence iteration
budget, not a contact buffer:

```bash
CCD_ITERATIONS=200 bash experiments/rapid_motor_adaptation/run_all_skrl.sh 1000000 1024
```

If that warning still appears, try `CCD_ITERATIONS=300` or `CCD_ITERATIONS=500`.
Higher values can slow the simulation, but they are the correct fix for this
warning.

Debug the effective allocation and whether overflow starts at reset or step:

```bash
uv run --no-sync python -m experiments.rapid_motor_adaptation.debug_buffers --num-envs 512 --steps 1 --naconmax 26384 --naccdmax 26384 --njmax 556
```

For CPU-only debugging, override Warp:

```bash
JAX_PLATFORMS=cpu EXTRA_ARGS="--impl jax --flat" ITERATIONS=1 NUM_ENVS=1 UNROLL_LENGTH=1 EPOCHS=1 ./experiments/rapid_motor_adaptation/run_phase1_full.sh
```

## Phase 2 training

Phase 2 freezes Phase-1 `mu` and `pi`, collects privileged policy rollouts, and
trains `phi(rma_history)` to regress to `mu(env_factors)`:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m experiments.rapid_motor_adaptation.train_phase2 \
  --phase1 experiments/rapid_motor_adaptation/.runs/phase1.pkl \
  --iterations 2 \
  --num-envs 4 \
  --steps-per-iteration 8 \
  --epochs 2 \
  --output experiments/rapid_motor_adaptation/.runs
```

Expected output: per-iteration adaptation MSE/reward logs and:

```text
saved experiments/rapid_motor_adaptation/.runs/phase2.pkl
progress experiments/rapid_motor_adaptation/.runs/phase2_progress.csv
```

To monitor a full run live:

```bash
tail -f experiments/rapid_motor_adaptation/.runs/full_warp/phase1_progress.csv
```

To create progress plots:

```bash
uv run --no-sync python -m experiments.rapid_motor_adaptation.plot_progress --run-dir experiments/rapid_motor_adaptation/.runs/full_warp
```

This writes `phase1_progress.png` and, after Phase 2 has run,
`phase2_progress.png`.

For the configured Warp backend, the convenience full-run launcher is:

```bash
./experiments/rapid_motor_adaptation/run_phase2_full.sh
```

Default full-run Phase 2 settings:

- `ITERATIONS=1000`
- `NUM_ENVS=512`
- `STEPS_PER_ITERATION=32`
- `EPOCHS=4`
- input: `experiments/rapid_motor_adaptation/.runs/full_warp/phase1.pkl`
- output: `experiments/rapid_motor_adaptation/.runs/full_warp/phase2.pkl`

Override from the shell without editing the script:

```bash
ITERATIONS=100 NUM_ENVS=128 ./experiments/rapid_motor_adaptation/run_phase2_full.sh
```

## Evaluation

Evaluate RMA with the adaptation module:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m experiments.rapid_motor_adaptation.evaluate \
  --checkpoint experiments/rapid_motor_adaptation/.runs/phase2.pkl \
  --mode rma \
  --num-envs 4 \
  --steps 32
```

Evaluate the privileged expert, which uses `mu(env_factors)` directly:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m experiments.rapid_motor_adaptation.evaluate \
  --checkpoint experiments/rapid_motor_adaptation/.runs/phase2.pkl \
  --mode privileged \
  --num-envs 4 \
  --steps 32
```

Evaluate the no-adaptation baseline, which uses a zero extrinsics vector:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m experiments.rapid_motor_adaptation.evaluate \
  --checkpoint experiments/rapid_motor_adaptation/.runs/phase2.pkl \
  --mode no_adapt \
  --num-envs 4 \
  --steps 32
```

Expected output includes:

```text
mode=rma return_mean=... episode_length_mean=... early_done_rate=... last_reward_mean=...
```

To evaluate all three modes against the full-run checkpoint:

```bash
./experiments/rapid_motor_adaptation/run_evaluate_full.sh
```

Default full-run evaluation settings:

- `NUM_ENVS=128`
- `STEPS=1000`
- checkpoint: `experiments/rapid_motor_adaptation/.runs/full_warp/phase2.pkl`

Run a single comparison rollout for the three policy modes and save JSON:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m experiments.rapid_motor_adaptation.rollout_compare \
  --checkpoint experiments/rapid_motor_adaptation/.runs/full_warp/phase2.pkl \
  --num-envs 128 \
  --steps 1000
```

This writes `experiments/rapid_motor_adaptation/.runs/full_warp/rollout_compare.json`
unless `--output` is provided. The rollout comparison script is implemented in
`rollout_compare.py:12-38` and calls the shared evaluator for each mode.

## Full training flow

The intended process is:

1. Phase 1: train `mu` and `pi` jointly with PPO.
   - Input: `rma_state`, previous action, and `z = mu(env_factors)`.
   - Output: 12-D Spot action.
   - Saves `phase1.pkl`.
2. Phase 2: freeze `mu` and `pi`, train `phi`.
   - Collect on-policy rollouts.
   - Train `phi(rma_history)` to regress to `mu(env_factors)`.
   - Saves `phase2.pkl`.
3. Evaluation:
   - Compare privileged expert: `pi(x, a_prev, mu(e))`.
   - Compare RMA: `pi(x, a_prev, phi(history))`.
   - Compare no-adaptation baseline: `pi(x, a_prev, zero_or_default_z)`.

Minimum metrics to report:

- Episode return.
- Success / early termination rate.
- Time to fall.
- Distance traveled.
- Command-tracking error.
- Energy or torque cost.
- Performance across held-out friction, payload, motor-strength, and rough
  terrain settings.

## Algorithm-to-code map

This section maps each RMA algorithm piece to the current implementation. Line
references are for the current experiment files.

| Algorithm piece | Current implementation |
| --- | --- |
| Environment default ranges and randomization metadata | `env.default_config` in `env.py:28-46` defines friction, payload, COM shift, motor-strength, and terrain-height ranges. |
| Rough terrain Spot wrapper | `RmaSpotJoystick.__init__` in `env.py:64-104` loads Playground Spot and the local hfield XML/assets. |
| Reset-time privileged labels | `RmaSpotJoystick.reset` in `env.py:106-115` samples `env_factors` and initializes the RMA history buffer. |
| Compact proprioceptive state `x_t` | `_rma_state` in `env.py:214-225` builds the 30-D state from joint state, gravity projection, and foot contacts. |
| History buffer `h_t` | `RmaSpotJoystick.step` in `env.py:196-205` appends `[rma_state, last_action]` to the 25-step history. |
| Environment factor encoder `mu(e)` | Flax `EnvFactorEncoder` in `networks.py:48-55`; PyTorch `EnvFactorEncoder` in `torch_networks.py:75-87`. |
| Base policy `pi(x_t, a_{t-1}, z_t)` | Flax `BasePolicy` in `networks.py:58-71`; PyTorch `BasePolicy` in `torch_networks.py:90-109`. |
| Value function | Flax `ValueFunction` in `networks.py:74-85`; PyTorch `ValueFunction` in `torch_networks.py:112-130`. |
| Adaptation module `phi(h_t)` | Flax `AdaptationModule` in `networks.py:88-108`; PyTorch `AdaptationModule` in `torch_networks.py:133-162`. |
| Phase-1 privileged action/value path | `_policy_value` in `train_phase1.py:35-45` computes `z = mu(e)`, action mean, and value. |
| Phase-1 rollout collection | `_collect_rollout` in `train_phase1.py:48-97` collects PPO batches and resets done envs. |
| PPO objective | `_ppo_update` in `train_phase1.py:100-163` implements clipped PPO, value loss, entropy, and KL metrics. |
| Done-state reset helper | `reset_done_envs` in `rl.py:76-87` replaces terminated per-env leaves without touching shared MJX/Warp buffers. |
| Phase-2 privileged teacher | `_privileged_action` in `train_phase2.py:24-32` computes teacher action and target latent `mu(e)`. |
| Phase-2 data collection | `_collect_supervised_batch` in `train_phase2.py:35-71` collects histories and target latents. |
| Phase-2 adaptation loss | `_adaptation_update` in `train_phase2.py:74-89` regresses `phi(history)` to `mu(env_factors)`. |
| Evaluation policy switch | `_action_for_mode` in `evaluate.py:17-45` implements `privileged`, `rma`, and `no_adapt` actions. |
| Evaluation metrics | `evaluate` in `evaluate.py:48-94` computes return, episode length, early-done rate, and last reward. |
| Rollout comparison artifact | `rollout_compare.py:12-38` runs all three evaluation modes and saves JSON. |
| PyTorch/skrl policy wrapper | `RmaSkrlPolicy` in `torch_networks.py:196-235` exposes RMA as a skrl Gaussian policy. |
| PyTorch/skrl value wrapper | `RmaSkrlValue` in `torch_networks.py:238-250` exposes the value model for skrl PPO. |
| PyTorch/skrl model factory | `make_skrl_models` in `torch_networks.py:253-258` returns the `{"policy", "value"}` model dict expected by skrl. |

## Ablation study structure

The ablation study should be run against the same checkpoint, seeds, number of
envs, rollout length, terrain, and buffer settings. The controlled variable is
only how the latent extrinsics vector `z` is produced or which environment
randomizations are active.

Primary policy-utility ablations:

| Ablation | Mode / code path | Purpose | Expected interpretation |
| --- | --- | --- | --- |
| Privileged expert | `--mode privileged`; `evaluate.py:32-33` uses `z = mu(env_factors)` | Upper bound for the trained Phase-1 base policy. | If this performs poorly, Phase 1 has not learned locomotion. |
| RMA adaptation | `--mode rma`; `evaluate.py:34-39` uses `z = phi(history)` | Tests whether online adaptation recovers privileged performance. | Good RMA should approach privileged return and episode length. |
| No adaptation | `--mode no_adapt`; `evaluate.py:40-41` uses zero `z` | Tests utility of any extrinsics inference. | If this matches RMA, the adaptation module is not adding useful information. |

Run these with:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m experiments.rapid_motor_adaptation.rollout_compare \
  --checkpoint experiments/rapid_motor_adaptation/.runs/full_warp/phase2.pkl \
  --num-envs 128 \
  --steps 1000
```

Report at minimum:

- `return_mean(privileged)`, `return_mean(rma)`, `return_mean(no_adapt)`.
- `episode_length_mean` for the same three modes.
- `early_done_rate` for the same three modes.
- `RMA / privileged return ratio`.
- `RMA - no_adapt return delta`.

Environment-factor ablations:

| Ablation | Code to modify/control | Purpose |
| --- | --- | --- |
| No rough terrain | Pass `--flat`; `make_env` in `env.py:260-283` constructs flat vs rough terrain. | Separates adaptation to terrain from dynamics adaptation. |
| Narrow friction range | `default_config` ranges in `env.py:34-40` and `_sample_env_factors` in `env.py:227-257`. | Tests whether friction diversity is needed for useful `z`. |
| No payload variation | Set payload range to zero in `env.py:36` and `_sample_env_factors` payload branch in `env.py:230-232`. | Tests mass/COM adaptation. |
| No motor-strength variation | Set motor strength range to `1.0, 1.0` in `env.py:38` and `_sample_env_factors` in `env.py:239-244`. | Tests actuator-strength adaptation. |
| Shorter history | Change `RMA_HISTORY_LEN` in `config.py` and history update in `env.py:196-205`. | Tests whether 25-step history is required at 50 Hz. |
| Adaptation architecture removed | Compare `--mode no_adapt` to `--mode rma` via `evaluate.py:34-41`. | Measures value of learned `phi`. |

Suggested experiment table:

| Experiment | Train setting | Eval modes | Metrics |
| --- | --- | --- | --- |
| A0 | Full randomization + rough terrain | privileged / rma / no_adapt | Return, length, early-done, RMA/privileged ratio |
| A1 | Flat terrain only | privileged / rma / no_adapt | Same |
| A2 | Full terrain, no payload variation | privileged / rma / no_adapt | Same |
| A3 | Full terrain, no motor variation | privileged / rma / no_adapt | Same |
| A4 | Full terrain, narrow friction | privileged / rma / no_adapt | Same |
| A5 | Full terrain, no adaptation | no_adapt only against A0 checkpoint | Utility of `z` |

The cleanest first pass is A0 only, using `rollout_compare.py`, because it
directly answers whether the policy is useful with and without the adaptation
encoder. The factor ablations require retraining separate checkpoints because
the distribution of `env_factors` is part of Phase 1 and Phase 2 training.

## PyTorch/skrl implementation status

A PyTorch implementation has been added locally for promotion work:

- `torch_networks.py:26-55`: flat observation layout for skrl-style tensor input.
- `torch_networks.py:75-162`: PyTorch `mu`, `pi`, value, and `phi` modules.
- `torch_networks.py:165-193`: convenience `RmaTorchBundle` with privileged,
  RMA, and no-adaptation action methods.
- `torch_networks.py:196-258`: skrl-compatible Gaussian policy, deterministic
  value model, and `make_skrl_models` factory.
- `export_jax_to_torch.py`: converts Flax/JAX RMA checkpoints into a PyTorch
  `RmaTorchBundle` state dict and checks numerical parity.
- `torch_smoke.py:15-37`: shape checks for native PyTorch and skrl wrappers.

Validate the PyTorch/skrl path:

```bash
UV_CACHE_DIR=/tmp/uv-cache CUDA_VISIBLE_DEVICES= uv run --no-sync python -m experiments.rapid_motor_adaptation.torch_smoke
```

Expected output includes:

```text
flat_observation_dim: 1109
privileged_action_shape: (4, 12)
rma_action_shape: (4, 12)
no_adapt_action_shape: (4, 12)
skrl_policy_mean_shape: (4, 12)
skrl_policy_log_std_shape: (4, 12)
skrl_value_shape: (4, 1)
```

Export a trained JAX checkpoint to PyTorch:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python -m experiments.rapid_motor_adaptation.export_jax_to_torch \
  --checkpoint experiments/rapid_motor_adaptation/.runs/full_warp/phase2.pkl \
  --output experiments/rapid_motor_adaptation/.runs/full_warp/rma_torch.pt
```

Use a Phase-2 checkpoint for the complete RMA policy. Phase 1 contains the
privileged encoder, base policy, value function, and action log standard
deviation; Phase 2 adds the trained adaptation encoder `phi(history)`. The
exporter copies Flax dense kernels `[in_dim, out_dim]` into PyTorch linear
weights `[out_dim, in_dim]`, copies Conv1d kernels `[kernel, in, out]` into
`[out, in, kernel]`, saves the torch checkpoint, and checks JAX-vs-Torch output
parity on random observations.

The exporter treats `--strict-tolerance` as a warning threshold and
`--warning-tolerance` as the hard failure threshold. The defaults are `1e-5` and
`1e-2`, respectively. CPU parity is usually near `1e-7`; GPU parity can be
looser because XLA and PyTorch may choose different float32/TF32 matmul and
convolution kernels. If the action-level differences stay below `1e-2`, the
weight mapping is still considered valid for this export check.

The first middle-ground adapter now exists:

- `skrl_env.py`: `RmaSkrlSpotJoystick` packs the RMA observation into
  `obs["state"]` and `obs["privileged_state"]` so skrl's Playground wrapper can
  feed the PyTorch policy/value models.
- `train_skrl_phase1.py`: registers the skrl-facing RMA env, creates
  `RmaSkrlPolicy` and `RmaSkrlValue`, and trains them with skrl PPO.
- `run_all_skrl.sh`: convenience launcher with the same positional interface as
  `run_all.sh`: `[TIMESTEPS] [NUM_ENVS]`.

Run the skrl/PyTorch middle-ground trainer:

```bash
bash experiments/rapid_motor_adaptation/run_all_skrl.sh 1000000 1024
```

The skrl launcher accepts the same positional `[TIMESTEPS] [NUM_ENVS]` and these
environment overrides: `IMPL`, `ROLL_OUTS`, `LEARNING_EPOCHS`, `MINI_BATCHES`,
`LEARNING_RATE`, `RESULTS_DIR`, `NCONMAX`, `NACONMAX`, `NACCDMAX`, `NJMAX`, and
`CCD_ITERATIONS`.

For a small CPU smoke run:

```bash
IMPL=jax EXTRA_ARGS="--cpu --flat" ROLL_OUTS=2 LEARNING_EPOCHS=1 MINI_BATCHES=1 RESULTS_DIR=/tmp/rma_skrl_script_smoke \
  bash experiments/rapid_motor_adaptation/run_all_skrl.sh 4 2
```

Current limitation: `run_all_skrl.sh` covers Phase 1 PPO with PyTorch/skrl
models. Phase 2 supervised adaptation training still needs a PyTorch checkpoint
loader and teacher-data path for skrl-trained `mu`/`pi`.

## GPU note

The code is designed for JAX/MJX and should use GPU automatically when the local
NVIDIA 5080 CUDA 13 is visible to JAX. If CUDA plugin or cuSolver errors appear, use
`JAX_PLATFORMS=cpu` for component verification and fix the CUDA/JAX installation
before running serious PPO training.
