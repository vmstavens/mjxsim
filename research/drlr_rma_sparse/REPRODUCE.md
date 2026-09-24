# Reproduce on another machine

## Transfer

Create a bundle as described in README.md, retain its printed SHA256, and transfer
it and the verifier. Verify before extracting into an empty directory:

```bash
python3 export_bundle.py --verify drlr-rma-sparse.zip
python3 -m zipfile -e drlr-rma-sparse.zip reproduction
cd reproduction/lab
```

The bundle preserves working files, including local modifications, rather than
assuming Git HEAD describes the implementation. Its manifest records provenance.

Included teacher:
`artifacts/models/diffusion_policy/align_target_align_state48_spacemouse_rot45_stiff10m_stride3_v1`.
Included RL replay dataset:
`data/grasped_end_dlo_target_align_sparse/processed/align_processed_target_align_state48_spacemouse_large_stride3_v1`.
The teacher manifest names a different teacher-training dataset. The supplied RL
replay dataset does not reproduce the original DP training.

## Environment

Use Linux, Python 3.11, uv, and an NVIDIA GPU/driver compatible with the lab's
locked CUDA 13 JAX environment. The lab pins `jax[cuda13]==0.10.2`. The portable
mjxsim dependency set alone is insufficient to match it. Consult the exported
GPU/driver and package inventory. Minimum GPU memory has not been validated.

```bash
uv sync --frozen --dev
uv pip install --python .venv/bin/python --no-deps -e ../mjxsim
uv run --no-sync python -c "import jax, mjxsim; print(jax.devices()); print(mjxsim.__file__)"
```

The first step uses the lab lockfile, including Git-pinned mjsim and mjxsim.
The editable install then selects the bundled mjxsim working tree. Continue
using `--no-sync` so uv does not restore the Git dependency. Verify the printed
module path points into the extracted sibling mjxsim and JAX sees the GPU.
A different CUDA/JAX build is a new environment variant; record it explicitly.

## Plan and train

Run from the extracted lab root:

```bash
uv run --no-sync python -m workflows.train_drlr_target_align_geometry_only_dp_rma_sparse --cfg job --resolve
uv run --no-sync python -m workflows.train_drlr_target_align_geometry_only_dp_rma_sparse runtime.dry_run=true hydra.run.dir=runs/reproduction_plan
```

These compose and validate configuration without entering the sweep runner or
launching GPU training. The from-scratch job builder includes DR0 followed by
DR5 through DR100; inspect the resolved config for historical parent paths.
The reference snapshots are for comparison; use the complete lab config tree.

```bash
uv run --no-sync python -m workflows.train_drlr_target_align_geometry_only_dp_rma_sparse runtime.dry_run=false hydra.run.dir=runs/reproduction_geometry_seed0
```

This is a substantial full curriculum. A reduced-budget hardware smoke test must
use another output directory and record its overrides; it establishes execution,
not reproduction of learning results. Save resolved configs, Hydra overrides,
stage configs, summaries, evaluation records, accepted actors, matching training
states, and reference replay. Preserve the transfer manifest with the run.

## Continuation and recovery

| Operation | Required artifacts |
| --- | --- |
| Actor inference | Actor, normalizer, observation/action/RMA contracts |
| Domain handoff | Accepted actor, matching learner state when configured, reference replay |
| Interrupted-stage recovery | Supported recovery.zip, original config/assets/parents, run metadata |

Reusing completed stages in the same sweep directory differs from resuming an
interrupted stage. The base reference does not enable crash recovery by default.
Fixed-stage DRLR recovery without HIL-SERL is supported; continuous/HIL-SERL
recovery is rejected. See the exported `docs/drlr_crash_recovery.md` for current
mechanics and diagnostic modes. Its polish command requires historical artifacts
not included in this from-scratch package. An actor checkpoint alone cannot
restore optimizer, replay, simulator, or RNG state. Cross-machine GPU trajectories
are not guaranteed bitwise identical.

## Evaluate

Inspect actor-only gate records and evaluate held-out starts and preceding
levels. Record success count and episode count, DR level, tolerances, seeds,
episode lengths, and checkpoint hash. Low teacher usage or teacher-assisted
success alone does not establish actor-only performance. Phase 1 requires
privileged factors; train and evaluate Phase 2 before sensor-only deployment.

The bundle integrity checks do not establish full-training convergence or
cross-machine GPU equivalence. This package preserves inputs and implementation;
it does not promise a particular final score.
