# Validation record

Validated on the source machine on 2026-09-24:

- Three standard-library unit checks passed: archive corruption detection,
  missing-asset rejection, and working-tree capture with run-output exclusion.
- Ruff checks passed for the Python tooling.
- The reference workflow composed and resolved successfully, and its dry run
  completed configuration validation without launching training.
- A full source/teacher/dataset archive passed per-file SHA256 and size checks.
- After extraction into an unrelated temporary directory, the workflow config
  composed successfully using the source machine's installed dependencies.
- The extracted replay dataset passed its loader validation: 46,102 transitions.
- The job builder produced 21 stages: DR0, DR5, ..., DR100.

`reference/resolved_config.yaml` captures the validated composition. The source
config snapshots and their hashes are recorded in `reference/provenance.json`.
The transfer archive manifest inventories the complete exported payload.

This does not validate a clean dependency installation, GPU training convergence,
performance on another machine, or a Phase 2 sensor-only deployment. The current
lab implementation is preserved; it has not yet been refactored into a standalone
public mjxsim curriculum API.
