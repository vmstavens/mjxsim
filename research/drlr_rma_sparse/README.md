# Sparse DRLR(DP) + RMA reproduction

This is the research reference for the DLO lab's **discrete geometry-only
sparse target-alignment curriculum**. The compiled learner and curriculum still
require the lab adapter; this is not yet a standalone mjxsim training API.

- [Reproduce on another machine](REPRODUCE.md): setup, assets, commands, evaluation, recovery.
- [Algorithm contract](ALGORITHM.md): semantics and distinctions between variants.
- [Validation record](VALIDATION.md): completed checks and remaining limits.
- [Promotion map](PROMOTION.md): reusable components and task boundaries.
- `reference/`: verbatim config/contract snapshots with source revisions and hashes.
- `export_bundle.py`: standard-library-only source/asset exporter and verifier.

From the lab root with mjxsim cloned inside it:

```bash
python3 mjxsim/research/drlr_rma_sparse/export_bundle.py --lab . --output /tmp/drlr-rma-sparse.zip
python3 mjxsim/research/drlr_rma_sparse/export_bundle.py --verify /tmp/drlr-rma-sparse.zip
```

The archive contains `lab/`, `mjxsim/`, and `manifest.json`, including current
source modifications, configs, tests, both lockfiles, the DP teacher, and the
RL demonstration dataset. Each payload file has a SHA256 and size. Provenance
includes Git revisions/dirty status, installed packages when the lab venv
exists, GPU/driver information when available, and selected runtime flags.
It excludes virtual environments, Git history, and arbitrary run outputs.
Dependency installation still requires network access; this is not an offline
wheelhouse. Keep archives outside source trees and do not commit them.

`--code-only` omits the teacher and dataset and cannot train by itself.
This reference starts RL from an existing DP. Retraining that DP, reproducing a
particular historical score, resuming a live run, and sensor-only deployment
require additional artifacts. No new full GPU training result is claimed.
