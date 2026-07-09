"""Checkpoint helpers for the RMA experiment."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Saves a pickle checkpoint and returns its path."""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with ckpt_path.open("wb") as f:
        pickle.dump(payload, f)
    return ckpt_path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Loads a pickle checkpoint."""
    with Path(path).open("rb") as f:
        return pickle.load(f)
