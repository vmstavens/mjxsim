"""Manifest-backed state/action sequences for Diffusion Policy training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProcessedSequenceDataset:
    """Memory-mapped processed demonstrations with episode-safe windows."""

    path: Path
    manifest: dict[str, Any]
    observations: np.ndarray
    actions: np.ndarray
    episode_ends: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> ProcessedSequenceDataset:
        root = Path(path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("kind") != "processed":
            raise ValueError("Expected a processed demonstration dataset")

        observations = np.load(root / "states.npy", mmap_mode="r")
        actions = np.load(root / "actions.npy", mmap_mode="r")
        episode_ends = np.load(root / "episode_ends.npy", mmap_mode="r")
        transition_count = int(manifest["transition_count"])
        observation_dim = int(manifest["observation_dim"])
        action_dim = int(manifest["action_dim"])
        episode_count = int(manifest["episode_count"])
        if observations.shape != (transition_count, observation_dim):
            raise ValueError("states.npy shape does not match the manifest")
        if actions.shape != (transition_count, action_dim):
            raise ValueError("actions.npy shape does not match the manifest")
        if (
            episode_ends.shape != (episode_count,)
            or int(episode_ends[-1]) != transition_count
            or np.any(np.diff(episode_ends) <= 0)
        ):
            raise ValueError("episode_ends.npy does not match the manifest")
        return cls(root, manifest, observations, actions, episode_ends)

    @property
    def observation_dim(self) -> int:
        return int(self.manifest["observation_dim"])

    @property
    def action_dim(self) -> int:
        return int(self.manifest["action_dim"])

    @property
    def episode_starts(self) -> np.ndarray:
        return np.concatenate([np.asarray([0], dtype=np.int64), self.episode_ends[:-1]])

    @property
    def action_low(self) -> np.ndarray:
        value = np.asarray(self.manifest.get("action_low"), dtype=np.float32)
        if value.shape != (self.action_dim,):
            raise ValueError("The manifest requires fixed action_low bounds")
        return value

    @property
    def action_high(self) -> np.ndarray:
        value = np.asarray(self.manifest.get("action_high"), dtype=np.float32)
        if value.shape != (self.action_dim,):
            raise ValueError("The manifest requires fixed action_high bounds")
        return value

    def split_episodes(
        self,
        *,
        validation_fraction: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        episode_indices = np.arange(len(self.episode_ends), dtype=np.int64)
        shuffled = np.random.default_rng(seed).permutation(episode_indices)
        validation_count = round(len(episode_indices) * validation_fraction)
        validation_count = min(max(1, validation_count), len(episode_indices) - 1)
        return (
            np.sort(shuffled[validation_count:]),
            np.sort(shuffled[:validation_count]),
        )

    def sequence_starts(
        self,
        episode_indices: np.ndarray,
        *,
        obs_horizon: int,
        pred_horizon: int,
    ) -> np.ndarray:
        window_length = obs_horizon + pred_horizon - 1
        starts = []
        episode_starts = self.episode_starts
        for episode_index in episode_indices:
            start = int(episode_starts[episode_index])
            end = int(self.episode_ends[episode_index])
            count = end - start - window_length + 1
            if count > 0:
                starts.append(np.arange(start, start + count, dtype=np.int64))
        if not starts:
            raise ValueError("The episode split contains no valid sequences")
        return np.concatenate(starts)

    def batch(
        self,
        starts: np.ndarray,
        *,
        obs_horizon: int,
        pred_horizon: int,
    ) -> dict[str, np.ndarray]:
        starts = np.asarray(starts, dtype=np.int64)
        observation_indices = starts[:, None] + np.arange(obs_horizon)
        action_indices = starts[:, None] + obs_horizon - 1 + np.arange(pred_horizon)
        return {
            "observations": np.asarray(
                self.observations[observation_indices],
                dtype=np.float32,
            ),
            "actions": np.asarray(
                self.actions[action_indices],
                dtype=np.float32,
            ),
        }

    def observation_stats(
        self,
        episode_indices: np.ndarray,
    ) -> dict[str, np.ndarray]:
        minimum = np.full(self.observation_dim, np.inf, dtype=np.float32)
        maximum = np.full(self.observation_dim, -np.inf, dtype=np.float32)
        episode_starts = self.episode_starts
        for episode_index in episode_indices:
            values = self.observations[
                int(episode_starts[episode_index]) : int(
                    self.episode_ends[episode_index]
                )
            ]
            minimum = np.minimum(minimum, values.min(axis=0))
            maximum = np.maximum(maximum, values.max(axis=0))
        return {"min": minimum, "max": maximum}


def iter_sequence_batches(
    starts: np.ndarray,
    *,
    batch_size: int,
    shuffle_seed: int | None = None,
):
    """Yield fixed-size slices of sequence starts, including the final partial batch."""
    indices = np.arange(len(starts), dtype=np.int64)
    if shuffle_seed is not None:
        indices = np.random.default_rng(shuffle_seed).permutation(indices)
    for offset in range(0, len(indices), batch_size):
        yield starts[indices[offset : offset + batch_size]]


__all__ = ["ProcessedSequenceDataset", "iter_sequence_batches"]
