import json

import numpy as np

from mjxsim.datasets.processed_sequences import (
    ProcessedSequenceDataset,
    iter_sequence_batches,
)


def _write_dataset(path):
    observations = np.arange(24, dtype=np.float32).reshape(12, 2)
    actions = np.arange(36, dtype=np.float32).reshape(12, 3)
    episode_ends = np.asarray([5, 12], dtype=np.int64)
    manifest = {
        "schema_version": 1,
        "dataset_id": "test",
        "kind": "processed",
        "environment": "test",
        "transition_count": 12,
        "episode_count": 2,
        "observation_dim": 2,
        "action_dim": 3,
        "action_low": [-1, -1, -1],
        "action_high": [1, 1, 1],
    }
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    np.save(path / "states.npy", observations)
    np.save(path / "actions.npy", actions)
    np.save(path / "episode_ends.npy", episode_ends)
    return observations, actions


def test_sequences_never_cross_episode_boundaries(tmp_path):
    observations, actions = _write_dataset(tmp_path / "dataset")
    dataset = ProcessedSequenceDataset.load(tmp_path / "dataset")

    starts = dataset.sequence_starts(
        np.asarray([0, 1]),
        obs_horizon=2,
        pred_horizon=3,
    )
    batch = dataset.batch(starts, obs_horizon=2, pred_horizon=3)

    np.testing.assert_array_equal(starts, [0, 1, 5, 6, 7, 8])
    np.testing.assert_array_equal(batch["observations"][2], observations[5:7])
    np.testing.assert_array_equal(batch["actions"][2], actions[6:9])


def test_split_stats_and_batches_are_deterministic(tmp_path):
    observations, _ = _write_dataset(tmp_path / "dataset")
    dataset = ProcessedSequenceDataset.load(tmp_path / "dataset")

    train, validation = dataset.split_episodes(validation_fraction=0.5, seed=7)
    repeated = dataset.split_episodes(validation_fraction=0.5, seed=7)
    np.testing.assert_array_equal(train, repeated[0])
    np.testing.assert_array_equal(validation, repeated[1])
    assert set(train).isdisjoint(validation)

    stats = dataset.observation_stats(train)
    episode_start = int(dataset.episode_starts[train[0]])
    episode_end = int(dataset.episode_ends[train[0]])
    np.testing.assert_array_equal(
        stats["min"],
        observations[episode_start:episode_end].min(axis=0),
    )
    starts = np.arange(7)
    batches = list(iter_sequence_batches(starts, batch_size=3, shuffle_seed=5))
    np.testing.assert_array_equal(np.sort(np.concatenate(batches)), starts)
    assert [len(batch) for batch in batches] == [3, 3, 1]
