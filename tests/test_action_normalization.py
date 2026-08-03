from __future__ import annotations

import gymnasium
import numpy as np
import pytest

from mjxsim.agents.action_normalization import ActionNormalization


def test_fixed_action_normalization_round_trip_and_endpoints() -> None:
    normalization = ActionNormalization.from_bounds(
        low=[-2.0, 1.0],
        high=[4.0, 5.0],
        contract_id="asymmetric_v1",
        units=["m", "rad"],
    )
    physical = np.array([[-2.0, 1.0], [1.0, 3.0], [4.0, 5.0]])

    normalized = normalization.normalize(physical)

    assert np.allclose(normalized, [[-1, -1], [0, 0], [1, 1]])
    assert np.allclose(normalization.denormalize(normalized), physical)
    assert normalization.to_dict()["kind"] == "fixed_bounds"


def test_dataset_minmax_is_explicit_and_not_an_rl_contract() -> None:
    actions = np.array([[-0.2, 1.0], [0.4, 3.0]], dtype=np.float32)
    normalization = ActionNormalization.from_dataset(actions)
    space = gymnasium.spaces.Box(
        low=np.array([-0.2, 1.0], dtype=np.float32),
        high=np.array([0.4, 3.0], dtype=np.float32),
    )

    assert normalization.kind == "dataset_minmax"
    with pytest.raises(ValueError, match="requires fixed action bounds"):
        normalization.assert_matches_space(space)


def test_out_of_contract_demonstration_is_rejected_not_clipped() -> None:
    normalization = ActionNormalization.from_bounds(
        [-1.0, -2.0], [1.0, 2.0], contract_id="expert_v1"
    )

    with pytest.raises(ValueError, match="outside the contract"):
        normalization.normalize(np.array([[1.1, 0.0]], dtype=np.float32))

    assert np.allclose(
        normalization.denormalize(np.array([[1.2, -1.5]], dtype=np.float32)),
        [[1.0, -2.0]],
    )


def test_environment_bounds_must_match_checkpoint_contract() -> None:
    normalization = ActionNormalization.from_bounds(
        [-0.4], [0.4], contract_id="expert_v1"
    )
    matching = gymnasium.spaces.Box(-0.4, 0.4, shape=(1,), dtype=np.float32)
    wider = gymnasium.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    normalization.assert_matches_space(matching)
    with pytest.raises(ValueError, match="do not match"):
        normalization.assert_matches_space(wider)
