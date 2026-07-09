"""Smoke checks for the PyTorch/skrl RMA modules."""

from __future__ import annotations

import torch

from experiments.rapid_motor_adaptation.config import SPOT_ACTION_DIM
from experiments.rapid_motor_adaptation.torch_networks import (
    RmaTorchBundle,
    RmaTorchObservationLayout,
    make_skrl_models,
)


def main() -> None:
    layout = RmaTorchObservationLayout()
    model = RmaTorchBundle(layout)
    observations = torch.zeros(4, layout.flat_dim)
    print(f"flat_observation_dim: {layout.flat_dim}")
    print(f"privileged_action_shape: {tuple(model.privileged_action(observations).shape)}")
    print(f"rma_action_shape: {tuple(model.rma_action(observations).shape)}")
    print(f"no_adapt_action_shape: {tuple(model.no_adapt_action(observations).shape)}")
    assert model.privileged_action(observations).shape == (4, SPOT_ACTION_DIM)
    assert model.rma_action(observations).shape == (4, SPOT_ACTION_DIM)
    assert model.no_adapt_action(observations).shape == (4, SPOT_ACTION_DIM)

    skrl_models = make_skrl_models(layout.flat_dim, SPOT_ACTION_DIM, "cpu")
    policy_mean, policy_outputs = skrl_models["policy"].compute(
        {"states": observations}, role="policy"
    )
    value, _ = skrl_models["value"].compute({"states": observations}, role="value")
    print(f"skrl_policy_mean_shape: {tuple(policy_mean.shape)}")
    print(f"skrl_policy_log_std_shape: {tuple(policy_outputs['log_std'].shape)}")
    print(f"skrl_value_shape: {tuple(value.shape)}")
    assert policy_mean.shape == (4, SPOT_ACTION_DIM)
    assert policy_outputs["log_std"].shape == (4, SPOT_ACTION_DIM)
    assert value.shape == (4, 1)


if __name__ == "__main__":
    main()
