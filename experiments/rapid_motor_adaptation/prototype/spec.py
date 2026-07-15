"""Pipe-specific RMA specification and compatibility imports."""

from mjxsim.rma import RmaObservationLayout, RmaSpec

PIPE_INSERT_RMA_SPEC = RmaSpec(
    observation_dim=60,
    action_dim=6,
    factor_dim=4,
    latent_dim=8,
    history_len=100,
)

__all__ = ["PIPE_INSERT_RMA_SPEC", "RmaObservationLayout", "RmaSpec"]
