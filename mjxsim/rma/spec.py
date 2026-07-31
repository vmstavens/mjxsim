"""Framework-neutral dimensions and observation layouts for RMA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, TypeVar


class _TensorLike(Protocol):
    shape: tuple[int, ...]

    def __getitem__(self, key): ...


TensorT = TypeVar("TensorT", bound=_TensorLike)


@dataclass(frozen=True)
class RmaSpec:
    """Problem-specific dimensions used by an RMA policy.

    The Phase-1 observation is kept compact for off-policy replay:
    ``[observation, previous_action, privileged_factors]``. Temporal history is
    stored separately and is only required by Phase 2 and deployment.
    """

    observation_dim: int
    action_dim: int
    factor_dim: int
    latent_dim: int = 8
    history_len: int = 100

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    @property
    def history_feature_dim(self) -> int:
        return self.observation_dim + self.action_dim

    @property
    def phase1_observation_dim(self) -> int:
        return self.observation_dim + self.action_dim + self.factor_dim

    @property
    def history_dim(self) -> int:
        return self.history_len * self.history_feature_dim


@dataclass(frozen=True)
class RmaObservationLayout:
    """Index the compact Phase-1 observation without depending on a framework."""

    spec: RmaSpec

    def split_phase1(self, values: TensorT) -> tuple[TensorT, TensorT, TensorT]:
        if values.shape[-1] != self.spec.phase1_observation_dim:
            raise ValueError(
                "invalid Phase-1 observation width: "
                f"expected {self.spec.phase1_observation_dim}, got {values.shape[-1]}"
            )
        obs_end = self.spec.observation_dim
        action_end = obs_end + self.spec.action_dim
        return (
            values[..., :obs_end],
            values[..., obs_end:action_end],
            values[..., action_end:],
        )

    def base_observation(self, values: TensorT) -> TensorT:
        if values.shape[-1] < self.spec.observation_dim:
            raise ValueError(
                f"expected at least {self.spec.observation_dim} features, "
                f"got {values.shape[-1]}"
            )
        return values[..., : self.spec.observation_dim]
