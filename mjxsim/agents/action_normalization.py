"""Framework-neutral action-normalization contracts.

Action bounds are part of the task definition. They may be supplied directly
from the expert controller/action space or, explicitly, inferred from a
dataset. The provenance is serialized so consumers never have to guess.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import gymnasium
import numpy as np

ActionNormalizationKind = Literal["fixed_bounds", "dataset_minmax"]


@dataclasses.dataclass(frozen=True)
class ActionNormalization:
    """Describe the physical action envelope mapped to normalized ``[-1, 1]``."""

    low: np.ndarray
    high: np.ndarray
    kind: ActionNormalizationKind = "fixed_bounds"
    contract_id: str | None = None
    units: tuple[str, ...] | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        low = np.asarray(self.low, dtype=np.float32)
        high = np.asarray(self.high, dtype=np.float32)
        if low.ndim != 1 or high.shape != low.shape:
            raise ValueError(
                "Action bounds must be one-dimensional and have equal shape"
            )
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("Action bounds must be finite")
        if not np.all(high > low):
            raise ValueError("Every action upper bound must exceed its lower bound")
        if self.kind not in ("fixed_bounds", "dataset_minmax"):
            raise ValueError(f"Unsupported action normalization kind: {self.kind}")
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported action-normalization schema version: {self.schema_version}"
            )
        if self.kind == "fixed_bounds" and not self.contract_id:
            raise ValueError("fixed_bounds requires a non-empty contract_id")
        if self.units is not None and len(self.units) != low.size:
            raise ValueError("Action units must have one entry per action dimension")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        if self.units is not None:
            object.__setattr__(self, "units", tuple(self.units))

    @property
    def size(self) -> int:
        return int(self.low.size)

    @classmethod
    def from_bounds(
        cls,
        low: Sequence[float] | np.ndarray,
        high: Sequence[float] | np.ndarray,
        *,
        contract_id: str,
        units: Sequence[str] | None = None,
    ) -> ActionNormalization:
        return cls(
            low=np.asarray(low),
            high=np.asarray(high),
            kind="fixed_bounds",
            contract_id=contract_id,
            units=None if units is None else tuple(units),
        )

    @classmethod
    def from_action_space(
        cls,
        action_space: gymnasium.Space,
        *,
        contract_id: str,
        units: Sequence[str] | None = None,
    ) -> ActionNormalization:
        if not isinstance(action_space, gymnasium.spaces.Box):
            raise TypeError("Action normalization requires a gymnasium.spaces.Box")
        return cls.from_bounds(
            action_space.low,
            action_space.high,
            contract_id=contract_id,
            units=units,
        )

    @classmethod
    def from_dataset(cls, actions: Any) -> ActionNormalization:
        """Explicitly infer bounds from a dataset.

        This is intended for exploratory or legacy datasets whose configured
        expert-controller limits are unknown. Fixed bounds are preferred.
        """

        values = np.asarray(actions, dtype=np.float32)
        if values.ndim < 2:
            raise ValueError(
                "Dataset actions must include samples and action dimensions"
            )
        axes = tuple(range(values.ndim - 1))
        return cls(
            low=np.min(values, axis=axes),
            high=np.max(values, axis=axes),
            kind="dataset_minmax",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActionNormalization:
        if "low" not in value or "high" not in value:
            raise KeyError("Action normalization requires 'low' and 'high'")
        return cls(
            low=np.asarray(value["low"]),
            high=np.asarray(value["high"]),
            kind=value.get("kind", "fixed_bounds"),
            contract_id=value.get("contract_id"),
            units=None if value.get("units") is None else tuple(value["units"]),
            schema_version=int(value.get("schema_version", 1)),
        )

    @classmethod
    def coerce(
        cls, value: ActionNormalization | Mapping[str, Any]
    ) -> ActionNormalization:
        return value if isinstance(value, cls) else cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "low": self.low.copy(),
            "high": self.high.copy(),
            "contract_id": self.contract_id,
            "units": self.units,
        }

    def assert_matches_space(
        self,
        action_space: gymnasium.Space,
        *,
        atol: float = 1e-6,
        require_fixed: bool = True,
    ) -> None:
        if require_fixed and self.kind != "fixed_bounds":
            raise ValueError(
                "RL integration requires fixed action bounds; the policy checkpoint "
                f"uses {self.kind!r}"
            )
        if not isinstance(action_space, gymnasium.spaces.Box):
            raise TypeError("Action contract matching requires a gymnasium.spaces.Box")
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        if low.shape != self.low.shape:
            raise ValueError(
                f"Action-contract shape {self.low.shape} does not match "
                f"environment shape {low.shape}"
            )
        if not np.allclose(low, self.low, rtol=0, atol=atol) or not np.allclose(
            high, self.high, rtol=0, atol=atol
        ):
            raise ValueError(
                "Imitation-policy action bounds do not match the environment "
                f"(policy low/high={self.low.tolist()}/{self.high.tolist()}, "
                f"environment low/high={low.tolist()}/{high.tolist()})"
            )

    def normalize(self, actions: Any, *, check_bounds: bool = True) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32)
        self._assert_last_dimension(values)
        if not np.all(np.isfinite(values)):
            raise ValueError("Actions contain non-finite values")
        if check_bounds and (np.any(values < self.low) or np.any(values > self.high)):
            index = np.argwhere((values < self.low) | (values > self.high))[0]
            dim = int(index[-1])
            actual = float(values[tuple(index)])
            raise ValueError(
                f"Physical action is outside the contract at index {tuple(index)}: "
                f"{actual} not in [{self.low[dim]}, {self.high[dim]}]"
            )
        return 2.0 * (values - self.low) / (self.high - self.low) - 1.0

    def denormalize(self, actions: Any, *, clip: bool = True) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float32)
        self._assert_last_dimension(values)
        if not np.all(np.isfinite(values)):
            raise ValueError("Actions contain non-finite values")
        if clip:
            values = np.clip(values, -1.0, 1.0)
        elif np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("Expected normalized actions in [-1, 1]")
        return 0.5 * (values + 1.0) * (self.high - self.low) + self.low

    def _assert_last_dimension(self, actions: np.ndarray) -> None:
        if actions.ndim == 0 or actions.shape[-1] != self.size:
            raise ValueError(
                f"Expected last action dimension {self.size}, got {actions.shape}"
            )


def action_normalization_from_legacy_stats(
    stats: Mapping[str, Any],
) -> ActionNormalization:
    """Read old DP action statistics without pretending they are fixed bounds."""

    action = stats["action"]
    return ActionNormalization(
        low=np.asarray(action["min"]),
        high=np.asarray(action["max"]),
        kind="dataset_minmax",
    )


def policy_action_normalization(policy: Any) -> ActionNormalization | None:
    """Return normalization metadata exposed by a policy, if present."""

    value = getattr(policy, "action_normalization", None)
    if value is None:
        return None
    return ActionNormalization.coerce(value)


def validate_policy_action_space(policy: Any, action_space: gymnasium.Space) -> None:
    """Validate metadata-aware imitation policies against an environment."""

    if not hasattr(policy, "action_normalization"):
        return
    normalization = policy_action_normalization(policy)
    if normalization is None:
        raise ValueError(
            "Imitation policy exposes action normalization support but has no "
            "action-normalization contract"
        )
    normalization.assert_matches_space(action_space)


__all__ = [
    "ActionNormalization",
    "ActionNormalizationKind",
    "action_normalization_from_legacy_stats",
    "policy_action_normalization",
    "validate_policy_action_space",
]
