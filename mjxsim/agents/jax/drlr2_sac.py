"""JAX DRLR2 policy-composition shell."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jp


@dataclasses.dataclass(kw_only=True)
class ExplorationCfg:
    """Configuration for exploration noise scheduling."""

    initial_scale: float = 1.0
    final_scale: float = 1e-3
    timesteps: int | None = None


@dataclasses.dataclass(kw_only=True)
class DRLR2_SAC_CFG:
    """Configuration for the JAX DRLR2 SAC shell."""

    gradient_steps: int = 1
    batch_size: int = 64
    decision_block: bool = False
    warmup_timesteps: int = 10_000
    il_ctrl_scale: float = 1.0
    rl_ctrl_scale: float = 1.0
    discount_factor: float = 0.99
    polyak: float = 0.005
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    random_timesteps: int = 0
    learning_starts: int = 0
    grad_norm_clip: float = 0.0
    exploration: ExplorationCfg = dataclasses.field(default_factory=ExplorationCfg)
    learn_entropy: bool = True
    entropy_learning_rate: float = 3e-4
    initial_entropy_value: float = 0.2
    target_entropy: float | None = None
    soft_update_beta: float = 0.2
    actor: str = "both"
    num_envs: int = 1
    action_trans_high: list[float] = dataclasses.field(default_factory=list)
    action_trans_low: list[float] = dataclasses.field(default_factory=list)
    action_rot_high: list[float] = dataclasses.field(default_factory=list)
    action_rot_low: list[float] = dataclasses.field(default_factory=list)
    a_min_lim: list[float] = dataclasses.field(default_factory=list)
    a_max_lim: list[float] = dataclasses.field(default_factory=list)
    action_ema_alpha: float = 1.0

    def expand(self) -> None:
        if self.actor not in {"rl", "il", "both"}:
            raise ValueError("actor must be one of 'rl', 'il', or 'both'")
        if not 0.0 < self.action_ema_alpha <= 1.0:
            raise ValueError("action_ema_alpha must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        self.expand()
        return dataclasses.asdict(self)

    def copy(self) -> dict[str, Any]:
        return copy.deepcopy(self.to_dict())


DRLR2_SAC_DEFAULT_CONFIG = DRLR2_SAC_CFG()


PolicyFn = Callable[[Any, jax.Array], jax.Array]


class DRLR2:
    """JAX DRLR2 actor combiner."""

    def __init__(
        self,
        *,
        policy: PolicyFn,
        imitation_policy: PolicyFn,
        params: Any,
        imitation_params: Any,
        cfg: DRLR2_SAC_CFG | Mapping[str, Any] | None = None,
    ):
        self.cfg = _coerce_cfg(cfg)
        self.policy = policy
        self.imitation_policy = imitation_policy
        self.params = params
        self.imitation_params = imitation_params

    def act(
        self,
        observations: Any,
        *,
        actor: str | None = None,
        beta: float | None = None,
    ) -> jax.Array:
        mode = self.cfg.actor if actor is None else actor
        obs = jp.asarray(observations, dtype=jp.float32)
        if mode == "rl":
            return self.cfg.rl_ctrl_scale * self.policy(self.params, obs)
        if mode == "il":
            return self.cfg.il_ctrl_scale * self.imitation_policy(
                self.imitation_params,
                obs,
            )
        if mode != "both":
            raise ValueError("actor must be one of 'rl', 'il', or 'both'")
        mix = self.cfg.soft_update_beta if beta is None else beta
        rl_action = self.cfg.rl_ctrl_scale * self.policy(self.params, obs)
        il_action = self.cfg.il_ctrl_scale * self.imitation_policy(
            self.imitation_params,
            obs,
        )
        return mix * il_action + (1.0 - mix) * rl_action

    def update(self, batch: Any) -> None:
        del batch
        raise NotImplementedError(
            "JAX DRLR2 SAC replay-buffer updates are not implemented yet. "
            "Use this class for policy composition or port the SAC update loop next."
        )


def _coerce_cfg(cfg: DRLR2_SAC_CFG | Mapping[str, Any] | None) -> DRLR2_SAC_CFG:
    if cfg is None:
        result = DRLR2_SAC_CFG()
    elif isinstance(cfg, DRLR2_SAC_CFG):
        result = copy.deepcopy(cfg)
    elif isinstance(cfg, Mapping):
        result = DRLR2_SAC_CFG()
        for key, value in cfg.items():
            if not hasattr(result, key):
                raise ValueError(f"Invalid DRLR2 config key: {key}")
            setattr(result, key, copy.deepcopy(value))
    else:
        raise TypeError(
            "cfg must be a DRLR2_SAC_CFG, mapping, or None "
            f"(got {type(cfg).__name__})"
        )
    result.expand()
    return result
