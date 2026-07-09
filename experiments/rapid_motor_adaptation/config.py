"""Configuration defaults for the Rapid Motor Adaptation experiment."""

from __future__ import annotations

from dataclasses import dataclass


RMA_STATE_DIM = 30
SPOT_ACTION_DIM = 12
RMA_HISTORY_LEN = 25
RMA_HISTORY_FEATURE_DIM = RMA_STATE_DIM + SPOT_ACTION_DIM
RMA_EXTRINSICS_DIM = 8

# The RMA paper reports a 17-D environment vector. We use:
# payload mass, planar payload COM offset, 12 motor strengths, friction,
# and local terrain height.
RMA_ENV_FACTOR_DIM = 17


@dataclass(frozen=True)
class RandomizationRanges:
    """Environment-factor ranges used by the local RMA experiment."""

    friction: tuple[float, float] = (0.05, 4.5)
    payload_mass: tuple[float, float] = (0.0, 6.0)
    payload_com_xy: tuple[float, float] = (-0.15, 0.15)
    motor_strength: tuple[float, float] = (0.9, 1.1)
    terrain_height: tuple[float, float] = (0.0, 0.27)


@dataclass(frozen=True)
class RmaConfig:
    """Top-level RMA experiment defaults."""

    env_name: str = "SpotFlatTerrainJoystick"
    rough_terrain: bool = True
    history_len: int = RMA_HISTORY_LEN
    state_dim: int = RMA_STATE_DIM
    action_dim: int = SPOT_ACTION_DIM
    env_factor_dim: int = RMA_ENV_FACTOR_DIM
    extrinsics_dim: int = RMA_EXTRINSICS_DIM
    ctrl_dt: float = 0.02
    sim_dt: float = 0.004
    episode_length: int = 1000
    randomization: RandomizationRanges = RandomizationRanges()
