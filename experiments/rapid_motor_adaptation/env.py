"""Spot environment wrapper for Rapid Motor Adaptation."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import io as mjx_io
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.spot import base as spot_base
from mujoco_playground._src.locomotion.spot import joystick as spot_joystick
from mujoco_playground._src.locomotion.spot import spot_constants as spot_consts

from experiments.rapid_motor_adaptation.config import (
    RMA_ENV_FACTOR_DIM,
    RMA_HISTORY_FEATURE_DIM,
    RMA_HISTORY_LEN,
    RMA_STATE_DIM,
)
from experiments.rapid_motor_adaptation.terrain import ROUGH_SPOT_XML, make_hfield_png


def default_config() -> config_dict.ConfigDict:
    """Returns the default Spot joystick config with RMA metadata."""
    cfg = spot_joystick.default_config()
    cfg.history_len = 3
    cfg.rma_history_len = RMA_HISTORY_LEN
    cfg.rma_rough_terrain = True
    cfg.rma_randomization = config_dict.create(
        friction=[0.05, 4.5],
        payload_mass=[0.0, 6.0],
        payload_com_xy=[-0.15, 0.15],
        motor_strength=[0.9, 1.1],
        terrain_height=[0.0, 0.27],
    )
    cfg.impl = "warp"
    cfg.nconmax = 131072
    cfg.njmax = 256
    cfg.naconmax = 8192
    cfg.naccdmax = 8192
    cfg.ccd_iterations = 200
    return cfg


class RmaSpotJoystick(spot_joystick.Joystick):
    """Spot joystick task augmented with RMA observations and history.

    The environment keeps Playground's normal `state` and `privileged_state`
    observation keys and adds:

    - `rma_state`: compact 30-D proprioceptive state.
    - `rma_history`: `(25, 42)` state-action history.
    - `env_factors`: privileged environment factors for `mu(e)`.

    Dynamics randomization for vectorized PPO should be supplied by
    `domain_randomize`; the reset-time `env_factors` are the privileged labels
    consumed by the RMA encoder/adaptation losses.
    """

    def __init__(
        self,
        *,
        rough_terrain: bool = True,
        config: config_dict.ConfigDict | None = None,
        config_overrides: dict[str, Any] | None = None,
    ):
        cfg = config or default_config()
        self._rough_terrain = rough_terrain

        if rough_terrain:
            mjx_env.MjxEnv.__init__(self, config=cfg, config_overrides=config_overrides)
            self._model_assets = spot_base.get_assets()
            self._model_assets["hfield.png"] = make_hfield_png()
            self._mj_model = mujoco.MjModel.from_xml_string(
                ROUGH_SPOT_XML, assets=self._model_assets
            )
            self._mj_model.opt.timestep = self._config.sim_dt
            self._mj_model.opt.ccd_iterations = self._config.ccd_iterations
            self._mj_model.dof_damping[6:] = self._config.Kd
            self._mj_model.actuator_gainprm[:, 0] = self._config.Kp
            self._mj_model.actuator_biasprm[:, 1] = -self._config.Kp
            self._mj_model.vis.global_.offwidth = 3840
            self._mj_model.vis.global_.offheight = 2160
            self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
            self._xml_path = "<rma_rough_spot_xml>"
            self._feet_floor_found_sensor = [
                self._mj_model.sensor(f"{geom}_floor_found").id
                for geom in spot_consts.FEET_GEOMS
            ]
            self._post_init()
            self._pert_func = (
                self._maybe_apply_perturbation
                if cfg.pert_config.enable
                else lambda state, _: state
            )
        else:
            super().__init__(
                task="flat_terrain",
                config=cfg,
                config_overrides=config_overrides,
            )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = self._reset_spot(rng)
        env_factors = self._sample_env_factors(state.info["rng"])
        info = dict(state.info)
        info["env_factors"] = env_factors
        info["rma_history"] = jp.zeros(
            (self._config.rma_history_len, RMA_HISTORY_FEATURE_DIM)
        )
        state = state.replace(info=info)
        return self._with_rma_obs(state)

    def _make_data(self, qpos: jax.Array, qvel: jax.Array) -> mjx.Data:
        naccdmax = getattr(self._config, "naccdmax", None)
        if self.mjx_model.impl.value == "warp" and naccdmax is not None:
            data = _make_data_warp_with_naccdmax(
                self.mj_model,
                naconmax=getattr(self._config, "naconmax", None),
                naccdmax=naccdmax,
                njmax=self._config.njmax,
            )
        else:
            data = mjx.make_data(
                self.mj_model,
                impl=self.mjx_model.impl.value,
                nconmax=self._config.nconmax,
                naconmax=getattr(self._config, "naconmax", None),
                njmax=self._config.njmax,
            )
        data = data.replace(qpos=qpos, qvel=qvel)
        return mjx.forward(self.mjx_model, data)

    def _reset_spot(self, rng: jax.Array) -> mjx_env.State:
        data = self._make_data(
            qpos=self._init_q,
            qvel=jp.zeros(self.mjx_model.nv),
        )

        rng, key1, key2, key3 = jax.random.split(rng, 4)
        time_until_next_pert = jax.random.uniform(
            key1,
            minval=self._config.pert_config.kick_wait_times[0],
            maxval=self._config.pert_config.kick_wait_times[1],
        )
        steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(
            jp.int32
        )
        pert_duration_seconds = jax.random.uniform(
            key2,
            minval=self._config.pert_config.kick_durations[0],
            maxval=self._config.pert_config.kick_durations[1],
        )
        pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(
            jp.int32
        )
        pert_mag = jax.random.uniform(
            key3,
            minval=self._config.pert_config.velocity_kick[0],
            maxval=self._config.pert_config.velocity_kick[1],
        )

        rng, cmd_rng, noise_rng = jax.random.split(rng, 3)
        info = {
            "rng": rng,
            "step": 0,
            "command": self.sample_command(cmd_rng),
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": jp.zeros(self.mjx_model.nu),
            "qpos_error_history": jp.zeros(self._config.history_len * 12),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
            "steps_until_next_pert": steps_until_next_pert,
            "pert_duration_seconds": pert_duration_seconds,
            "pert_duration": pert_duration_steps,
            "steps_since_last_pert": 0,
            "pert_steps": 0,
            "pert_dir": jp.zeros(3),
            "pert_mag": pert_mag,
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        obs = self._get_obs(data, info, noise_rng)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = super().step(state, action)
        rma_state = self._rma_state(state.data, state.info)
        transition = jp.concatenate([rma_state, state.info["last_act"]])
        history = jp.roll(state.info["rma_history"], shift=-1, axis=0)
        history = history.at[-1].set(transition)
        info = dict(state.info)
        info["rma_history"] = history
        state = state.replace(info=info)
        return self._with_rma_obs(state)

    def _with_rma_obs(self, state: mjx_env.State) -> mjx_env.State:
        obs = dict(state.obs)
        obs["rma_state"] = self._rma_state(state.data, state.info)
        obs["rma_history"] = state.info["rma_history"]
        obs["env_factors"] = state.info["env_factors"]
        return state.replace(obs=obs)

    def _rma_state(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        joint_pos = data.qpos[7:] - self._default_pose
        joint_vel = data.qvel[6:]
        gravity_xy = self.get_gravity(data)[:2]
        contacts = jp.array(
            [
                data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
                for sensor_id in self._feet_floor_found_sensor
            ]
        ).astype(jp.float32)
        state = jp.concatenate([joint_pos, joint_vel, gravity_xy, contacts])
        return jp.asarray(state[:RMA_STATE_DIM], dtype=jp.float32)

    def _sample_env_factors(self, rng: jax.Array) -> jax.Array:
        ranges = self._config.rma_randomization
        rng, k_mass, k_com, k_motor, k_friction = jax.random.split(rng, 5)
        payload_mass = jax.random.uniform(
            k_mass, (), minval=ranges.payload_mass[0], maxval=ranges.payload_mass[1]
        )
        payload_com_xy = jax.random.uniform(
            k_com,
            (2,),
            minval=ranges.payload_com_xy[0],
            maxval=ranges.payload_com_xy[1],
        )
        motor_strength = jax.random.uniform(
            k_motor,
            (12,),
            minval=ranges.motor_strength[0],
            maxval=ranges.motor_strength[1],
        )
        friction = jax.random.uniform(
            k_friction, (), minval=ranges.friction[0], maxval=ranges.friction[1]
        )
        terrain_height = jp.where(self._rough_terrain, ranges.terrain_height[1], 0.0)
        factors = jp.concatenate(
            [
                jp.array([payload_mass]),
                payload_com_xy,
                motor_strength,
                jp.array([friction, terrain_height]),
            ]
        )
        return jp.asarray(factors[:RMA_ENV_FACTOR_DIM], dtype=jp.float32)


def make_env(
    *,
    rough_terrain: bool = True,
    impl: str | None = None,
    nconmax: int | None = None,
    njmax: int | None = None,
    naconmax: int | None = None,
    naccdmax: int | None = None,
    ccd_iterations: int | None = None,
) -> RmaSpotJoystick:
    """Creates the local RMA Spot joystick environment."""
    config = default_config()
    if impl is not None:
        config.impl = impl
    if nconmax is not None:
        config.nconmax = nconmax
    if njmax is not None:
        config.njmax = njmax
    if naconmax is not None:
        config.naconmax = naconmax
    if naccdmax is not None:
        config.naccdmax = naccdmax
    if ccd_iterations is not None:
        config.ccd_iterations = ccd_iterations
    if config.naccdmax > config.naconmax:
        config.naconmax = config.naccdmax
    return RmaSpotJoystick(rough_terrain=rough_terrain, config=config)


def _make_data_warp_with_naccdmax(
    model: mujoco.MjModel,
    *,
    naconmax: int | None,
    naccdmax: int,
    njmax: int | None,
) -> mjx.Data:
    """Allocates Warp MJX data while exposing MuJoCo Warp's CCD buffer size.

    Public `mjx.make_data` forwards `naconmax` but not `naccdmax`. The RMA rough
    hfield can exhaust the CCD-specific pool even when the contact pool is large,
    so keep this workaround local to the experiment until MJX exposes the knob.
    """
    if naconmax is not None and naccdmax > naconmax:
        raise ValueError(f"naccdmax ({naccdmax}) must be <= naconmax ({naconmax})")

    _, device = mjx_io._resolve_impl_and_device("warp", None)  # pylint: disable=protected-access

    with mjx_io.wp.ScopedDevice("cpu"):
        warp_data = mjx_io.mjwp.make_data(
            model,
            nworld=1,
            naconmax=naconmax,
            naccdmax=naccdmax,
            njmax=njmax,
        )

    fields = mjx_io._make_data_public_fields(model)  # pylint: disable=protected-access
    for name in fields:
        if name in {"userdata", "plugin_state", "history"}:
            continue
        if not hasattr(warp_data, name):
            raise ValueError(f"Public data field {name} not found in Warp data.")
        field = mjx_io._wp_to_np_type(getattr(warp_data, name))  # pylint: disable=protected-access
        if mjx_io.mjxw.types._BATCH_DIM["Data"][name]:  # pylint: disable=protected-access
            field = field.reshape(field.shape[1:])
        fields[name] = field

    impl_fields = {}
    for name in mjx_io.mjxw.types.DataWarp.__annotations__:
        field = mjx_io._get_nested_attr(  # pylint: disable=protected-access
            warp_data, name, split="__"
        )
        field = mjx_io._wp_to_np_type(field)  # pylint: disable=protected-access
        if mjx_io.mjxw.types._BATCH_DIM["Data"][name]:  # pylint: disable=protected-access
            field = field.reshape(field.shape[1:])
        impl_fields[name] = field

    data = mjx_io.types.Data(
        qpos=model.qpos0.astype(np.float32),
        eq_active=model.eq_active0.astype(bool),
        **fields,
        _impl=mjx_io.mjxw.types.DataWarp(**impl_fields),
    )

    data = jax.device_put(data, device=device)

    with mjx_io.wp.ScopedDevice("cuda:0"):
        warp_data = mjx_io.mjwp.make_data(
            model,
            nworld=1,
            naconmax=naconmax,
            naccdmax=naccdmax,
            njmax=njmax,
        )
        warp_model = mjx_io.mjwp.put_model(model)
        _ = mjx_io.mjwp.step(warp_model, warp_data)
    del warp_data, warp_model

    return data


def domain_randomize(model: mjx.Model, rng: jax.Array):
    """Domain-randomizes Spot dynamics for vectorized Playground/Brax training."""

    floor_geom_id = 0
    torso_body_id = 1

    @jax.vmap
    def rand_dynamics(key):
        key, friction_key, mass_key, com_key, motor_key = jax.random.split(key, 5)

        friction = jax.random.uniform(friction_key, (), minval=0.05, maxval=4.5)
        geom_friction = model.geom_friction.at[floor_geom_id, 0].set(friction)

        payload = jax.random.uniform(mass_key, (), minval=0.0, maxval=6.0)
        body_mass = model.body_mass.at[torso_body_id].set(
            model.body_mass[torso_body_id] + payload
        )

        dpos_xy = jax.random.uniform(com_key, (2,), minval=-0.15, maxval=0.15)
        body_ipos = model.body_ipos.at[torso_body_id, :2].set(
            model.body_ipos[torso_body_id, :2] + dpos_xy
        )

        motor_strength = jax.random.uniform(
            motor_key, (model.nu,), minval=0.9, maxval=1.1
        )
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(
            model.actuator_gainprm[:, 0] * motor_strength
        )
        actuator_biasprm = model.actuator_biasprm.at[:, 1].set(
            model.actuator_biasprm[:, 1] * motor_strength
        )

        return geom_friction, body_mass, body_ipos, actuator_gainprm, actuator_biasprm

    (
        geom_friction,
        body_mass,
        body_ipos,
        actuator_gainprm,
        actuator_biasprm,
    ) = rand_dynamics(rng)

    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "body_mass": 0,
            "body_ipos": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    )
    model = model.tree_replace(
        {
            "geom_friction": geom_friction,
            "body_mass": body_mass,
            "body_ipos": body_ipos,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
        }
    )
    return model, in_axes
