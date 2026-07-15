import enum
import json
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import jaxlie as jaxl
import mjsim as ms
import mujoco
import mujoco as mj
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from mjxsim import ObjType, cable, get_pose, pipe, set_pose, set_state

from .helpers import (
    apply_random_pre_curvature,
    cable_ball_joint_qpos_adrs,
)


class Params(enum.Enum):
    THICK = enum.auto()
    THIN = enum.auto()
    MING_LI = enum.auto()


MODE = Params.MING_LI

if MODE is Params.THICK:
    # thick ################################
    scale = 1000
    DEFAULT_PIPE_INNER_RADIUS = 0.033 / 2
    DEFAULT_PIPE_DR_INNER_RADIUS = (0.031 / 2, 0.035 / 2)
    DEFAULT_CABLE_LENGTH = 0.4
    DEFAULT_CABLE_THICKNESS = 0.025 / 2
    DEFAULT_CABLE_SHEAR_MODULUS = 5e7 / scale
    DEFAULT_CABLE_YOUNGS_MODULUS = 4e7 / scale
    DEFAULT_CABLE_DR_LENGTH = (0.36, 0.44)
    DEFAULT_CABLE_DR_THICKNESS = (0.02 / 2, 0.025 / 2)
    DEFAULT_CABLE_DR_SHEAR_MODULUS = (4.5e7 / scale, 5e7 / scale)
    DEFAULT_CABLE_DR_YOUNGS_MODULUS = (3.5e7 / scale, 4e7 / scale)
elif MODE is Params.THIN:
    # thin #################################
    DEFAULT_PIPE_INNER_RADIUS = 0.033 / 2
    DEFAULT_PIPE_DR_INNER_RADIUS = (0.031 / 2, 0.035 / 2)
    DEFAULT_CABLE_LENGTH = 0.4
    DEFAULT_CABLE_THICKNESS = 0.002
    DEFAULT_CABLE_SHEAR_MODULUS = 6e7
    DEFAULT_CABLE_YOUNGS_MODULUS = 5e7
    DEFAULT_CABLE_DR_LENGTH = (0.36, 0.44)
    DEFAULT_CABLE_DR_THICKNESS = (0.0018, 0.0022)
    DEFAULT_CABLE_DR_SHEAR_MODULUS = (5e7, 7e7)
    DEFAULT_CABLE_DR_YOUNGS_MODULUS = (4e7, 6e7)
elif MODE is Params.MING_LI:
    scale = 30
    DEFAULT_PIPE_INNER_RADIUS = 0.01  # radius of 1 cm
    DEFAULT_PIPE_DR_INNER_RADIUS = (0.01, 0.02)  # keep below 0.02175 m outer radius
    DEFAULT_CABLE_LENGTH = 0.4
    DEFAULT_CABLE_THICKNESS = 0.005  # radius in m
    DEFAULT_CABLE_SHEAR_MODULUS = 6e7 / scale
    DEFAULT_CABLE_YOUNGS_MODULUS = 5e7 / scale
    DEFAULT_CABLE_DR_LENGTH = (0.36, 0.44)
    DEFAULT_CABLE_DR_THICKNESS = (0.0018, 0.0022)
    DEFAULT_CABLE_DR_SHEAR_MODULUS = (5e7 / scale, 7e7 / scale)
    DEFAULT_CABLE_DR_YOUNGS_MODULUS = (4e7 / scale, 6e7 / scale)


# DEFAULT_CABLE_LENGTH = 0.4
# DEFAULT_CABLE_THICKNESS = 0.025 / 2
# # DEFAULT_CABLE_THICKNESS = 0.002
# DEFAULT_CABLE_SHEAR_MODULUS = 5e7 / 1000
# DEFAULT_CABLE_YOUNGS_MODULUS = 4e7 / 1000
# # DEFAULT_CABLE_SHEAR_MODULUS = 6e7
# # DEFAULT_CABLE_YOUNGS_MODULUS = 5e7

# DEFAULT_CABLE_DR_LENGTH = (0.36, 0.44)
# DEFAULT_CABLE_DR_THICKNESS = (0.02 / 2, 0.025 / 2)
# # DEFAULT_CABLE_DR_THICKNESS = (0.0018, 0.0022)

# scale = 1000

# DEFAULT_CABLE_DR_SHEAR_MODULUS = (4.5e7 / scale, 5e7 / scale)
# DEFAULT_CABLE_DR_YOUNGS_MODULUS = (3.5e7 / scale, 4e7 / scale)
# DEFAULT_CABLE_DR_SHEAR_MODULUS = (5e7, 7e7)
# DEFAULT_CABLE_DR_YOUNGS_MODULUS = (4e7, 6e7)
# DEFAULT_RESET_POS_MIN = (0, 0, 0)
# DEFAULT_RESET_POS_MAX = (0, 0, 0)
# DEFAULT_RESET_ROT_MIN = (0, 0, 0)
# DEFAULT_RESET_ROT_MAX = (0, 0, 0)
DEFAULT_RESET_POS_MIN = (-0.05, -0.05, -0.05)
DEFAULT_RESET_POS_MAX = (0.05, 0.05, 0.05)
DEFAULT_RESET_ROT_MIN = (-np.deg2rad(15), -np.deg2rad(15), -np.deg2rad(15))
DEFAULT_RESET_ROT_MAX = (np.deg2rad(15), np.deg2rad(15), np.deg2rad(15))
DEFAULT_PIPE_OUTER_RADIUS = 0.0435 / 2
DEFAULT_PIPE_LENGTH = 0.125 / 2
DEFAULT_PIPE_RESOLUTION = 20


def _as_3_vector(value: Any, *, name: str) -> jp.ndarray:
    array = jp.asarray(value, dtype=jp.float32)
    if array.ndim == 0:
        array = jp.repeat(array[None], 3)
    if array.shape != (3,):
        raise ValueError(
            f"{name} must be a scalar or 3-vector, got shape {array.shape}"
        )
    return array


def cable_ball_joint_stiffness(
    *,
    radius: Union[float, jax.Array],
    segment_length: Union[float, jax.Array],
    youngs_modulus: Union[float, jax.Array],
    shear_modulus: Union[float, jax.Array],
) -> Union[float, jax.Array]:
    """Match the mjxsim cable helper ball-joint stiffness approximation."""
    polar_moment = jp.pi * radius**4 / 2.0
    area_moment = jp.pi * radius**4 / 4.0
    length = jp.maximum(segment_length, 1e-9)
    k_twist = (polar_moment * shear_modulus) / length
    k_bend_y = (area_moment * youngs_modulus) / length
    k_bend_z = (area_moment * youngs_modulus) / length
    return (k_bend_y + k_bend_z + k_twist) / 3.0


def domain_randomize(
    model: mjx.Model,
    rng: jax.Array,
    *,
    pipe_inner_radius_min: float = DEFAULT_PIPE_DR_INNER_RADIUS[0],
    pipe_inner_radius_max: float = DEFAULT_PIPE_DR_INNER_RADIUS[1],
    length_min: float = DEFAULT_CABLE_DR_LENGTH[0],
    length_max: float = DEFAULT_CABLE_DR_LENGTH[1],
    youngs_modulus_min: float = DEFAULT_CABLE_DR_YOUNGS_MODULUS[0],
    youngs_modulus_max: float = DEFAULT_CABLE_DR_YOUNGS_MODULUS[1],
    shear_modulus_min: float = DEFAULT_CABLE_DR_SHEAR_MODULUS[0],
    shear_modulus_max: float = DEFAULT_CABLE_DR_SHEAR_MODULUS[1],
    thickness_min: float = DEFAULT_CABLE_DR_THICKNESS[0],
    thickness_max: float = DEFAULT_CABLE_DR_THICKNESS[1],
) -> tuple[mjx.Model, Any]:
    if pipe_inner_radius_min <= 0 or pipe_inner_radius_max <= 0:
        raise ValueError("Pipe inner radius randomization bounds must be positive.")
    if pipe_inner_radius_min > pipe_inner_radius_max:
        raise ValueError("pipe_inner_radius_min must be <= pipe_inner_radius_max.")
    if pipe_inner_radius_max >= DEFAULT_PIPE_OUTER_RADIUS:
        raise ValueError("Pipe inner radius must be smaller than pipe outer radius.")
    if length_min <= 0 or length_max <= 0:
        raise ValueError("Cable length randomization bounds must be positive.")
    if length_min > length_max:
        raise ValueError("length_min must be <= length_max.")
    if youngs_modulus_min < 0 or youngs_modulus_max < 0:
        raise ValueError("Youngs modulus randomization bounds must be non-negative.")
    if youngs_modulus_min > youngs_modulus_max:
        raise ValueError("youngs_modulus_min must be <= youngs_modulus_max.")
    if shear_modulus_min < 0 or shear_modulus_max < 0:
        raise ValueError("Shear modulus randomization bounds must be non-negative.")
    if shear_modulus_min > shear_modulus_max:
        raise ValueError("shear_modulus_min must be <= shear_modulus_max.")
    if thickness_min <= 0 or thickness_max <= 0:
        raise ValueError("Cable thickness randomization bounds must be positive.")
    if thickness_min > thickness_max:
        raise ValueError("thickness_min must be <= thickness_max.")

    cable_body_ids = np.array(
        [
            mjx.name2id(model, ObjType.BODY.value, name)
            for name in (
                "cable:B1",
                "cable:B2",
                "cable:B3",
                "cable:B4",
                "cable:B5",
                "cable:B6",
                "cable:B7",
                "cable:B8",
                "cable:Blast",
            )
        ],
        dtype=np.int32,
    )
    cable_geom_ids = np.array(
        [
            mjx.name2id(model, ObjType.GEOM.value, name)
            for name in (
                "cable:Gfirst",
                "cable:G1",
                "cable:G2",
                "cable:G3",
                "cable:G4",
                "cable:G5",
                "cable:G6",
                "cable:G7",
                "cable:G8",
                "cable:Glast",
            )
        ],
        dtype=np.int32,
    )
    cable_joint_ids = np.array(
        [
            mjx.name2id(model, ObjType.JOINT.value, name)
            for name in (
                "cable:Jfirst",
                "cable:J2",
                "cable:J3",
                "cable:J4",
                "cable:J5",
                "cable:J6",
                "cable:J7",
                "cable:J8",
                "cable:Jlast",
            )
        ],
        dtype=np.int32,
    )
    if (
        (cable_body_ids < 0).any()
        or (cable_geom_ids < 0).any()
        or (cable_joint_ids < 0).any()
    ):
        raise ValueError("Expected cable bodies, geoms, and joints were not found.")

    pipe_geom_ids = np.arange(
        1,
        1 + DEFAULT_PIPE_RESOLUTION,
        dtype=np.int32,
    )
    if model.ngeom <= int(pipe_geom_ids[-1]):
        raise ValueError("Expected pipe geoms were not found.")
    pipe_angles = jp.linspace(
        0.0,
        2.0 * jp.pi,
        DEFAULT_PIPE_RESOLUTION,
        endpoint=False,
        dtype=jp.float32,
    )
    pipe_radial_dirs = jp.stack([jp.cos(pipe_angles), jp.sin(pipe_angles)], axis=1)

    num_segments = len(cable_geom_ids)

    def rand(rng: jax.Array):
        (
            rng,
            pipe_inner_key,
            length_key,
            youngs_key,
            shear_key,
            thickness_key,
        ) = jax.random.split(rng, 6)
        pipe_inner_radius = jax.random.uniform(
            pipe_inner_key,
            shape=(),
            minval=pipe_inner_radius_min,
            maxval=pipe_inner_radius_max,
        )
        pipe_center_radius = 0.5 * (DEFAULT_PIPE_OUTER_RADIUS + pipe_inner_radius)
        pipe_radial_half_width = 0.5 * (DEFAULT_PIPE_OUTER_RADIUS - pipe_inner_radius)

        cable_length = jax.random.uniform(
            length_key, shape=(), minval=length_min, maxval=length_max
        )
        segment_length = cable_length / num_segments
        half_segment_length = 0.5 * segment_length
        youngs_modulus = jax.random.uniform(
            youngs_key,
            shape=(),
            minval=youngs_modulus_min,
            maxval=youngs_modulus_max,
        )
        shear_modulus = jax.random.uniform(
            shear_key, shape=(), minval=shear_modulus_min, maxval=shear_modulus_max
        )
        thickness = jax.random.uniform(
            thickness_key, shape=(), minval=thickness_min, maxval=thickness_max
        )
        stiffness = cable_ball_joint_stiffness(
            radius=thickness,
            segment_length=segment_length,
            youngs_modulus=youngs_modulus,
            shear_modulus=shear_modulus,
        )

        pipe_geom_pos_xy = pipe_radial_dirs * pipe_center_radius
        geom_pos = model.geom_pos.at[pipe_geom_ids, :2].set(pipe_geom_pos_xy)
        body_pos = model.body_pos.at[cable_body_ids, 1].set(segment_length)
        geom_size = model.geom_size.at[pipe_geom_ids, 0].set(pipe_radial_half_width)
        geom_size = model.geom_size.at[cable_geom_ids, 0].set(thickness)
        geom_size = geom_size.at[cable_geom_ids, 1].set(half_segment_length)
        jnt_pos = model.jnt_pos.at[cable_joint_ids, 1].set(-half_segment_length)
        jnt_stiffness = model.jnt_stiffness.at[cable_joint_ids].set(stiffness)

        return geom_pos, body_pos, geom_size, jnt_pos, jnt_stiffness

    batched = rng.ndim > 1
    if batched:
        outputs = jax.vmap(rand)(rng)
    else:
        outputs = rand(rng)

    (
        geom_pos,
        body_pos,
        geom_size,
        jnt_pos,
        jnt_stiffness,
    ) = outputs

    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    if batched:
        in_axes = in_axes.tree_replace(
            {
                "geom_pos": 0,
                "body_pos": 0,
                "geom_size": 0,
                "jnt_pos": 0,
                "jnt_stiffness": 0,
            }
        )

    model = model.tree_replace(
        {
            "geom_pos": geom_pos,
            "body_pos": body_pos,
            "geom_size": geom_size,
            "jnt_pos": jnt_pos,
            "jnt_stiffness": jnt_stiffness,
        }
    )

    return model, in_axes


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.002,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        vision=False,
        sparse_reward=False,
        sparse_reward_scale=100.0,
        a_max=0,
        a_min=0,
        success_threshold_pos=0.01,
        reset_pos_min=list(DEFAULT_RESET_POS_MIN),
        reset_pos_max=list(DEFAULT_RESET_POS_MAX),
        reset_rot_min=list(DEFAULT_RESET_ROT_MIN),
        reset_rot_max=list(DEFAULT_RESET_ROT_MAX),
        initial_curvature_randomization=False,
        curvature_theta_min=float(np.deg2rad(45.0)),
        curvature_theta_max=float(np.deg2rad(60.0)),
        curvature_phi_min=0.0,
        curvature_phi_max=float(2.0 * np.pi),
        curvature_update_ref=True,
        homo=True,
        impl="warp",
    )


def _parse_float_list(value: Optional[str]) -> list[float]:
    if not value:
        return []
    return [float(item) for item in value.strip().split()]


OBSERVATION_LAYOUT = (
    "keypoint_to_target_x_keypoint_frame",
    "keypoint_to_target_y_keypoint_frame",
    "keypoint_to_target_z_keypoint_frame",
    "target_z_axis_x_keypoint_frame",
    "target_z_axis_y_keypoint_frame",
    "target_z_axis_z_keypoint_frame",
)

with open("experiments/pipe_insert/config/state_1.json", "r") as f:
    _PIPE_INSERT_INIT_KEYFRAME = json.load(f)

# _PIPE_INSERT_INIT_KEYFRAME = {
#     "name": "init",
#     "time": 0,
#     "qpos": _parse_float_list(
#         "-0.00947511 0.180653 -0.0887441 0.851438 0.524092 -0.0102263 -0.0166138 0.999999 -0.00146951 -4.03258e-12 -1.68616e-09 0.999983 -0.00590368 -8.03079e-11 -6.97733e-09 0.999909 -0.0134648 -5.6518e-10 -1.6602e-08 0.999698 -0.02459 -2.51178e-09 -3.21214e-08 0.999194 -0.0401316 -8.61218e-09 -5.60426e-08 0.998108 -0.0614894 -2.48804e-08 -9.04947e-08 0.995881 -0.0906679 -6.23136e-08 -1.30835e-07 0.99151 -0.130033 -1.32017e-07 -1.46397e-07 0.983451 -0.181172 -2.12483e-07 -4.69784e-08"
#     ),
#     "qvel": _parse_float_list(
#         "-6.55311e-08 3.12602e-07 4.81199e-07 8.20477e-06 -8.22193e-07 -3.26604e-07 -1.92035e-07 1.12256e-13 3.03087e-09 -7.59123e-07 2.20739e-11 1.25879e-08 -1.67801e-06 3.05132e-10 2.99011e-08 -2.88816e-06 1.87116e-09 5.73999e-08 -4.21045e-06 7.7683e-09 9.88912e-08 -5.14747e-06 2.55825e-08 1.57635e-07 -4.51174e-06 7.129e-08 2.26798e-07 -1.61511e-08 1.7075e-07 2.59132e-07 1.12037e-05 3.46293e-07 1.05207e-07"
#     ),
#     "mpos": _parse_float_list("-0.000600281 0.407976 0.168044"),
#     "mquat": _parse_float_list("1 0 0 0"),
# }


class PipeInsertBase(mjx_env.MjxEnv):
    """Shared pipe-insertion model and default environment behavior."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides=config_overrides)
        if config.impl not in {"jax", "warp", "c"}:
            raise ValueError(
                f"Invalid MJX impl '{config.impl}' (expected: jax, warp, c)"
            )

        self.impl = config.impl
        self._mj_model = self._init()
        self._mjx_model: mjx.Model = mjx.put_model(self._mj_model, impl=self.impl)

        # Terminate on stricter success limits.
        self._termination_threshold_pos = 0.02  # m
        # self._termination_threshold_pos = 0.02  # m
        # self._termination_threshold_pos = 0.01  # m
        self._termination_threshold_rot = jp.deg2rad(10)  # rad
        # self._termination_threshold_rot = jp.deg2rad(1)  # rad
        # self._termination_threshold_rot = jp.deg2rad(7)  # rad
        # self._termination_threshold_rot = jp.deg2rad(1)  # rad

        self._reset_pos_min = _as_3_vector(
            config.get("reset_pos_min", DEFAULT_RESET_POS_MIN),
            name="reset_pos_min",
        )
        self._reset_pos_max = _as_3_vector(
            config.get("reset_pos_max", DEFAULT_RESET_POS_MAX),
            name="reset_pos_max",
        )
        self._reset_rot_min = _as_3_vector(
            config.get("reset_rot_min", DEFAULT_RESET_ROT_MIN),
            name="reset_rot_min",
        )
        self._reset_rot_max = _as_3_vector(
            config.get("reset_rot_max", DEFAULT_RESET_ROT_MAX),
            name="reset_rot_max",
        )
        if bool((self._reset_pos_max < self._reset_pos_min).any()):
            raise ValueError("reset_pos_min must be <= reset_pos_max on every axis.")
        if bool((self._reset_rot_max < self._reset_rot_min).any()):
            raise ValueError("reset_rot_min must be <= reset_rot_max on every axis.")

        self._initial_curvature_randomization = bool(
            config.get("initial_curvature_randomization", False)
        )
        self._curvature_theta_min = float(
            config.get("curvature_theta_min", np.deg2rad(45.0))
        )
        self._curvature_theta_max = float(
            config.get("curvature_theta_max", np.deg2rad(60.0))
        )
        self._curvature_phi_min = float(config.get("curvature_phi_min", 0.0))
        self._curvature_phi_max = float(config.get("curvature_phi_max", 2.0 * np.pi))
        self._curvature_update_ref = bool(config.get("curvature_update_ref", True))
        self._cable_joint_qpos_adrs = cable_ball_joint_qpos_adrs(self.mj_model)

        self._termination_threshold = (
            self._termination_threshold_pos,
            self._termination_threshold_rot,
        )

        self._w_pos = 1.0
        self._w_rot = 1.0
        self._w_homo_false = -0.2
        self._w_homo_true = 0.05
        # self._w_pos = 100.0
        # self._w_rot = 100.0
        # self._w_pos = 10.0

        self._homo = config.homo

        self._sparse_reward = bool(config["sparse_reward"])
        self._sparse_reward_scale = float(config.get("sparse_reward_scale", 1.0))
        self._episode_length = config["episode_length"]

        self._post_init()

    def _init(self) -> mj.MjModel:
        scene = ms.empty_scene(
            sim_name="empty scene",
            memory=None,
            add_floor=False,
            sdf_iterations=10,
            sdf_initpoints=40,
            noslip_iterations=0,
            enable_multiccd=False,
            nativeccd=True,
            option_overrides={
                "disableflags": int(mj.mjtDisableBit.mjDSBL_EULERDAMP),
                "enableflags": 0,
            },
        )

        self.pipe_inner_radius = DEFAULT_PIPE_INNER_RADIUS
        self.pipe_outer_radius = DEFAULT_PIPE_OUTER_RADIUS
        self.pipe_length = DEFAULT_PIPE_LENGTH  # 121 mm from schematic

        pip = pipe(
            inner_radius=self.pipe_inner_radius,
            outer_radius=self.pipe_outer_radius,
            length=self.pipe_length,
            rgba=[0.2, 0.2, 0.2, 0.2],
            resolution=DEFAULT_PIPE_RESOLUTION,
            solref=[0.00001, 2],
        )

        pip.worldbody.first_body().add_site(
            name="target",
            pos=[0, 0, 0.062],
            # pos=[0, 0, 0.07],
            # pos=[0, 0, 0.1],
            group=1,
            rgba=[1, 0, 0, 1],
            # name="target", pos=[0, 0, -0.062], group=1, rgba=[1, 0, 0, 1]
        )
        pip.worldbody.first_body().add_site(
            name="target_align",
            pos=[0, 0, -0.062],
            # pos=[0, 0, 0.07],
            # pos=[0, 0, 0.1],
            group=1,
            rgba=[1, 0, 0, 1],
            # name="target", pos=[0, 0, -0.062], group=1, rgba=[1, 0, 0, 1]
        )

        spec_cable = cable(
            twist=DEFAULT_CABLE_SHEAR_MODULUS,
            bend=DEFAULT_CABLE_YOUNGS_MODULUS,
            size=DEFAULT_CABLE_LENGTH,
            segment_size=DEFAULT_CABLE_THICKNESS,
            initial="free",
        )

        spec_cable.body("cable:Bfirst").add_site(
            name="cable_weld",
            pos=[0.0, 0.0, 0.0],
            quat=[np.cos(np.pi / 4), 0.0, 0.0, -np.sin(np.pi / 4)],
            group=1,
            rgba=[0, 1, 1, 1],
        )

        spec_cable.body("cable:Blast").add_site(
            name="keypoint",
            group=1,
            rgba=[1, 0, 0, 1],
            pos=[0, 0, 0],
            quat=[0.7071068, -0.7071068, 0, 0],
            # quat=[0, 0, -0.7071068, 0.7071068],
        )

        _c = scene.worldbody.add_camera(
            name="cam",
            pos=[1.2, 0.234, 0.156],
            # pos=[0.721, 0.234, 0.156],
            xyaxes=[-0.037, 0.999, 0.000, -0.001, -0.000, 1.000],
            resolution=[640, 480],
        )

        gripper_spawn = [0.0, 0.4, 0.4]

        mocap = scene.worldbody.add_body(
            name="mocap", mocap=True, pos=gripper_spawn, euler=[0, 0, 0]
        )
        mocap.add_geom(
            name="mocap",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.02, 0.02, 0.02],
            contype=0,
            conaffinity=0,
        )
        mocap.add_site(name="mocap_site", pos=[0, 0, 0], euler=[0, 0, np.pi / 2])

        scene.worldbody.add_frame(pos=[0, 0, 0.1], euler=[1.57, 0, 3.14]).attach_body(
            pip.worldbody.first_body(),
        )
        cable_root = spec_cable.worldbody.first_body()
        scene.worldbody.add_frame(pos=[0, 0.2, 0.1], euler=[0, 0, 0]).attach_body(
            cable_root
        )

        solref = [-50_000, -362.71080706]
        solimp = [0.999, 0.9999, 0.0001, 0.5, 1]

        scene.add_equality(
            name="mocap_cable_weld",
            type=mj.mjtEq.mjEQ_WELD,
            objtype=mj.mjtObj.mjOBJ_SITE,
            name1="mocap_site",
            name2="cable_weld",
            # Keep the current relative pose at creation time.
            solref=solref,
            solimp=solimp,
        )

        scene.add_key(**_PIPE_INSERT_INIT_KEYFRAME)

        self._xml_path = "generated_scene.xml"  # dummy path
        return scene.compile()

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        return self._get_obs_for_model(self.mjx_model, data)

    def get_mj_obs(self, data: mj.MjData) -> np.ndarray:
        """Return the demo/NumPy equivalent of the MJX observation vector."""
        target_pose = ms.get_pose(self.mj_model, data, "target", ms.ObjType.SITE)
        keypoint_pose = ms.get_pose(self.mj_model, data, "keypoint", ms.ObjType.SITE)

        e_pos_world = target_pose.t - keypoint_pose.t
        e_pos = keypoint_pose.R.T @ e_pos_world
        z_rel = keypoint_pose.R.T @ target_pose.R[:, 2]
        return np.concatenate([e_pos, z_rel], axis=0).astype(np.float64)

    def get_demo_observation_space(self) -> dict[str, Any]:
        return {
            "shape": [len(OBSERVATION_LAYOUT)],
            "dtype": "float64",
            "layout": list(OBSERVATION_LAYOUT),
        }

    def _is_homo_for_model(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        T_w_keypoint = get_pose(model, data, "keypoint", ObjType.SITE)
        T_w_target = get_pose(model, data, "target", ObjType.SITE)

        target_to_keypoint_w = T_w_keypoint.translation() - T_w_target.translation()
        radial_distance = jp.linalg.norm(
            jp.array([target_to_keypoint_w[0], target_to_keypoint_w[2]])
        )
        return radial_distance <= jp.asarray(self.pipe_inner_radius)

    def _is_homo(self, data: mjx.Data) -> bool:
        return self._is_homo_for_model(self.mjx_model, data)

    def _get_reward_for_model(
        self,
        model: mjx.Model,
        data0: mjx.Data,
        data1: mjx.Data,
        done: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        _, pos_err0, rot_err0 = self._get_error_metrics_for_model(model, data0)
        _, pos_err1, rot_err1 = self._get_error_metrics_for_model(model, data1)

        # Normalize by success thresholds so meters and radians are comparable,
        # then weight the two components with the configured error weights.
        eps = jp.asarray(1e-6)
        pos_scale = jp.maximum(jp.asarray(self._termination_threshold_pos), eps)
        rot_scale = jp.maximum(jp.asarray(self._termination_threshold_rot), eps)
        pos0 = pos_err0 / pos_scale
        pos1 = pos_err1 / pos_scale
        rot0 = rot_err0 / rot_scale
        rot1 = rot_err1 / rot_scale

        position_weight = jp.asarray(self._w_pos)
        rotation_weight = jp.asarray(self._w_rot)
        delta_pose_0 = position_weight * pos0 + rotation_weight * rot0
        delta_pose_1 = position_weight * pos1 + rotation_weight * rot1

        pose_progress = delta_pose_0 - delta_pose_1
        pos_progress = position_weight * (pos0 - pos1)
        rot_distance_reward = -rotation_weight * rot1

        success = self._get_success_for_model(model, data1)
        done_value = jp.array(0.0) if done is None else done
        success_bonus = jp.where(success, 25.0, 0.0)
        terminal_penalty = jp.where((done_value > 0.0) & (~success), -25.0, 0.0)

        is_homo = jp.array(False)
        homo_reward = jp.array(0.0)
        if self._homo:
            is_homo = self._is_homo_for_model(model, data1)
            homo_reward = jp.where(
                is_homo,
                jp.asarray(self._w_homo_true),
                jp.asarray(self._w_homo_false),
            )

        reward = pose_progress + homo_reward + success_bonus + terminal_penalty

        return {
            "position_error_prev": pos_err0,
            "rotation_error_prev": rot_err0,
            "position_error": pos_err1,
            "rotation_error": rot_err1,
            "reward_delta_pose_prev": delta_pose_0,
            "reward_delta_pose": delta_pose_1,
            "reward_position_progress": pos_progress,
            "reward_rotation_distance_component": rot_distance_reward,
            "is_success": success.astype(float),
            "is_homo": is_homo.astype(float),
            "reward_homo": homo_reward,
            "reward_total": reward,
        }

    def _get_reward(
        self, data0: mjx.Data, data1: mjx.Data, done: jax.Array | None = None
    ) -> dict[str, jax.Array]:
        return self._get_reward_for_model(self.mjx_model, data0, data1, done)

    def _get_empty_reward_metrics(self) -> dict[str, jax.Array]:
        zero = jp.array(0.0)
        return {
            "position_error_prev": zero,
            "rotation_error_prev": zero,
            "position_error": zero,
            "rotation_error": zero,
            "reward_delta_pose_prev": zero,
            "reward_delta_pose": zero,
            "reward_position_progress": zero,
            "reward_rotation_distance_component": zero,
            "is_success": zero,
            "is_homo": zero,
            "reward_homo": zero,
            "reward_total": zero,
        }

    def _get_reward_vel(self, data0: mjx.Data, data1: mjx.Data) -> float:
        return self._get_reward(data0, data1)["reward_total"]

    def _get_obs_for_model(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        T_w_target = get_pose(model, data, "target", ObjType.SITE)
        T_w_keypoint = get_pose(model, data, "keypoint", ObjType.SITE)
        R_w_target = T_w_target.rotation().as_matrix()
        R_w_keypoint = T_w_keypoint.rotation().as_matrix()
        e_pos_world = T_w_target.translation() - T_w_keypoint.translation()
        e_pos = jp.linalg.matrix_transpose(R_w_keypoint) @ e_pos_world
        z_rel = jp.linalg.matrix_transpose(R_w_keypoint) @ R_w_target[:, 2]
        return jp.concatenate([e_pos, z_rel], axis=0)

    def _get_error_metrics_for_model(
        self, model: mjx.Model, data: mjx.Data
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        return pos and orientation error in cm and rad
        """
        obs = self._get_obs_for_model(model, data)
        pos_err = jp.linalg.norm(obs[:3])
        z_dot = jp.clip(obs[5], -1.0, 1.0)
        rot_err = jp.arccos(z_dot)
        return obs, pos_err, rot_err

    def _get_error_metrics(
        self, data: mjx.Data
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return self._get_error_metrics_for_model(self.mjx_model, data)

    def _get_weighted_error_metrics(
        self, pos_err: jax.Array, rot_err: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return self._w_pos * pos_err, self._w_rot * rot_err

    def _get_success_for_model(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        _, pos_err, rot_err = self._get_error_metrics_for_model(model, data)

        if isinstance(self._termination_threshold, (tuple, list)):
            pos_thresh, rot_thresh = self._termination_threshold
        else:
            pos_thresh = self._termination_threshold
            rot_thresh = self._termination_threshold

        return (pos_err < pos_thresh) & (rot_err < rot_thresh)

    def _get_done_for_model(
        self,
        model: mjx.Model,
        data: mjx.Data,
        info: dict,
    ) -> jax.Array:
        next_step_count = info["step"] + 1

        is_unstable = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        timeout = next_step_count >= self._episode_length

        if self._sparse_reward:
            done = is_unstable | timeout
        else:
            success = self._get_success_for_model(model, data)
            done = is_unstable | timeout | success

        # done = is_unstable | timeout | success | failed
        return done.astype(float)

    def _get_done(self, data: mjx.Data, info: dict) -> float:
        return self._get_done_for_model(self.mjx_model, data, info)

    def _post_init(self) -> None:
        self.target_id = self._mj_model.site("target").id
        self.keypoint_id = self._mj_model.site("keypoint").id

        self._mocap_id = None
        try:
            mocap_body_id = self._mj_model.body("mocap").id
            self._mocap_id = int(self._mj_model.body_mocapid[mocap_body_id])
            if self._mocap_id < 0:
                self._mocap_id = None
        except Exception:
            self._mocap_id = None

        self._key_id: Optional[int] = None
        for key_name in ("bent", "init"):
            try:
                self._key_id = self._mj_model.key(key_name).id
                break
            except Exception:
                continue
        if self._key_id is None and getattr(self._mj_model, "nkey", 0) > 0:
            self._key_id = 0

        if self._key_id is not None:
            self._qpos0 = jp.array(self._mj_model.key_qpos[self._key_id])
            self._qvel0 = jp.array(self._mj_model.key_qvel[self._key_id])
            self._ctrl0 = jp.array(self._mj_model.key_ctrl[self._key_id])

    def reset(self, rng: jax.Array) -> mjx_env.State:
        # Split RNG (advance per reset even if we don't add noise).
        if self._initial_curvature_randomization:
            rng, rng_pos, rng_rot, rng_curvature = jax.random.split(rng, 4)
        else:
            rng, rng_pos, rng_rot = jax.random.split(rng, 3)
        model = self.mjx_model

        data = mjx.make_data(self.mj_model, impl=self.impl, naconmax=4500, njmax=512)
        if self._key_id is not None:
            data = set_state(
                model,
                data,
                self._key_id,
                ObjType.KEYFRAME,
                forward=False,
            )

        # Randomize cable root + mocap pose around the keyframe.
        pos_delta = jax.random.uniform(
            rng_pos,
            (3,),
            minval=self._reset_pos_min,
            maxval=self._reset_pos_max,
        )
        rot_delta = jax.random.uniform(
            rng_rot,
            (3,),
            minval=self._reset_rot_min,
            maxval=self._reset_rot_max,
        )
        reset_rotation = jaxl.SO3.exp(rot_delta)
        qpos = data.qpos

        # for debugging
        # pos_delta = jp.zeros(3)
        # rot_delta = jp.zeros(3)

        try:
            cable_root_jnt = self._mj_model.joint("cable:free").id
            adr = int(self._mj_model.jnt_qposadr[cable_root_jnt])
            qpos = qpos.at[adr : adr + 3].add(pos_delta)
            qpos_quat = qpos[adr + 3 : adr + 7]
            qpos_quat = reset_rotation.multiply(jaxl.SO3(wxyz=qpos_quat)).normalize()
            qpos = qpos.at[adr + 3 : adr + 7].set(qpos_quat.wxyz)
        except Exception:
            pass

        if self._initial_curvature_randomization:
            model, qpos, curvature_theta, curvature_phi = apply_random_pre_curvature(
                model,
                qpos,
                rng_curvature,
                cable_joint_qpos_adrs=self._cable_joint_qpos_adrs,
                theta_min=self._curvature_theta_min,
                theta_max=self._curvature_theta_max,
                phi_min=self._curvature_phi_min,
                phi_max=self._curvature_phi_max,
                update_ref=self._curvature_update_ref,
            )

        mocap_pos = data.mocap_pos + pos_delta if model.nmocap else None
        mocap_quat = None
        if model.nmocap:
            mocap_quat = jax.vmap(
                lambda quat: (
                    reset_rotation.multiply(jaxl.SO3(wxyz=quat)).normalize().wxyz
                )
            )(data.mocap_quat)
        data = set_state(
            model,
            data,
            qpos=qpos,
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
            forward=True,
        )
        data = set_state(
            model,
            data,
            qacc=jp.zeros(self.mj_model.nv),
            qfrc_applied=jp.zeros(self.mj_model.nv),
            xfrc_applied=jp.zeros((self.mj_model.nbody, 6)),
            forward=False,
        )

        obs, pos_err, rot_err = self._get_error_metrics_for_model(model, data)
        weighted_pos_err, weighted_rot_err = self._get_weighted_error_metrics(
            pos_err, rot_err
        )
        reward_metrics = self._get_empty_reward_metrics()
        metrics = {
            "position_error": pos_err,
            "rotation_error": rot_err,
            "weighted_position_error": weighted_pos_err,
            "weighted_rotation_error": weighted_rot_err,
            **reward_metrics,
        }

        reward, done = jp.zeros(2)

        info = {
            "rng": rng,
            "step": jp.array(0),
            "mjx_model": model,
            "position_error": pos_err,
            "rotation_error": rot_err,
            "weighted_position_error": weighted_pos_err,
            "weighted_rotation_error": weighted_rot_err,
            **reward_metrics,
        }
        if self._initial_curvature_randomization:
            info["curvature_theta"] = curvature_theta
            info["curvature_phi"] = curvature_phi

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

        model = state.info.get("mjx_model", self.mjx_model)
        data0 = state.data

        dpos = action[:3]
        drot = action[3:6]

        T_w_mocap = get_pose(model, data0, "mocap", ObjType.BODY)
        T_w_mocap_target = jaxl.SE3.from_rotation_and_translation(
            rotation=jaxl.SO3.exp(drot).multiply(T_w_mocap.rotation()).normalize(),
            translation=T_w_mocap.translation() + dpos,
        )
        data = set_pose(model, data0, "mocap", ObjType.BODY, T_w_mocap_target)
        data = mjx_env.step(model, data, data.ctrl, self.n_substeps)

        obs, pos_err, rot_err = self._get_error_metrics_for_model(model, data)

        weighted_pos_err, weighted_rot_err = self._get_weighted_error_metrics(
            pos_err, rot_err
        )
        done = self._get_done_for_model(model, data, state.info)
        reward_metrics = self._get_empty_reward_metrics()

        if self._sparse_reward:
            success = self._get_success_for_model(model, data)
            reward = success.astype(float) * self._sparse_reward_scale
            reward_metrics = {
                **reward_metrics,
                "is_success": success.astype(float),
                "reward_total": reward,
            }
        else:
            reward_metrics = self._get_reward_for_model(model, data0, data, done)
            reward = reward_metrics["reward_total"]

        rng, _ = jax.random.split(state.info["rng"])
        info = {**state.info}
        info["rng"] = rng
        info["step"] = info["step"] + 1
        info["mjx_model"] = model
        info["position_error"] = pos_err
        info["rotation_error"] = rot_err
        info["weighted_position_error"] = weighted_pos_err
        info["weighted_rotation_error"] = weighted_rot_err
        info.update(reward_metrics)

        metrics = {
            **state.metrics,
            "position_error": pos_err,
            "rotation_error": rot_err,
            "weighted_position_error": weighted_pos_err,
            "weighted_rotation_error": weighted_rot_err,
            **reward_metrics,
        }

        return mjx_env.State(data, obs, reward, done, metrics, info)

    def get_obs(
        R_current: jp.ndarray,
        t_current: jp.ndarray,
        R_target: jp.ndarray,
        t_target: jp.ndarray,
    ) -> jp.ndarray:
        """
        Compute an SE(3)-style observation where translation is standard and
        orientation only measures alignment of the z-axis.

        Args:
            R_current: (3, 3) current rotation matrix
            t_current: (3,) current position
            R_target:  (3, 3) target rotation matrix
            t_target:  (3,) target position
            position_in_body_frame: if True, express translation error in current/body frame
            include_z_dot: if True, include the z-axis cosine term for robustness

        Returns:
            obs:
                if include_z_dot:
                    shape (6,) = [dp_x, dp_y, dp_z, z_rel_x, z_rel_y, z_rel_z]
                else:
                    shape (5,) = [dp_x, dp_y, dp_z, z_rel_x, z_rel_y]
        """
        # Position error in current tip frame
        dp_world = t_target - t_current
        dp = R_current.T @ dp_world

        # Relative target z-axis expressed in the current frame
        # This is the 3rd column of R_current.T @ R_target
        z_rel = R_current.T @ R_target[:, 2]

        obs = jp.concatenate([dp, z_rel], axis=0)

        return obs

    @property
    def observation_size(self) -> int:
        """Compute observation size by calling _get_obs with dummy data"""
        # Create dummy data for size computation
        dummy_data = mjx.make_data(self.mj_model, impl=self.impl)

        # Get observation and check its shape
        obs = self._get_obs(dummy_data)
        return obs.shape[0]  # Get the last dimension (feature size)

    @property
    def action_size(self) -> int:
        if self._mocap_id is not None and self.mj_model.nmocap:
            return 6
        return self.mj_model.nu

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


class PipeInsert(PipeInsertBase):
    """Default shared pipe-insertion environment."""


MOCAP_SITE_LOCAL_ROT = jp.asarray(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=jp.float32,
)


class TargetAlignPipeInsert(PipeInsert):
    """PipeInsert v0 variant that emits the 24D target_align observation."""

    def _get_obs_for_model(self, model, data) -> jax.Array:
        T_w_target = self._pose(model, data, "target", "site")
        T_w_target_align = self._pose(model, data, "target_align", "site")
        T_w_keypoint = self._pose(model, data, "keypoint", "site")
        T_w_mocap = self._pose(model, data, "mocap", "body")

        R_w_target = T_w_target.rotation().as_matrix()
        R_w_target_align = T_w_target_align.rotation().as_matrix()
        R_w_keypoint = T_w_keypoint.rotation().as_matrix()
        R_w_mocap_site = T_w_mocap.rotation().as_matrix() @ MOCAP_SITE_LOCAL_ROT

        target_to_keypoint = jp.linalg.matrix_transpose(R_w_target) @ (
            T_w_keypoint.translation() - T_w_target.translation()
        )
        keypoint_z_target = jp.linalg.matrix_transpose(R_w_target) @ R_w_keypoint[:, 2]
        target_align_to_keypoint = jp.linalg.matrix_transpose(R_w_target_align) @ (
            T_w_keypoint.translation() - T_w_target_align.translation()
        )
        keypoint_z_target_align = (
            jp.linalg.matrix_transpose(R_w_target_align) @ R_w_keypoint[:, 2]
        )
        target_to_mocap_site = jp.linalg.matrix_transpose(R_w_target) @ (
            T_w_mocap.translation() - T_w_target.translation()
        )
        R_target_mocap_site = jp.linalg.matrix_transpose(R_w_target) @ R_w_mocap_site

        return jp.concatenate(
            [
                target_to_keypoint,
                keypoint_z_target,
                target_align_to_keypoint,
                keypoint_z_target_align,
                target_to_mocap_site,
                R_target_mocap_site.reshape(-1),
            ],
            axis=0,
        )

    @staticmethod
    def _pose(model, data, name: str, obj_type: str):
        kind = ObjType.SITE if obj_type == "site" else ObjType.BODY
        return get_pose(model, data, name, kind)

    def _get_error_metrics_for_model(self, model, data):
        obs = self._get_obs_for_model(model, data)
        pos_err = jp.linalg.norm(obs[:3])
        z_dot = jp.clip(obs[5], -1.0, 1.0)
        rot_err = jp.arccos(z_dot)
        return obs, pos_err, rot_err

    def _get_obs(self, data) -> jax.Array:
        return self._get_obs_for_model(self.mjx_model, data)

    def get_obs(
        self,
        data,
        rng: jax.Array | None = None,
        add_noise: bool = False,
    ) -> jax.Array:
        del rng, add_noise
        return self._get_obs_for_model(self.mjx_model, data)


CABLE_POINT_NAMES = [
    "cable:Bfirst",
    "cable:B1",
    "cable:B2",
    "cable:B3",
    "cable:B4",
    "cable:B5",
    "cable:B6",
    "cable:B7",
    "cable:B8",
    "cable:Blast",
]


class LatentPipeInsert(TargetAlignPipeInsert):
    """Pipe-insert env that emits the 60D state used by latent distillation."""

    @staticmethod
    def _pose_vector(model, data, name: str, obj_type: ObjType, target_t, target_r):
        pose = get_pose(model, data, name, obj_type)
        target_r_inv = jp.linalg.matrix_transpose(target_r)
        position = target_r_inv @ (pose.translation() - target_t)
        rotation = target_r_inv @ pose.rotation().as_matrix()
        return jp.concatenate([position, rotation.reshape(-1)], axis=0)

    @staticmethod
    def _position_vector(model, data, name: str, obj_type: ObjType, target_t, target_r):
        pose = get_pose(model, data, name, obj_type)
        return jp.linalg.matrix_transpose(target_r) @ (pose.translation() - target_t)

    def _get_obs_for_model(self, model, data) -> jax.Array:
        target = get_pose(model, data, "target", ObjType.SITE)
        target_t = target.translation()
        target_r = target.rotation().as_matrix()

        target_align = self._pose_vector(
            model, data, "target_align", ObjType.SITE, target_t, target_r
        )
        keypoint = get_pose(model, data, "keypoint", ObjType.SITE)
        keypoint_pos = jp.linalg.matrix_transpose(target_r) @ (
            keypoint.translation() - target_t
        )
        keypoint_z = (
            jp.linalg.matrix_transpose(target_r) @ keypoint.rotation().as_matrix()[:, 2]
        )
        mocap = self._pose_vector(
            model, data, "mocap", ObjType.BODY, target_t, target_r
        )
        cable_points = [
            self._position_vector(model, data, name, ObjType.BODY, target_t, target_r)
            for name in CABLE_POINT_NAMES
        ]
        return jp.concatenate(
            [target_align, keypoint_pos, keypoint_z, mocap, *cable_points], axis=0
        )

    def _get_task_obs_for_model(self, model, data) -> jax.Array:
        target = get_pose(model, data, "target", ObjType.SITE)
        keypoint = get_pose(model, data, "keypoint", ObjType.SITE)
        target_r = target.rotation().as_matrix()
        keypoint_r = keypoint.rotation().as_matrix()
        e_pos = jp.linalg.matrix_transpose(target_r) @ (
            keypoint.translation() - target.translation()
        )
        z_rel = jp.linalg.matrix_transpose(target_r) @ keypoint_r[:, 2]
        return jp.concatenate([e_pos, z_rel], axis=0)

    def _get_error_metrics_for_model(self, model, data):
        obs = self._get_obs_for_model(model, data)
        task_obs = self._get_task_obs_for_model(model, data)
        pos_err = jp.linalg.norm(task_obs[:3])
        z_dot = jp.clip(task_obs[5], -1.0, 1.0)
        rot_err = jp.arccos(z_dot)
        return obs, pos_err, rot_err

    def _get_error_metrics(self, data):
        return self._get_error_metrics_for_model(self.mjx_model, data)

    def _get_obs(self, data) -> jax.Array:
        return self._get_obs_for_model(self.mjx_model, data)

    def get_obs(
        self, data, rng: jax.Array | None = None, add_noise: bool = False
    ) -> jax.Array:
        del rng, add_noise
        return self._get_obs_for_model(self.mjx_model, data)
