import logging
import os
import time
from typing import Optional, Sequence

import glfw
import jax
import jax.numpy as jp
import jaxlie as jaxl
import mujoco
import mujoco as mj
import mujoco.mjx as mjx
import mujoco.viewer
import numpy as np
import spatialmath as sm
import spatialmath.base as smb
from absl import flags
from mujoco.mjx._src.forward import _integrate_pos, rungekutta4
from robot_descriptions import ur5e_mj_description, ur10e_mj_description

import utils.jax as xu
from ctrl.base_ctrl import BaseController
from robots import BaseRobot
from utils.mj import ObjType, RobotInfo, name2id

os.environ["XLA_FLAGS"] = "--xla_gpu_graph_min_graph_size=1"


# import warp as wp
from absl import app


class RobotX:
    def __init__(
        self,
        model: mj.MjModel,
        name: str,
        base_name: Optional[str] = None,
        tcp_name: Optional[str] = None,
    ):
        self._model = model
        self._name = name
        self._info = RobotInfo(self._model, name)

        self.base_id = (
            name2id(self._model, base_name, ObjType.BODY)
            if base_name is not None
            else self.info.body_ids[0]
        )
        self.tcp_id = (
            name2id(self._model, tcp_name, ObjType.SITE)
            if tcp_name is not None
            else self.info.site_ids[-1]
        )

    # Properties
    @property
    def name(self) -> str:
        return self._name

    @property
    def info(self) -> RobotInfo:
        return self._info

    def get_q(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        return data.qpos[model.jnt_qposadr[self._info.joint_ids]]

    def set_q(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        pass

    def get_dq(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        return data.qvel[model.jnt_dofadr[self._info.joint_ids]]

    def get_ddq(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        return data.qacc[model.jnt_dofadr[self._info.joint_ids]]

    def get_ctrl(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        return data.ctrl[self._info.actuator_ids]

    def get_Mq(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        sys_M = mjx.full_m(model, data)
        # print("sys_m:")
        # print(sys_M)
        dof_indices = jp.ravel(self._info._dof_indxs)  # Flatten to 1D if not already
        # print("dof_indices:")
        # print(dof_indices)
        # print("self._info._dof_indxs:")
        # print(self._info._dof_indxs)

        Mq = sys_M[jp.ix_(dof_indices, dof_indices)]
        # print("Mq:")
        # print(Mq)
        # input()
        return Mq

    def get_Mx(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        """
        handbook of robotics page 45 equation 2.52

        https://users.dimi.uniud.it/~antonio.dangelo/Robotica/2018/helper/Handbook-dynamics.pdf
        """

        J = self.get_J(model, data)
        # print(f"{J.shape=}")

        # J = jp.hstack([Jp, Jo])

        Mq = self.get_Mq(model, data)

        Mx_inv = J @ jp.linalg.inv(Mq) @ J.T

        if abs(jp.linalg.det(Mx_inv)) >= 1e-2:
            Mx = jp.linalg.inv(Mx_inv)
        else:
            Mx = jp.linalg.pinv(Mx_inv, rtol=1e-2)

        return Mx

    def get_J(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        """
        Compute the geometric Jacobian for the TCP site
        """
        tcp_id = self._info.site_ids[-1]  # id of tcp site

        # Get the body ID that the TCP site belongs to
        site_body_id = model.site_bodyid[tcp_id]

        # Get TCP position in world frame
        T_w_tcp = xu.get_pose(model, data, tcp_id, obj_type=ObjType.SITE)
        tcp_pos_world = T_w_tcp.translation()

        # Compute Jacobian in world frame for the TCP's body
        jacp_world, jacr_world = mjx.jac(
            model, data, point=tcp_pos_world, body_id=site_body_id
        )

        # Get base frame rotation for transformation (if needed)
        base_id = self._info.body_ids[0]
        T_w_base = xu.get_pose(model, data, base_id, obj_type=ObjType.BODY)
        R_w_base = T_w_base.rotation().as_matrix()
        R_base_w = R_w_base.T  # Rotation from world to base frame

        # Rotate to base frame (optional - depends on your needs)
        # Jp = R_base_w @ jacp_world.T  # (nv, 3) - Linear Jacobian in base frame
        # Jo = R_base_w @ jacr_world.T  # (nv, 3) - Angular Jacobian in base frame
        Jp = jacp_world @ R_base_w  # (nv, 3) - Linear Jacobian in base frame
        Jo = jacr_world @ R_base_w  # (nv, 3) - Angular Jacobian in base frame

        # Extract the parts regarding the robot's joints
        # Jp = Jp[:, self._info.dof_indxs].T
        # Jo = Jo[:, self._info.dof_indxs].T
        Jp = Jp[self._info.dof_indxs, :]
        Jo = Jo[self._info.dof_indxs, :]

        return np.hstack([Jp, Jo])

    def get_c(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        # print(f"{data.qfrc_bias=}")
        # print(f"{data.qfrc_bias[jp.ravel(self._info._dof_indxs)]=}")
        return data.qfrc_bias[jp.ravel(self._info._dof_indxs)]


class OpSpaceX:
    def __init__(self, robot: RobotX, gravity_comp: bool = True) -> None:
        self.gravity_comp = gravity_comp

        # Cartesian impedance control gains.
        self.impedance_pos = jp.array([10000.0, 10000.0, 10000.0])  # [N/m]
        self.impedance_ori = jp.array([5000.0, 5000.0, 5000.0])  # [Nm/rad]

        # Joint impedance control gains.
        self.Kp_null = jp.array([75.0, 75.0, 50.0, 50.0, 40.0, 25.0])

        # Damping ratio for both Cartesian and joint impedance control.
        self.damping_ratio = 1.0

        # Gains for the twist computation.
        self.Kpos: float = 0.95
        self.Kori: float = 0.95

        # Integration timestep in seconds.
        self.integration_dt: float = 1.0

        # Compute damping and stiffness matrices.
        self.damping_pos = self.damping_ratio * 2 * jp.sqrt(self.impedance_pos)
        self.damping_ori = self.damping_ratio * 2 * jp.sqrt(self.impedance_ori)
        self.Kp = jp.concatenate([self.impedance_pos, self.impedance_ori], axis=0)
        self.Kd = jp.concatenate([self.damping_pos, self.damping_ori], axis=0)
        self.Kd_null = self.damping_ratio * 2 * jp.sqrt(self.Kp_null)

        self.robot = robot
        self.T_target = None
        self.q0 = None

    def _quat2vel(self, error_quat: jax.Array, dt: float) -> jax.Array:
        """
        Convert quaternion error to angular velocity in the same frame
        Matches MuJoCo's mju_quat2Vel implementation
        """
        vec = error_quat[1:4]  # [x, y, z]
        w = error_quat[0]  # w

        # Normalize the axis and get its magnitude (sin(a/2))
        axis_norm = jp.linalg.norm(vec)

        def compute_velocity(axis_norm_val):
            # Normalize the axis
            axis = vec / axis_norm_val

            # Compute angle using atan2(sin(a/2), cos(a/2)) = atan2(|vec|, w)
            sin_a_2 = axis_norm_val
            speed = 2.0 * jp.arctan2(sin_a_2, w)

            # When axis-angle is larger than pi, rotation is in the opposite direction
            speed = jax.lax.cond(
                speed > jp.pi, lambda s: s - 2.0 * jp.pi, lambda s: s, speed
            )

            # Scale by time
            speed /= dt

            # Return scaled axis
            return axis * speed

        # Handle the case where the norm is very small (no rotation)
        return jax.lax.cond(
            axis_norm > 1e-6, compute_velocity, lambda n: jp.zeros(3), axis_norm
        )

    def _r2q(self, R: jax.Array) -> jax.Array:
        """Convert rotation matrix to quaternion (w, x, y, z)"""
        # Simple implementation - you might want to use a more robust one
        trace = jp.trace(R)
        if trace > 0:
            s = 0.5 / jp.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * jp.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * jp.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * jp.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s

        return jp.array([w, x, y, z])

    def _neg_quat(self, quat: jax.Array) -> jax.Array:
        """Negate quaternion (conjugate for unit quaternions)"""
        return jp.array([quat[0], -quat[1], -quat[2], -quat[3]])

    def _mul_quat(self, q1: jax.Array, q2: jax.Array) -> jax.Array:
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return jp.array([w, x, y, z])

    def _get_tcp_pose(self, model: mjx.Model, data: mjx.Data):
        """Get TCP pose in base frame"""
        # Get current base and TCP poses in WORLD frame
        T_w_base = xu.get_pose(model, data, self.robot.base_id, ObjType.BODY)
        T_w_tcp = xu.get_pose(model, data, self.robot.tcp_id, ObjType.SITE)

        # Convert TCP pose to BASE frame: T_base_tcp = T_w_base⁻¹ * T_w_tcp
        return T_w_base.inverse() @ T_w_tcp

    def step(self, model: mjx.Model, data: mjx.Data) -> jax.Array:
        """
        Perform a control step to compute the control signal using JAX/MJX.
        """
        # Initialize target if not set
        if self.T_target is None:
            T_base_tcp = self._get_tcp_pose(model, data)
            self.T_target = T_base_tcp
            self.q0 = self.robot.get_q(model, data)

        # self.q0 = self.robot.get_q(model, data)

        # Get current TCP pose in base frame
        T_base_tcp = self._get_tcp_pose(model, data)

        # print(f"{self.T_target.translation()=}")
        # print(f"{T_base_tcp.translation()=}")

        # Compute spatial velocity (twist)
        dx = self.T_target.translation() - T_base_tcp.translation()
        twist_pos = self.Kpos * dx / self.integration_dt

        # print(f"{dx=} | {twist_pos=}")
        # quit()

        # Compute orientation error
        Q_current = T_base_tcp.rotation()
        Q_target = self.T_target.rotation()

        # error_quat = Q_current⁻¹ * Q_target
        error_quat = (Q_current.inverse() @ Q_target).wxyz
        # print(f"{error_quat=}")

        # Convert quaternion error to angular velocity
        twist_ori = self._quat2vel(error_quat, self.integration_dt)
        twist_ori = twist_ori * self.Kori / self.integration_dt

        twist = jp.concatenate([twist_pos, twist_ori])
        # print(f"{twist=}")
        # Get Jacobian, mass matrices, and bias forces
        J = self.robot.get_J(model, data)
        # print(f"{J.shape=}")
        # Jp, Jo = self.robot.get_J(model, data)
        # J = jp.hstack([Jp, Jo])
        Mx = self.robot.get_Mx(model, data)
        dq = self.robot.get_dq(model, data)
        c = self.robot.get_c(model, data)

        # Compute generalized forces
        tau = J.T @ Mx @ (self.Kp * twist - self.Kd * (J @ dq))
        # tau = J.T @ Mx @ (self.Kp * twist - self.Kd @ (J @ dq))

        Mq = self.robot.get_Mq(model, data)

        # print(f"{J=}")
        # print(f"{Mx=}")
        # print(f"{Mq=}")
        # print(f"{dq=}")
        # print(f"{c=}")
        # print(f"{tau=}")
        # input()
        # quit()

        # Add joint task in nullspace for overactuated manipulators
        if self.robot.info.n_joints > self.robot.info.n_actuators:
            Mq = self.robot.get_Mq(model, data)
            Jbar = jp.linalg.inv(Mq) @ J.T @ Mx
            ddq = (
                self.Kp_null * (self.q0 - self.robot.get_q(model, data))
                - self.Kd_null * dq
            )
            tau += (jp.eye(self.robot.info.n_joints) - J.T @ Jbar.T) @ ddq

        # print(f"before grav comp: {tau=}")
        # Add gravity compensation
        if self.gravity_comp:
            tau += c

        # print(f"after grav comp: {tau=}")
        # Clip to actuator limits
        # print(self.robot.info.actuator_limits)
        tau = jp.clip(tau, *self.robot.info.actuator_limits)
        # print(f"end: {tau=}")
        # input()
        return tau


def mk_model_data() -> tuple[mj.MjModel, mj.MjData]:
    robot_name = "robot"
    spec = mj.MjSpec().from_file("scenes/empty.xml")
    # b = spec.worldbody.add_body(name="test", pos=[1, 1, 1])
    # b.add_geom(name="test2", size=[0.1, 0.1, 0.1])
    # b.add_freejoint()
    arm = mj.MjSpec().from_file(ur5e_mj_description.MJCF_PATH)
    # arm = mj.MjSpec().from_file(ur10e_mj_description.MJCF_PATH)
    body = spec.worldbody.add_frame(pos=[0, 0, 0], euler=[0, 0, np.pi]).attach_body(
        # body = spec.worldbody.add_frame(pos=[0, 0, 0.1]).attach_body(
        arm.worldbody.first_body(),
        f"{robot_name}/",
    )

    # a.classname
    # motor sizes from https://www.universal-robots.com/articles/ur/robot-care-maintenance/max-joint-torques-cb3-and-e-series/
    size0 = 9  # Nm
    size1 = 28  # Nm
    size2 = 54  # Nm
    size3 = 150  # Nm
    size4 = 330  # Nm

    for a in spec.actuators:
        cls_name = a.classname.name
        print(cls_name)
        if "size0" in cls_name:
            a.set_to_motor()
            a.ctrlrange = np.array([-size0, size0])
        elif "size1" in cls_name:
            a.set_to_motor()
            a.ctrlrange = np.array([-size1, size1])
        elif "size2" in cls_name:
            a.set_to_motor()
            a.ctrlrange = np.array([-size2, size2])
        elif "size3" in cls_name:
            a.set_to_motor()
            a.ctrlrange = np.array([-size3, size3])
        elif "size4" in cls_name:
            a.set_to_motor()
            a.ctrlrange = np.array([-size4, size4])
        else:
            print("error")
            quit()

    # for a in spec.actuators:
    #     a.set_to_position(kp=10_000, kv=10_000)

    # from mujoco_playground._src import mjx_env
    # mjx_env.get_qvel_ids()
    # mjx_env.get_qpos_ids()

    model = spec.compile()
    data = mj.MjData(model)
    return model, data


def key_callback(key: int) -> None:
    if key == glfw.KEY_SPACE:  # Space bar
        # if key == 32:  # Space bar

        print("running...")


def _main() -> None:
    """Launches MuJoCo passive viewer fed by MJX."""

    jax.config.update("jax_debug_nans", True)

    # create mujoco model and data
    m, d = mk_model_data()

    # create MJX model and data
    mx = mjx.put_model(m)
    dx = mjx.put_data(m, d)

    print(f"Default backend: {jax.default_backend()}")
    step_fn = mjx.step
    JIT = True
    RUNNING = True

    robot = RobotX(m, "robot")

    if JIT:
        print("JIT-compiling the model physics step...")
        start = time.time()
        step_fn = jax.jit(step_fn).lower(mx, dx).compile()
        elapsed = time.time() - start
        print(f"Compilation took {elapsed}s.")

    viewer = mujoco.viewer.launch_passive(m, d, key_callback=key_callback)

    home_q = jp.deg2rad(jp.array([-90, -90, -90, -90, 90, 0]))

    # controller = DiffIKX(robot)
    controller = OpSpaceX(robot)
    T_w_base = xu.get_pose(mx, dx, robot.base_id, ObjType.BODY)
    T_w_tcp = xu.get_pose(mx, dx, robot.tcp_id, ObjType.SITE)
    controller.T_target = T_w_base.inverse() @ T_w_tcp

    def get_ee_pose(m, d) -> jaxl.SE3:
        T_w_base = xu.get_pose(m, d, robot.base_id, ObjType.BODY)
        T_w_tcp = xu.get_pose(m, d, robot.tcp_id, ObjType.SITE)
        return T_w_base.inverse() @ T_w_tcp

    i = 0
    j = -1

    with viewer:
        d.qpos = home_q
        # d.ctrl = home_q
        dx = dx.replace(qpos=home_q)

        dx = step_fn(mx, dx)

        controller.T_target = get_ee_pose(mx, dx)
        N = 100
        # print(f"{controller.T_target=}")
        while True:
            start = time.time()

            print(f"{i} / {N}...")
            if i % N == 0:
                i = 0
                controller.T_target_base = jaxl.SE3.from_translation(
                    jp.array([0, 0, j * 0.1])
                ) @ get_ee_pose(mx, dx)
                # print(f"{controller.T_target_base=}")
                print(f"{d.ctrl=}")
                j *= -1
            i += 1

            # TODO(robotics-simulation): recompile when changing disable flags, etc.
            dx = dx.replace(
                ctrl=jp.array(d.ctrl),
                act=jp.array(d.act),
                xfrc_applied=jp.array(d.xfrc_applied),
            )
            dx = dx.replace(
                qpos=jp.array(d.qpos), qvel=jp.array(d.qvel), time=jp.array(d.time)
            )  # handle resets
            mx = mx.tree_replace(
                {
                    "opt.gravity": m.opt.gravity,
                    "opt.tolerance": m.opt.tolerance,
                    "opt.ls_tolerance": m.opt.ls_tolerance,
                    "opt.timestep": m.opt.timestep,
                }
            )

            if RUNNING:
                dx = step_fn(mx, dx)

            mjx.get_data_into(d, m, dx)
            viewer.sync()

            elapsed = time.time() - start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)

            des_tau = controller.step(mx, dx)
            # print(controller.T_target_base)
            # print(des_qpos)
            # print(dx.qpos)
            d.ctrl = des_tau


if __name__ == "__main__":
    _main()
