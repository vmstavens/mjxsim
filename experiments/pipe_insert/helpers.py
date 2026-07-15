"""Helpers shared by pipe-insert experiment variants."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco as mj
from mujoco import mjx


def cable_ball_joint_qpos_adrs(
    model: mj.MjModel,
    *,
    joint_prefix: str = "cable:",
) -> jnp.ndarray:
    """Return qpos addresses for cable ball joints in MuJoCo model order."""
    qpos_adrs = [
        int(model.jnt_qposadr[joint_id])
        for joint_id in range(model.njnt)
        if int(model.jnt_type[joint_id]) == int(mj.mjtJoint.mjJNT_BALL)
        and (mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id) or "").startswith(
            joint_prefix
        )
    ]
    if not qpos_adrs:
        raise ValueError(f"No cable ball joints found with prefix {joint_prefix!r}.")
    return jnp.asarray(qpos_adrs, dtype=jnp.int32)


def rotvec_to_quat(rotvec: jax.Array) -> jax.Array:
    """Convert a rotation vector to a wxyz quaternion."""
    angle = jnp.linalg.norm(rotvec)
    half_angle = 0.5 * angle
    scale = jnp.where(
        angle > 1e-8,
        jnp.sin(half_angle) / angle,
        0.5 - angle**2 / 48.0,
    )
    quat = jnp.concatenate(
        [jnp.cos(half_angle)[None], rotvec * scale],
        axis=0,
    )
    return quat / jnp.maximum(jnp.linalg.norm(quat), 1e-8)


def constant_pre_curvature_quat(
    theta: jax.Array,
    phi: jax.Array,
    *,
    joints_per_cable: int,
) -> jax.Array:
    """Return the per-ball-joint quaternion for constant cable pre-curvature."""
    joint_theta = theta / joints_per_cable
    return rotvec_to_quat(
        jnp.stack(
            [
                jnp.asarray(0.0, dtype=jnp.float32),
                joint_theta * jnp.cos(phi),
                joint_theta * jnp.sin(phi),
            ]
        ).astype(jnp.float32)
    )


def curve_dlo_qpos(
    qpos: jax.Array,
    theta: jax.Array,
    phi: jax.Array,
    *,
    cable_joint_qpos_adrs: jax.Array,
    num_cables: int = 1,
) -> jax.Array:
    """Apply constant pre-curvature to DLO cable ball-joint qpos values."""
    if num_cables <= 0:
        raise ValueError(f"num_cables must be > 0, got {num_cables}")

    qpos = jnp.asarray(qpos)
    qpos_adrs = jnp.asarray(cable_joint_qpos_adrs, dtype=jnp.int32)
    num_joints = int(qpos_adrs.shape[0])
    if num_joints % num_cables != 0:
        raise ValueError(
            f"Expected cable joints ({num_joints}) to be divisible by "
            f"num_cables ({num_cables})."
        )

    quat = constant_pre_curvature_quat(
        theta,
        phi,
        joints_per_cable=num_joints // num_cables,
    ).astype(qpos.dtype)
    return qpos.at[qpos_adrs[:, None] + jnp.arange(4)].set(quat)


def update_dlo_ref_from_qpos(
    model: mjx.Model,
    qpos: jax.Array,
    *,
    cable_joint_qpos_adrs: jax.Array,
) -> mjx.Model:
    """Set MJX cable ball-joint spring references from qpos."""
    qpos_adrs = jnp.asarray(cable_joint_qpos_adrs, dtype=jnp.int32)
    quat_ids = qpos_adrs[:, None] + jnp.arange(4)
    qpos_spring = model.qpos_spring.at[quat_ids].set(qpos[quat_ids])
    return model.tree_replace({"qpos_spring": qpos_spring})


def sample_pre_curvature(
    rng: jax.Array,
    *,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
) -> tuple[jax.Array, jax.Array]:
    """Sample a total bend angle and bend-plane azimuth."""
    rng_theta, rng_phi = jax.random.split(rng)
    theta = jax.random.uniform(rng_theta, (), minval=theta_min, maxval=theta_max)
    phi = jax.random.uniform(rng_phi, (), minval=phi_min, maxval=phi_max)
    return theta, phi


def apply_random_pre_curvature(
    model: mjx.Model,
    qpos: jax.Array,
    rng: jax.Array,
    *,
    cable_joint_qpos_adrs: jax.Array,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
    num_cables: int = 1,
    update_ref: bool = True,
) -> tuple[mjx.Model, jax.Array, jax.Array, jax.Array]:
    """Apply sampled DLO pre-curvature and optionally update spring references."""
    theta, phi = sample_pre_curvature(
        rng,
        theta_min=theta_min,
        theta_max=theta_max,
        phi_min=phi_min,
        phi_max=phi_max,
    )
    qpos = curve_dlo_qpos(
        qpos,
        theta,
        phi,
        cable_joint_qpos_adrs=cable_joint_qpos_adrs,
        num_cables=num_cables,
    )
    if update_ref:
        model = update_dlo_ref_from_qpos(
            model,
            qpos,
            cable_joint_qpos_adrs=cable_joint_qpos_adrs,
        )
    return model, qpos, theta, phi
