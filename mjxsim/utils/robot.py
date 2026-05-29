from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jp
import jaxlie as jaxl
import mujoco as mj
import mujoco.mjx as mjx

from mjxsim.utils.mjx import ObjType, get_pose


_QPOS_WIDTH = {
    int(mj.mjtJoint.mjJNT_FREE): 7,
    int(mj.mjtJoint.mjJNT_BALL): 4,
    int(mj.mjtJoint.mjJNT_SLIDE): 1,
    int(mj.mjtJoint.mjJNT_HINGE): 1,
}

_DOF_WIDTH = {
    int(mj.mjtJoint.mjJNT_FREE): 6,
    int(mj.mjtJoint.mjJNT_BALL): 3,
    int(mj.mjtJoint.mjJNT_SLIDE): 1,
    int(mj.mjtJoint.mjJNT_HINGE): 1,
}


def _is_robot_entity(entity_name: str | None, robot_name: str) -> bool:
    if not entity_name:
        return False

    entity_parts = [p for p in entity_name.lower().split("/") if p]
    robot_parts = [p for p in robot_name.lower().split("/") if p]
    if not robot_parts:
        return False

    return (
        entity_parts[: len(robot_parts)] == robot_parts
        and len(entity_parts) == len(robot_parts) + 1
    )


def _name2id(model: mjx.Model, name: str, obj_type: int) -> int:
    obj_id = int(mjx.name2id(model, obj_type, name))
    if obj_id == -1:
        raise ValueError(f"{name!r} not found for MuJoCo object type {obj_type}.")
    return obj_id


def _id2name(model: mjx.Model, obj_id: int, obj_type: int) -> str:
    name = mjx.id2name(model, obj_type, obj_id)
    return "" if name is None else name


def _joint_qpos_indices(model: mjx.Model, joint_id: int) -> list[int]:
    start = int(model.jnt_qposadr[joint_id])
    width = _QPOS_WIDTH[int(model.jnt_type[joint_id])]
    return list(range(start, start + width))


def _joint_dof_indices(model: mjx.Model, joint_id: int) -> list[int]:
    start = int(model.jnt_dofadr[joint_id])
    width = _DOF_WIDTH[int(model.jnt_type[joint_id])]
    return list(range(start, start + width))


def _idx(ids: list[int]) -> jp.ndarray:
    return jp.asarray(ids, dtype=jp.int32)


@dataclass(frozen=True)
class RobotInfoX:
    """Static MJX model ids and indices belonging to one robot namespace."""

    model: mjx.Model
    name: str

    def __post_init__(self) -> None:
        body_ids = [
            i
            for i in range(self.model.nbody)
            if _is_robot_entity(
                _id2name(self.model, i, int(mj.mjtObj.mjOBJ_BODY)), self.name
            )
        ]
        joint_ids = [
            i
            for i in range(self.model.njnt)
            if _is_robot_entity(
                _id2name(self.model, i, int(mj.mjtObj.mjOBJ_JOINT)), self.name
            )
        ]
        actuator_ids = [
            i
            for i in range(self.model.nu)
            if _is_robot_entity(
                _id2name(self.model, i, int(mj.mjtObj.mjOBJ_ACTUATOR)), self.name
            )
        ]
        site_ids = [
            i
            for i in range(self.model.nsite)
            if _is_robot_entity(
                _id2name(self.model, i, int(mj.mjtObj.mjOBJ_SITE)), self.name
            )
        ]
        geom_ids = [
            i
            for i in range(self.model.ngeom)
            if int(self.model.geom_bodyid[i]) in body_ids
        ]

        object.__setattr__(self, "_body_ids", body_ids)
        object.__setattr__(
            self,
            "_body_names",
            [_id2name(self.model, i, int(mj.mjtObj.mjOBJ_BODY)) for i in body_ids],
        )
        object.__setattr__(self, "_geom_ids", geom_ids)
        object.__setattr__(
            self,
            "_geom_names",
            [_id2name(self.model, i, int(mj.mjtObj.mjOBJ_GEOM)) for i in geom_ids],
        )
        object.__setattr__(self, "_joint_ids", joint_ids)
        object.__setattr__(
            self,
            "_joint_names",
            [_id2name(self.model, i, int(mj.mjtObj.mjOBJ_JOINT)) for i in joint_ids],
        )
        object.__setattr__(self, "_actuator_ids", actuator_ids)
        object.__setattr__(
            self,
            "_actuator_names",
            [
                _id2name(self.model, i, int(mj.mjtObj.mjOBJ_ACTUATOR))
                for i in actuator_ids
            ],
        )
        object.__setattr__(self, "_site_ids", site_ids)
        object.__setattr__(
            self,
            "_site_names",
            [_id2name(self.model, i, int(mj.mjtObj.mjOBJ_SITE)) for i in site_ids],
        )
        object.__setattr__(
            self,
            "_joint_indxs",
            [idx for jid in joint_ids for idx in _joint_qpos_indices(self.model, jid)],
        )
        object.__setattr__(
            self,
            "_dof_indxs",
            [idx for jid in joint_ids for idx in _joint_dof_indices(self.model, jid)],
        )

    @property
    def body_ids(self) -> list[int]:
        return self._body_ids

    @property
    def body_names(self) -> list[str]:
        return self._body_names

    @property
    def geom_ids(self) -> list[int]:
        return self._geom_ids

    @property
    def geom_names(self) -> list[str]:
        return self._geom_names

    @property
    def joint_ids(self) -> list[int]:
        return self._joint_ids

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def joint_indxs(self) -> list[int]:
        return self._joint_indxs

    @property
    def dof_indxs(self) -> list[int]:
        return self._dof_indxs

    @property
    def actuator_ids(self) -> list[int]:
        return self._actuator_ids

    @property
    def actuator_names(self) -> list[str]:
        return self._actuator_names

    @property
    def site_ids(self) -> list[int]:
        return self._site_ids

    @property
    def site_names(self) -> list[str]:
        return self._site_names

    @property
    def n_actuators(self) -> int:
        return len(self._actuator_ids)

    @property
    def n_joints(self) -> int:
        return len(self._joint_ids)

    @property
    def joint_limits(self) -> jp.ndarray:
        return self.model.jnt_range[_idx(self._joint_ids)].T

    @property
    def actuator_limits(self) -> jp.ndarray:
        return self.model.actuator_ctrlrange[_idx(self._actuator_ids)].T


class RobotX:
    """MJX robot wrapper for querying namespace-scoped robot quantities."""

    def __init__(
        self,
        model: mjx.Model,
        data: mjx.Data,
        namespace: str,
        base_identifier: int | str | None = None,
    ):
        self._model = model
        self._data = data
        self._name = namespace
        self._info = RobotInfoX(model, namespace)
        self._base = 0 if base_identifier is None else base_identifier

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> mjx.Model:
        return self._model

    @property
    def data(self) -> mjx.Data:
        return self._data

    @property
    def info(self) -> RobotInfoX:
        return self._info

    def replace_data(self, data: mjx.Data) -> RobotX:
        self._data = data
        return self

    def set_ctrl(self, x: Any) -> mjx.Data:
        x = jp.asarray(x)
        if x.shape[0] != self.info.n_actuators:
            raise ValueError(
                "Control input length must match robot actuator count: "
                f"{x.shape[0]} != {self.info.n_actuators}."
            )
        ctrl = self.data.ctrl.at[_idx(self.info.actuator_ids)].set(x)
        self._data = self.data.replace(ctrl=ctrl)
        return self._data

    @property
    def ctrl(self) -> jp.ndarray:
        return self.data.ctrl[_idx(self.info.actuator_ids)]

    @property
    def q(self) -> jp.ndarray:
        return self.data.qpos[_idx(self.info.joint_indxs)]

    @property
    def dq(self) -> jp.ndarray:
        return self.data.qvel[_idx(self.info.dof_indxs)]

    @property
    def ddq(self) -> jp.ndarray:
        return self.data.qacc[_idx(self.info.dof_indxs)]

    @property
    def c(self) -> jp.ndarray:
        return self.data.qfrc_bias[_idx(self.info.dof_indxs)]

    @property
    def Mq(self) -> jp.ndarray:
        mass = mjx.full_m(self.model, self.data)
        dof = _idx(self.info.dof_indxs)
        return mass[jp.ix_(dof, dof)]

    def Jp(
        self, base_frame: int | str | None = None, site_frame: int | str | None = None
    ) -> jp.ndarray:
        return self.J(base_frame=base_frame, site_frame=site_frame)[:3]

    def Jo(
        self, base_frame: int | str | None = None, site_frame: int | str | None = None
    ) -> jp.ndarray:
        return self.J(base_frame=base_frame, site_frame=site_frame)[3:]

    def J(
        self, base_frame: int | str | None = None, site_frame: int | str | None = None
    ) -> jp.ndarray:
        base_id = self._resolve_body(base_frame)
        site_id = self._resolve_site(site_frame)

        jacp, jacr = mjx.jac(
            self.model,
            self.data,
            self.data.site_xpos[site_id],
            self.model.site_bodyid[site_id],
        )
        sys_j = jp.concatenate([jacp.T, jacr.T], axis=0)
        sys_j = sys_j[:, _idx(self.info.dof_indxs)]

        if base_id != 0:
            base_pose = get_pose(self.model, self.data, base_id, ObjType.BODY)
            r_world_base = base_pose.rotation().inverse().as_matrix()
            sys_j = sys_j.at[:3].set(r_world_base @ sys_j[:3])
            sys_j = sys_j.at[3:].set(r_world_base @ sys_j[3:])

        return sys_j

    def Mx(
        self, base_frame: int | str | None = None, site_frame: int | str | None = None
    ) -> jp.ndarray:
        jac = self.J(base_frame=base_frame, site_frame=site_frame)
        mx_inv = jac @ jp.linalg.pinv(self.Mq, rcond=1e-2) @ jac.T
        return jp.linalg.pinv(mx_inv, rcond=1e-2)

    def fk(
        self,
        q: Any,
        sites: str | int | list[str] | list[int] | None = None,
        base_frame: jaxl.SE3 | int | str | None = None,
    ) -> jaxl.SE3 | list[jaxl.SE3]:
        q = jp.asarray(q)
        joint_indxs = _idx(self.info.joint_indxs)
        if q.shape[0] != len(self.info.joint_indxs):
            raise ValueError(
                "Joint configuration length must match robot qpos width: "
                f"{q.shape[0]} != {len(self.info.joint_indxs)}."
            )

        if sites is None:
            site_ids = self.info.site_ids
        elif isinstance(sites, (str, int)):
            site_ids = [self._resolve_site(sites)]
        else:
            site_ids = [self._resolve_site(site) for site in sites]

        data = self.data.replace(qpos=self.data.qpos.at[joint_indxs].set(q))
        data = mjx.forward(self.model, data)

        if base_frame is None:
            base_frame = self._base
        if isinstance(base_frame, jaxl.SE3):
            base_pose = base_frame
        else:
            base_pose = get_pose(
                self.model, data, self._resolve_body(base_frame), ObjType.BODY
            )

        poses = [
            base_pose.inverse() @ get_pose(self.model, data, site_id, ObjType.SITE)
            for site_id in site_ids
        ]
        return poses if len(poses) != 1 else poses[0]

    def _resolve_body(self, body: int | str | None) -> int:
        if body is None:
            body = self._base
        if isinstance(body, str):
            name = body if "/" in body else f"{self.name}/{body}"
            return _name2id(self.model, name, int(mj.mjtObj.mjOBJ_BODY))
        return int(body)

    def _resolve_site(self, site: int | str | None) -> int:
        if site is None:
            if not self.info.site_ids:
                raise ValueError(f"Robot {self.name!r} has no namespace-scoped sites.")
            return self.info.site_ids[0]
        if isinstance(site, str):
            name = site if "/" in site else f"{self.name}/{site}"
            return _name2id(self.model, name, int(mj.mjtObj.mjOBJ_SITE))
        return int(site)


__all__ = ["RobotInfoX", "RobotX"]
