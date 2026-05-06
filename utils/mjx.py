from enum import Enum
from typing import Any, Union

import jax.numpy as jp
import jaxlie as jaxl
import mujoco as mj
import mujoco.mjx as mjx


class ObjType(Enum):
    """
    Enumeration of object types used in MuJoCo simulations.

    Attributes
    ----------
    UNKNOWN : int
        Unknown object type (0)
    BODY : int
        Body object type (1)
    XBODY : int
        Body object type for accessing regular frame instead of i-frame (2)
    GEOM : int
        Geometric object type (5)
    SITE : int
        Site object type (6)
    CAMERA : int
        Camera object type (7)
    """

    UNKNOWN = int(mjx.ObjType.UNKNOWN)  # unknown object type
    BODY = int(mjx.ObjType.BODY)  # body
    XBODY = int(
        mjx.ObjType.XBODY
    )  # body, used to access regular frame instead of i-frame
    GEOM = int(mjx.ObjType.GEOM)
    SITE = int(mjx.ObjType.SITE)
    CAMERA = int(mjx.ObjType.CAMERA)
    JOINT = int(mj.mjtObj.mjOBJ_JOINT)
    KEYFRAME = int(mj.mjtObj.mjOBJ_KEY)
    KEY = int(mj.mjtObj.mjOBJ_KEY)


def get_number_of(model: mjx.Model, obj_type: ObjType) -> int:
    """
    Retrieves the count of objects of a specified type in a MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model from which to count objects.
    obj_type : ObjType
        The type of objects to count, e.g., actuators, bodies, joints.

    Returns
    -------
    int
        The number of objects of the specified type in the model.

    Raises
    ------
    ValueError
        If the specified object type is not recognized.
    """
    type_to_attribute = {
        ObjType.BODY: model.nbody,
        ObjType.GEOM: model.ngeom,
        ObjType.SITE: model.nsite,
        ObjType.CAMERA: model.ncam,
        ObjType.JOINT: model.njnt,
        ObjType.KEYFRAME: model.nkey,
    }

    if obj_type not in type_to_attribute:
        raise ValueError(f"Object type {obj_type} not recognized.")

    return type_to_attribute[obj_type]


def get_names(model: mjx.Model, obj_type: ObjType) -> list[str]:
    """
    Retrieves the names of all objects of a specified type in a MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model containing the objects.
    obj_type : ObjType
        The type of objects to retrieve names for, e.g., actuators, bodies.

    Returns
    -------
    List[str]
        A list of names for all objects of the specified type in the model.
    """
    return [
        mjx.id2name(model, obj_type.value, id)
        for id in range(get_number_of(model, obj_type))
    ]


def get_ids(model: mjx.Model, obj_type: ObjType) -> list[int]:
    """
    Retrieves the names of all objects of a specified type in a MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model containing the objects.
    obj_type : ObjType
        The type of objects to retrieve names for, e.g., actuators, bodies.

    Returns
    -------
    List[str]
        A list of names for all objects of the specified type in the model.
    """
    return [
        mjx.id2name(model, obj_type.value, id)
        for id in range(get_number_of(model, obj_type))
    ]


def does_exist(model: mjx.Model, identifier: Union[int, str], obj_type: mjx.ObjType):
    if isinstance(identifier, str):
        exists = True if mjx.name2id(model, obj_type.value, identifier) != -1 else False
        if not exists:
            raise ValueError(
                f"{obj_type.name} name '{identifier}' not found in the model. The model contain the {obj_type.name}s {get_names(model, obj_type)}"
            )
    elif isinstance(identifier, int):
        exists = (identifier < get_number_of(model, obj_type)) and (identifier >= 0)
        if not exists:
            raise ValueError(
                f"{obj_type.name} id '{identifier}' not found in the model. The model contain the {obj_type.name}s {get_ids(model, obj_type)}"
            )
    else:
        raise ValueError(
            f"Invalid type input id with value '{identifier}' and type {type(identifier)} use either string or int."
        )
    return exists


def _rotation_as_wxyz(rotation: Any, quat_order: str = "xyzw") -> Any:
    if hasattr(rotation, "normalize") and hasattr(rotation, "as_quaternion_xyzw"):
        quat_xyzw = rotation.normalize().as_quaternion_xyzw()
        return jp.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    quat = jp.asarray(rotation)
    quat = quat / jp.linalg.norm(quat)
    if quat_order == "xyzw":
        return jp.array([quat[3], quat[0], quat[1], quat[2]])
    if quat_order == "wxyz":
        return quat

    raise ValueError(f"Unsupported quat_order: {quat_order}")


def set_pose(
    model: mjx.Model,
    data: mjx.Data,
    identifier: Union[int, str],
    obj_type: ObjType,
    T: jaxl.SE3,
    quat_order: str = "xyzw",
) -> mjx.Data:
    """
    Sets the pose (position and orientation) of an object in a MuJoCo model, if allowed.

    Parameters
    ----------
    model : mjx.Model
        The MuJoCo model containing the object.
    data : mjx.Data
        The simulation data where the pose is set.
    identifier : int or str
        The ID or name of the object.
    obj_type : mjx.ObjType
        The type of the object, e.g., body, joint.
    T : jaxl.SE3
        The desired pose as an SE3 transformation matrix.
    quat_order : str
        Quaternion order to use if ``T.rotation()`` returns a raw quaternion
        array. ``jaxlie.SO3`` rotations are read with
        ``as_quaternion_xyzw()`` regardless of this value.
    """
    assert does_exist(model, identifier, obj_type)

    # Convert name to id if needed
    if isinstance(identifier, str):
        obj_id = mjx.name2id(model, obj_type.value, identifier)
    else:
        obj_id = identifier

    def set_position_and_orientation(pos_array, quat_array, index):
        """Helper to set position and orientation at given index."""
        new_pos = T.translation()
        new_quat_wxyz = _rotation_as_wxyz(T.rotation(), quat_order=quat_order)

        updated_pos = pos_array.at[index].set(new_pos)
        updated_quat = quat_array.at[index].set(new_quat_wxyz)
        return updated_pos, updated_quat

    # Process based on object type
    if obj_type is ObjType.BODY:
        # Check if the body is a mocap body
        mocap_id = model.body_mocapid[obj_id]
        if mocap_id != -1:
            new_mocap_pos, new_mocap_quat = set_position_and_orientation(
                data.mocap_pos, data.mocap_quat, mocap_id
            )
            data = data.replace(mocap_pos=new_mocap_pos, mocap_quat=new_mocap_quat)
            return data

        # Check if the body has a freejoint
        body_jntadr = model.body_jntadr[obj_id]
        if (
            body_jntadr != -1 and model.jnt_type[body_jntadr] == 0
        ):  # 0 = free joint in MJX
            # Get the qpos address for this joint
            jnt_qposadr = model.jnt_qposadr[body_jntadr]

            # Update qpos for free joint: [x, y, z, qw, qx, qy, qz]
            new_pos = T.translation()
            new_quat_wxyz = _rotation_as_wxyz(T.rotation(), quat_order=quat_order)

            # Create the full 7D pose for free joint
            new_qpos = jp.concatenate([new_pos, new_quat_wxyz])

            # Update qpos at the correct position
            updated_qpos = data.qpos.at[jnt_qposadr : jnt_qposadr + 7].set(new_qpos)
            data = data.replace(qpos=updated_qpos)
            return data

    elif obj_type is ObjType.JOINT:
        # Check if the joint is a free joint
        if model.jnt_type[obj_id] == 0:  # 0 = free joint in MJX
            # Get the qpos address for this joint
            jnt_qposadr = model.jnt_qposadr[obj_id]

            # Update qpos for free joint: [x, y, z, qw, qx, qy, qz]
            new_pos = T.translation()
            new_quat_wxyz = _rotation_as_wxyz(T.rotation(), quat_order=quat_order)

            # Create the full 7D pose for free joint
            new_qpos = jp.concatenate([new_pos, new_quat_wxyz])

            # Update qpos at the correct position
            updated_qpos = data.qpos.at[jnt_qposadr : jnt_qposadr + 7].set(new_qpos)
            data = data.replace(qpos=updated_qpos)
            return data

    # If no valid option found, raise an error
    raise ValueError(
        f"{obj_type.name} '{identifier}' cannot have its pose set. Only mocap bodies, bodies with freejoints, or freejoints are allowed."
    )


def get_pose(
    model: mjx.Model, data: mjx.Data, identifier: Union[int, str], obj_type: mjx.ObjType
) -> jaxl.SE3:
    assert does_exist(model, identifier, obj_type)

    if isinstance(identifier, str):
        id = mjx.name2id(model, obj_type.value, identifier)
    else:
        id = identifier

    if obj_type is ObjType.BODY:
        # Check if the body is a mocap body
        mocap_id = model.body_mocapid[id]
        if mocap_id != -1:
            # For mocap bodies, use mocap_pos and mocap_quat
            xt = data.mocap_pos[mocap_id]
            xquat = data.mocap_quat[mocap_id]  # wxyz format
            # Convert wxyz to xyzw for jaxl
            xquat_xyzw = jp.array([xquat[1], xquat[2], xquat[3], xquat[0]])
            xR = jaxl.SO3.from_quaternion_xyzw(xquat_xyzw)
            return jaxl.SE3.from_rotation_and_translation(rotation=xR, translation=xt)

        # Check if the body has a freejoint
        body_jntadr = model.body_jntadr[id]
        if (
            body_jntadr != -1 and model.jnt_type[body_jntadr] == 0
        ):  # 0 = free joint in MJX
            # Get the qpos address for this joint
            jnt_qposadr = model.jnt_qposadr[body_jntadr]

            # Read qpos for free joint: [x, y, z, qw, qx, qy, qz]
            qpos_slice = data.qpos[jnt_qposadr : jnt_qposadr + 7]
            xt = qpos_slice[:3]  # position
            xquat = qpos_slice[3:]  # wxyz quaternion

            # Convert wxyz to xyzw for jaxl
            xquat_xyzw = jp.array([xquat[1], xquat[2], xquat[3], xquat[0]])
            xR = jaxl.SO3.from_quaternion_xyzw(xquat_xyzw)
            return jaxl.SE3.from_rotation_and_translation(rotation=xR, translation=xt)

        # For regular bodies, use xpos and xmat
        xt = data.xpos[id]
        xR = data.xmat[id]
        xR = jaxl.SO3.from_matrix(xR.reshape(3, 3))
        return jaxl.SE3.from_rotation_and_translation(rotation=xR, translation=xt)

    elif obj_type is ObjType.JOINT:
        # Check if the joint is a free joint
        if model.jnt_type[id] == 0:  # 0 = free joint in MJX
            # Get the qpos address for this joint
            jnt_qposadr = model.jnt_qposadr[id]

            # Read qpos for free joint: [x, y, z, qw, qx, qy, qz]
            qpos_slice = data.qpos[jnt_qposadr : jnt_qposadr + 7]
            xt = qpos_slice[:3]  # position
            xquat = qpos_slice[3:]  # wxyz quaternion

            # Convert wxyz to xyzw for jaxl
            xquat_xyzw = jp.array([xquat[1], xquat[2], xquat[3], xquat[0]])
            xR = jaxl.SO3.from_quaternion_xyzw(xquat_xyzw)
            return jaxl.SE3.from_rotation_and_translation(rotation=xR, translation=xt)
        else:
            raise ValueError(
                f"Joint {identifier} is not a free joint and cannot provide a pose"
            )

    # For other object types (geom, site, camera)
    pose_mapping = {
        ObjType.GEOM: (data.geom_xpos, data.geom_xmat),
        ObjType.SITE: (data.site_xpos, data.site_xmat),
        ObjType.CAMERA: (data.cam_xpos, data.cam_xmat),
    }

    if obj_type not in pose_mapping:
        raise ValueError(f"obj_type {obj_type.name} cannot provide a pose...")

    _xpos, _xmat = pose_mapping[obj_type]
    xt = _xpos[id]
    xR = _xmat[id]
    xR = jaxl.SO3.from_matrix(xR.reshape(3, 3))
    return jaxl.SE3.from_rotation_and_translation(rotation=xR, translation=xt)


def set_state(
    model: mjx.Model,
    data: mjx.Data,
    identifier: Union[int, str] | None = None,
    obj_type: ObjType = ObjType.KEYFRAME,
    *,
    time: Any | None = None,
    qpos: Any | None = None,
    qvel: Any | None = None,
    act: Any | None = None,
    ctrl: Any | None = None,
    mocap_pos: Any | None = None,
    mocap_quat: Any | None = None,
    qacc: Any | None = None,
    qfrc_applied: Any | None = None,
    xfrc_applied: Any | None = None,
    forward: bool = True,
) -> mjx.Data:
    """
    Sets simulation state components on MJX data.

    If ``identifier`` is provided, the matching MuJoCo keyframe is applied
    first. Explicit keyword components are then applied as overrides. Components
    that do not exist for the model, such as ``ctrl`` for a model without
    actuators or ``mocap`` arrays for a model without mocap bodies, are skipped
    when reading from a keyframe.

    Parameters
    ----------
    model : mjx.Model
        The MuJoCo MJX model.
    data : mjx.Data
        The simulation data to update.
    identifier : int or str
        Optional keyframe ID or name.
    obj_type : ObjType
        Must be ``ObjType.KEYFRAME`` or ``ObjType.KEY`` when ``identifier`` is
        provided.
    time, qpos, qvel, act, ctrl, mocap_pos, mocap_quat, qacc, qfrc_applied,
    xfrc_applied
        Optional state components to set directly. ``mocap_pos`` and
        ``mocap_quat`` are reshaped to ``(model.nmocap, 3)`` and
        ``(model.nmocap, 4)``.
    forward : bool
        Run ``mjx.forward`` after replacing components. Disable this if callers
        need to set acceleration or applied-force fields after a forward pass.

    Returns
    -------
    mjx.Data
        Updated data.
    """
    if identifier is not None and obj_type is not ObjType.KEYFRAME:
        raise ValueError(
            f"set_state only supports keyframes, got obj_type {obj_type.name}."
        )

    replace_kwargs = {}

    if identifier is not None:
        assert does_exist(model, identifier, obj_type)

        if isinstance(identifier, str):
            key_id = mjx.name2id(model, obj_type.value, identifier)
        else:
            key_id = identifier

        replace_kwargs["time"] = jp.asarray(model.key_time[key_id])

        if model.nq:
            replace_kwargs["qpos"] = jp.asarray(model.key_qpos[key_id])
        if model.nv:
            replace_kwargs["qvel"] = jp.asarray(model.key_qvel[key_id])
        if model.na:
            replace_kwargs["act"] = jp.asarray(model.key_act[key_id])
        if model.nu:
            replace_kwargs["ctrl"] = jp.asarray(model.key_ctrl[key_id])
        if model.nmocap:
            replace_kwargs["mocap_pos"] = jp.asarray(model.key_mpos[key_id]).reshape(
                model.nmocap, 3
            )
            replace_kwargs["mocap_quat"] = jp.asarray(model.key_mquat[key_id]).reshape(
                model.nmocap, 4
            )

    component_values = {
        "time": time,
        "qpos": qpos,
        "qvel": qvel,
        "act": act,
        "ctrl": ctrl,
        "qacc": qacc,
        "qfrc_applied": qfrc_applied,
        "xfrc_applied": xfrc_applied,
    }
    replace_kwargs.update(
        {
            field_name: jp.asarray(value)
            for field_name, value in component_values.items()
            if value is not None
        }
    )

    if mocap_pos is not None:
        replace_kwargs["mocap_pos"] = jp.asarray(mocap_pos).reshape(model.nmocap, 3)
    if mocap_quat is not None:
        replace_kwargs["mocap_quat"] = jp.asarray(mocap_quat).reshape(model.nmocap, 4)

    if not replace_kwargs:
        return data

    data = data.replace(**replace_kwargs)
    if not forward or model.nv == 0:
        return data
    return mjx.forward(model, data)
