import jax.numpy as jp
import jaxlie as jaxl
import mujoco as mj
import mujoco.mjx as mjx

from utils.mjx import ObjType, set_pose, set_state


class _RawQuatPose:
    def __init__(self, translation, rotation):
        self._translation = translation
        self._rotation = rotation

    def translation(self):
        return self._translation

    def rotation(self):
        return self._rotation


def test_set_state_applies_keyframe_components_by_name():
    xml = """
    <mujoco>
      <worldbody>
        <body name="mocap" mocap="true" pos="0 0 0"/>
        <body name="free_body" pos="0 0 0">
          <freejoint/>
          <geom type="sphere" size="0.01" mass="1"/>
        </body>
        <body name="hinge_body" pos="0 0 0">
          <joint name="hinge" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.01" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="hinge_motor" joint="hinge" gear="1"/>
      </actuator>
      <keyframe>
        <key name="ready"
             time="1.25"
             qpos="1 2 3 1 0 0 0 0.5"
             qvel="0.1 0.2 0.3 0.4 0.5 0.6 0.7"
             ctrl="0.9"
             mpos="4 5 6"
             mquat="0 1 0 0"/>
      </keyframe>
    </mujoco>
    """
    model = mjx.put_model(mj.MjModel.from_xml_string(xml))
    data = mjx.make_data(model)

    data = set_state(model, data, "ready", ObjType.KEYFRAME)

    assert data.time == model.key_time[0]
    assert jp.allclose(data.qpos, model.key_qpos[0])
    assert jp.allclose(data.qvel, model.key_qvel[0])
    assert jp.allclose(data.ctrl, model.key_ctrl[0])
    assert jp.allclose(data.mocap_pos, model.key_mpos[0].reshape(model.nmocap, 3))
    assert jp.allclose(data.mocap_quat, model.key_mquat[0].reshape(model.nmocap, 4))


def test_set_state_applies_keyframe_components_by_id():
    xml = """
    <mujoco>
      <worldbody/>
      <keyframe>
        <key name="empty" time="2.0"/>
      </keyframe>
    </mujoco>
    """
    model = mjx.put_model(mj.MjModel.from_xml_string(xml))
    data = mjx.make_data(model)

    data = set_state(model, data, 0, ObjType.KEYFRAME)

    assert data.time == model.key_time[0]


def test_set_state_applies_direct_components():
    xml = """
    <mujoco>
      <worldbody>
        <body name="mocap" mocap="true" pos="0 0 0"/>
        <body name="body" pos="0 0 0">
          <joint name="hinge" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.01" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="motor" joint="hinge" gear="1"/>
      </actuator>
    </mujoco>
    """
    model = mjx.put_model(mj.MjModel.from_xml_string(xml))
    data = mjx.make_data(model)

    qpos = jp.array([0.2])
    qvel = jp.array([0.3])
    ctrl = jp.array([0.4])
    qacc = jp.array([0.5])
    qfrc_applied = jp.array([0.6])
    xfrc_applied = jp.ones((model.nbody, 6))
    mocap_pos = jp.array([1.0, 2.0, 3.0])
    mocap_quat = jp.array([1.0, 0.0, 0.0, 0.0])

    data = set_state(
        model,
        data,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
        qacc=qacc,
        qfrc_applied=qfrc_applied,
        xfrc_applied=xfrc_applied,
        mocap_pos=mocap_pos,
        mocap_quat=mocap_quat,
        forward=False,
    )

    assert jp.allclose(data.qpos, qpos)
    assert jp.allclose(data.qvel, qvel)
    assert jp.allclose(data.ctrl, ctrl)
    assert jp.allclose(data.qacc, qacc)
    assert jp.allclose(data.qfrc_applied, qfrc_applied)
    assert jp.allclose(data.xfrc_applied, xfrc_applied)
    assert jp.allclose(data.mocap_pos, mocap_pos.reshape(model.nmocap, 3))
    assert jp.allclose(data.mocap_quat, mocap_quat.reshape(model.nmocap, 4))


def test_set_state_allows_direct_components_to_override_keyframe():
    xml = """
    <mujoco>
      <worldbody>
        <body name="body" pos="0 0 0">
          <joint name="hinge" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.01" mass="1"/>
        </body>
      </worldbody>
      <keyframe>
        <key name="ready" qpos="0.1" qvel="0.2"/>
      </keyframe>
    </mujoco>
    """
    model = mjx.put_model(mj.MjModel.from_xml_string(xml))
    data = mjx.make_data(model)

    data = set_state(model, data, "ready", ObjType.KEYFRAME, qpos=jp.array([0.9]))

    assert jp.allclose(data.qpos, jp.array([0.9]))
    assert jp.allclose(data.qvel, model.key_qvel[0])


def test_set_pose_handles_jaxlie_so3_rotation_for_mocap_body():
    xml = """
    <mujoco>
      <worldbody>
        <body name="mocap" mocap="true" pos="0 0 0"/>
      </worldbody>
    </mujoco>
    """
    model = mjx.put_model(mj.MjModel.from_xml_string(xml))
    data = mjx.make_data(model)
    quat_xyzw = jp.array([0.0, 0.0, 0.70710678, 0.70710678])
    rotation = jaxl.SO3.from_quaternion_xyzw(quat_xyzw)
    translation = jp.array([1.0, 2.0, 3.0])
    pose = jaxl.SE3.from_rotation_and_translation(rotation, translation)

    data = set_pose(model, data, "mocap", ObjType.BODY, pose)

    assert jp.allclose(data.mocap_pos[0], translation)
    assert jp.allclose(data.mocap_quat[0], jp.array([0.70710678, 0.0, 0.0, 0.70710678]))


def test_set_pose_handles_raw_quaternion_rotation_order_for_freejoint():
    xml = """
    <mujoco>
      <worldbody>
        <body name="free_body" pos="0 0 0">
          <freejoint/>
          <geom type="sphere" size="0.01" mass="1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mjx.put_model(mj.MjModel.from_xml_string(xml))
    data = mjx.make_data(model)
    translation = jp.array([1.0, 2.0, 3.0])
    quat_wxyz = jp.array([0.1, 0.2, 0.3, 0.4])
    pose = _RawQuatPose(translation, quat_wxyz)

    data = set_pose(model, data, "free_body", ObjType.BODY, pose, quat_order="wxyz")

    expected_quat = quat_wxyz / jp.linalg.norm(quat_wxyz)
    assert jp.allclose(data.qpos[:3], translation)
    assert jp.allclose(data.qpos[3:7], expected_quat)
