import jax.numpy as jp
import mujoco as mj
import mujoco.mjx as mjx

from utils.mjx import ObjType, set_state


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
