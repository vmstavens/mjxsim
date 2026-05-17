import jax.numpy as jp
import mujoco as mj
import mujoco.mjx as mjx

from mjxsim.utils.robot import RobotX


def test_robot_x_returns_namespace_scoped_mjx_quantities():
    xml = """
    <mujoco>
      <worldbody>
        <body name="robot/base" pos="0 0 0">
          <joint name="robot/j1" type="hinge" axis="0 0 1" range="-1 1"/>
          <geom name="robot/link" type="capsule" size="0.02 0.1" mass="1"/>
          <site name="robot/tcp" pos="0 0 0.1"/>
        </body>
        <body name="other/base" pos="1 0 0">
          <joint name="other/j1" type="hinge"/>
          <geom name="other/link" type="sphere" size="0.02" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="robot/m1" joint="robot/j1" gear="1" ctrlrange="-2 2"/>
        <motor name="other/m1" joint="other/j1" gear="1"/>
      </actuator>
    </mujoco>
    """
    model = mjx.put_model(mj.MjModel.from_xml_string(xml))
    data = mjx.forward(
        model,
        mjx.make_data(model).replace(
            qpos=jp.array([0.25, 0.75]),
            qvel=jp.array([0.5, 1.5]),
            qacc=jp.array([0.1, 0.2]),
            ctrl=jp.array([0.0, 0.0]),
        ),
    )

    robot = RobotX(model, data, "robot")
    data = robot.set_ctrl(jp.array([1.25]))

    assert robot.data is data
    assert robot.info.body_names == ["robot/base"]
    assert robot.info.joint_names == ["robot/j1"]
    assert robot.info.actuator_names == ["robot/m1"]
    assert robot.info.site_names == ["robot/tcp"]
    assert jp.allclose(robot.q, jp.array([0.25]))
    assert jp.allclose(robot.dq, jp.array([0.5]))
    assert jp.allclose(robot.ddq, jp.array([0.1]))
    assert jp.allclose(robot.ctrl, jp.array([1.25]))
    assert robot.J().shape == (6, 1)
    assert robot.Mq.shape == (1, 1)
    assert robot.Mx().shape == (6, 6)


def test_robot_x_fk_returns_jaxlie_pose():
    xml = """
    <mujoco>
      <worldbody>
        <body name="robot/base" pos="0 0 0">
          <joint name="robot/j1" type="hinge" axis="0 0 1"/>
          <geom name="robot/link" type="capsule" size="0.02 0.1" mass="1"/>
          <site name="robot/tcp" pos="0 0 0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mjx.put_model(mj.MjModel.from_xml_string(xml))
    data = mjx.forward(model, mjx.make_data(model))
    robot = RobotX(model, data, "robot")

    pose = robot.fk(jp.array([0.3]), "tcp")

    assert jp.allclose(pose.translation(), jp.array([0.0, 0.0, 0.1]))
