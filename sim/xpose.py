from pathlib import Path

import glfw
import mujoco as mj
import spatialmath as sm
from dm_control import mjcf
from robots.base_robot import BaseRobot
from sims import BaseSim
from sims.base_sim import SimSync, sleep
from utils.mj import (
    ObjType,
    RobotInfo,
    get_contact_states,
    set_pose,
)


class MjSim(BaseSim):
    def __init__(self):
        super().__init__()

        self._model, self._data = self.init()

        self.threads = [self.spin]

        print(self.data.joint(0))

    def init(self):
        # root
        _HERE = Path(__file__).parent.parent
        # scene path
        _XML_SCENE = Path(_HERE / "scenes/empty.xml")
        scene = mjcf.from_path(_XML_SCENE)

        prop = scene.worldbody.add("body", name="prop", pos="0 0 0.13")
        prop.add("freejoint", name="freejoint")
        prop.add("geom", name="prop", type="sphere", size="0.1")

        m = mj.MjModel.from_xml_string(scene.to_xml_string(), scene.get_assets())
        d = mj.MjData(m)

        m.actuator_ctrlrange

        # step once to compute the poses of objects
        mj.mj_step(m, d)

        return m, d

    def spin(self, ss: SimSync):
        while True:
            ss.step()
            sleep()

    @property
    def data(self) -> mj.MjData:
        return self._data

    @property
    def model(self) -> mj.MjModel:
        return self._model

    def keyboard_callback(self, key: int):
        if key is glfw.KEY_SPACE:
            set_pose(self.model, self.data, "freejoint", ObjType.JOINT, sm.SE3.Tz(1))
            print("You pressed space...")


if __name__ == "__main__":
    sim = MjSim()
    sim.run()
