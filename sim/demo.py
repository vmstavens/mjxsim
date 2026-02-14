import time
from collections import deque
from copy import deepcopy

import glfw
import mujoco
import mujoco as mj
import mujoco.viewer
import numpy as np
import torch
from robot_descriptions import ur10e_mj_description

from agents.diffusion_policy_state import (
    DIFFUSION_POLICY_STATE_DEFAULT_CONFIG,
    ConditionalUnet1D,
    DiffusionPolicy,
    EMAModel,
)


def get_state(m: mj.MjModel, d: mj.MjData) -> torch.Tensor:
    """
    Get the TCP position relative to the base frame.

    Args:
        m: MuJoCo model
        d: MuJoCo data

    Returns:
        torch.Tensor: TCP position [x, y, z] relative to base frame
    """
    # Get base body position and orientation
    base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_body_id == -1:
        # Try common base names
        base_names = ["base_link", "base", "world"]
        for name in base_names:
            base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            if base_body_id != -1:
                break

    # Get TCP site position in world coordinates
    tcp_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tool0")
    if tcp_site_id == -1:
        # Fallback to last body
        last_body_id = m.nbody - 1
        tcp_pos_world = d.xpos[last_body_id].copy()
    else:
        tcp_pos_world = d.site_xpos[tcp_site_id].copy()

    if base_body_id != -1:
        # Get base position and rotation
        base_pos = d.xpos[base_body_id].copy()
        base_rot = d.xmat[base_body_id].reshape(3, 3).copy()

        # Transform TCP position to base frame
        tcp_pos_base = base_rot.T @ (tcp_pos_world - base_pos)
    else:
        # If base not found, assume world frame is base frame
        tcp_pos_base = tcp_pos_world.copy()

    return torch.tensor(tcp_pos_base, dtype=torch.float32)


def key_cb(key: int):
    if key is glfw.KEY_SPACE:
        print("space")


def init_sim() -> tuple[mj.MjModel, mj.MjData]:
    _XML = """
        <mujoco model="empty scene">


            <extension>
                <plugin plugin="mujoco.sensor.touch_grid" />
            </extension>


            <compiler angle="radian" autolimits="true" />
            <option timestep="0.002" integrator="implicitfast" solver="Newton" gravity="0 0 -9.82"
                cone="elliptic"/>

            <statistic center="0.3 0 0.3" extent="0.8" meansize="0.08" />

            <visual>
                <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0" />
                <rgba haze="0.15 0.25 0.35 1" />
                <global azimuth="120" elevation="-20" />

            </visual>

            <asset>
                <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512"
                    height="3072" />
                <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
                    rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
                <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"
                    reflectance="0.2" />
            </asset>

            <worldbody>

                <light pos="0 0 1.5" dir="0 0 -1" directional="true" />
                <geom name="floor" size="0 0 0.5" type="plane" material="groundplane" />

            </worldbody>

        </mujoco>
    """

    world = mj.MjSpec().from_string(_XML)

    arm = mj.MjSpec().from_file(ur10e_mj_description.MJCF_PATH)

    world.worldbody.add_frame(pos=[0, 0, 0.2]).attach_body(
        arm.worldbody.first_body(), "robot/"
    )

    m = world.compile()
    d = mj.MjData(m)

    key = np.array(
        [-1.56715757, -1.8205482, -2.08289099, -0.72139986, 1.52121496, 0.00535272]
    )

    d.qpos = key
    d.ctrl = key
    # d.qpos = arm.keys[0].qpos
    # d.ctrl = arm.keys[0].ctrl

    mj.mj_step(m, d)

    return m, d


def init_model() -> DiffusionPolicy:
    dp_models = {}
    input_dim = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dp_config = DIFFUSION_POLICY_STATE_DEFAULT_CONFIG

    dp_models["model"] = ConditionalUnet1D(input_dim, dp_config).to(device)
    ema = EMAModel(dp_models["model"].parameters(), power=dp_config["ema_power"])
    dp_models["ema_model"] = ConditionalUnet1D(input_dim, dp_config).to(device)
    dp = DiffusionPolicy(models=dp_models, ema=ema, config=dp_config)
    dp.load(".runs/supervised_trainer/checkpoints/agent_0.pt")
    dp.set_mode("eval")
    return dp


def main(show_left_ui=True, show_right_ui=True):
    m, d = init_sim()

    dp = init_model()

    i = 0

    state0 = get_state(m, d)

    # state_deque = deque(
    #     [deepcopy(state0) for _ in range(dp.cfg["obs_horizon"])],
    #     maxlen=dp.cfg["obs_horizon"],
    # )

    with mujoco.viewer.launch_passive(
        model=m,
        data=d,
        key_callback=key_cb,
        show_left_ui=show_left_ui,
        show_right_ui=show_right_ui,
    ) as viewer:
        while viewer.is_running():
            step_start = time.time()

            state = get_state(m, d)
            # state_deque.append(state)

            print(f"{state=}")

            a, _, _ = dp.act(states=state, timestep=i, timesteps=np.inf)
            # dp.act(states=state_deque, timestep=i, timesteps=np.inf)
            i += 1

            print(f"{a=}")

            input()

            # step simulation one time step
            mj.mj_step(m, d)

            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
