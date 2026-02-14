"""

optimal ppo brax cartpole trainer

"""

import datetime
import functools
import itertools
import os
import time
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jax
import mujoco
import numpy as np
from brax import base, envs, math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
from jax import numpy as jp
from matplotlib import pyplot as plt
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import registry
from mujoco_playground._src import wrapper
from mujoco_playground.config import dm_control_suite_params
from orbax import checkpoint as ocp

env_name = "CartpoleBalance"

ppo_params = dm_control_suite_params.brax_ppo_config(env_name)
env = registry.load("CartpoleBalance")
env_cfg = registry.get_default_config("CartpoleBalance")
env_cfg
x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    clear_output(wait=True)

    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])

    plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
    plt.ylim([0, 1100])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")
    plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
    plt.savefig("tmp/test.png")

    # display(plt.gcf())


ppo_training_params = dict(ppo_params)
# ppo_training_params["num_envs"] = 1

network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )

train_fn = functools.partial(
    ppo.train,
    **dict(ppo_training_params),
    network_factory=network_factory,
    progress_fn=progress,
)

make_inference_fn, params, metrics = train_fn(
    environment=env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")
