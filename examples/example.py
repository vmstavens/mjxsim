from envs.mocap_control import MocapReach, default_config
from utils.load import register

# import the environment loader
from skrl.envs.loaders.torch import load_playground_env


def example_config():
    config = default_config()
    config.impl = "warp"
    return config


register(env_name="MocapControl", env_type=MocapReach, config=example_config)

# load environment
env = load_playground_env(
    task_name="MocapControl",
    num_envs=1024,
    episode_length=300,
)
