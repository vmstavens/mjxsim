import copy
import glob
import logging
import os
from pathlib import Path

import numpy as np
import torch

from agents.ibrl_sac import IBRL, IBRL_SAC_DEFAULT_CONFIG
from agents.models import ibrl_sac as ibrl
from skrl.memories.torch import RandomMemory
from skrl.utils import set_seed
from trainers.sequential_trainer_plus import SequentialTrainerPlus
from utils.envs import mk_env

import agents.diffusion_policy_state as dp
from agents.diffusion_policy_state import DiffusionPolicy
from datasets.demonstration import DemonstrationDataset

logging.basicConfig(level=logging.WARN)  # This adds a default handler
relative_path = os.path.relpath(__file__)  # Relative to current working directory
logger = logging.getLogger(relative_path)
logger.setLevel(logging.DEBUG)


# seed for reproducibility
set_seed(10)  # e.g. `set_seed(42)` for fixed seed

_TRAIN = Path(__file__).parent

logger.info("Loading configs...")
dp_config = copy.deepcopy(dp.DIFFUSION_POLICY_STATE_DEFAULT_CONFIG)
dp_config.num_envs = 10
num_envs = dp_config.num_envs
dp_config.obs_horizon = 2
logger.info("Setting up env...")
env = mk_env("ant")  # noqa: F821

a_dim = env.action_space.shape[0]
o_dim = env.observation_space.shape[0]

dp_config.obs_dim = o_dim
dp_config.global_cond_dim = dp_config.obs_horizon * dp_config.obs_dim

logger.info("Loading memories...")

train_path = Path(".data/gen_traj_ant_ppo/train")
valid_path = Path(".data/gen_traj_ant_ppo/valid")
train_data_files = glob.glob(train_path.as_posix() + "/*.json")
valid_data_files = glob.glob(valid_path.as_posix() + "/*.json")
pred_horizon = dp_config.pred_horizon
obs_horizon = dp_config.obs_horizon
action_horizon = dp_config.action_horizon
dataset = DemonstrationDataset(
    json_paths=train_data_files,
    sequence_length=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon,
)

states = np.concatenate([episode["states"] for episode in dataset.episodes])
actions = np.concatenate([episode["actions"] for episode in dataset.episodes])
next_states = np.concatenate([episode["next_states"] for episode in dataset.episodes])
rewards = np.concatenate([episode["rewards"] for episode in dataset.episodes])
dones = np.concatenate([episode["dones"] for episode in dataset.episodes])


# Convert to tensors
states = torch.tensor(states)
actions = torch.tensor(actions)
next_states = torch.tensor(next_states)
rewards = torch.tensor(rewards)
terminated = torch.tensor(dones)

memory_size = len(states)

# Create expert memory
expert_memory = RandomMemory(memory_size=memory_size)
expert_memory.create_tensor(name="states", size=o_dim)
expert_memory.create_tensor(name="actions", size=a_dim)
expert_memory.create_tensor(name="next_states", size=o_dim)
expert_memory.create_tensor(name="rewards", size=1)
expert_memory.create_tensor(name="terminated", size=1)

expert_memory.add_samples(
    states=states,
    actions=actions,
    next_states=next_states,
    rewards=rewards,
    dones=dones,
)


logger.info(f"Expert memory filled with {len(expert_memory)} samples")
device = env.device

# create empty memory
memory = RandomMemory(memory_size=10000, device=device, replacement=True)


# ------------------IL DP--------------------------
logger.info("Building DP...")

dp_models = {}

input_dim = a_dim

# Build models
dp_models = {}
dp_models["model"] = dp.ConditionalUnet1D(input_dim, dp_config).to(device)
# dp_models["model"] = dp.ConditionalUnet1D(input_dim, dp_config).to(device)
ema = dp.EMAModel(dp_models["model"].parameters(), power=dp_config.ema_power)
# ema = dp.EMAModel(dp_models["model"].parameters(), power=dp_config["ema_power"])
dp_models["ema_model"] = dp.ConditionalUnet1D(input_dim, dp_config).to(device)

il_agent = DiffusionPolicy(models=dp_models, ema=ema, config=dp_config)

il_agent.load(".demo/state/models/model_epoch_90.pth")
il_agent.ema.copy_to(il_agent.ema_model.parameters())

dp_models["policy"] = il_agent

# ------------------RL SAC-----------------------------
logger.info("Building RL (SAC)...")

logger.info("Configuring IBRL... ")

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
# configure and instantiate the agent
cfg = copy.deepcopy(IBRL_SAC_DEFAULT_CONFIG)
cfg.discount_factor = 0.99
cfg.batch_size = 32
# cfg["batch_size"] = 128
# cfg["batch_size"] = 10
# cfg["batch_size"] = 128
cfg.num_envs = dp_config.num_envs
cfg.random_timesteps = 0  # Add some random exploration at the start
cfg.learning_starts = 0  # Start learning after some experience
cfg.learn_entropy = True
cfg.grad_norm_clip = 1.0  # Add gradient clipping for stability
cfg.actor_learning_rate = 3e-4
cfg.critic_learning_rate = 3e-4
cfg.initial_entropy_value = 0.1
cfg.offline = False  # not important here
# cfg["num_envs"] = env.num_envs

# logging to TensorBoard and write checkpoints (in timesteps)
cfg.experiment.wandb = True
cfg.experiment.write_interval = 50
cfg.experiment.checkpoint_interval = 100
cfg.experiment.experiment_name = Path(__file__).stem
model_path = Path(__file__).parent.parent / ".runs"
model_path.mkdir(parents=True, exist_ok=True)
cfg.experiment.directory = model_path.as_posix()
cfg.experiment.experiment_name = Path(__file__).stem


# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = {}
models["policy"] = ibrl.StochasticActor(
    env.observation_space, env.action_space, device, clip_actions=True
)
models["critic_1"] = ibrl.Critic(env.observation_space, env.action_space, device)
models["critic_2"] = ibrl.Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = ibrl.Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = ibrl.Critic(env.observation_space, env.action_space, device)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


red_q_enable = False
if red_q_enable:
    # Create Emseabling Q networks
    ensemble_size = 5
    # Create ensemble of critics (each with unique parameters)
    critics = []
    target_critics = []
    for i in range(ensemble_size):
        # Create new critic instance for each position
        critic = ibrl.Critic(env.observation_space, env.action_space, device)

        # Initialize the models' parameters (weights and biases) using a Gaussian distribution
        critic.init_parameters(method_name="normal_", mean=0.0, std=0.1)
        target_critic = copy.deepcopy(critic)  # Create target as deep copy

        critics.append(critic)
        target_critics.append(target_critic)
    # models["critics"] = critics[0]
    models["critics"] = critics
    # models["target_critics"] = target_critics[0]
    models["target_critics"] = target_critics


logger.info("Building IBRL... ")
logger.info(f"{num_envs=}")
logger.info(f"{dp_config.num_envs=}")


agent = IBRL(
    models=models,
    models_il=dp_models,
    memory=memory,
    expert_memory=expert_memory,
    cfg=cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)


cfg_trainer = {"timesteps": 50_000, "headless": True}

# cfg_trainer = {"timesteps": 350000, "headless": True}
trainer = SequentialTrainerPlus(cfg=cfg_trainer, env=env, agents=agent)
# trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

logger.info("Start Training... ")

# # start training
trainer.train()
