import glob
import logging
import os
from pathlib import Path

import pandas as pd
import torch
from brax import envs
from skrl.agents.torch.sac.sac import SAC_DEFAULT_CONFIG
from skrl.envs.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

import agents.diffusion_policy_state as dp
from agents.diffusion_policy_state import DiffusionPolicy
from agents.ibrl_sac import IBRL
from agents.models import ibrl_sac as ibrl
from envs.brax.ur10e import UR10e
from utils.datasets import DataHandler, folder_to_memory

logging.basicConfig(level=logging.WARN)  # This adds a default handler
relative_path = os.path.relpath(__file__)  # Relative to current working directory
logger = logging.getLogger(relative_path)
logger.setLevel(logging.DEBUG)

# seed for reproducibility
set_seed(10)  # e.g. `set_seed(42)` for fixed seed

_TRAIN = Path(__file__).parent


logger.info("Setting up env...")
# define environment
env_name = "ur10e"
envs.register_environment(env_name, UR10e)
env = envs.create(env_name=env_name, batch_size=1)
a_lim = 0.1
o_lim = 5

env = wrap_env(env)

device = env.device

logger.info("Loading memories...")

# Define goal
goal = torch.Tensor(
    [-1.13769275, -1.84879365, -2.23483372, -0.61194004, 1.53735435, 0.48228303]
)

# Column labels in CSV files
state_labels = [f"actual_q_{i}" for i in range(6)]
action_labels = [f"actual_qd_{i}" for i in range(6)]


# Reward function
def get_reward(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return -torch.linalg.norm(state - goal)


# Get all CSV files
file_paths = glob.glob("data/robotB_data_trimmed/*.csv")
print(f"Found {len(file_paths)} files")

# Lists to accumulate data
all_states = []
all_actions = []
all_next_states = []
all_rewards = []
all_terminated = []

for fp in file_paths:
    df = pd.read_csv(fp)

    # Extract states and actions
    states = df[state_labels].to_numpy(dtype=float)
    actions = df[action_labels].to_numpy(dtype=float)

    # Compute next_states
    next_states = states[1:]
    next_states = torch.tensor(next_states, dtype=torch.float32)

    # Last state repeats for terminal
    last_state = states[-1].reshape(1, -1)
    next_states = torch.cat(
        [next_states, torch.tensor(last_state, dtype=torch.float32)], dim=0
    )

    # Compute rewards
    rewards = torch.tensor(
        [
            get_reward(
                torch.tensor(s, dtype=torch.float32),
                torch.tensor(a, dtype=torch.float32),
            )
            for s, a in zip(states, actions)
        ],
        dtype=torch.float32,
    ).unsqueeze(1)

    # Terminal flags: last sample = 1, others = 0
    terminated = torch.zeros((len(states), 1), dtype=torch.float32)
    terminated[-1] = 1.0

    # Append to global lists
    all_states.append(torch.tensor(states, dtype=torch.float32))
    all_actions.append(torch.tensor(actions, dtype=torch.float32))
    all_next_states.append(next_states)
    all_rewards.append(rewards)
    all_terminated.append(terminated)

# Concatenate all files
states = torch.cat(all_states, dim=0)
actions = torch.cat(all_actions, dim=0)
next_states = torch.cat(all_next_states, dim=0)
rewards = torch.cat(all_rewards, dim=0)
terminated = torch.cat(all_terminated, dim=0)

# Create expert memory
expert_memory = RandomMemory(memory_size=len(states))
expert_memory.create_tensor(name="states", size=6)
expert_memory.create_tensor(name="actions", size=6)
expert_memory.create_tensor(name="next_states", size=6)
expert_memory.create_tensor(name="rewards", size=1)
expert_memory.create_tensor(name="terminated", size=1)

# Add all samples
expert_memory.add_samples(
    states=states,
    actions=actions,
    next_states=next_states,
    rewards=rewards,
    terminated=terminated,
)

print(f"Expert memory filled with {len(expert_memory)} samples")

# create empty memory
memory = RandomMemory(memory_size=10000, device=device, replacement=True)


# ------------------IL DP--------------------------
logger.info("Building DP...")

dp_config = dp.DIFFUSION_POLICY_STATE_DEFAULT_CONFIG
dp_config["num_envs"] = 1

dp_models = {}

act_dim = 6
obs_dim = 6

input_dim = act_dim

# Build models
dp_models = {}
dp_models["model"] = dp.ConditionalUnet1D(input_dim, dp_config).to(device)
# dp_models["model"] = dp.ConditionalUnet1D(input_dim, dp_config).to(device)
ema = dp.EMAModel(dp_models["model"].parameters(), power=dp_config["ema_power"])
# ema = dp.EMAModel(dp_models["model"].parameters(), power=dp_config["ema_power"])
dp_models["ema_model"] = dp.ConditionalUnet1D(input_dim, dp_config).to(device)

il_agent = DiffusionPolicy(models=dp_models, ema=ema, config=dp_config)

dp_models["policy"] = il_agent

# ------------------RL SAC-----------------------------
logger.info("Building RL (SAC)...")

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

# # Create Emseabling Q networks
# ensemble_size = 5
# # Create ensemble of critics (each with unique parameters)
# critics = []
# target_critics = []
# for i in range(ensemble_size):
#     # Create new critic instance for each position
#     critic = ibrl.Critic(env.observation_space, env.action_space, device)

#     # Initialize the models' parameters (weights and biases) using a Gaussian distribution
#     critic.init_parameters(method_name="normal_", mean=0.0, std=0.1)
#     target_critic = copy.deepcopy(critic)  # Create target as deep copy

#     critics.append(critic)
#     target_critics.append(target_critic)
# models["critics"] = critics[0]
# # models["critics"] = critics
# models["target_critics"] = target_critics[0]
# # models["target_critics"] = target_critics


logger.info("Configuring IBRL... ")

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
# configure and instantiate the agent
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["discount_factor"] = 0.99
cfg["batch_size"] = 10
# cfg["batch_size"] = 128
# cfg["batch_size"] = 10
# cfg["batch_size"] = 128
cfg["random_timesteps"] = 0  # Add some random exploration at the start
cfg["learning_starts"] = 0  # Start learning after some experience
cfg["learn_entropy"] = True
cfg["grad_norm_clip"] = 1.0  # Add gradient clipping for stability
cfg["learning_rate"] = 3e-4  # Standard SAC learning rate
cfg["initial_entropy_value"] = 0.1  # Entropy learning rate
cfg["RED-Q_enable"] = False  # enable RED-Q
cfg["offline"] = False  # not important here
cfg["num_envs"] = env.num_envs

# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 50
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["experiment_name"] = Path(__file__).stem
cfg["experiment"]["wandb"] = True
model_path = Path(__file__).parent / "results/models"
model_path.mkdir(parents=True, exist_ok=True)

cfg["experiment"]["directory"] = model_path.as_posix()
cfg["experiment"]["experiment_name"] = Path(__file__).stem
cfg["experiment"]["wandb"] = True

logger.info("Building IBRL... ")

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


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 350000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
# trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# # start training
trainer.train()
