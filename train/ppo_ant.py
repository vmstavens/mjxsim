# import isaacgym
# import isaacgymenvs
import copy
from pathlib import Path

import torch
import torch.nn as nn
from skrl.agents.torch.ppo import PPO, PPO_CFG

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory

# Import the skrl components to build the RL system
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from mjxsim.utils.envs import ENV_TYPE, mk_env

# set the seed for reproducibility
set_seed(27)


# Define the shared model (stochastic and deterministic models) for the agent using mixins.
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64), nn.ELU(), nn.Linear(64, 32), nn.ELU()
        )

        self.mean_layer = nn.Linear(32, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(32, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        observations = inputs.get("states", inputs.get("observations"))
        if role == "policy":
            return (
                self.mean_layer(self.net(observations)),
                self.log_std_parameter,
                {},
            )
        elif role == "value":
            return self.value_layer(self.net(observations)), {}


class DeterministicActor(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        dropout_rate=0.5,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.Dropout(dropout_rate),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.Dropout(dropout_rate),
            nn.ELU(),
            nn.Linear(32, self.num_actions),
        )

    def compute(self, inputs, role):
        observations = inputs.get("states", inputs.get("observations"))
        return self.net(observations), {}


env_name = "ant"
num_envs = 256
env = mk_env(env_name, num_envs=num_envs)
device = env.device

# ============================================================================
# # Instantiate replay and expert memory buffer, expert memory is used for storing trained PPO demo
# ============================================================================
memory = RandomMemory(memory_size=128, num_envs=env.num_envs, device=device)
# expert_memory = RandomMemory(
#     memory_size=5000, num_envs=env.num_envs, device=device, replacement=False
# )

# Instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
models_ppo = {}
models_ppo["policy"] = Shared(env.observation_space, env.action_space, device)
models_ppo["value"] = Shared(
    env.observation_space, env.action_space, device
)  # same instance: shared model

# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_ppo = PPO_CFG()
cfg_ppo.rollouts = 128  # memory_size  ## 16 horizon_length
cfg_ppo.learning_epochs = 2  # mini_epochs
cfg_ppo.mini_batches = 64
cfg_ppo.discount_factor = 0.99
cfg_ppo.gae_lambda = 0.95
cfg_ppo.learning_rate = 3e-4
cfg_ppo.learning_rate_scheduler = KLAdaptiveLR
cfg_ppo.learning_rate_scheduler_kwargs = {"kl_threshold": 0.008}
cfg_ppo.random_timesteps = 0
cfg_ppo.learning_starts = 0
cfg_ppo.grad_norm_clip = 1.0
cfg_ppo.ratio_clip = 0.2
cfg_ppo.value_clip = 0.2
cfg_ppo.clip_predicted_values = True
cfg_ppo.entropy_loss_scale = 0.0
cfg_ppo.value_loss_scale = 2.0
cfg_ppo.kl_threshold = 0
cfg_ppo.rewards_shaper = lambda rewards, timestep, timesteps: rewards * 0.1
cfg_ppo.state_preprocessor = RunningStandardScaler
cfg_ppo.state_preprocessor_kwargs = {"size": env.observation_space, "device": device}
cfg_ppo.value_preprocessor = RunningStandardScaler
cfg_ppo.value_preprocessor_kwargs = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints each 20 and 200 timesteps respectively
cfg_ppo.experiment.write_interval = 50
cfg_ppo.experiment.checkpoint_interval = 200


cfg_ppo.experiment.experiment_name = Path(__file__).stem
cfg_ppo.experiment.wandb = True
model_path = Path(__file__).parent / "results/models"
model_path.mkdir(parents=True, exist_ok=True)

cfg_ppo.experiment.directory = model_path.as_posix()
cfg_ppo.experiment.experiment_name = Path(__file__).stem

agent = PPO(
    models=models_ppo,
    memory=memory,
    # expert_memory=expert_memory,
    cfg=cfg_ppo,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 50_000, "headless": True}

trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()
