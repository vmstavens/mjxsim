import copy
import dataclasses
import itertools
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from skrl import config, logger
from skrl.agents.torch.base import AgentCfg, ExperimentCfg
from skrl.memories.torch import Memory
from skrl.models.torch import Model

from .ibrl_base_agent import Agent


@dataclasses.dataclass(kw_only=True)
class DRLR_CFG(AgentCfg):
    """Configuration for the DRLR SAC agent."""

    gradient_steps: int = 1
    batch_size: int = 64
    discount_factor: float = 0.99
    polyak: float = 0.005
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    learning_rate_scheduler: type | None = None
    learning_rate_scheduler_kwargs: dict[str, Any] = dataclasses.field(
        default_factory=dict
    )
    state_preprocessor: type | None = None
    state_preprocessor_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    random_timesteps: int = 0
    learning_starts: int = 0
    grad_norm_clip: float = 0
    learn_entropy: bool = True
    entropy_learning_rate: float = 3e-4
    initial_entropy_value: float = 0.01
    target_entropy: float | None = None
    offline: bool = False
    demo_file: str = ""
    num_envs: int = 1
    rewards_shaper: Any = None
    mixed_precision: bool = False
    experiment: ExperimentCfg = dataclasses.field(
        default_factory=lambda: ExperimentCfg(
            write_interval="auto",
            checkpoint_interval="auto",
        )
    )

    def expand(self) -> None:
        super().expand()


DRLR_DEFAULT_CONFIG = DRLR_CFG()


class DRLR(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        models_il: Dict[str, Model],
        memory: Optional[Memory] = None,
        expert_memory: Optional[Memory] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[DRLR_CFG] = None,
    ) -> None:
        """Deep Reinforcement Learning with Reference (DRLR)

        https://arxiv.org/abs/2509.04069

        :param models: Models used by the agent
        :type models:  Working as RL agent, dictionary of skrl.models.torch.Model
        :type models_il:  Working as IL agent, dictionary of skrl.models.torch.Model
        :param memory: Working as Replay buffer, to storage the transitions from online interactions.
        :type memory: skrl.memory.torch.Memory or None
        :param expert_memory: Working as expert buffer, to storage the offline expert demonstrations. Same type with memory.
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Agent configuration
        :type cfg: DRLR_CFG

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = DRLR_CFG() if cfg is None else copy.deepcopy(cfg)
        _cfg.expand()
        super().__init__(
            models=models,
            models_il=models_il,
            memory=memory,
            expert_memory=expert_memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )
        # IL model
        self.IL_policy = self.models_il.get("policy", None)
        # models
        self.policy = self.models.get("policy", None)
        self.critic_1 = self.models.get("critic_1", None)
        self.critic_2 = self.models.get("critic_2", None)
        self.target_critic_1 = self.models.get("target_critic_1", None)
        self.target_critic_2 = self.models.get("target_critic_2", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["critic_1"] = self.critic_1
        self.checkpoint_modules["critic_2"] = self.critic_2
        self.checkpoint_modules["target_critic_1"] = self.target_critic_1
        self.checkpoint_modules["target_critic_2"] = self.target_critic_2

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info("Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.critic_1 is not None:
                self.critic_1.broadcast_parameters()
            if self.critic_2 is not None:
                self.critic_2.broadcast_parameters()

        if self.target_critic_1 is not None and self.target_critic_2 is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_critic_1.freeze_parameters(True)
            self.target_critic_2.freeze_parameters(True)

            # update target networks (hard update)
            self.target_critic_1.update_parameters(self.critic_1, polyak=1)
            self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        # configuration
        self._gradient_steps = _cfg.gradient_steps
        self._batch_size = _cfg.batch_size

        self._discount_factor = _cfg.discount_factor
        self._polyak = _cfg.polyak

        self._actor_learning_rate = _cfg.actor_learning_rate
        self._critic_learning_rate = _cfg.critic_learning_rate
        self._learning_rate_scheduler = _cfg.learning_rate_scheduler

        self._state_preprocessor = _cfg.state_preprocessor

        self._random_timesteps = _cfg.random_timesteps
        self._learning_starts = _cfg.learning_starts

        self._grad_norm_clip = _cfg.grad_norm_clip

        self._entropy_learning_rate = _cfg.entropy_learning_rate
        self._learn_entropy = _cfg.learn_entropy
        self._entropy_coefficient = _cfg.initial_entropy_value

        self._rewards_shaper = _cfg.rewards_shaper

        self._mixed_precision = _cfg.mixed_precision
        self._offline = _cfg.offline
        self._num_envs = _cfg.num_envs

        self._demo_file = _cfg.demo_file

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(
                device=self._device_type, enabled=self._mixed_precision
            )
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # entropy
        if self._learn_entropy:
            self._target_entropy = _cfg.target_entropy
            if self._target_entropy is None:
                if issubclass(type(self.action_space), gymnasium.spaces.Box):
                    self._target_entropy = -np.prod(self.action_space.shape).astype(
                        np.float32
                    )
                elif issubclass(type(self.action_space), gymnasium.spaces.Discrete):
                    self._target_entropy = -self.action_space.n
                else:
                    self._target_entropy = 0

            self.log_entropy_coefficient = torch.log(
                torch.ones(1, device=self.device) * self._entropy_coefficient
            ).requires_grad_(True)
            self.entropy_optimizer = torch.optim.Adam(
                [self.log_entropy_coefficient], lr=self._entropy_learning_rate
            )

            self.checkpoint_modules["entropy_optimizer"] = self.entropy_optimizer

        # set up optimizers and learning rate schedulers
        if (
            self.policy is not None
            and self.critic_1 is not None
            and self.critic_2 is not None
        ):
            self.policy_optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=self._actor_learning_rate
            )
            self.critic_optimizer = torch.optim.Adam(
                itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                lr=self._critic_learning_rate,
            )
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(
                    self.policy_optimizer, **_cfg.learning_rate_scheduler_kwargs
                )
                self.critic_scheduler = self._learning_rate_scheduler(
                    self.critic_optimizer, **_cfg.learning_rate_scheduler_kwargs
                )

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(
                **_cfg.state_preprocessor_kwargs
            )
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[dict[str, Any]] = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.enable_training_mode(False, apply_to_models=True)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(
                name="states", size=self.observation_space, dtype=torch.float32
            )
            self.memory.create_tensor(
                name="next_states", size=self.observation_space, dtype=torch.float32
            )
            self.memory.create_tensor(
                name="actions", size=self.action_space, dtype=torch.float32
            )
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self._tensors_names = [
                "states",
                "actions",
                "rewards",
                "next_states",
                "terminated",
            ]

            # create tensors in memory
            # # self.expert_memory
            if self.expert_memory is not None:
                self.expert_memory.create_tensor(
                    name="states", size=self.observation_space, dtype=torch.float32
                )
                self.expert_memory.create_tensor(
                    name="next_states", size=self.observation_space, dtype=torch.float32
                )
                self.expert_memory.create_tensor(
                    name="actions", size=self.action_space, dtype=torch.float32
                )
                self.expert_memory.create_tensor(
                    name="rewards", size=1, dtype=torch.float32
                )
                self.expert_memory.create_tensor(
                    name="terminated", size=1, dtype=torch.bool
                )
                self._tensors_names = [
                    "states",
                    "actions",
                    "rewards",
                    "next_states",
                    "terminated",
                ]

    def _select_act(
        self, obs: torch.Tensor, exp_obs: torch.Tensor, soft: bool, target: bool
    ):
        if target:
            # target policy smoothing
            rl_actions, policy_outputs = self._unpack_act_result(
                self.policy.act(self._state_inputs(obs), role="policy")
            )
            next_log_prob = policy_outputs["log_prob"]
        else:
            # sample stochastic actions
            with torch.autocast(
                device_type=self._device_type, enabled=self._mixed_precision
            ):
                rl_actions, outputs = self._unpack_act_result(
                    self.policy.act(
                        self._state_inputs(self._state_preprocessor(obs)),
                        role="policy",
                    )
                )

        # Get IL actions
        self.IL_policy.eval()
        il_actions, _ = self._unpack_act_result(
            self.IL_policy.act(
                self._state_inputs(self._state_preprocessor(exp_obs)), role="policy"
            )
        )

        # Stack actions and get batch dimensions
        rl_bc_actions = torch.stack([rl_actions, il_actions], dim=1)
        batch_size, num_action, _ = (
            rl_bc_actions.size()
        )  # get dimensions values, bsize:batch size

        # Compute min Q-values for both policies
        target_q_il = self._compute_min_q_values(exp_obs, il_actions)
        target_q_rl = self._compute_min_q_values(obs, rl_actions)

        # Stack Q-values
        target_q_values = torch.stack([target_q_rl, target_q_il], dim=1).view(
            batch_size, num_action
        )

        if torch.mean(target_q_il) > torch.mean(target_q_rl):
            il_actions, _ = self._unpack_act_result(
                self.IL_policy.act(
                    self._state_inputs(self._state_preprocessor(obs)), role="policy"
                )
            )
            actions = il_actions

        else:
            actions = rl_actions

        if not target:
            self.track_data(
                "Q-network / select_rl_Q (mean)", torch.mean(target_q_rl).item()
            )
            self.track_data(
                "Q-network / select_il_Q (mean)", torch.mean(target_q_il).item()
            )
            return actions, None, outputs
        else:
            return actions, next_log_prob, {}

    def act(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None = None,
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        states = observations

        # sample random actions
        if timestep < self._random_timesteps:
            return self.policy.random_act(
                self._state_inputs(self._state_preprocessor(states)), role="policy"
            )

        expert_states_r = self.expert_memory.sample(
            names=["states"], batch_size=self._num_envs
        )[0][0]

        # select actions
        actions, _, outputs = self._select_act(
            states, expert_states_r, soft=True, target=False
        )

        return actions, outputs or {}

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor | None,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor | None,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        states = observations
        next_states = next_observations

        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        if timestep < self._random_timesteps + self._learning_starts - 1:
            self.expert_memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
            )

        # storage transition in memory
        self.memory.add_samples(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
        )
        # if timestep == timesteps - 1:
        #     self.memory.save("./Demos", "csv")

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if self.training and timestep >= self._learning_starts:
            self.enable_models_training_mode(True)
            self.update(timestep=timestep, timesteps=timesteps)
            self.enable_models_training_mode(False)

        # write tracking data and checkpoints
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        self._update(timestep, timesteps)

    def _compute_min_q_values(self, states, actions):
        """Helper to compute target Q-values using both critics"""
        target_q1_value, _ = self._unpack_act_result(
            self.target_critic_1.act(
                self._state_inputs(states, taken_actions=actions),
                role="target_critic_1",
            )
        )
        target_q2_value, _ = self._unpack_act_result(
            self.target_critic_2.act(
                self._state_inputs(states, taken_actions=actions),
                role="target_critic_2",
            )
        )
        return torch.min(target_q1_value, target_q2_value)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        # gradient steps
        for gradient_step in range(self._gradient_steps):
            if self._offline:
                (
                    sampled_states,
                    sampled_actions,
                    sampled_rewards,
                    sampled_next_states,
                    sampled_dones,
                ) = self.expert_memory.sample(
                    names=self._tensors_names, batch_size=self._batch_size
                )[0]

            else:
                (
                    sampled_states,
                    sampled_actions,
                    sampled_rewards,
                    sampled_next_states,
                    sampled_dones,
                ) = self.memory.sample(
                    names=self._tensors_names, batch_size=int(self._batch_size)
                )[0]
                (
                    expert_states,
                    expert_actions,
                    expert_rewards,
                    expert_next_states,
                    expert_dones,
                ) = self.expert_memory.sample(
                    names=self._tensors_names, batch_size=int(self._batch_size)
                )[0]

            with torch.autocast(
                device_type=self._device_type, enabled=self._mixed_precision
            ):
                sampled_states = self._state_preprocessor(sampled_states, train=True)
                sampled_next_states = self._state_preprocessor(
                    sampled_next_states, train=True
                )

                # compute target values
                with torch.no_grad():
                    next_actions, next_log_prob, _ = self._select_act(
                        sampled_next_states, expert_next_states, soft=True, target=True
                    )  # DRLR core modification
                    # next_actions, next_log_prob, _ = self._select_act(sampled_next_states, sampled_next_states, soft=True,
                    #                                       target=True)  # DRLR core modification

                    target_q1_values, _ = self._unpack_act_result(
                        self.target_critic_1.act(
                            self._state_inputs(
                                sampled_next_states, taken_actions=next_actions
                            ),
                            role="target_critic_1",
                        )
                    )
                    target_q2_values, _ = self._unpack_act_result(
                        self.target_critic_2.act(
                            self._state_inputs(
                                sampled_next_states, taken_actions=next_actions
                            ),
                            role="target_critic_2",
                        )
                    )
                    target_q_values = (
                        torch.min(target_q1_values, target_q2_values)
                        - self._entropy_coefficient * next_log_prob
                    )
                    target_values = (
                        sampled_rewards
                        + self._discount_factor
                        * (sampled_dones).logical_not()
                        * target_q_values
                    )

                    discout_q_values = (
                        self._discount_factor
                        * (sampled_dones).logical_not()
                        * target_q_values
                    ).mean()

            critic_1_values, _ = self._unpack_act_result(
                self.critic_1.act(
                    self._state_inputs(sampled_states, taken_actions=sampled_actions),
                    role="critic_1",
                )
            )
            critic_2_values, _ = self._unpack_act_result(
                self.critic_2.act(
                    self._state_inputs(sampled_states, taken_actions=sampled_actions),
                    role="critic_2",
                )
            )
            critic_loss = F.mse_loss(
                critic_1_values, target_values
            ) + F.mse_loss(critic_2_values, target_values)

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self._grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(
                    itertools.chain(
                        self.critic_1.parameters(), self.critic_2.parameters()
                    ),
                    self._grad_norm_clip,
                )
            self.critic_optimizer.step()

            # if config.torch.is_distributed:
            #     self.critic_1.reduce_parameters()
            #     self.critic_2.reduce_parameters()
            #
            # if self._grad_norm_clip > 0:
            #     self.scaler.unscale_(self.critic_optimizer)
            #     nn.utils.clip_grad_norm_(
            #         itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()), self._grad_norm_clip
            #     )
            #
            # self.scaler.step(self.critic_optimizer)

            with torch.autocast(
                device_type=self._device_type, enabled=self._mixed_precision
            ):
                # compute policy (actor) loss
                actions, outputs = self._unpack_act_result(
                    self.policy.act(self._state_inputs(sampled_states), role="policy")
                )
                log_prob = outputs["log_prob"]
                critic_1_values, _ = self._unpack_act_result(
                    self.critic_1.act(
                        self._state_inputs(sampled_states, taken_actions=actions),
                        role="critic_1",
                    )
                )
                critic_2_values, _ = self._unpack_act_result(
                    self.critic_2.act(
                        self._state_inputs(sampled_states, taken_actions=actions),
                        role="critic_2",
                    )
                )

                policy_loss = (
                    self._entropy_coefficient * log_prob
                    - torch.min(critic_1_values, critic_2_values)
                ).mean()

            # optimization step (policy)
            self.policy_optimizer.zero_grad()
            self.scaler.scale(policy_loss).backward()

            if config.torch.is_distributed:
                self.policy.reduce_parameters()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)

            self.scaler.step(self.policy_optimizer)

            # entropy learning
            if self._learn_entropy:
                with torch.autocast(
                    device_type=self._device_type, enabled=self._mixed_precision
                ):
                    # compute entropy loss
                    entropy_loss = -(
                        self.log_entropy_coefficient
                        * (log_prob + self._target_entropy).detach()
                    ).mean()

                # optimization step (entropy)
                self.entropy_optimizer.zero_grad()
                self.scaler.scale(entropy_loss).backward()
                self.scaler.step(self.entropy_optimizer)

                # compute entropy coefficient
                self._entropy_coefficient = torch.exp(
                    self.log_entropy_coefficient.detach()
                )

            self.scaler.update()  # called once, after optimizers have been stepped

            # update target networks
            self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.policy_scheduler.step()
                self.critic_scheduler.step()

            # record data
            if self.write_interval > 0:
                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Loss / Critic loss", critic_loss.item())

                self.track_data(
                    "Q-network / Q1 (max)", torch.max(critic_1_values).item()
                )
                self.track_data(
                    "Q-network / Q1 (min)", torch.min(critic_1_values).item()
                )
                self.track_data(
                    "Q-network / Q1 (mean)", torch.mean(critic_1_values).item()
                )

                self.track_data(
                    "Q-network / Q2 (max)", torch.max(critic_2_values).item()
                )
                self.track_data(
                    "Q-network / Q2 (min)", torch.min(critic_2_values).item()
                )
                self.track_data(
                    "Q-network / Q2 (mean)", torch.mean(critic_2_values).item()
                )

                self.track_data(
                    "Target / Target (max)", torch.max(target_values).item()
                )
                self.track_data(
                    "Target / Target (min)", torch.min(target_values).item()
                )
                self.track_data(
                    "Target / Target (mean)", torch.mean(target_values).item()
                )
                self.track_data(
                    "Target / sampled_rewards (mean)",
                    torch.mean(sampled_rewards).item(),
                )
                self.track_data(
                    "Target / discount (mean)", torch.mean(discout_q_values).item()
                )

                if self._learn_entropy:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())
                    self.track_data(
                        "Coefficient / Entropy coefficient",
                        self._entropy_coefficient.item(),
                    )

                if self._learning_rate_scheduler:
                    self.track_data(
                        "Learning / Policy learning rate",
                        self.policy_scheduler.get_last_lr()[0],
                    )
                    self.track_data(
                        "Learning / Critic learning rate",
                        self.critic_scheduler.get_last_lr()[0],
                    )
