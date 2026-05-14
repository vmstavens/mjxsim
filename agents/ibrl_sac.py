import copy
import dataclasses
import itertools
import logging
import os
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

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

logging.basicConfig(level=logging.WARNING)  # This adds a default handler
relative_path = os.path.relpath(__file__)  # Relative to current working directory
logger = logging.getLogger(relative_path)
logger.setLevel(logging.WARNING)


@dataclasses.dataclass(kw_only=True)
class ExplorationCfg:
    """Configuration for exploration noise scheduling."""

    noise: Any = None
    initial_scale: float = 1.0
    final_scale: float = 1e-3
    timesteps: int | None = None


@dataclasses.dataclass(kw_only=True)
class IBRL_SAC_CFG(AgentCfg):
    """Configuration for the IBRL SAC agent."""

    gradient_steps: int = 1
    batch_size: int = 64
    warmup_timesteps: int = 10_000
    il_ctrl_scale: float = 1.0
    rl_ctrl_scale: float = 1.0
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
    exploration: ExplorationCfg = dataclasses.field(default_factory=ExplorationCfg)
    learn_entropy: bool = True
    entropy_learning_rate: float = 3e-4
    initial_entropy_value: float = 0.2
    target_entropy: float | None = None
    rewards_shaper: Callable | None = None
    mixed_precision: bool = False
    experiment: ExperimentCfg = dataclasses.field(
        default_factory=lambda: ExperimentCfg(
            write_interval=1000,
            checkpoint_interval=1000,
        )
    )
    soft_update_beta: float = 0.2
    actor: str = "both"
    num_envs: int = 1

    def expand(self) -> None:
        super().expand()


IBRL_SAC_DEFAULT_CONFIG = IBRL_SAC_CFG()


class IBRL(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        models_il: Dict[str, Model],
        memory: Optional[Memory],
        expert_memory: Optional[Memory],
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[IBRL_SAC_CFG] = None,
    ) -> None:
        """Imitation Bootstrapped Reinforcement Learning (IBRL) - SAC Based

        https://arxiv.org/abs/2311.02198

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param models_il: Imitation learning models used by the agent
        :type models_il: dictionary of skrl.models.torch.Model
        :param memory: Memory to store the transitions.
        :type memory: skrl.memories.torch.Memory or None
        :param expert_memory: Expert demonstration memory buffer
        :type expert_memory: skrl.memories.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Agent configuration
        :type cfg: IBRL_SAC_CFG

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = IBRL_SAC_CFG() if cfg is None else copy.deepcopy(cfg)
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

        # memories
        self._tensors_names = [
            "states",
            "actions",
            "rewards",
            "next_states",
            "terminated",
        ]

        # Ensure that memory and expert_memory have the correct tensors
        assert set(self.memory.get_tensor_names()).issubset(self._tensors_names), (
            "Memory error: memory should have the tensor names "
            f"{self._tensors_names}, but got {self.memory.get_tensor_names()}"
        )
        assert set(self.expert_memory.get_tensor_names()).issubset(
            self._tensors_names
        ), (
            "Memory error: expert_memory should have the tensor names "
            f"{self._tensors_names}, but got {self.expert_memory.get_tensor_names()}"
        )

        # IL model
        self.IL_policy = self.models_il["policy"]

        # models
        self.policy = self.models["policy"]

        self.critic_1 = self.models["critic_1"]
        self.critic_2 = self.models["critic_2"]
        self.critics = [self.critic_1, self.critic_2]

        self.target_critic_1 = self.models["target_critic_1"]
        self.target_critic_2 = self.models["target_critic_2"]
        self.target_critics = [self.target_critic_1, self.target_critic_2]

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["critic_1"] = self.critic_1
        self.checkpoint_modules["critic_2"] = self.critic_2
        self.checkpoint_modules["target_critic_1"] = self.target_critic_1
        self.checkpoint_modules["target_critic_2"] = self.target_critic_2

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
        self._gradient_steps: int = _cfg.gradient_steps
        self._batch_size: int = _cfg.batch_size
        self._discount_factor: float = _cfg.discount_factor
        self._polyak: float = _cfg.polyak
        self._actor_learning_rate: float = _cfg.actor_learning_rate
        self._critic_learning_rate: float = _cfg.critic_learning_rate
        self._learning_rate_scheduler = _cfg.learning_rate_scheduler
        self._state_preprocessor = _cfg.state_preprocessor
        self._random_timesteps: int = _cfg.random_timesteps
        self._learning_starts: int = _cfg.learning_starts  # 0
        self._grad_norm_clip = _cfg.grad_norm_clip
        self._exploration_noise = _cfg.exploration.noise
        self._exploration_initial_scale = _cfg.exploration.initial_scale
        self._exploration_final_scale = _cfg.exploration.final_scale
        self._exploration_timesteps = _cfg.exploration.timesteps
        self._entropy_learning_rate: float = _cfg.entropy_learning_rate
        self._learn_entropy: bool = _cfg.learn_entropy
        self._entropy_coefficient: float = _cfg.initial_entropy_value
        self._rewards_shaper = _cfg.rewards_shaper
        self._mixed_precision: bool = _cfg.mixed_precision
        self._soft_update_beta = _cfg.soft_update_beta
        self._actor: str = _cfg.actor
        self._num_envs: int = _cfg.num_envs

        self._actors = ["rl", "il", "both"]

        self._il_ctrl_scale = _cfg.il_ctrl_scale
        self._rl_ctrl_scale = _cfg.rl_ctrl_scale
        self._warmup_timesteps = _cfg.warmup_timesteps

        assert self._actor in self._actors, (
            f"In config: 'actor' should be one of {self._actors} but got {self._actor}"
        )
        if isinstance(self.action_space, gymnasium.spaces.Box):
            self.clip_actions_min = torch.tensor(
                self.action_space.low, device=self.device, dtype=torch.float32
            )
            self.clip_actions_max = torch.tensor(
                self.action_space.high, device=self.device, dtype=torch.float32
            )
        else:
            self.clip_actions_min = None
            self.clip_actions_max = None

        # used to keep track of observations for diffusion policy
        self._states: torch.Tensor = None
        self._prev_states: torch.Tensor = None
        self._state_window_timestep: int | None = None

        # set up automatic mixed precision
        self._device_type = torch.device(self.device).type
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

    def _select_act(
        self,
        rl_obs: torch.Tensor,
        il_obs: torch.Tensor,
        exp_obs: torch.Tensor,
        soft: bool,
        target: bool,
        timestep: int,
    ):
        """Select an action by comparing RL and IL policies and their Q-values.

        :param rl_obs: Observations for the RL policy
        :type rl_obs: torch.Tensor
        :param il_obs: Observations for the IL policy
        :type il_obs: torch.Tensor
        :param exp_obs: Expert observations used for IL Q evaluation
        :type exp_obs: torch.Tensor
        :param soft: Whether to sample actions with a softmax strategy
        :type soft: bool
        :param target: Whether to use target policy/Q networks
        :type target: bool
        :return: Selected actions, log-probabilities (if available), and extra outputs
        :rtype: tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]
        """
        if target:
            # target policy smoothing
            rl_actions, policy_outputs = self._unpack_act_result(
                self.policy.act(self._state_inputs(rl_obs), role="policy")
            )
            next_log_prob = policy_outputs["log_prob"]
        else:
            # sample stochastic actions
            with torch.autocast(
                device_type=self._device_type, enabled=self._mixed_precision
            ):
                rl_actions, outputs = self._unpack_act_result(
                    self.policy.act(
                        self._state_inputs(self._state_preprocessor(rl_obs)),
                        role="policy",
                    )
                )

        il_actions, _ = self._unpack_act_result(
            self.IL_policy.act(
                self._state_inputs(self._state_preprocessor(il_obs)),
                role="policy",
            )
        )

        il_actions = il_actions[:, 0, :]
        il_states = il_obs[:, -1, :]

        # scale il_actions
        il_actions = il_actions * self._il_ctrl_scale
        rl_actions = rl_actions * self._rl_ctrl_scale

        # Change this line - instead of concatenating, stack the actions
        rl_il_actions = torch.stack([rl_actions, il_actions], dim=1)

        batch_size, _, num_action = rl_il_actions.size()

        target_q_il = self._compute_min_q_values(il_states, il_actions)
        target_q_rl = self._compute_min_q_values(rl_obs, rl_actions)

        target_q_values = torch.hstack([target_q_rl, target_q_il])

        if soft:
            # Boltzmann exploration
            # convert q values to a probability distribution using softmax
            probs = F.softmax(target_q_values * self._soft_update_beta, dim=1)
            action_indices = probs.multinomial(1)  # Shape: [num_envs]

            # how many percent of the action taken is il
            pct_of_il = action_indices.sum().item() / len(action_indices)

            # OBS Better logging needed
            if self._actor == "rl":
                il_ratio = 0.0
            elif self._actor == "il":
                il_ratio = 1.0
            elif self._actor == "both":
                il_ratio = 1.0 if timestep < self._warmup_timesteps else pct_of_il
            else:
                raise ValueError(
                    f"[ERROR]: {self._actor=}, has to be ('rl', 'il' or 'both')"
                )
            # self.track_data("Which / il_selection_ratio", il_ratio)

            # we here index over the entire batch (using torch.arange(batch_size)) in the rows and
            # use the action indecies for either rl or il in the columns

            # OBS: holy shit this might be it!
            actions = rl_actions * (1 - action_indices) + il_actions * action_indices
        else:
            # Greedy selection
            action_indices = target_q_values.argmax(dim=1)  # Shape: [num_envs]
            actions = rl_il_actions[torch.arange(batch_size), action_indices]
            il_ratio = action_indices.sum().item() / len(action_indices)

        # here "actions" is the product of both agents
        self.track_data("Which / il_selection_ratio", il_ratio)

        actor_id = 0

        if self._actor == "rl":
            actions = rl_actions
            actor_id = 1

        elif self._actor == "il":
            actions = il_actions
            actor_id = 2

        elif self._actor == "both":
            actor_id = 3
            if timestep < self._warmup_timesteps:
                actor_id = 4
                actions = il_actions

        else:
            raise ValueError("Wrong actor")

        self.track_data("Which / Actor", actor_id)

        if not target:
            self.track_data(
                "Q-network / select_rl_Q (mean)", torch.mean(target_q_rl).item()
            )
            self.track_data(
                "Q-network / select_il_Q (mean)", torch.mean(target_q_il).item()
            )
            il_selected = (torch.mean(target_q_il) > torch.mean(target_q_rl)).float()
            self.track_data("Online / IL selection probability", il_selected.item())
            return actions, None, outputs
        else:
            self.track_data(
                "Q-network / select_rl_Q (max)", torch.max(target_q_rl).item()
            )
            self.track_data(
                "Q-network / select_il_Q (max)", torch.max(target_q_il).item()
            )
            self.track_data(
                "Bootstrap / IL selection probability",
                action_indices.sum().item() / self._batch_size,
            )
            self.track_data(
                "Bootstrap / select_rl_Q (mean)", torch.mean(target_q_rl).item()
            )
            self.track_data(
                "Bootstrap / select_il_Q (mean)", torch.mean(target_q_il).item()
            )

            return actions, next_log_prob, {}

    def _set_current_states(self, states: torch.Tensor, timestep: int) -> None:
        if self._states is None or self._states.shape[0] != states.shape[0]:
            self._prev_states = states
        else:
            self._prev_states = self._states
        self._states = states
        self._state_window_timestep = timestep

    def act(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None = None,
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Process environment states and return an action tuple.

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions, log-probabilities, and extra outputs (unused)
        :rtype: tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]
        """
        # sample random actions
        states = observations
        if self._state_window_timestep != timestep:
            self._set_current_states(states, timestep)

        if timestep < self._random_timesteps:
            return self.policy.random_act(
                self._state_inputs(self._state_preprocessor(states)), role="policy"
            )

        # sample from expert buffer
        (
            expert_states,
            expert_actions,
            expert_rewards,
            expert_next_states,
            expert_dones,
        ) = self.expert_memory.sample(
            names=self._tensors_names,
            # OBS: Obs, we are sampling num envs instead of batch_size in order to follow the shape
            # of the observations coming from the environment.
            batch_size=self._num_envs,
        )[0]

        # here states (num_envs, o_dim) and next_states (num_envs, o_dim)
        # we need il_states (num_envs, pred_horizon, o_dim)
        # therefore axis=1

        il_states = torch.stack([self._prev_states, self._states], axis=1)

        actions, _, outputs = self._select_act(
            rl_obs=states,
            il_obs=il_states,
            exp_obs=expert_states,
            soft=True,
            target=False,
            timestep=timestep,
        )

        return actions, outputs or {}

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor | None = None,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor | None = None,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any = None,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory and expert buffer.

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
        if states is None or next_states is None:
            raise ValueError(
                "states/observations and next_states/next_observations are required"
            )

        super().record_transition(
            observations=states,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_states,
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

    def pre_interaction(
        self,
        *,
        observations: torch.Tensor | None = None,
        states: torch.Tensor | None = None,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Callback called before the interaction with the environment.

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        states = observations if states is None else states
        if states is not None:
            self._set_current_states(states, timestep)

    def post_interaction(
        self,
        *,
        next_states: torch.Tensor | None = None,
        next_observations: torch.Tensor | None = None,
        terminated: torch.Tensor | None = None,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Callback called after the interaction with the environment.

        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.BoolTensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        next_states = next_observations if next_states is None else next_states
        # save the next state for diffusion policy observation horizon
        if next_states is not None:
            self._prev_states = next_states

        if self.training and timestep >= self._learning_starts:
            self.enable_models_training_mode(True)
            self.update(timestep=timestep, timesteps=timesteps)
            self.enable_models_training_mode(False)

        # write tracking data and checkpoints
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def enable_training_mode(
        self, enabled: bool = True, *, apply_to_models: bool = False
    ) -> None:
        super().enable_training_mode(enabled, apply_to_models=apply_to_models)

    def update(self, *, timestep: int, timesteps: int) -> None:
        self._update(timestep, timesteps)

    def _compute_min_q_values(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-sample min Q-values from both target critics.

        :param states: Batch of states
        :type states: torch.Tensor
        :param actions: Batch of actions
        :type actions: torch.Tensor
        :return: Min Q-values across critics
        :rtype: torch.Tensor
        """

        target_q_values_list = []

        for idx in [0, 1]:
            if len(states.shape) == 1:
                states = states.unsqueeze(0)
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(0)
            target_q_val, _ = self._unpack_act_result(
                self.target_critics[idx].act(
                    self._state_inputs(states, taken_actions=actions),
                    role=f"target_critic_{idx + 1}",
                )
            )
            target_q_values_list.append(target_q_val)

        target_q_values = torch.hstack(target_q_values_list)  # (num_envs, 2)

        target_q_value = torch.min(target_q_values, dim=1).values.unsqueeze(-1)
        return target_q_value

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step.

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # gradient steps
        for gradient_step in range(self._gradient_steps):
            # here we mix expert memory and sampled memory
            (
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_dones,
            ) = self.memory.sample(
                names=self._tensors_names, batch_size=self._batch_size
            )[0]

            (
                expert_states,
                expert_actions,
                expert_rewards,
                expert_next_states,
                expert_dones,
            ) = self.expert_memory.sample(
                names=self._tensors_names, batch_size=self._batch_size
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
                    # next_actions, next_log_prob, _ = self.policy.act({"states": sampled_next_states}, role="policy")
                    # TODO: here we add o2 as well
                    # TODO: here we attempt to build the DP state queue. We here choose the expert memory
                    # due to the Bootstrapping error as described in https://arxiv.org/pdf/2509.04069

                    il_next_states = torch.stack(
                        [sampled_states, sampled_next_states], axis=1
                    )
                    il_expert_states = torch.stack(
                        [expert_states, expert_next_states], axis=1
                    )

                    # TODO: Here we need to figure out if we should use the expert or the samples
                    # obs for DP
                    # OBS: soft = true, is that okay? in ibrl it uses soft false
                    next_actions, next_log_prob, _ = self._select_act(
                        rl_obs=sampled_next_states,
                        il_obs=il_next_states,
                        exp_obs=il_expert_states,
                        soft=True,
                        target=True,
                        timestep=timestep,
                    )

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

                # compute critic loss
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

                # OBS sum, not average
                # critic_loss = F.mse_loss(critic_1_values, target_values) + F.mse_loss(
                #     critic_2_values, target_values
                # )
                critic_loss = (
                    F.mse_loss(critic_1_values, target_values)
                    + F.mse_loss(critic_2_values, target_values)
                ) / 2

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(
                        self.critic_1.parameters(), self.critic_2.parameters()
                    ),
                    self._grad_norm_clip,
                )

            self.scaler.step(self.critic_optimizer)

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

                bc_loss = F.mse_loss(actions, sampled_actions)

                policy_loss = (
                    self._entropy_coefficient * log_prob
                    - torch.min(critic_1_values, critic_2_values)
                ).mean()

                self.track_data(
                    "Loss / self._entropy_coefficient * log_prob",
                    (self._entropy_coefficient * log_prob).mean().item(),
                )
                self.track_data(
                    "Loss / torch.min(critic_1_values, critic_2_values)",
                    (torch.min(critic_1_values, critic_2_values)).mean().item(),
                )

                # optimization step (policy)
                self.policy_optimizer.zero_grad()
                self.scaler.scale(policy_loss).backward()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)

            self.scaler.step(self.policy_optimizer)

            self.track_data("Loss / Target Entropy", self._target_entropy)
            self.track_data("Loss / Log Prob", log_prob.mean().item())
            self.track_data(
                "Loss / Log Entropy Coeff", self.log_entropy_coefficient.item()
            )

            # entropy learning
            if self._learn_entropy:
                with torch.autocast(
                    device_type=self._device_type, enabled=self._mixed_precision
                ):
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
                    "Target / Target Q (max)", torch.max(target_q_values).item()
                )
                self.track_data(
                    "Target / Target Q (min)", torch.min(target_q_values).item()
                )
                self.track_data(
                    "Target / Target Q (mean)", torch.mean(target_q_values).item()
                )
                self.track_data(
                    "Target / Next log prob (max)", torch.max(next_log_prob).item()
                )
                self.track_data(
                    "Target / Next log prob (min)", torch.min(next_log_prob).item()
                )
                self.track_data(
                    "Target / Next log prob (mean)", torch.mean(next_log_prob).item()
                )

                self.track_data("Loss / BC loss", bc_loss.item())

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

                if self._learn_entropy:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())
                    self.track_data(
                        "Coefficient / Entropy coefficient",
                        self._entropy_coefficient.item(),
                    )
                    if hasattr(self.policy, "get_log_std"):
                        policy_log_std = self.policy.get_log_std()
                        self.track_data(
                            "Policy / Log std (max)", torch.max(policy_log_std).item()
                        )
                        self.track_data(
                            "Policy / Log std (min)", torch.min(policy_log_std).item()
                        )
                        self.track_data(
                            "Policy / Log std (mean)", torch.mean(policy_log_std).item()
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
