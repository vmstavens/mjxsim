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

# from skrl import config, logger
# from algorithms.IBRLbase import Agent
from skrl.agents.torch.base import AgentCfg, ExperimentCfg
from skrl.memories.torch import Memory
from skrl.models.torch import Model

# from skrl.agents.torch import Agent
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
class DRLR2_SAC_CFG(AgentCfg):
    """Configuration for the DRLR SAC O/O2 agent."""

    gradient_steps: int = 1
    batch_size: int = 64
    decision_block: bool = False
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
            write_interval="auto",
            checkpoint_interval="auto",
        )
    )
    soft_update_beta: float = 0.2
    actor: str = "both"
    num_envs: int = 1
    action_trans_high: list[float] = dataclasses.field(default_factory=list)
    action_trans_low: list[float] = dataclasses.field(default_factory=list)
    action_rot_high: list[float] = dataclasses.field(default_factory=list)
    action_rot_low: list[float] = dataclasses.field(default_factory=list)
    a_min_lim: list[float] = dataclasses.field(default_factory=list)
    a_max_lim: list[float] = dataclasses.field(default_factory=list)

    def expand(self) -> None:
        super().expand()


DRLR2_SAC_DEFAULT_CONFIG = DRLR2_SAC_CFG()


class DRLR2(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        models_il: Dict[str, Model],
        memory: Optional[Memory],
        expert_memory: Optional[Memory],
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[DRLR2_SAC_CFG] = None,
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
        :type cfg: DRLR2_SAC_CFG

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = DRLR2_SAC_CFG() if cfg is None else copy.deepcopy(cfg)
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
        self.expert_memory = expert_memory
        self.memory = memory

        self._tensors_names = [
            "states",
            "actions",
            "rewards",
            "next_states",
            "terminated",
        ]

        # Ensure that memory and expert_memory have the correct tensors
        assert set(self.memory.get_tensor_names()).issubset(self._tensors_names), (
            f"Memory error: memory should have the tensor names {self._tensors_names}, but got {self.memory.get_tensor_names()}"
        )
        assert set(self.expert_memory.get_tensor_names()).issubset(
            self._tensors_names
        ), (
            f"Memory error: expert_memory should have the tensor names {self._tensors_names}, but got {self.expert_memory.get_tensor_names()}"
        )

        # IL model
        self.IL_policy: Agent = self.models_il["policy"]

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

        self.expert_mean_states = 0
        self.expert_cov_states = 0

        # OBS I will try :----------------------------------------------------------:
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
        self._decision_block: bool = _cfg.decision_block
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

        self._action_trans_high: list[float] = _cfg.action_trans_high
        self._action_trans_low: list[float] = _cfg.action_trans_low
        self._action_rot_high: list[float] = _cfg.action_rot_high
        self._action_rot_low: list[float] = _cfg.action_rot_low

        self._a_min_lim: list[float] = _cfg.a_min_lim
        self._a_max_lim: list[float] = _cfg.a_max_lim

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

        expert_states = self.expert_memory.sample(
            names=["states"], batch_size=self._batch_size
        )[0][0]
        expert_rewards = self.expert_memory.sample(
            names=["rewards"], batch_size=self._batch_size
        )[0][0]

        # compute the expert mean and covariance of states based on a sampled batch
        self.expert_mean_states = torch.mean(expert_states, dim=0)
        self.expert_mean_rewards = torch.mean(expert_rewards, dim=0)
        cov = torch.cov(expert_states.T)
        self.expert_cov_states = torch.inverse(
            cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device)
        )

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
            rl_actions, next_log_prob, _ = self.policy.act(
                {"states": self._state_preprocessor(rl_obs)}, role="policy"
            )
        else:
            # sample stochastic actions
            with torch.autocast(
                device_type=self._device_type, enabled=self._mixed_precision
            ):
                rl_actions, _, outputs = self.policy.act(
                    {"states": self._state_preprocessor(rl_obs)},
                    role="policy",
                )

        # here rl_actions are [-1, 1]

        # Get IL actions
        # a_{il} ← µ ( s_{t} )
        il_actions, _, _ = self.IL_policy.act(
            {"states": exp_obs},
            role="policy",
            unnormalize_act=False,
        )
        # Here il_actions are [-1, 1]

        # act out the 0'th index
        il_actions = il_actions[:, 0, :]

        if len(exp_obs.shape) == 3:
            exp_obs = exp_obs[:, 0, :]

        target_q_il = self._compute_min_q_values(
            self._state_preprocessor(exp_obs), il_actions
        )
        target_q_rl = self._compute_min_q_values(
            self._state_preprocessor(rl_obs), rl_actions
        )

        if torch.mean(target_q_il) > torch.mean(target_q_rl):
            # IL wins: reuse the single-step IL action to keep shape (num_envs, a_dim)
            il_actions, _, _ = self.IL_policy.act(
                {"states": il_obs},
                role="policy",
                unnormalize_act=True,
            )
            actions = il_actions[:, 0, :]
            self.track_data("Which / Actor", 1)
        else:
            self.track_data("Which / Actor", 0)
            actions = self._unnormalize_action(rl_actions)

        if not target:
            self.track_data(
                "Q-network / select_rl_Q (mean)", torch.mean(target_q_rl).item()
            )
            self.track_data(
                "Q-network / select_il_Q (mean)", torch.mean(target_q_il).item()
            )
            self.track_data("Q-network / debug (mean)", self.expert_mean_rewards.item())
            il_selected = (torch.mean(target_q_il) > torch.mean(target_q_rl)).float()
            self.track_data("Online / IL selection probability", il_selected.item())

            return actions, _, outputs
        else:
            il_selected = (torch.mean(target_q_il) > torch.mean(target_q_rl)).float()
            # -------
            self.track_data(
                "Bootstrap / select_rl_Q (mean)", torch.mean(target_q_rl).item()
            )
            self.track_data(
                "Bootstrap / select_il_Q (mean)", torch.mean(target_q_il).item()
            )
            self.track_data("Bootstrap / IL selection probability", il_selected.item())
            return actions, next_log_prob, _

    def _unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if action is None:
            return action

        if self.clip_actions_min is None or self.clip_actions_max is None:
            raise ValueError(
                "Action normalization requires a gymnasium.spaces.Box action space"
            )

        low = self.clip_actions_min.to(device=action.device, dtype=action.dtype)
        high = self.clip_actions_max.to(device=action.device, dtype=action.dtype)
        scale = torch.where(high > low, high - low, torch.ones_like(high))

        action = action.clamp(-1.0, 1.0)
        return 0.5 * (action + 1.0) * scale + low

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if action is None:
            return action

        if self.clip_actions_min is None or self.clip_actions_max is None:
            raise ValueError(
                "Action normalization requires a gymnasium.spaces.Box action space"
            )

        low = self.clip_actions_min.to(device=action.device, dtype=action.dtype)
        high = self.clip_actions_max.to(device=action.device, dtype=action.dtype)
        scale = torch.where(high > low, high - low, torch.ones_like(high))

        normalized = 2.0 * (action - low) / scale - 1.0
        return normalized.clamp(-1.0, 1.0)

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
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
        if timestep < self._random_timesteps:
            return self.policy.random_act(
                {"states": self._state_preprocessor(states)}, role="policy"
            )

        if timestep < self._warmup_timesteps:
            il_states = torch.stack([self._prev_states, self._states], axis=1)
            il_actions, _, _ = self.IL_policy.act(
                {"states": il_states},
                role="policy",
                unnormalize_act=True,
            )
            il_actions = il_actions[:, 0, :]

            return il_actions, None, None
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

        # compute states BC loss to track state-OOD behavior
        diff = states - self.expert_mean_states
        left = torch.matmul(diff, self.expert_cov_states)
        dist_sq = (left * diff).sum(dim=1)
        M_dist = torch.sqrt(dist_sq)

        self.track_data("Loss / states BC loss", torch.mean(M_dist).item())

        il_states = torch.stack([self._prev_states, self._states], axis=1)
        il_expert_states = torch.stack([expert_states, expert_next_states], axis=1)

        # actions here are un-normalized [min, max]
        actions, _, output = self._select_act(
            rl_obs=states,
            il_obs=il_states,
            exp_obs=il_expert_states,
            soft=True,
            target=False,
            timestep=timestep,
        )

        return actions, None, output

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
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
        super().record_transition(
            states,
            actions,
            rewards,
            next_states,
            terminated,
            truncated,
            infos,
            timestep,
            timesteps,
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
        self, states: torch.Tensor, timestep: int, timesteps: int
    ) -> None:
        """Callback called before the interaction with the environment.

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # Shift the last observation so the DP sees [previous_state, current_state].
        if self._states is None or self._states.shape[0] != states.shape[0]:
            self._prev_states = states
        else:
            self._prev_states = self._states
        self._states = states

    def post_interaction(
        self, next_states: torch.Tensor, timestep: int, timesteps: int
    ) -> None:
        """Callback called after the interaction with the environment.

        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.BoolTensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        # save the next state for diffusion policy observation horizon
        self._prev_states = next_states

        if timestep >= self._learning_starts:
            self.enable_training_mode(True, apply_to_models=True)
            self._update(timestep, timesteps)
            self.enable_training_mode(False, apply_to_models=True)

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

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
            target_q_val, _, _ = self.target_critics[idx].act(
                {"states": states, "taken_actions": actions},
                role=f"target_critic_{idx + 1}",
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
                sampled_states_rl = self._state_preprocessor(sampled_states, train=True)
                sampled_next_states_rl = self._state_preprocessor(
                    sampled_next_states, train=True
                )

                with torch.no_grad():
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

                    target_q1_values, _, _ = self.target_critic_1.act(
                        {
                            "states": sampled_next_states_rl,
                            "taken_actions": next_actions,
                        },
                        role="target_critic_1",
                    )
                    target_q2_values, _, _ = self.target_critic_2.act(
                        {
                            "states": sampled_next_states_rl,
                            "taken_actions": next_actions,
                        },
                        role="target_critic_2",
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

                    self.track_data(
                        "Q-network / reward_to_Q",
                        sampled_rewards.mean().item() / target_values.mean().item(),
                    )

                # make action [-1, 1]
                sampled_actions = self._normalize_action(sampled_actions)

                # compute critic loss
                critic_1_values, _, _ = self.critic_1.act(
                    {"states": sampled_states_rl, "taken_actions": sampled_actions},
                    role="critic_1",
                )
                critic_2_values, _, _ = self.critic_2.act(
                    {"states": sampled_states_rl, "taken_actions": sampled_actions},
                    role="critic_2",
                )

                # OBS sum, not average
                critic_loss = F.mse_loss(critic_1_values, target_values) + F.mse_loss(
                    critic_2_values, target_values
                )

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
                actions, log_prob, _ = self.policy.act(
                    {"states": sampled_states_rl}, role="policy"
                )
                critic_1_values, _, _ = self.critic_1.act(
                    {"states": sampled_states_rl, "taken_actions": actions},
                    role="critic_1",
                )
                critic_2_values, _, _ = self.critic_2.act(
                    {"states": sampled_states_rl, "taken_actions": actions},
                    role="critic_2",
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

            if self._learn_entropy:
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

                self.track_data("Loss / actions BC loss", bc_loss.item())

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
