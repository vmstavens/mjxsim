"""Imitation-Bootstrapped Reinforcement Learning with a TD3 backbone."""

from __future__ import annotations

import copy
import dataclasses
import itertools
from collections.abc import Mapping
from typing import Any

import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl import config
from skrl.agents.torch.td3 import TD3, TD3_CFG
from skrl.memories.torch import Memory
from skrl.models.torch import Model


@dataclasses.dataclass(kw_only=True)
class IBRL_TD3_CFG(TD3_CFG):
    """Configuration for TD3 with imitation bootstrapping."""

    warmup_timesteps: int = 10_000
    """Use only the imitation policy before this timestep."""

    actor: str = "both"
    """Behavior policy: ``"rl"``, ``"il"``, or Q-selected ``"both"``."""

    expert_batch_ratio: float = 0.5
    """Fraction of each critic batch sampled from expert memory."""

    action_ema_alpha: float = 1.0
    """EMA coefficient applied to selected RL behavior actions."""

    def validate(self) -> bool:
        if self.actor not in {"rl", "il", "both"}:
            raise ValueError("actor must be 'rl', 'il', or 'both'")
        if not 0 <= self.expert_batch_ratio <= 1:
            raise ValueError("expert_batch_ratio must be in [0, 1]")
        if not 0 < self.action_ema_alpha <= 1:
            raise ValueError("action_ema_alpha must be in (0, 1]")
        if self.warmup_timesteps < 0:
            raise ValueError("warmup_timesteps cannot be negative")
        return super().validate()

    def expand(self) -> None:
        self.validate()
        super().expand()


IBRL_TD3_DEFAULT_CONFIG = IBRL_TD3_CFG()


class IBRL(TD3):
    """TD3 agent bootstrapped by a frozen imitation policy and demonstrations.

    The imitation policy must accept a two-step observation history with shape
    ``[batch, 2, observation_dim]`` and return either ``[batch, action_dim]`` or
    an action plan whose first action is ``[:, 0, :]``. Its actions must use the
    same coordinates and bounds as ``action_space``.
    """

    def __init__(
        self,
        *,
        models: Mapping[str, Model],
        models_il: Mapping[str, Model],
        memory: Memory,
        expert_memory: Memory | None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: IBRL_TD3_CFG | dict[str, Any] | None = None,
    ) -> None:
        agent_cfg = IBRL_TD3_CFG() if cfg is None else copy.deepcopy(cfg)
        if isinstance(agent_cfg, dict):
            agent_cfg = IBRL_TD3_CFG(**agent_cfg)
        agent_cfg.expand()

        self.models_il = dict(models_il)
        self.expert_memory = expert_memory
        self.IL_policy = self.models_il.get("policy")
        if self.IL_policy is None:
            raise KeyError("models_il must contain a 'policy'")

        super().__init__(
            models=dict(models),
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=agent_cfg,
        )

        self.IL_policy.to(self.IL_policy.device)
        self.IL_policy.freeze_parameters(True)
        self.checkpoint_modules["il_policy"] = self.IL_policy
        self._previous_observations: torch.Tensor | None = None
        self._state_window_timestep: int | None = None
        self._ema_actions: torch.Tensor | None = None
        self._ema_action_valid: torch.Tensor | None = None

    @property
    def cfg(self) -> IBRL_TD3_CFG:
        return self._cfg

    @cfg.setter
    def cfg(self, value: IBRL_TD3_CFG) -> None:
        self._cfg = value

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.IL_policy.enable_training_mode(False)
        if self.expert_memory is not None:
            for name, size, dtype in (
                ("states", self.state_space, torch.float32),
                ("actions", self.action_space, torch.float32),
                ("rewards", 1, torch.float32),
                ("next_states", self.state_space, torch.float32),
                ("terminated", 1, torch.bool),
            ):
                self.expert_memory.create_tensor(name=name, size=size, dtype=dtype)

    def enable_models_training_mode(self, enabled: bool = True) -> None:
        super().enable_models_training_mode(enabled)
        self.IL_policy.enable_training_mode(False)

    @staticmethod
    def _unpack_act(result: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        if not isinstance(result, tuple):
            return result, {}
        if len(result) == 2:
            values, outputs = result
            return values, outputs or {}
        if len(result) == 3:
            values, log_prob, outputs = result
            result_outputs = dict(outputs or {})
            if log_prob is not None:
                result_outputs.setdefault("log_prob", log_prob)
            return values, result_outputs
        raise ValueError(f"unsupported act result with {len(result)} values")

    @staticmethod
    def _first_action(actions: torch.Tensor) -> torch.Tensor:
        return actions[:, 0, :] if actions.ndim == 3 else actions

    def _il_action(self, history: torch.Tensor) -> torch.Tensor:
        actions, _ = self._unpack_act(
            self.IL_policy.act(
                {"observations": history, "states": history},
                role="policy",
            )
        )
        actions = self._first_action(actions)
        if self._min_actions is not None and self._max_actions is not None:
            actions = torch.clamp(
                actions,
                min=self._min_actions,
                max=self._max_actions,
            )
        return actions

    def _minimum_q(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        *,
        target: bool,
    ) -> torch.Tensor:
        critic_1 = self.target_critic_1 if target else self.critic_1
        critic_2 = self.target_critic_2 if target else self.critic_2
        inputs = {
            "observations": states,
            "states": states,
            "taken_actions": actions,
        }
        q1, _ = self._unpack_act(
            critic_1.act(
                inputs,
                role="target_critic_1" if target else "critic_1",
            )
        )
        q2, _ = self._unpack_act(
            critic_2.act(
                inputs,
                role="target_critic_2" if target else "critic_2",
            )
        )
        return torch.minimum(q1, q2)

    def _selection_mask(
        self,
        q_rl: torch.Tensor,
        q_il: torch.Tensor,
    ) -> torch.Tensor:
        """Return a per-sample mask selecting imitation actions."""

        return q_il > q_rl

    def _select_candidates(
        self,
        states: torch.Tensor,
        histories: torch.Tensor,
        rl_actions: torch.Tensor,
        *,
        target: bool,
        behavior: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        il_actions = self._il_action(histories)
        q_rl = self._minimum_q(states, rl_actions, target=target)
        q_il = self._minimum_q(states, il_actions, target=target)
        use_il = self._selection_mask(q_rl, q_il)
        actions = torch.where(use_il, il_actions, rl_actions)
        actions = self._transform_selected_actions(
            actions,
            use_il,
            behavior=behavior,
        )
        return actions, use_il

    def _transform_selected_actions(
        self,
        actions: torch.Tensor,
        use_il: torch.Tensor,
        *,
        behavior: bool,
    ) -> torch.Tensor:
        """Transform selected actions before execution or target evaluation."""

        del use_il, behavior
        return actions

    def _smooth_rl_actions(
        self,
        actions: torch.Tensor,
        rl_mask: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.cfg.action_ema_alpha
        if alpha >= 1:
            return actions
        if self._ema_actions is None or self._ema_actions.shape != actions.shape:
            self._ema_actions = torch.zeros_like(actions)
            self._ema_action_valid = torch.zeros(
                (actions.shape[0], 1),
                dtype=torch.bool,
                device=actions.device,
            )
        valid = self._ema_action_valid
        smoothed = torch.where(
            valid,
            alpha * actions + (1 - alpha) * self._ema_actions,
            actions,
        )
        update = rl_mask.reshape(-1, 1)
        self._ema_actions = torch.where(update, smoothed.detach(), self._ema_actions)
        self._ema_action_valid = torch.logical_or(valid, update)
        return torch.where(update, smoothed, actions)

    def act(
        self,
        observations: torch.Tensor,
        states: torch.Tensor | None,
        *,
        timestep: int,
        timesteps: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del states
        if timestep < self.cfg.random_timesteps:
            inputs = {"observations": observations, "states": observations}
            return self.policy.random_act(inputs, role="policy")

        current = self._observation_preprocessor(observations)
        previous = (
            current
            if self._previous_observations is None
            or self._previous_observations.shape != current.shape
            else self._previous_observations
        )
        histories = torch.stack((previous, current), dim=1)
        self._previous_observations = current.detach()
        self._state_window_timestep = timestep

        rl_actions, outputs = self._unpack_act(
            self.policy.act(
                {"observations": current, "states": current},
                role="policy",
            )
        )
        il_actions = self._il_action(histories)
        if self.cfg.actor == "rl":
            actions = rl_actions
            use_il = torch.zeros(
                (actions.shape[0], 1), dtype=torch.bool, device=actions.device
            )
        elif self.cfg.actor == "il" or timestep < self.cfg.warmup_timesteps:
            actions = il_actions
            use_il = torch.ones(
                (actions.shape[0], 1), dtype=torch.bool, device=actions.device
            )
        else:
            actions, use_il = self._select_candidates(
                current,
                histories,
                rl_actions,
                target=True,
                behavior=True,
            )

        rl_mask = torch.logical_not(use_il)
        actions = self._smooth_rl_actions(actions, rl_mask)
        if self._exploration_noise is not None and torch.any(rl_mask):
            noise = self._exploration_noise.sample(actions.shape)
            if self.cfg.exploration_scheduler is not None:
                noise.mul_(self.cfg.exploration_scheduler(timestep, timesteps))
            actions = actions + noise * rl_mask.to(actions.dtype)
        if self._min_actions is not None and self._max_actions is not None:
            actions = torch.clamp(
                actions,
                min=self._min_actions,
                max=self._max_actions,
            )
        self.track_data(
            "Which / IL selection ratio",
            use_il.to(torch.float32).mean().item(),
        )
        return actions, outputs

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
        effective_states = observations if states is None else states
        effective_next_states = (
            next_observations if next_states is None else next_states
        )
        super().record_transition(
            observations=observations,
            states=effective_states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=effective_next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )
        done = torch.logical_or(terminated, truncated).reshape(-1, 1)
        if self._previous_observations is not None:
            self._previous_observations = torch.where(
                done,
                next_observations.detach(),
                self._previous_observations,
            )
        if self._ema_action_valid is not None:
            self._ema_action_valid = torch.where(
                done,
                torch.zeros_like(self._ema_action_valid),
                self._ema_action_valid,
            )
            self._ema_actions = torch.where(
                done,
                torch.zeros_like(self._ema_actions),
                self._ema_actions,
            )

    def _sample_training_batch(self):
        expert_size = 0
        if self.expert_memory is not None and self.cfg.expert_batch_ratio > 0:
            expert_size = round(self.cfg.batch_size * self.cfg.expert_batch_ratio)
        online_size = self.cfg.batch_size - expert_size
        if online_size == 0 and self.expert_memory is None:
            online_size = self.cfg.batch_size

        names = [
            "states",
            "actions",
            "rewards",
            "next_states",
            "terminated",
        ]
        batches = []
        if online_size:
            batches.append(self.memory.sample(names=names, batch_size=online_size)[0])
        if expert_size:
            batches.append(
                self.expert_memory.sample(names=names, batch_size=expert_size)[0]
            )
        if len(batches) == 1:
            return batches[0]
        return tuple(torch.cat(parts, dim=0) for parts in zip(*batches, strict=True))

    def update(self, *, timestep: int, timesteps: int) -> None:
        del timesteps
        for _ in range(self.cfg.gradient_steps):
            (
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminated,
            ) = self._sample_training_batch()
            states = self._state_preprocessor(sampled_states, train=True)
            next_states = self._state_preprocessor(
                sampled_next_states,
                train=True,
            )
            inputs = {"observations": states, "states": states}
            next_inputs = {
                "observations": next_states,
                "states": next_states,
            }

            with torch.autocast(
                device_type=self._device_type,
                enabled=self.cfg.mixed_precision,
            ):
                with torch.no_grad():
                    next_rl_actions, _ = self._unpack_act(
                        self.target_policy.act(
                            next_inputs,
                            role="target_policy",
                        )
                    )
                    if self._smooth_regularization_noise is not None:
                        noise = torch.clamp(
                            self._smooth_regularization_noise.sample(
                                next_rl_actions.shape
                            ),
                            min=-self.cfg.smooth_regularization_clip,
                            max=self.cfg.smooth_regularization_clip,
                        )
                        next_rl_actions = next_rl_actions + noise
                        if self._min_actions is not None:
                            next_rl_actions = torch.clamp(
                                next_rl_actions,
                                min=self._min_actions,
                                max=self._max_actions,
                            )
                    histories = torch.stack(
                        (states, next_states),
                        dim=1,
                    )
                    next_actions, _ = self._select_candidates(
                        next_states,
                        histories,
                        next_rl_actions,
                        target=True,
                    )
                    target_q = self._minimum_q(
                        next_states,
                        next_actions,
                        target=True,
                    )
                    target_values = (
                        sampled_rewards
                        + self.cfg.discount_factor
                        * sampled_terminated.logical_not()
                        * target_q
                    )

                critic_inputs = {
                    **inputs,
                    "taken_actions": sampled_actions,
                }
                critic_1_values, _ = self._unpack_act(
                    self.critic_1.act(critic_inputs, role="critic_1")
                )
                critic_2_values, _ = self._unpack_act(
                    self.critic_2.act(critic_inputs, role="critic_2")
                )
                critic_loss = F.mse_loss(critic_1_values, target_values) + F.mse_loss(
                    critic_2_values, target_values
                )

            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            if config.torch.is_distributed:
                self.critic_1.reduce_parameters()
                self.critic_2.reduce_parameters()
            if self.cfg.grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(
                        self.critic_1.parameters(),
                        self.critic_2.parameters(),
                    ),
                    self.cfg.grad_norm_clip,
                )
            self.scaler.step(self.critic_optimizer)

            self._update_counter += 1
            if not self._update_counter % self.cfg.policy_delay:
                with torch.autocast(
                    device_type=self._device_type,
                    enabled=self.cfg.mixed_precision,
                ):
                    policy_actions, _ = self._unpack_act(
                        self.policy.act(inputs, role="policy")
                    )
                    policy_q, _ = self._unpack_act(
                        self.critic_1.act(
                            {**inputs, "taken_actions": policy_actions},
                            role="critic_1",
                        )
                    )
                    policy_loss = -policy_q.mean()
                self.policy_optimizer.zero_grad()
                self.scaler.scale(policy_loss).backward()
                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.policy_optimizer)
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.cfg.grad_norm_clip,
                    )
                self.scaler.step(self.policy_optimizer)
                self.target_policy.update_parameters(
                    self.policy,
                    polyak=self.cfg.polyak,
                )
                self.target_critic_1.update_parameters(
                    self.critic_1,
                    polyak=self.cfg.polyak,
                )
                self.target_critic_2.update_parameters(
                    self.critic_2,
                    polyak=self.cfg.polyak,
                )
                self.track_data("Loss / Policy loss", policy_loss.item())

            self.scaler.update()
            if self.policy_scheduler is not None:
                self.policy_scheduler.step()
            if self.critic_scheduler is not None:
                self.critic_scheduler.step()
            self.track_data("Loss / Critic loss", critic_loss.item())


__all__ = ["IBRL", "IBRL_TD3_CFG", "IBRL_TD3_DEFAULT_CONFIG"]
