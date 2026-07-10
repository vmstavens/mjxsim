"""SKRL 2.x JAX/Flax implementation of DRLR2 SAC.

The implementation intentionally mirrors :mod:`mjxsim.agents.torch.drlr2_sac`:
the imitation policy is frozen, actor selection is made from batch-mean target
Q values, and the behaviour-cloning loss is diagnostic only.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Mapping
from typing import Any

import gymnasium
import jax
import jax.numpy as jnp

from skrl import config
from skrl.agents.jax.sac import SAC, SAC_CFG
from skrl.memories.jax import Memory
from skrl.models.jax import Model

from .diffusion_policy_state import DiffusionPolicy


@dataclasses.dataclass(kw_only=True)
class ExplorationCfg:
    """Exploration scheduling retained for Torch configuration parity."""

    noise: Any = None
    initial_scale: float = 1.0
    final_scale: float = 1e-3
    timesteps: int | None = None


@dataclasses.dataclass(kw_only=True)
class DRLR2_SAC_CFG(SAC_CFG):
    """DRLR2 configuration on top of SKRL's JAX SAC configuration."""

    decision_block: bool = False
    warmup_timesteps: int = 10_000
    il_ctrl_scale: float = 1.0
    rl_ctrl_scale: float = 1.0
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    entropy_learning_rate: float = 3e-4
    exploration: ExplorationCfg = dataclasses.field(default_factory=ExplorationCfg)
    soft_update_beta: float = 0.2
    actor: str = "both"
    num_envs: int = 1
    action_trans_high: list[float] = dataclasses.field(default_factory=list)
    action_trans_low: list[float] = dataclasses.field(default_factory=list)
    action_rot_high: list[float] = dataclasses.field(default_factory=list)
    action_rot_low: list[float] = dataclasses.field(default_factory=list)
    a_min_lim: list[float] = dataclasses.field(default_factory=list)
    a_max_lim: list[float] = dataclasses.field(default_factory=list)
    action_ema_alpha: float = 1.0

    def expand(self) -> None:
        if self.actor not in {"rl", "il", "both"}:
            raise ValueError("actor must be one of 'rl', 'il', or 'both'")
        if not 0 < self.action_ema_alpha <= 1:
            raise ValueError("action_ema_alpha must be in (0, 1]")
        self.learning_rate = (
            self.actor_learning_rate,
            self.critic_learning_rate,
            self.entropy_learning_rate,
        )
        super().expand()

    def to_dict(self) -> dict[str, Any]:
        self.expand()
        return dataclasses.asdict(self)

    def copy(self) -> dict[str, Any]:
        return self.to_dict()


DRLR2_SAC_DEFAULT_CONFIG = DRLR2_SAC_CFG()


class DiffusionPolicyAdapter:
    """Expose the local Flax diffusion policy through SKRL's ``act`` surface."""

    def __init__(self, policy: DiffusionPolicy):
        self.policy = policy
        self.rng = policy.rng

    def act(
        self,
        inputs: Mapping[str, jax.Array],
        *,
        role: str = "policy",
        unnormalize_act: bool = False,
        rng: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, Any]]:
        del role
        observations = inputs.get("observations", inputs.get("states"))
        if rng is None:
            self.rng, rng = jax.random.split(self.rng)
        return (
            self.policy.act(
                observations,
                rng=rng,
                unnormalize_act=unnormalize_act,
            ),
            {},
        )

    def enable_training_mode(self, enabled: bool = True) -> None:
        del enabled  # frozen during RL training

    def state_dict(self) -> dict[str, Any]:
        return self.policy.state_dict()


@functools.partial(jax.jit, static_argnames=("critic_act", "role"))
def _critic_gradient(critic_act, state_dict, inputs, targets, role):
    def loss(params):
        values, _ = critic_act(inputs, role=role, params=params)
        return jnp.square(values - targets).mean(), values

    return jax.value_and_grad(loss, has_aux=True)(state_dict.params)


@functools.partial(
    jax.jit, static_argnames=("policy_act", "critic_1_act", "critic_2_act")
)
def _policy_gradient(
    policy_act,
    critic_1_act,
    critic_2_act,
    policy_state_dict,
    critic_1_state_dict,
    critic_2_state_dict,
    entropy_coefficient,
    inputs,
    sampled_actions,
):
    def loss(policy_params):
        actions, outputs = policy_act(inputs, role="policy", params=policy_params)
        log_prob = outputs["log_prob"]
        critic_inputs = {**inputs, "taken_actions": actions}
        q1, _ = critic_1_act(
            critic_inputs, role="critic_1", params=critic_1_state_dict.params
        )
        q2, _ = critic_2_act(
            critic_inputs, role="critic_2", params=critic_2_state_dict.params
        )
        policy_loss = (entropy_coefficient * log_prob - jnp.minimum(q1, q2)).mean()
        bc_loss = jnp.square(actions - sampled_actions).mean()
        return policy_loss, (log_prob, bc_loss)

    return jax.value_and_grad(loss, has_aux=True)(policy_state_dict.params)


class DRLR2(SAC):
    """DRLR2 SAC agent compatible with SKRL 2.x JAX trainers and memories."""

    def __init__(
        self,
        *,
        models: Mapping[str, Model],
        models_il: Mapping[str, Any],
        memory: Memory | None,
        expert_memory: Memory | None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | jax.Device | None = None,
        cfg: DRLR2_SAC_CFG | dict[str, Any] | None = None,
    ) -> None:
        cfg = DRLR2_SAC_CFG() if cfg is None else cfg
        if isinstance(cfg, dict):
            cfg = DRLR2_SAC_CFG(**cfg)
        cfg.expand()
        self.models_il = dict(models_il)
        self.expert_memory = expert_memory
        super().__init__(
            models=dict(models),
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )
        self.cfg: DRLR2_SAC_CFG
        self.IL_policy = self.models_il["policy"]
        if isinstance(self.IL_policy, DiffusionPolicy):
            self.IL_policy = DiffusionPolicyAdapter(self.IL_policy)
        self._states = None
        self._prev_states = None
        self._state_window_timestep = None
        self._ema_actions = None
        self._ema_action_valid = None
        self.expert_mean_states = None
        self.expert_mean_rewards = None
        self.expert_cov_states = None
        if isinstance(action_space, gymnasium.spaces.Box):
            self.clip_actions_min = jnp.asarray(action_space.low, dtype=jnp.float32)
            self.clip_actions_max = jnp.asarray(action_space.high, dtype=jnp.float32)
        else:
            self.clip_actions_min = self.clip_actions_max = None

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        if self.expert_memory is None:
            raise ValueError("DRLR2 training requires expert_memory")
        for name, size, dtype in (
            ("observations", self.observation_space, jnp.float32),
            ("states", self.state_space, jnp.float32),
            ("actions", self.action_space, jnp.float32),
            ("rewards", 1, jnp.float32),
            ("next_observations", self.observation_space, jnp.float32),
            ("next_states", self.state_space, jnp.float32),
            ("terminated", 1, jnp.int8),
        ):
            self.expert_memory.create_tensor(name=name, size=size, dtype=dtype)
        if len(self.expert_memory):
            self._refresh_expert_statistics()

    def _refresh_expert_statistics(self) -> None:
        """Refresh diagnostics used by the Torch implementation."""
        if self.expert_memory is None or not len(self.expert_memory):
            raise ValueError("expert_memory must contain demonstrations")
        expert_states, expert_rewards = self.expert_memory.sample(
            names=["states", "rewards"], batch_size=self.cfg.batch_size
        )[0]
        self.expert_mean_states = jnp.mean(expert_states, axis=0)
        self.expert_mean_rewards = jnp.mean(expert_rewards, axis=0)
        centered = expert_states - self.expert_mean_states
        cov = centered.T @ centered / jnp.maximum(expert_states.shape[0] - 1, 1)
        self.expert_cov_states = jnp.linalg.inv(cov + 1e-6 * jnp.eye(cov.shape[0]))

    def _normalize_action(self, action: jax.Array) -> jax.Array:
        if self.clip_actions_min is None or self.clip_actions_max is None:
            raise ValueError("Action normalization requires a Box action space")
        scale = jnp.where(
            self.clip_actions_max > self.clip_actions_min,
            self.clip_actions_max - self.clip_actions_min,
            1,
        )
        return jnp.clip(2 * (action - self.clip_actions_min) / scale - 1, -1, 1)

    def _unnormalize_action(self, action: jax.Array) -> jax.Array:
        if self.clip_actions_min is None or self.clip_actions_max is None:
            raise ValueError("Action normalization requires a Box action space")
        scale = jnp.where(
            self.clip_actions_max > self.clip_actions_min,
            self.clip_actions_max - self.clip_actions_min,
            1,
        )
        return 0.5 * (jnp.clip(action, -1, 1) + 1) * scale + self.clip_actions_min

    def _set_current_states(self, states: jax.Array, timestep: int) -> None:
        self._prev_states = states if self._states is None else self._states
        self._states = states
        self._state_window_timestep = timestep

    def _smooth_action_ema(self, actions: jax.Array) -> jax.Array:
        if self.cfg.action_ema_alpha >= 1:
            return actions
        if self._ema_actions is None or self._ema_actions.shape != actions.shape:
            self._ema_actions = jnp.zeros_like(actions)
            self._ema_action_valid = jnp.zeros((actions.shape[0], 1), dtype=bool)
        smoothed = jnp.where(
            self._ema_action_valid,
            self.cfg.action_ema_alpha * actions
            + (1 - self.cfg.action_ema_alpha) * self._ema_actions,
            actions,
        )
        self._ema_actions = jax.lax.stop_gradient(smoothed)
        self._ema_action_valid = jnp.ones_like(self._ema_action_valid)
        return smoothed

    def _reset_action_ema(self, terminated, truncated) -> None:
        if self._ema_actions is None:
            return
        done = jnp.reshape(jnp.logical_or(terminated, truncated), (-1, 1))
        self._ema_actions = jnp.where(done, 0, self._ema_actions)
        self._ema_action_valid = jnp.where(done, False, self._ema_action_valid)

    def _compute_min_q_values(self, states, actions):
        inputs = {
            "observations": states,
            "states": states,
            "taken_actions": actions,
        }
        q1, _ = self.target_critic_1.act(inputs, role="target_critic_1")
        q2, _ = self.target_critic_2.act(inputs, role="target_critic_2")
        return jnp.minimum(q1, q2)

    def _select_act(
        self,
        *,
        rl_obs,
        il_obs,
        exp_obs,
        soft,
        target,
        timestep,
        smooth_rl_action=False,
    ):
        del soft, timestep  # retained for exact Torch call compatibility
        inputs = {
            "observations": self._observation_preprocessor(rl_obs),
            "states": self._state_preprocessor(rl_obs),
        }
        rl_actions, outputs = self.policy.act(inputs, role="policy")
        il_eval, _ = self.IL_policy.act(
            {"observations": exp_obs, "states": exp_obs},
            role="policy",
            unnormalize_act=False,
        )
        il_eval = il_eval[:, 0, :]
        exp_flat = exp_obs[:, 0, :] if exp_obs.ndim == 3 else exp_obs
        q_il = self._compute_min_q_values(self._state_preprocessor(exp_flat), il_eval)
        q_rl = self._compute_min_q_values(self._state_preprocessor(rl_obs), rl_actions)
        use_il = jnp.mean(q_il) > jnp.mean(q_rl)
        il_actions, _ = self.IL_policy.act(
            {"observations": il_obs, "states": il_obs},
            role="policy",
            unnormalize_act=True,
        )
        env_rl_actions = self._unnormalize_action(rl_actions)
        if smooth_rl_action:
            env_rl_actions = self._smooth_action_ema(env_rl_actions)
        actions = jnp.where(use_il, il_actions[:, 0, :], env_rl_actions)
        return actions, outputs.get("log_prob") if target else None, outputs

    def act(self, observations, states=None, *, timestep, timesteps):
        del states, timesteps
        if self._state_window_timestep != timestep:
            self._set_current_states(observations, timestep)
        if timestep < self.cfg.random_timesteps:
            return self.policy.random_act(
                {"observations": observations, "states": observations}, role="policy"
            )
        history = jnp.stack((self._prev_states, self._states), axis=1)
        if timestep < self.cfg.warmup_timesteps or self.cfg.actor == "il":
            actions, _ = self.IL_policy.act(
                {"observations": history, "states": history},
                role="policy",
                unnormalize_act=True,
            )
            return actions[:, 0, :], {}
        if self.expert_mean_states is None:
            self._refresh_expert_statistics()
        if self.cfg.actor == "rl":
            actions, outputs = self.policy.act(
                {"observations": observations, "states": observations}, role="policy"
            )
            return self._smooth_action_ema(self._unnormalize_action(actions)), outputs
        expert_states, expert_next_states = self.expert_memory.sample(
            names=["states", "next_states"], batch_size=observations.shape[0]
        )[0]
        expert_history = jnp.stack((expert_states, expert_next_states), axis=1)
        actions, _, outputs = self._select_act(
            rl_obs=observations,
            il_obs=history,
            exp_obs=expert_history,
            soft=True,
            target=False,
            timestep=timestep,
            smooth_rl_action=True,
        )
        return actions, outputs

    def record_transition(self, **kwargs) -> None:
        super().record_transition(**kwargs)
        self._reset_action_ema(kwargs["terminated"], kwargs["truncated"])

    def update(self, *, timestep: int, timesteps: int) -> None:
        del timesteps
        if self.expert_mean_states is None:
            self._refresh_expert_statistics()
        for _ in range(self.cfg.gradient_steps):
            batch = self.memory.sample(
                names=self._tensors_names, batch_size=self.cfg.batch_size
            )[0]
            (obs, states, actions, rewards, next_obs, next_states, terminated) = batch
            expert_states, expert_next_states = self.expert_memory.sample(
                names=["states", "next_states"], batch_size=self.cfg.batch_size
            )[0]
            inputs = {
                "observations": self._observation_preprocessor(obs, train=True),
                "states": self._state_preprocessor(states, train=True),
            }
            history = jnp.stack((obs, next_obs), axis=1)
            expert_history = jnp.stack((expert_states, expert_next_states), axis=1)
            next_actions, next_log_prob, _ = self._select_act(
                rl_obs=next_obs,
                il_obs=history,
                exp_obs=expert_history,
                soft=True,
                target=True,
                timestep=timestep,
            )
            next_actions = self._normalize_action(next_actions)
            next_inputs = {
                "observations": self._observation_preprocessor(next_obs, train=True),
                "states": self._state_preprocessor(next_states, train=True),
                "taken_actions": next_actions,
            }
            tq1, _ = self.target_critic_1.act(next_inputs, role="target_critic_1")
            tq2, _ = self.target_critic_2.act(next_inputs, role="target_critic_2")
            targets = jax.lax.stop_gradient(
                rewards
                + self.cfg.discount_factor
                * jnp.logical_not(terminated)
                * (jnp.minimum(tq1, tq2) - self._entropy_coefficient * next_log_prob)
            )
            critic_inputs = {**inputs, "taken_actions": self._normalize_action(actions)}
            (loss1, q1), grad1 = _critic_gradient(
                self.critic_1.act,
                self.critic_1.state_dict,
                critic_inputs,
                targets,
                "critic_1",
            )
            (loss2, q2), grad2 = _critic_gradient(
                self.critic_2.act,
                self.critic_2.state_dict,
                critic_inputs,
                targets,
                "critic_2",
            )
            if config.jax.is_distributed:
                grad1 = self.critic_1.reduce_parameters(grad1)
                grad2 = self.critic_2.reduce_parameters(grad2)
            self.critic_1_optimizer = self.critic_1_optimizer.step(
                grad=grad1, model=self.critic_1
            )
            self.critic_2_optimizer = self.critic_2_optimizer.step(
                grad=grad2, model=self.critic_2
            )
            (policy_loss, (log_prob, bc_loss)), policy_grad = _policy_gradient(
                self.policy.act,
                self.critic_1.act,
                self.critic_2.act,
                self.policy.state_dict,
                self.critic_1.state_dict,
                self.critic_2.state_dict,
                self._entropy_coefficient,
                inputs,
                self._normalize_action(actions),
            )
            if config.jax.is_distributed:
                policy_grad = self.policy.reduce_parameters(policy_grad)
            self.policy_optimizer = self.policy_optimizer.step(
                grad=policy_grad, model=self.policy
            )
            if self.cfg.learn_entropy:
                from skrl.agents.jax.sac.sac import _update_entropy

                entropy_grad, entropy_loss = _update_entropy(
                    self.log_entropy_coefficient.state_dict,
                    self._target_entropy,
                    log_prob,
                )
                self.entropy_optimizer = self.entropy_optimizer.step(
                    grad=entropy_grad, model=self.log_entropy_coefficient
                )
                self._entropy_coefficient = jnp.exp(self.log_entropy_coefficient.value)
            self.target_critic_1.update_parameters(self.critic_1, polyak=self.cfg.polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self.cfg.polyak)
            if self.write_interval > 0:
                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Loss / Critic loss", (loss1 + loss2).item())
                self.track_data("Loss / actions BC loss", bc_loss.item())
                self.track_data("Q-network / Q1 (mean)", q1.mean().item())
                self.track_data("Q-network / Q2 (mean)", q2.mean().item())
                if self.cfg.learn_entropy:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())


__all__ = [
    "DRLR2",
    "DRLR2_SAC_CFG",
    "DRLR2_SAC_DEFAULT_CONFIG",
    "DiffusionPolicyAdapter",
    "ExplorationCfg",
]
