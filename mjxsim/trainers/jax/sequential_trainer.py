"""Repository-owned JAX sequential trainer.

The compatibility path preserves the skrl agent/environment interaction contract.
The compiled path is deliberately opt-in and delegates chunks to a functional
kernel, allowing future PPO and DRLR2 kernels to use ``jax.lax.scan`` without
changing the public APIs of the existing agents.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import tqdm
from skrl.trainers.jax import (
    SequentialTrainer as SkrlSequentialTrainer,
    SequentialTrainerCfg,
)


@dataclasses.dataclass(kw_only=True)
class JaxSequentialTrainerCfg(SequentialTrainerCfg):
    """Configuration for :class:`JaxSequentialTrainer`.

    ``compatibility`` uses the standard skrl interaction hooks. ``compiled``
    requires an explicit functional kernel and never falls back silently.
    """

    mode: Literal["compatibility", "compiled"] = "compatibility"
    compiled_chunk_size: int = 32
    measurement_warmup_steps: int = 0
    throughput_log_interval: int = 0
    synchronize_on_measurement: bool = True

    def validate(self) -> bool:
        if self.mode not in {"compatibility", "compiled"}:
            raise ValueError("mode must be 'compatibility' or 'compiled'")
        if self.compiled_chunk_size < 1:
            raise ValueError("compiled_chunk_size must be at least 1")
        if self.measurement_warmup_steps < 0:
            raise ValueError("measurement_warmup_steps cannot be negative")
        if self.throughput_log_interval < 0:
            raise ValueError("throughput_log_interval cannot be negative")
        return super().validate()

    def expand(self) -> None:
        self.validate()
        super().expand()


@dataclasses.dataclass(frozen=True)
class ThroughputStats:
    """Wall-clock training throughput.

    A vector step advances every parallel environment once. ``transitions``
    therefore equals ``vector_steps * num_envs``.
    """

    mode: str
    num_envs: int
    vector_steps: int
    transitions: int
    elapsed_seconds: float
    warmup_vector_steps: int = 0
    warmup_seconds: float = 0.0

    @property
    def vector_steps_per_second(self) -> float:
        return (
            self.vector_steps / self.elapsed_seconds
            if self.elapsed_seconds > 0
            else 0.0
        )

    @property
    def transitions_per_second(self) -> float:
        return (
            self.transitions / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0
        )

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            **dataclasses.asdict(self),
            "vector_steps_per_second": self.vector_steps_per_second,
            "transitions_per_second": self.transitions_per_second,
        }


@runtime_checkable
class CompiledTrainingKernel(Protocol):
    """Functional chunk interface for fully compiled algorithm kernels."""

    def initialize(self, *, env: Any, agents: Any, cfg: JaxSequentialTrainerCfg) -> Any:
        """Create and return device-resident training state."""

    def train_chunk(
        self,
        state: Any,
        *,
        start_timestep: int,
        timesteps: int,
        total_timesteps: int,
    ) -> tuple[Any, Mapping[str, Any]]:
        """Run exactly ``timesteps`` vector steps and return state and metrics."""


def _coerce_cfg(
    cfg: JaxSequentialTrainerCfg | SequentialTrainerCfg | dict[str, Any] | None,
) -> JaxSequentialTrainerCfg:
    if cfg is None:
        return JaxSequentialTrainerCfg()
    if isinstance(cfg, JaxSequentialTrainerCfg):
        return dataclasses.replace(cfg)
    if isinstance(cfg, dict):
        return JaxSequentialTrainerCfg(**cfg)
    if isinstance(cfg, SequentialTrainerCfg):
        values = {
            field.name: getattr(cfg, field.name)
            for field in dataclasses.fields(SequentialTrainerCfg)
        }
        return JaxSequentialTrainerCfg(**values)
    raise TypeError(
        "cfg must be a JaxSequentialTrainerCfg, SequentialTrainerCfg, dict, "
        f"or None (got {type(cfg).__name__})"
    )


def _synchronize(value: Any | None = None) -> None:
    """Wait for queued JAX work without copying arrays to the host."""

    if value is not None:
        jax.block_until_ready(value)
    effects_barrier = getattr(jax, "effects_barrier", None)
    if effects_barrier is not None:
        effects_barrier()


class JaxSequentialTrainer(SkrlSequentialTrainer):
    """Additive JAX trainer with compatible and compiled execution modes."""

    def __init__(
        self,
        *,
        env: Any,
        agents: Any,
        scopes: list[int] | None = None,
        cfg: JaxSequentialTrainerCfg
        | SequentialTrainerCfg
        | dict[str, Any]
        | None = None,
        compiled_kernel: CompiledTrainingKernel | None = None,
        throughput_callback: Callable[[ThroughputStats], None] | None = None,
    ) -> None:
        trainer_cfg = _coerce_cfg(cfg)
        if trainer_cfg.mode == "compiled" and compiled_kernel is None:
            raise ValueError(
                "compiled mode requires an explicit CompiledTrainingKernel; "
                "use mode='compatibility' for skrl agents"
            )
        self.compiled_kernel = compiled_kernel
        self.throughput_callback = throughput_callback
        self.last_throughput: ThroughputStats | None = None
        self.throughput_history: list[ThroughputStats] = []
        super().__init__(
            env=env,
            agents=agents,
            scopes=scopes if scopes is not None else [],
            cfg=trainer_cfg,
        )

    @property
    def cfg(self) -> JaxSequentialTrainerCfg:
        return self._cfg

    @cfg.setter
    def cfg(self, value: JaxSequentialTrainerCfg) -> None:
        self._cfg = value

    def _agents_iter(self) -> list[Any]:
        if self.num_simultaneous_agents > 1:
            return list(self.agents)
        return [self.agents]

    def _set_training_mode(self, enabled: bool) -> None:
        for agent in self._agents_iter():
            agent.enable_training_mode(enabled)

    def _emit_stats(self, stats: ThroughputStats) -> None:
        self.last_throughput = stats
        self.throughput_history.append(stats)
        if self.throughput_callback is not None:
            self.throughput_callback(stats)

    def _make_stats(
        self,
        *,
        vector_steps: int,
        elapsed_seconds: float,
        warmup_vector_steps: int,
        warmup_seconds: float,
    ) -> ThroughputStats:
        return ThroughputStats(
            mode=self.cfg.mode,
            num_envs=int(self.env.num_envs),
            vector_steps=vector_steps,
            transitions=vector_steps * int(self.env.num_envs),
            elapsed_seconds=elapsed_seconds,
            warmup_vector_steps=warmup_vector_steps,
            warmup_seconds=warmup_seconds,
        )

    def train(self) -> ThroughputStats:
        """Train and return correctly-scaled vector-step/transition throughput."""

        self._set_training_mode(True)
        if self.cfg.mode == "compiled":
            return self._train_compiled()
        return self._train_compatibility()

    def _train_compatibility(self) -> ThroughputStats:
        observations, infos = self.env.reset()
        del infos
        states = self.env.state()

        total_steps = int(self.cfg.timesteps)
        warmup_steps = min(int(self.cfg.measurement_warmup_steps), total_steps)
        wall_start = time.perf_counter()
        measured_start = wall_start
        measured_steps = 0
        last_log_start = wall_start
        last_log_step = 0

        for timestep in tqdm.tqdm(
            range(total_steps),
            disable=self.cfg.disable_progressbar,
            file=sys.stdout,
        ):
            for agent in self._agents_iter():
                agent.pre_interaction(timestep=timestep, timesteps=total_steps)

            if self.num_simultaneous_agents == 1:
                actions, outputs = self.agents.act(
                    observations,
                    states,
                    timestep=timestep,
                    timesteps=total_steps,
                )
            else:
                action_rows = []
                for agent, scope in zip(self.agents, self.scopes):
                    scoped_states = (
                        states[scope[0] : scope[1]] if states is not None else None
                    )
                    actions, outputs = agent.act(
                        observations[scope[0] : scope[1]],
                        scoped_states,
                        timestep=timestep,
                        timesteps=total_steps,
                    )
                    action_rows.append(actions)
                actions = jnp.vstack(action_rows)
            del outputs

            next_observations, rewards, terminated, truncated, infos = self.env.step(
                actions
            )
            next_states = self.env.state()

            if not self.cfg.headless and not timestep % self.cfg.render_interval:
                self.env.render()

            if self.num_simultaneous_agents == 1:
                self.agents.record_transition(
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
                    timesteps=total_steps,
                )
            else:
                for agent, scope in zip(self.agents, self.scopes):
                    agent.record_transition(
                        observations=observations[scope[0] : scope[1]],
                        states=states[scope[0] : scope[1]]
                        if states is not None
                        else None,
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_observations=next_observations[scope[0] : scope[1]],
                        next_states=next_states[scope[0] : scope[1]]
                        if next_states is not None
                        else None,
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=total_steps,
                    )

            for agent in self._agents_iter():
                agent.post_interaction(timestep=timestep, timesteps=total_steps)

            if self.env.num_envs > 1:
                observations = next_observations
                states = next_states
            else:
                should_reset = (
                    not self.env.agents
                    if self.env.num_agents > 1
                    else bool(terminated.any() or truncated.any())
                )
                if should_reset:
                    observations, _ = self.env.reset()
                    states = self.env.state()
                else:
                    observations = next_observations
                    states = next_states

            completed_steps = timestep + 1
            if completed_steps == warmup_steps:
                if self.cfg.synchronize_on_measurement:
                    _synchronize(observations)
                measured_start = time.perf_counter()
                last_log_start = measured_start
                last_log_step = completed_steps

            if completed_steps > warmup_steps:
                measured_steps = completed_steps - warmup_steps

            interval = int(self.cfg.throughput_log_interval)
            if (
                interval > 0
                and completed_steps > warmup_steps
                and completed_steps - last_log_step >= interval
            ):
                if self.cfg.synchronize_on_measurement:
                    _synchronize(observations)
                now = time.perf_counter()
                interval_steps = completed_steps - last_log_step
                self._emit_stats(
                    self._make_stats(
                        vector_steps=interval_steps,
                        elapsed_seconds=now - last_log_start,
                        warmup_vector_steps=warmup_steps,
                        warmup_seconds=measured_start - wall_start,
                    )
                )
                last_log_start = now
                last_log_step = completed_steps

        if self.cfg.synchronize_on_measurement:
            _synchronize(observations)
        end = time.perf_counter()
        if warmup_steps == total_steps:
            measured_start = end
        stats = self._make_stats(
            vector_steps=measured_steps,
            elapsed_seconds=max(end - measured_start, 0.0),
            warmup_vector_steps=warmup_steps,
            warmup_seconds=max(measured_start - wall_start, 0.0),
        )
        self._emit_stats(stats)
        return stats

    def _train_compiled(self) -> ThroughputStats:
        if self.compiled_kernel is None:  # guarded during initialization
            raise RuntimeError("compiled mode has no compiled kernel")

        total_steps = int(self.cfg.timesteps)
        warmup_steps = min(int(self.cfg.measurement_warmup_steps), total_steps)
        state = self.compiled_kernel.initialize(
            env=self.env,
            agents=self.agents,
            cfg=self.cfg,
        )
        wall_start = time.perf_counter()
        measured_start = wall_start
        completed_steps = 0
        measurement_started = warmup_steps == 0

        while completed_steps < total_steps:
            requested = min(
                int(self.cfg.compiled_chunk_size),
                total_steps - completed_steps,
            )
            if not measurement_started:
                requested = min(requested, warmup_steps - completed_steps)
            state, metrics = self.compiled_kernel.train_chunk(
                state,
                start_timestep=completed_steps,
                timesteps=requested,
                total_timesteps=total_steps,
            )
            del metrics
            completed_steps += requested
            if not measurement_started and completed_steps == warmup_steps:
                if self.cfg.synchronize_on_measurement:
                    _synchronize(state)
                measured_start = time.perf_counter()
                measurement_started = True

        if self.cfg.synchronize_on_measurement:
            _synchronize(state)
        end = time.perf_counter()
        measured_steps = max(total_steps - warmup_steps, 0)
        stats = self._make_stats(
            vector_steps=measured_steps,
            elapsed_seconds=max(end - measured_start, 0.0),
            warmup_vector_steps=warmup_steps,
            warmup_seconds=max(measured_start - wall_start, 0.0),
        )
        self._emit_stats(stats)
        return stats


__all__ = [
    "CompiledTrainingKernel",
    "JaxSequentialTrainer",
    "JaxSequentialTrainerCfg",
    "ThroughputStats",
]
