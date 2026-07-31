"""JAX Phase-2 latent distillation for Rapid Motor Adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import optax

from mjxsim.rma.spec import RmaSpec

from .modules import AdaptationEncoder, PrivilegedEncoder


class LatentDistillationTrainer:
    """Train ``phi(history)`` against a frozen privileged encoder ``mu(factors)``."""

    def __init__(
        self,
        spec: RmaSpec,
        privileged_encoder: PrivilegedEncoder,
        privileged_variables: Mapping[str, Any],
        *,
        adaptation_encoder: AdaptationEncoder | None = None,
        adaptation_variables: Mapping[str, Any] | None = None,
        key: jax.Array | None = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.0,
    ) -> None:
        self.spec = spec
        self.privileged_encoder = privileged_encoder
        self.privileged_variables = privileged_variables
        self.adaptation_encoder = adaptation_encoder or AdaptationEncoder(spec)
        if adaptation_variables is None:
            if key is None:
                raise ValueError(
                    "key is required when adaptation_variables are not provided"
                )
            adaptation_variables = self.adaptation_encoder.init(
                key,
                jnp.zeros(
                    (1, spec.history_len, spec.history_feature_dim),
                    dtype=jnp.float32,
                ),
            )
        self.adaptation_params = adaptation_variables["params"]
        self.optimizer = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self.optimizer_state = self.optimizer.init(self.adaptation_params)

    def update(
        self,
        history: jax.Array,
        factors: jax.Array,
    ) -> Mapping[str, float]:
        """Perform one adaptation-encoder update and return scalar metrics."""

        history = jnp.asarray(history, dtype=jnp.float32)
        factors = jnp.asarray(factors, dtype=jnp.float32)
        target = jax.lax.stop_gradient(
            self.privileged_encoder.apply(self.privileged_variables, factors)
        )

        def loss_fn(params):
            prediction = self.adaptation_encoder.apply(
                {"params": params},
                history,
            )
            loss = jnp.mean(jnp.square(prediction - target))
            return loss, prediction

        (loss, prediction), gradients = jax.value_and_grad(loss_fn, has_aux=True)(
            self.adaptation_params
        )
        updates, self.optimizer_state = self.optimizer.update(
            gradients,
            self.optimizer_state,
            self.adaptation_params,
        )
        self.adaptation_params = optax.apply_updates(
            self.adaptation_params,
            updates,
        )
        return {
            "adaptation_mse": float(loss),
            "predicted_latent_norm": float(jnp.linalg.norm(prediction, axis=-1).mean()),
            "target_latent_norm": float(jnp.linalg.norm(target, axis=-1).mean()),
        }

    @property
    def adaptation_variables(self) -> dict[str, Any]:
        """Return Flax variables suitable for actor-only deployment."""

        return {"params": self.adaptation_params}

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer and model state for trusted local checkpointing."""

        return {
            "adaptation_params": self.adaptation_params,
            "optimizer_state": self.optimizer_state,
        }
