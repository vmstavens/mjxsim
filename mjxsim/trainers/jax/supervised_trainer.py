"""JAX-native supervised trainer."""

from __future__ import annotations

import copy
import dataclasses
import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import jax
import jax.numpy as jp
import optax
import tqdm


@dataclasses.dataclass(kw_only=True)
class SupervisedTrainerCfg:
    """Configuration for offline supervised training."""

    epochs: int = 10
    """Legacy alias for num_epochs."""

    num_epochs: int | None = None
    """Number of training epochs."""

    eval_frequency: int = 10
    """Number of epochs between validation/callback execution."""

    early_stopping_patience: int | None = None
    """Number of validation evaluations without improvement before stopping."""

    early_stopping_min_delta: float = 0.0
    """Minimum validation-loss decrease required to count as an improvement."""

    restore_best_params: bool = True
    """Whether to restore the best validation params after early stopping."""

    jit_compile: bool = True
    """Whether to JIT compile the train and validation steps."""

    progressbar: bool = True
    """Whether to show a tqdm progress bar for training batches."""

    def expand(self) -> None:
        if self.num_epochs is None:
            self.num_epochs = self.epochs
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be at least 1")
        if self.eval_frequency < 1:
            raise ValueError("eval_frequency must be at least 1")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience < 1
        ):
            raise ValueError("early_stopping_patience must be at least 1 or None")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.expand()
        return dataclasses.asdict(self)

    def copy(self) -> dict[str, Any]:
        return copy.deepcopy(self.to_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


SUPERVISED_TRAINER_DEFAULT_CONFIG = SupervisedTrainerCfg()


def _coerce_supervised_trainer_cfg(
    cfg: SupervisedTrainerCfg | Mapping[str, Any] | None,
) -> SupervisedTrainerCfg:
    if cfg is None:
        result = SupervisedTrainerCfg()
    elif isinstance(cfg, SupervisedTrainerCfg):
        result = copy.deepcopy(cfg)
    elif isinstance(cfg, Mapping):
        result = SupervisedTrainerCfg()
        for key, value in cfg.items():
            if not hasattr(result, key):
                raise ValueError(f"Invalid supervised trainer config key: {key}")
            setattr(result, key, copy.deepcopy(value))
    else:
        raise TypeError(
            "trainer_config must be a SupervisedTrainerCfg, mapping, or None "
            f"(got {type(cfg).__name__})"
        )
    result.expand()
    return result


def _tree_to_jax(batch: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: jp.asarray(x), batch)


def _loss_takes_rng(loss_fn: Callable[..., Any]) -> bool:
    signature = inspect.signature(loss_fn)
    parameters = list(signature.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters):
        return True
    if "rng" in signature.parameters:
        return True
    positional = [
        p
        for p in parameters
        if p.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    return len(positional) >= 3


class SupervisedTrainer:
    """Small JAX/Optax trainer for pure supervised loss functions.

    ``loss_fn`` should accept ``(params, batch)`` or ``(params, batch, rng)`` and
    return a scalar loss. Batches may be any pytree of array-like values.
    """

    def __init__(
        self,
        *,
        params: Any,
        loss_fn: Callable[..., jax.Array],
        optimizer: optax.GradientTransformation,
        trainer_config: SupervisedTrainerCfg | Mapping[str, Any] | None = None,
        train_loader: Iterable[Any] | None = None,
        valid_loader: Iterable[Any] | None = None,
        rng: jax.Array | None = None,
        callback_fn: Callable[[int, float, float | None], None] | None = None,
    ):
        self.config = _coerce_supervised_trainer_cfg(trainer_config)
        self.params = params
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.opt_state = optimizer.init(params)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.rng = jax.random.PRNGKey(0) if rng is None else rng
        self._callback_fn = callback_fn
        self._loss_takes_rng = _loss_takes_rng(loss_fn)
        self.early_stopped = False
        self.best_epoch: int | None = None
        self.best_validation_loss: float | None = None

        self._train_step = self._make_train_step()
        self._validation_step = self._make_validation_step()

    def _call_loss(self, params: Any, batch: Any, rng: jax.Array) -> jax.Array:
        if self._loss_takes_rng:
            return self.loss_fn(params, batch, rng)
        return self.loss_fn(params, batch)

    def _make_train_step(self):
        def train_step(params, opt_state, batch, rng):
            loss, grads = jax.value_and_grad(self._call_loss)(params, batch, rng)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        return jax.jit(train_step) if self.config.jit_compile else train_step

    def _make_validation_step(self):
        def validation_step(params, batch, rng):
            return self._call_loss(params, batch, rng)

        return jax.jit(validation_step) if self.config.jit_compile else validation_step

    def _validate(self) -> float:
        if self.valid_loader is None:
            raise ValueError("valid_loader is required for validation")
        total_loss, batch_count = 0.0, 0
        for batch in self.valid_loader:
            self.rng, batch_rng = jax.random.split(self.rng)
            loss = self._validation_step(
                self.params,
                _tree_to_jax(batch),
                batch_rng,
            )
            total_loss += float(loss)
            batch_count += 1
        return total_loss / max(batch_count, 1)

    def train(self):
        """Run training and return the final parameter pytree."""
        if self.train_loader is None:
            raise ValueError("Set train_loader first")
        if (
            self.config.early_stopping_patience is not None
            and self.valid_loader is None
        ):
            raise ValueError("early stopping requires a valid_loader")

        best_params = None
        stale_evaluations = 0

        for epoch in range(self.config.num_epochs):
            epoch_loss, batch_count = 0.0, 0
            iterator = tqdm.tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                disable=not self.config.progressbar,
            )

            for batch in iterator:
                self.rng, batch_rng = jax.random.split(self.rng)
                self.params, self.opt_state, loss = self._train_step(
                    self.params,
                    self.opt_state,
                    _tree_to_jax(batch),
                    batch_rng,
                )
                epoch_loss += float(loss)
                batch_count += 1

            avg_loss = epoch_loss / max(batch_count, 1)
            val_loss = None
            if (
                self.valid_loader is not None
                and epoch % self.config.eval_frequency == 0
            ):
                val_loss = self._validate()
                if (
                    self.best_validation_loss is None
                    or val_loss
                    < self.best_validation_loss - self.config.early_stopping_min_delta
                ):
                    self.best_validation_loss = val_loss
                    self.best_epoch = epoch
                    stale_evaluations = 0
                    best_params = copy.deepcopy(jax.device_get(self.params))
                else:
                    stale_evaluations += 1

                if (
                    self.config.early_stopping_patience is not None
                    and stale_evaluations >= self.config.early_stopping_patience
                ):
                    self.early_stopped = True
                    if self.config.restore_best_params and best_params is not None:
                        self.params = best_params
                    if self._callback_fn:
                        self._callback_fn(epoch, avg_loss, val_loss)
                    break

            if self._callback_fn:
                self._callback_fn(epoch, avg_loss, val_loss)

        return self.params
