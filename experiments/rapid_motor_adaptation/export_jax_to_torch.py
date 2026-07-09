"""Exports JAX/Flax RMA checkpoints to equivalent PyTorch modules.

The exporter copies the local RMA MLP/Conv parameters from the Flax checkpoint
format into `RmaTorchBundle`. Dense kernels are transposed from Flax
`[in_dim, out_dim]` to PyTorch `[out_dim, in_dim]`; Conv1d kernels are
transposed from Flax `[kernel, in_channels, out_channels]` to PyTorch
`[out_channels, in_channels, kernel]`.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
import torch

from experiments.rapid_motor_adaptation.checkpoints import (
    load_checkpoint,
    save_checkpoint,
)
from experiments.rapid_motor_adaptation.networks import (
    dummy_inputs,
    initialize_networks,
    make_networks,
)
from experiments.rapid_motor_adaptation.torch_networks import (
    RmaTorchBundle,
    RmaTorchObservationLayout,
)

jax.config.update("jax_default_matmul_precision", "float32")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _numpy(value: Any) -> np.ndarray:
    return np.array(jax.device_get(value))


def _copy_dense(linear: torch.nn.Linear, params: dict[str, Any]) -> None:
    kernel = torch.from_numpy(_numpy(params["kernel"]).T).to(
        dtype=linear.weight.dtype,
        device=linear.weight.device,
    )
    bias = torch.from_numpy(_numpy(params["bias"])).to(
        dtype=linear.bias.dtype,
        device=linear.bias.device,
    )
    if linear.weight.shape != kernel.shape:
        raise ValueError(f"dense weight shape mismatch: {linear.weight.shape} != {kernel.shape}")
    if linear.bias.shape != bias.shape:
        raise ValueError(f"dense bias shape mismatch: {linear.bias.shape} != {bias.shape}")
    linear.weight.data.copy_(kernel)
    linear.bias.data.copy_(bias)


def _copy_conv(conv: torch.nn.Conv1d, params: dict[str, Any]) -> None:
    kernel = torch.from_numpy(np.transpose(_numpy(params["kernel"]), (2, 1, 0))).to(
        dtype=conv.weight.dtype,
        device=conv.weight.device,
    )
    bias = torch.from_numpy(_numpy(params["bias"])).to(
        dtype=conv.bias.dtype,
        device=conv.bias.device,
    )
    if conv.weight.shape != kernel.shape:
        raise ValueError(f"conv weight shape mismatch: {conv.weight.shape} != {kernel.shape}")
    if conv.bias.shape != bias.shape:
        raise ValueError(f"conv bias shape mismatch: {conv.bias.shape} != {bias.shape}")
    conv.weight.data.copy_(kernel)
    conv.bias.data.copy_(bias)


def _copy_mlp(torch_mlp, flax_mlp_params: dict[str, Any], dense_count: int) -> None:
    linear_layers = [layer for layer in torch_mlp.net if isinstance(layer, torch.nn.Linear)]
    if len(linear_layers) != dense_count:
        raise ValueError(f"expected {dense_count} linear layers, got {len(linear_layers)}")
    for index, linear in enumerate(linear_layers):
        _copy_dense(linear, flax_mlp_params[f"Dense_{index}"])


def copy_flax_params_to_torch(
    torch_model: RmaTorchBundle,
    *,
    phase1_params: dict[str, Any],
    adaptation_params: dict[str, Any] | None,
) -> list[str]:
    """Copies available Flax RMA parameters into a PyTorch RMA bundle."""
    converted = []
    phase1_params = phase1_params["params"] if "params" in phase1_params else phase1_params

    _copy_mlp(torch_model.encoder.net, phase1_params["encoder"]["params"]["Mlp_0"], 3)
    converted.append("encoder")

    _copy_mlp(torch_model.policy.net, phase1_params["policy"]["params"]["Mlp_0"], 4)
    converted.append("policy")

    _copy_mlp(torch_model.value.net, phase1_params["value"]["params"]["Mlp_0"], 4)
    converted.append("value")

    if adaptation_params is not None:
        adaptation_params = (
            adaptation_params["params"] if "params" in adaptation_params else adaptation_params
        )
        params = adaptation_params["params"] if "params" in adaptation_params else adaptation_params
        _copy_dense(torch_model.adaptation.step_embed[0], params["Dense_0"])
        _copy_dense(torch_model.adaptation.step_embed[2], params["Dense_1"])
        _copy_conv(torch_model.adaptation.temporal[0].conv, params["Conv_0"])
        _copy_conv(torch_model.adaptation.temporal[2].conv, params["Conv_1"])
        _copy_conv(torch_model.adaptation.temporal[4].conv, params["Conv_2"])
        _copy_dense(torch_model.adaptation.proj, params["Dense_2"])
        converted.append("adaptation")

    return converted


def _flat_observations(
    rma_state: np.ndarray,
    previous_action: np.ndarray,
    env_factors: np.ndarray,
    history: np.ndarray,
) -> torch.Tensor:
    history_flat = history.reshape(history.shape[0], -1)
    obs = np.concatenate([rma_state, previous_action, env_factors, history_flat], axis=-1)
    return torch.from_numpy(obs.astype(np.float32))


def _parity_check(
    torch_model: RmaTorchBundle,
    *,
    phase1_params: dict[str, Any],
    adaptation_params: dict[str, Any] | None,
    batch_size: int,
    strict_tolerance: float,
    warning_tolerance: float,
) -> dict[str, float]:
    networks = make_networks()
    inputs = dummy_inputs(batch_size=batch_size)
    rma_state = jax.random.normal(jax.random.PRNGKey(10), inputs.rma_state.shape)
    previous_action = jax.random.normal(jax.random.PRNGKey(11), inputs.previous_action.shape)
    env_factors = jax.random.normal(jax.random.PRNGKey(12), inputs.env_factors.shape)
    history = jax.random.normal(jax.random.PRNGKey(13), inputs.history.shape)

    z = networks.encoder.apply(phase1_params["encoder"], env_factors)
    jax_policy = networks.policy.apply(phase1_params["policy"], rma_state, previous_action, z)
    jax_value = networks.value.apply(phase1_params["value"], rma_state, previous_action, z)

    observations = _flat_observations(
        _numpy(rma_state),
        _numpy(previous_action),
        _numpy(env_factors),
        _numpy(history),
    )
    torch_model.eval()
    with torch.no_grad():
        torch_policy = torch_model.privileged_action(observations).cpu().numpy()
        rma_state_t, prev_action_t, env_factors_t, history_t = torch_model.layout.split(
            observations
        )
        torch_z = torch_model.encoder(env_factors_t)
        torch_value = torch_model.value(rma_state_t, prev_action_t, torch_z).squeeze(-1)

    metrics = {
        "privileged_action_max_abs_diff": float(
            np.max(np.abs(_numpy(jax_policy) - torch_policy))
        ),
        "value_max_abs_diff": float(np.max(np.abs(_numpy(jax_value) - torch_value.cpu().numpy()))),
    }

    if adaptation_params is not None:
        jax_adaptation = networks.adaptation.apply(adaptation_params, history)
        with torch.no_grad():
            torch_adaptation = torch_model.adaptation(history_t).cpu().numpy()
            torch_rma_policy = torch_model.policy(
                rma_state_t,
                prev_action_t,
                torch.from_numpy(torch_adaptation).to(rma_state_t),
            ).cpu().numpy()
        jax_rma_policy = networks.policy.apply(
            phase1_params["policy"], rma_state, previous_action, jax_adaptation
        )
        metrics["adaptation_max_abs_diff"] = float(
            np.max(np.abs(_numpy(jax_adaptation) - torch_adaptation))
        )
        metrics["rma_action_max_abs_diff"] = float(
            np.max(np.abs(_numpy(jax_rma_policy) - torch_rma_policy))
        )

    strict_failures = {
        name: value for name, value in metrics.items() if value > strict_tolerance
    }
    hard_failures = {
        name: value for name, value in metrics.items() if value > warning_tolerance
    }
    if hard_failures:
        raise AssertionError(f"JAX/Torch parity check failed: {hard_failures}")
    if strict_failures:
        warnings.warn(
            "JAX/Torch parity exceeded strict tolerance but stayed within "
            f"export tolerance: {strict_failures}",
            RuntimeWarning,
            stacklevel=2,
        )
    return metrics


def _checkpoint_params(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if "phase1_params" in payload:
        return payload["phase1_params"], payload.get("adaptation_params")
    if "params" in payload:
        return payload["params"], payload["params"].get("adaptation")
    raise KeyError("checkpoint must contain `params` or `phase1_params`")


def export_checkpoint(
    checkpoint: Path,
    output: Path,
    *,
    batch_size: int,
    strict_tolerance: float,
    warning_tolerance: float,
) -> dict[str, Any]:
    payload = load_checkpoint(checkpoint)
    phase1_params, adaptation_params = _checkpoint_params(payload)
    torch_model = RmaTorchBundle(RmaTorchObservationLayout())
    converted_modules = copy_flax_params_to_torch(
        torch_model,
        phase1_params=phase1_params,
        adaptation_params=adaptation_params,
    )
    parity = _parity_check(
        torch_model,
        phase1_params=phase1_params,
        adaptation_params=adaptation_params,
        batch_size=batch_size,
        strict_tolerance=strict_tolerance,
        warning_tolerance=warning_tolerance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": torch_model.state_dict(),
            "layout": RmaTorchObservationLayout().__dict__,
            "converted_modules": converted_modules,
            "log_std": _numpy(phase1_params["log_std"]) if "log_std" in phase1_params else None,
            "source_checkpoint": str(checkpoint),
            "parity": parity,
            "strict_tolerance": strict_tolerance,
            "warning_tolerance": warning_tolerance,
        },
        output,
    )
    return {
        "output": str(output),
        "converted_modules": converted_modules,
        "parity": parity,
    }


def self_test(
    output: Path,
    *,
    batch_size: int,
    strict_tolerance: float,
    warning_tolerance: float,
) -> dict[str, Any]:
    params = initialize_networks(jax.random.PRNGKey(0), batch_size=batch_size)
    phase1_params = {
        "encoder": params["encoder"],
        "policy": params["policy"],
        "value": params["value"],
        "log_std": jp.zeros((12,), dtype=jp.float32),
    }
    payload = {
        "phase": 2,
        "phase1_params": phase1_params,
        "adaptation_params": params["adaptation"],
    }
    temp = output.with_suffix(".self_test_jax.pkl")
    save_checkpoint(temp, payload)
    try:
        return export_checkpoint(
            temp,
            output,
            batch_size=batch_size,
            strict_tolerance=strict_tolerance,
            warning_tolerance=warning_tolerance,
        )
    finally:
        temp.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/rapid_motor_adaptation/.runs/full_warp/phase2.pkl"),
        help="JAX Phase-1 or Phase-2 checkpoint to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/rapid_motor_adaptation/.runs/full_warp/rma_torch.pt"),
        help="Torch checkpoint output path.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--strict-tolerance",
        type=float,
        default=1e-5,
        help="Warn if any parity metric exceeds this value.",
    )
    parser.add_argument(
        "--warning-tolerance",
        type=float,
        default=1e-2,
        help="Fail export if any parity metric exceeds this value.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Deprecated alias for --warning-tolerance.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Export a freshly initialized synthetic Flax checkpoint instead of reading --checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warning_tolerance = (
        args.tolerance if args.tolerance is not None else args.warning_tolerance
    )
    if args.self_test:
        result = self_test(
            args.output,
            batch_size=args.batch_size,
            strict_tolerance=args.strict_tolerance,
            warning_tolerance=warning_tolerance,
        )
    else:
        result = export_checkpoint(
            args.checkpoint,
            args.output,
            batch_size=args.batch_size,
            strict_tolerance=args.strict_tolerance,
            warning_tolerance=warning_tolerance,
        )
    print(f"saved {result['output']}")
    print(f"converted_modules={','.join(result['converted_modules'])}")
    for name, value in result["parity"].items():
        print(f"{name}={value:.8g}")


if __name__ == "__main__":
    main()
