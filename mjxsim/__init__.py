"""Public convenience API for the :mod:`mjxsim` robot-learning toolkit."""

from __future__ import annotations

from importlib import import_module, metadata, util
from typing import TYPE_CHECKING, Any

try:
    __version__ = metadata.version("mjxsim")
except metadata.PackageNotFoundError:
    __version__ = "0.1.0"

if TYPE_CHECKING:
    from agents.diffusion_policy_state import (  # noqa: F401
        DP_CFG,
        ConditionalUnet1D,
        DiffusionPolicy,
        EMAModel,
    )
    from agents.diffusion_policy_vision import (  # noqa: F401
        VISION_DP_CFG,
        DiffusionPolicyVision,
    )
    from agents.ibrl_base_agent import Agent  # noqa: F401
    from agents.ibrl_sac import IBRL, IBRL_SAC_CFG  # noqa: F401
    from agents.variational_autoencoder import (  # noqa: F401
        VAE_CFG,
        VAE_STATE_CFG,
        VAE_VISION_CFG,
        VariationalAutoencoder,
        VariationalAutoencoderAgent,
        VariationalAutoencoderState,
        VariationalAutoencoderStateAgent,
        VariationalAutoencoderVision,
        VariationalAutoencoderVisionAgent,
    )
    from datasets.attractor import AttractorTrajectoryDataset  # noqa: F401
    from datasets.demonstration import DemonstrationDataset  # noqa: F401
    from datasets.state import StateDataset  # noqa: F401
    from datasets.vision import ImageStateDataset  # noqa: F401
    from envs.mocap_control import MocapReach  # noqa: F401
    from envs.pipe_insert_2 import PipeInsert2  # noqa: F401
    from envs.pushert import PushTEnv  # noqa: F401
    from examples.datasets.pushert import (  # noqa: F401
        PushTStateDataset,
        download_dataset,
    )
    from trainers.sequential_trainer_plus import SequentialTrainerPlus  # noqa: F401
    from trainers.supervised_trainer import SupervisedTrainer  # noqa: F401
    from utils.datahandler import DataHandler  # noqa: F401
    from utils.datasets import split_dataset  # noqa: F401
    from utils.load import register  # noqa: F401
    from utils.mjx import ObjType, get_pose, set_pose  # noqa: F401

_SUBMODULE_EXPORTS = [
    "agents",
    "datasets",
    "envs",
    "models",
    "trainers",
    "utils",
]

_AGENT_EXPORTS = [
    "Agent",
    "ConditionalUnet1D",
    "DP_CFG",
    "DiffusionPolicy",
    "DiffusionPolicyVision",
    "EMAModel",
    "IBRL",
    "IBRL_SAC_CFG",
    "VAE_CFG",
    "VAE_STATE_CFG",
    "VAE_VISION_CFG",
    "VariationalAutoencoder",
    "VariationalAutoencoderAgent",
    "VariationalAutoencoderState",
    "VariationalAutoencoderStateAgent",
    "VariationalAutoencoderVision",
    "VariationalAutoencoderVisionAgent",
    "VISION_DP_CFG",
]

_DATASET_EXPORTS = [
    "AttractorTrajectoryDataset",
    "DemonstrationDataset",
    "ImageStateDataset",
    "PushTStateDataset",
    "StateDataset",
    "download_dataset",
]

_ENV_EXPORTS = ["MocapReach", "PipeInsert2", "PushTEnv"]

_TRAINER_EXPORTS = ["SequentialTrainerPlus", "SupervisedTrainer"]

_UTIL_EXPORTS = [
    "DataHandler",
    "ObjType",
    "get_pose",
    "register",
    "set_pose",
    "split_dataset",
]

_EXPORT_MODULES = {
    "agents": "mjxsim.agents",
    "datasets": "mjxsim.datasets",
    "envs": "mjxsim.envs",
    "models": "mjxsim.models",
    "trainers": "mjxsim.trainers",
    "utils": "mjxsim.utils",
    "Agent": "agents.ibrl_base_agent",
    "ConditionalUnet1D": "agents.diffusion_policy_state",
    "DP_CFG": "agents.diffusion_policy_state",
    "DiffusionPolicy": "agents.diffusion_policy_state",
    "DiffusionPolicyVision": "agents.diffusion_policy_vision",
    "EMAModel": "agents.diffusion_policy_state",
    "IBRL": "agents.ibrl_sac",
    "IBRL_SAC_CFG": "agents.ibrl_sac",
    "VAE_CFG": "agents.variational_autoencoder",
    "VAE_STATE_CFG": "agents.variational_autoencoder",
    "VAE_VISION_CFG": "agents.variational_autoencoder",
    "VariationalAutoencoder": "agents.variational_autoencoder",
    "VariationalAutoencoderAgent": "agents.variational_autoencoder",
    "VariationalAutoencoderState": "agents.variational_autoencoder",
    "VariationalAutoencoderStateAgent": "agents.variational_autoencoder",
    "VariationalAutoencoderVision": "agents.variational_autoencoder",
    "VariationalAutoencoderVisionAgent": "agents.variational_autoencoder",
    "VISION_DP_CFG": "agents.diffusion_policy_vision",
    "AttractorTrajectoryDataset": "datasets.attractor",
    "DemonstrationDataset": "datasets.demonstration",
    "ImageStateDataset": "datasets.vision",
    "PushTStateDataset": "datasets.pushert",
    "StateDataset": "datasets.state",
    "download_dataset": "datasets.pushert",
    "MocapReach": "envs.mocap_control",
    "PipeInsert2": "envs.pipe_insert_2",
    "PushTEnv": "envs.pushert",
    "SequentialTrainerPlus": "trainers.sequential_trainer_plus",
    "SupervisedTrainer": "trainers.supervised_trainer",
    "DataHandler": "utils.datahandler",
    "ObjType": "utils.mjx",
    "get_pose": "utils.mjx",
    "register": "utils.load",
    "set_pose": "utils.mjx",
    "split_dataset": "utils.datasets",
}

_OPTIONAL_DEPENDENCIES = {
    "DiffusionPolicy": "torch",
    "DiffusionPolicyVision": "torch",
    "IBRL": "skrl",
    "VariationalAutoencoder": "torch",
    "VariationalAutoencoderAgent": "skrl",
    "VariationalAutoencoderState": "torch",
    "VariationalAutoencoderStateAgent": "skrl",
    "VariationalAutoencoderVision": "torch",
    "VariationalAutoencoderVisionAgent": "skrl",
    "MocapReach": "mujoco",
    "PipeInsert2": "mujoco",
    "PushTEnv": "gym",
    "SequentialTrainerPlus": "skrl",
    "SupervisedTrainer": "torch",
}


def _missing_dependency(name: str) -> str | None:
    dependency = _OPTIONAL_DEPENDENCIES.get(name)
    if dependency is None:
        return None
    return dependency if util.find_spec(dependency) is None else None


def __getattr__(name: str) -> Any:
    """Lazily resolve public exports while keeping ``import mjxsim`` lightweight."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    missing_dependency = _missing_dependency(name)
    if missing_dependency is not None:
        msg = (
            f"Cannot import mjxsim.{name}: optional dependency "
            f"{missing_dependency!r} is not installed."
        )
        raise ImportError(msg)

    module = import_module(module_name)
    value = module if name in _SUBMODULE_EXPORTS else getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "__version__",
    *_SUBMODULE_EXPORTS,
    *_AGENT_EXPORTS,
    *_DATASET_EXPORTS,
    *_ENV_EXPORTS,
    *_TRAINER_EXPORTS,
    *_UTIL_EXPORTS,
]
