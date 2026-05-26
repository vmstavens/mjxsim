"""Public convenience API for the :mod:`mjxsim` robot-learning toolkit."""

from __future__ import annotations

from importlib import import_module, metadata, util
from typing import TYPE_CHECKING, Any

try:
    __version__ = metadata.version("mjxsim")
except metadata.PackageNotFoundError:
    __version__ = "0.1.0"

if TYPE_CHECKING:
    from mjxsim.agents.diffusion_policy_state import (  # noqa: F401
        DP_CFG,
        ConditionalUnet1D,
        DiffusionPolicy,
        EMAModel,
    )
    from mjxsim.agents.diffusion_policy_vision import (  # noqa: F401
        VISION_DP_CFG,
        DiffusionPolicyVision,
    )
    from mjxsim.agents.drlr_sac import DRLR, DRLR_CFG, DRLR_DEFAULT_CONFIG  # noqa: F401
    from mjxsim.agents.drlr2_sac import (  # noqa: F401
        DRLR2,
        DRLR2_SAC_CFG,
        DRLR2_SAC_DEFAULT_CONFIG,
    )
    from mjxsim.agents.gnn import (  # noqa: F401
        GNN_CFG,
        GNN_DEFAULT_CONFIG,
        GNNAgent,
        GraphConvolution,
        GraphRegressionGCN,
        normalized_chain_adjacency,
    )
    from mjxsim.agents.ibrl_base_agent import Agent  # noqa: F401
    from mjxsim.agents.ibrl_sac import IBRL, IBRL_SAC_CFG  # noqa: F401
    from mjxsim.agents.autoencoder import (  # noqa: F401
        AE_CFG,
        AE_DEFAULT_CONFIG,
        Autoencoder,
        AutoencoderAgent,
    )
    from mjxsim.agents.latent_distiller import (  # noqa: F401
        LATENT_DISTILLER_CFG,
        LATENT_DISTILLER_DEFAULT_CONFIG,
        LatentDistillerAgent,
    )
    from mjxsim.agents.variational_autoencoder import (  # noqa: F401
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
    from mjxsim.trainers.sequential_trainer_plus import SequentialTrainerPlus  # noqa: F401
    from mjxsim.trainers.supervised_trainer import SupervisedTrainer  # noqa: F401
    from mjxsim.utils.datahandler import DataHandler  # noqa: F401
    from mjxsim.utils.datasets import split_dataset  # noqa: F401
    from mjxsim.utils.load import (  # noqa: F401
        get_memory,
        override_action_space,
        register,
    )
    from mjxsim.utils.mjx import (  # noqa: F401
        ObjType,
        does_exist,
        get_ids,
        get_names,
        get_number_of,
        get_pose,
        set_pose,
        set_state,
    )
    from mjxsim.utils.modelling import cable, pipe  # noqa: F401

_SUBMODULE_EXPORTS = [
    "agents",
    "datasets",
    "envs",
    "models",
    "trainers",
    "utils",
]

_AGENT_EXPORTS = [
    "AE_CFG",
    "AE_DEFAULT_CONFIG",
    "Agent",
    "Autoencoder",
    "AutoencoderAgent",
    "ConditionalUnet1D",
    "DP_CFG",
    "DiffusionPolicy",
    "DiffusionPolicyVision",
    "DRLR",
    "DRLR2",
    "DRLR2_SAC_CFG",
    "DRLR2_SAC_DEFAULT_CONFIG",
    "DRLR_CFG",
    "DRLR_DEFAULT_CONFIG",
    "EMAModel",
    "GNN_CFG",
    "GNN_DEFAULT_CONFIG",
    "GNNAgent",
    "GraphConvolution",
    "GraphRegressionGCN",
    "IBRL",
    "IBRL_SAC_CFG",
    "LATENT_DISTILLER_CFG",
    "LATENT_DISTILLER_DEFAULT_CONFIG",
    "LatentDistillerAgent",
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
    "normalized_chain_adjacency",
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
    "cable",
    "does_exist",
    "get_ids",
    "get_memory",
    "get_names",
    "get_number_of",
    "get_pose",
    "override_action_space",
    "pipe",
    "register",
    "set_pose",
    "set_state",
    "split_dataset",
]

_EXPORT_MODULES = {
    "agents": "mjxsim.agents",
    "datasets": "mjxsim.datasets",
    "envs": "mjxsim.envs",
    "models": "mjxsim.models",
    "trainers": "mjxsim.trainers",
    "utils": "mjxsim.utils",
    "AE_CFG": "mjxsim.agents.autoencoder",
    "AE_DEFAULT_CONFIG": "mjxsim.agents.autoencoder",
    "Agent": "mjxsim.agents.ibrl_base_agent",
    "Autoencoder": "mjxsim.agents.autoencoder",
    "AutoencoderAgent": "mjxsim.agents.autoencoder",
    "ConditionalUnet1D": "mjxsim.agents.diffusion_policy_state",
    "DP_CFG": "mjxsim.agents.diffusion_policy_state",
    "DiffusionPolicy": "mjxsim.agents.diffusion_policy_state",
    "DiffusionPolicyVision": "mjxsim.agents.diffusion_policy_vision",
    "DRLR": "mjxsim.agents.drlr_sac",
    "DRLR2": "mjxsim.agents.drlr2_sac",
    "DRLR2_SAC_CFG": "mjxsim.agents.drlr2_sac",
    "DRLR2_SAC_DEFAULT_CONFIG": "mjxsim.agents.drlr2_sac",
    "DRLR_CFG": "mjxsim.agents.drlr_sac",
    "DRLR_DEFAULT_CONFIG": "mjxsim.agents.drlr_sac",
    "EMAModel": "mjxsim.agents.diffusion_policy_state",
    "GNN_CFG": "mjxsim.agents.gnn",
    "GNN_DEFAULT_CONFIG": "mjxsim.agents.gnn",
    "GNNAgent": "mjxsim.agents.gnn",
    "GraphConvolution": "mjxsim.agents.gnn",
    "GraphRegressionGCN": "mjxsim.agents.gnn",
    "IBRL": "mjxsim.agents.ibrl_sac",
    "IBRL_SAC_CFG": "mjxsim.agents.ibrl_sac",
    "LATENT_DISTILLER_CFG": "mjxsim.agents.latent_distiller",
    "LATENT_DISTILLER_DEFAULT_CONFIG": "mjxsim.agents.latent_distiller",
    "LatentDistillerAgent": "mjxsim.agents.latent_distiller",
    "VAE_CFG": "mjxsim.agents.variational_autoencoder",
    "VAE_STATE_CFG": "mjxsim.agents.variational_autoencoder",
    "VAE_VISION_CFG": "mjxsim.agents.variational_autoencoder",
    "VariationalAutoencoder": "mjxsim.agents.variational_autoencoder",
    "VariationalAutoencoderAgent": "mjxsim.agents.variational_autoencoder",
    "VariationalAutoencoderState": "mjxsim.agents.variational_autoencoder",
    "VariationalAutoencoderStateAgent": "mjxsim.agents.variational_autoencoder",
    "VariationalAutoencoderVision": "mjxsim.agents.variational_autoencoder",
    "VariationalAutoencoderVisionAgent": "mjxsim.agents.variational_autoencoder",
    "VISION_DP_CFG": "mjxsim.agents.diffusion_policy_vision",
    "normalized_chain_adjacency": "mjxsim.agents.gnn",
    "AttractorTrajectoryDataset": "datasets.attractor",
    "DemonstrationDataset": "datasets.demonstration",
    "ImageStateDataset": "datasets.vision",
    "PushTStateDataset": "datasets.pushert",
    "StateDataset": "datasets.state",
    "download_dataset": "datasets.pushert",
    "MocapReach": "envs.mocap_control",
    "PipeInsert2": "envs.pipe_insert_2",
    "PushTEnv": "envs.pushert",
    "SequentialTrainerPlus": "mjxsim.trainers.sequential_trainer_plus",
    "SupervisedTrainer": "mjxsim.trainers.supervised_trainer",
    "DataHandler": "mjxsim.utils.datahandler",
    "ObjType": "mjxsim.utils.mjx",
    "cable": "mjxsim.utils.modelling",
    "does_exist": "mjxsim.utils.mjx",
    "get_ids": "mjxsim.utils.mjx",
    "get_memory": "mjxsim.utils.load",
    "get_names": "mjxsim.utils.mjx",
    "get_number_of": "mjxsim.utils.mjx",
    "get_pose": "mjxsim.utils.mjx",
    "override_action_space": "mjxsim.utils.load",
    "pipe": "mjxsim.utils.modelling",
    "register": "mjxsim.utils.load",
    "set_pose": "mjxsim.utils.mjx",
    "set_state": "mjxsim.utils.mjx",
    "split_dataset": "mjxsim.utils.datasets",
}

_OPTIONAL_DEPENDENCIES = {
    "Autoencoder": "torch",
    "AutoencoderAgent": "torch",
    "DiffusionPolicy": "torch",
    "DiffusionPolicyVision": "torch",
    "DRLR": "skrl",
    "DRLR2": "skrl",
    "GNNAgent": "torch",
    "IBRL": "skrl",
    "LatentDistillerAgent": "torch",
    "VariationalAutoencoder": "torch",
    "VariationalAutoencoderAgent": "skrl",
    "VariationalAutoencoderState": "torch",
    "VariationalAutoencoderStateAgent": "skrl",
    "VariationalAutoencoderVision": "torch",
    "VariationalAutoencoderVisionAgent": "skrl",
    "MocapReach": "mujoco",
    "PipeInsert2": "mujoco",
    "PushTEnv": "gym",
    "get_memory": "skrl",
    "override_action_space": "gymnasium",
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
