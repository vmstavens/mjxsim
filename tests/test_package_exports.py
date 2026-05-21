from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path


def test_import_mjxsim_root() -> None:
    import mjxsim

    assert mjxsim.__version__
    assert "DiffusionPolicy" in mjxsim.__all__
    assert "PushTStateDataset" in mjxsim.__all__
    assert "pipe" in mjxsim.__all__
    assert "register" in mjxsim.__all__
    assert "set_state" in mjxsim.__all__
    assert resources.files("mjxsim").joinpath("py.typed").is_file()


def test_namespaced_submodule_imports() -> None:
    from mjxsim.agents import DiffusionPolicy, GNNAgent, PrivilegedAutoencoderAgent
    from mjxsim.datasets.pushert import PushTStateDataset
    from mjxsim.trainers import SupervisedTrainer
    from mjxsim.utils.datasets import split_dataset
    from mjxsim.utils.mjx import ObjType, get_names, get_number_of, set_state
    from mjxsim.utils.modelling import cable, pipe

    assert DiffusionPolicy.__name__ == "DiffusionPolicy"
    assert GNNAgent.__name__ == "GNNAgent"
    assert PrivilegedAutoencoderAgent.__name__ == "PrivilegedAutoencoderAgent"
    assert PushTStateDataset.__name__ == "PushTStateDataset"
    assert SupervisedTrainer.__name__ == "SupervisedTrainer"
    assert split_dataset.__name__ == "split_dataset"
    assert ObjType.BODY.value == 1
    assert get_names.__name__ == "get_names"
    assert get_number_of.__name__ == "get_number_of"
    assert set_state.__name__ == "set_state"
    assert cable.__name__ == "cable"
    assert pipe.__name__ == "pipe"


def test_utils_exports_register_lazily() -> None:
    import mjxsim.utils as utils

    assert "register" in utils.__all__
    assert "get_pose" in utils.__all__
    assert "pipe" in utils.__all__


def test_root_exports_modelling_and_mjx_helpers_lazily() -> None:
    import mjxsim

    assert mjxsim.get_number_of.__name__ == "get_number_of"
    assert mjxsim.get_names.__name__ == "get_names"
    assert mjxsim.get_ids.__name__ == "get_ids"
    assert mjxsim.does_exist.__name__ == "does_exist"
    assert mjxsim.set_state.__name__ == "set_state"
    assert mjxsim.pipe.__name__ == "pipe"
    assert mjxsim.cable.__name__ == "cable"


def test_mjxsim_utils_exports_are_not_shadowed_by_top_level_utils(
    tmp_path: Path, monkeypatch
) -> None:
    shadow_agents = tmp_path / "agents"
    shadow_agents.mkdir()
    shadow_agents.joinpath("__init__.py").write_text("", encoding="utf-8")
    shadow_agents.joinpath("diffusion_policy_state.py").write_text(
        "raise RuntimeError('shadowed agents.diffusion_policy_state imported')\n",
        encoding="utf-8",
    )
    shadow_trainers = tmp_path / "trainers"
    shadow_trainers.mkdir()
    shadow_trainers.joinpath("__init__.py").write_text("", encoding="utf-8")
    shadow_trainers.joinpath("supervised_trainer.py").write_text(
        "raise RuntimeError('shadowed trainers.supervised_trainer imported')\n",
        encoding="utf-8",
    )
    shadow_utils = tmp_path / "utils"
    shadow_utils.mkdir()
    shadow_utils.joinpath("__init__.py").write_text("", encoding="utf-8")
    shadow_utils.joinpath("mjx.py").write_text(
        "raise RuntimeError('shadowed utils.mjx imported')\n",
        encoding="utf-8",
    )
    shadow_utils.joinpath("modelling.py").write_text(
        "raise RuntimeError('shadowed utils.modelling imported')\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    for module_name in [
        "agents",
        "agents.diffusion_policy_state",
        "mjxsim.agents",
        "mjxsim.agents.diffusion_policy_state",
        "mjxsim.trainers",
        "mjxsim.trainers.supervised_trainer",
        "trainers",
        "trainers.supervised_trainer",
        "mjxsim.utils.mjx",
        "mjxsim.utils.modelling",
        "utils",
        "utils.mjx",
        "utils.modelling",
    ]:
        sys.modules.pop(module_name, None)

    import mjxsim

    for name in [
        "DiffusionPolicy",
        "ObjType",
        "SupervisedTrainer",
        "get_pose",
        "set_pose",
        "set_state",
        "pipe",
        "cable",
    ]:
        mjxsim.__dict__.pop(name, None)

    assert mjxsim.DiffusionPolicy.__name__ == "DiffusionPolicy"
    assert mjxsim.SupervisedTrainer.__name__ == "SupervisedTrainer"
    assert mjxsim.ObjType.BODY.value == 1
    assert mjxsim.get_pose.__name__ == "get_pose"
    assert mjxsim.set_pose.__name__ == "set_pose"
    assert mjxsim.set_state.__name__ == "set_state"
    assert mjxsim.pipe.__name__ == "pipe"
    assert mjxsim.cable.__name__ == "cable"
