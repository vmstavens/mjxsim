from __future__ import annotations

from importlib import resources


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
    from mjxsim.agents import DiffusionPolicy
    from mjxsim.datasets.pushert import PushTStateDataset
    from mjxsim.utils.datasets import split_dataset
    from mjxsim.utils.mjx import ObjType, get_names, get_number_of, set_state
    from mjxsim.utils.modelling import cable, pipe

    assert DiffusionPolicy.__name__ == "DiffusionPolicy"
    assert PushTStateDataset.__name__ == "PushTStateDataset"
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
