from __future__ import annotations

from importlib import resources


def test_import_mjxsim_root() -> None:
    import mjxsim

    assert mjxsim.__version__
    assert "DiffusionPolicy" in mjxsim.__all__
    assert "PushTStateDataset" in mjxsim.__all__
    assert "register" in mjxsim.__all__
    assert resources.files("mjxsim").joinpath("py.typed").is_file()


def test_namespaced_submodule_imports() -> None:
    from mjxsim.agents import DiffusionPolicy
    from mjxsim.datasets.pushert import PushTStateDataset
    from mjxsim.utils.datasets import split_dataset
    from mjxsim.utils.mjx import ObjType

    assert DiffusionPolicy.__name__ == "DiffusionPolicy"
    assert PushTStateDataset.__name__ == "PushTStateDataset"
    assert split_dataset.__name__ == "split_dataset"
    assert ObjType.BODY.value == 1


def test_utils_exports_register_lazily() -> None:
    import mjxsim.utils as utils

    assert "register" in utils.__all__
