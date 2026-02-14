"""Top-level helpers for the repository."""

from importlib import metadata

try:
    __version__ = metadata.version("mjx-sim")
except metadata.PackageNotFoundError:
    # Local/editable installs before building a wheel
    __version__ = "0.1.0"

__all__ = ["agents", "datasets", "envs", "sim", "trainers", "utils"]
