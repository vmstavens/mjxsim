"""Export the lab working tree and reference assets without importing ML packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

REFERENCE = "train_drlr_target_align_geometry_only_dp_rma_sparse"
ASSETS = (
    "artifacts/models/diffusion_policy/"
    "align_target_align_state48_spacemouse_rot45_stiff10m_stride3_v1",
    "data/grasped_end_dlo_target_align_sparse/processed/"
    "align_processed_target_align_state48_spacemouse_large_stride3_v1",
)
LAB_DIRS = (
    "learning",
    "envs",
    "configs",
    "workflows",
    "tests",
    "tools",
    "scripts",
    "docs",
    "notes",
    "typings",
)
LAB_FILES = ("pyproject.toml", "uv.lock", "README.md", "LICENSE", ".gitignore")


def command(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def git_files(root: Path) -> list[Path]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
    )
    return sorted({Path(os.fsdecode(p)) for p in raw.split(b"\0") if p})


def provenance(root: Path) -> dict:
    return {
        "commit": command(["git", "rev-parse", "HEAD"], root),
        "status": command(["git", "status", "--short"], root),
        "source": (
            "working tree, including tracked modifications "
            "and nonignored untracked files"
        ),
    }


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        expected = {entry["path"] for entry in manifest["files"]}
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != expected | {"manifest.json"}:
            raise ValueError("Archive members do not match manifest")
        for entry in manifest["files"]:
            member = PurePosixPath(entry["path"])
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"Unsafe archive path: {member}")
            with archive.open(str(member)) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(f"SHA256 mismatch: {member}")
            if archive.getinfo(str(member)).file_size != entry["bytes"]:
                raise ValueError(f"Size mismatch: {member}")
    print(f"Verified {len(expected)} files: {path}")


def export(lab: Path, mjxsim: Path, output: Path, code_only: bool) -> None:
    selected: dict[str, Path] = {}
    for relative in git_files(lab):
        if relative.parts[0] in LAB_DIRS or relative.as_posix() in LAB_FILES:
            selected[f"lab/{relative.as_posix()}"] = lab / relative
    for relative in git_files(mjxsim):
        selected[f"mjxsim/{relative.as_posix()}"] = mjxsim / relative
    if not code_only:
        for name in ASSETS:
            directory = lab / name
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Required asset directory missing: {directory}"
                )
            for path in directory.rglob("*"):
                if path.is_file():
                    selected[f"lab/{path.relative_to(lab).as_posix()}"] = path
        for name in (
            f"{ASSETS[0]}/final_model.pkl",
            f"{ASSETS[1]}/manifest.json",
            f"{ASSETS[1]}/states.npy",
            f"{ASSETS[1]}/actions.npy",
            f"{ASSETS[1]}/episode_ends.npy",
        ):
            if not (lab / name).is_file():
                raise FileNotFoundError(f"Required asset missing: {name}")
    for name in ("pyproject.toml", "uv.lock"):
        if f"lab/{name}" not in selected:
            raise FileNotFoundError(f"Lab is missing {name}")
    manifest = {
        "format_version": 1,
        "reference_config": REFERENCE,
        "includes_reference_assets": not code_only,
        "lab": provenance(lab),
        "mjxsim": provenance(mjxsim),
        "host": {"platform": platform.platform(), "export_python": sys.version},
        "runtime_environment": {
            key: os.environ[key]
            for key in (
                "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "JAX_PLATFORMS",
                "CUDA_VISIBLE_DEVICES",
                "MUJOCO_GL",
            )
            if key in os.environ
        },
        "files": [],
    }
    python = lab / ".venv/bin/python"
    if python.exists():
        manifest["installed_packages"] = json.loads(
            command(
                [
                    str(python),
                    "-c",
                    "import importlib.metadata as m,json; "
                    "print(json.dumps(sorted((d.metadata['Name'],d.version) "
                    "for d in m.distributions())))",
                ],
                lab,
            )
        )
    try:
        manifest["gpu"] = command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            lab,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        manifest["gpu"] = "unavailable"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting a previous reproduction snapshot.
    with output.open("xb") as destination:
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, path in sorted(selected.items()):
                if path.is_symlink():
                    raise ValueError(f"Symlinks need explicit handling: {path}")
                if not path.is_file():  # Tracked deletions are recorded in provenance.
                    continue
                digest = file_digest(path)
                size = path.stat().st_size
                archive.write(path, name)
                if file_digest(path) != digest:
                    raise RuntimeError(f"File changed during export: {path}")
                manifest["files"].append(
                    {
                        "path": name,
                        "sha256": digest,
                        "bytes": size,
                    }
                )
            archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
    verify(output)
    print(f"Bundle SHA256: {file_digest(output)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=Path, help="DLO lab checkout")
    parser.add_argument(
        "--mjxsim", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--code-only", action="store_true", help="Omit teacher and replay dataset"
    )
    parser.add_argument(
        "--verify", type=Path, help="Verify an existing bundle without extracting it"
    )
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
    elif args.lab and args.output:
        export(
            args.lab.resolve(),
            args.mjxsim.resolve(),
            args.output.resolve(),
            args.code_only,
        )
    else:
        parser.error("Supply --verify BUNDLE or both --lab PATH and --output BUNDLE")


if __name__ == "__main__":
    main()
