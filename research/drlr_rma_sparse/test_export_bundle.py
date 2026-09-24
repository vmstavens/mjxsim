"""Run with python3 -m unittest discover -s research/drlr_rma_sparse."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "export_bundle", Path(__file__).with_name("export_bundle.py")
)
bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bundle)


class BundleTest(unittest.TestCase):
    def test_integrity_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.zip"
            payload = b"original"
            manifest = {
                "files": [
                    {
                        "path": "lab/uv.lock",
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ]
            }
            for data in (payload, b"tampered"):
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("lab/uv.lock", data)
                    archive.writestr("manifest.json", json.dumps(manifest))
                if data == payload:
                    bundle.verify(path)
                else:
                    with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                        bundle.verify(path)

    def test_missing_assets_fail_before_creating_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "bundle.zip"
            with patch.object(bundle, "git_files", return_value=[]):
                with self.assertRaises(FileNotFoundError):
                    bundle.export(root, root, output, False)
            self.assertFalse(output.exists())

    def test_working_tree_capture_excludes_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lab, library = root / "lab", root / "mjxsim"
            lab.mkdir()
            library.mkdir()
            files = ["pyproject.toml", "uv.lock", "learning/local.py", "runs/live.bin"]
            for name in files:
                path = lab / name
                path.parent.mkdir(exist_ok=True)
                path.write_text("local modified content")
            (library / "README.md").write_text("library snapshot")
            with (
                patch.object(
                    bundle,
                    "git_files",
                    side_effect=[
                        list(map(Path, files)),
                        [Path("README.md")],
                    ],
                ),
                patch.object(bundle, "provenance", return_value={"commit": "test"}),
                patch.object(bundle, "command", side_effect=FileNotFoundError),
            ):
                output = root / "bundle.zip"
                bundle.export(lab, library, output, True)
            with zipfile.ZipFile(output) as archive:
                self.assertNotIn("lab/runs/live.bin", archive.namelist())
                self.assertEqual(
                    archive.read("lab/learning/local.py"), b"local modified content"
                )
                self.assertFalse(
                    json.loads(archive.read("manifest.json"))[
                        "includes_reference_assets"
                    ]
                )


if __name__ == "__main__":
    unittest.main()
