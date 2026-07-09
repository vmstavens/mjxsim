"""Progress logging helpers for RMA training."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CsvLogger:
    """Append-only CSV logger with a fixed schema."""

    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow({name: row.get(name, "") for name in self.fieldnames})
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "CsvLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()
