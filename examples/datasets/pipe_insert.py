import logging
import os
import re
from pathlib import Path

import gdown

logging.basicConfig(level=logging.WARN)
relative_path = os.path.relpath(__file__)
logger = logging.getLogger(relative_path)
logger.setLevel(logging.DEBUG)


def google_drive_file_id(url: str) -> str:
    """Extract a Google Drive file id from a common share/download URL."""
    patterns = (
        r"/file/d/([^/]+)",
        r"[?&]id=([^&]+)",
        r"/uc\?export=download&id=([^&]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not find a Google Drive file id in URL: {url}")


def download_dataset(
    *,
    file_id: str | None = None,
    url: str | None = None,
    filename: str = "pipe_insert_dataset.zip",
    verbose: bool = False,
) -> str:
    """Download a Pipe Insert dataset from Google Drive to a local cache.

    Pass either the Google Drive file id or a full Google Drive share URL.
    """
    if verbose:
        logger.setLevel(logging.INFO)

    if file_id is None:
        if url is None:
            raise ValueError("Pass either file_id or url to download_dataset")
        file_id = google_drive_file_id(url)

    cache_dir = Path.home() / ".cache" / "pipe_insert_data"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = cache_dir / filename

    if not dataset_path.exists():
        logger.info("Downloading Pipe Insert dataset...")
        gdown.download(id=file_id, output=str(dataset_path), quiet=not verbose)
        logger.info("Dataset downloaded to: %s", dataset_path)
    else:
        logger.info("Dataset already exists at: %s", dataset_path)

    logger.setLevel(logging.WARN)
    return str(dataset_path)


if __name__ == "__main__":
    # Example:
    # dataset_path = download_dataset(url="https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing", verbose=True)
    # dataset_path = download_dataset(file_id="<FILE_ID>", verbose=True)
    raise SystemExit("Call download_dataset(file_id=...) or download_dataset(url=...)")
