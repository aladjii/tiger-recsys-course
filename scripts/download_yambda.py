#!/usr/bin/env python3
"""Download Yambda-500M data and optionally build listens_fltrd."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "yandex/yambda"
ROOT = Path(__file__).resolve().parents[1]

DATA_PATTERNS = [
    "flat/500m/listens.parquet",
    "embeddings.parquet",
    "album_item_mapping.parquet",
    "artist_item_mapping.parquet",
]


def download(data_root: Path) -> None:
    print(f"[download] Yambda-500M -> {data_root}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(data_root),
        allow_patterns=DATA_PATTERNS,
        max_workers=8,
    )
    print("[download] done")


def prepare_listens_fltrd(data_root: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_listens_fltrd.py"),
        "--data-root",
        str(data_root),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-listens-fltrd",
        action="store_true",
        help="After download, build flat+sequential listens_fltrd.parquet.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "dataset" / "yambda",
    )
    args = parser.parse_args()

    download(args.data_root)
    if args.prepare_listens_fltrd:
        prepare_listens_fltrd(args.data_root)


if __name__ == "__main__":
    main()
