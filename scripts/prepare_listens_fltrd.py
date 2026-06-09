#!/usr/bin/env python3
"""Build listens_fltrd from flat listens: timestamp filter + full-listen threshold."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "dataset" / "yambda"

TIMESTAMP_MIN = 18_000_000
PLAYED_RATIO_MIN = 90


def build_listens_fltrd(
    data_root: Path,
    *,
    size: str = "500m",
    timestamp_min: int = TIMESTAMP_MIN,
    played_ratio_min: int = PLAYED_RATIO_MIN,
) -> tuple[Path, Path]:
    flat_dir = data_root / "flat" / size
    seq_dir = data_root / "sequential" / size
    src = flat_dir / "listens.parquet"
    if not src.exists():
        raise FileNotFoundError(
            f"Missing {src}. Run: python scripts/download_yambda.py"
        )

    flat_dir.mkdir(parents=True, exist_ok=True)
    seq_dir.mkdir(parents=True, exist_ok=True)
    flat_out = flat_dir / "listens_fltrd.parquet"
    seq_out = seq_dir / "listens_fltrd.parquet"

    print(
        f"[prepare] filtering listens: timestamp > {timestamp_min}, "
        f"played_ratio_pct > {played_ratio_min}%"
    )
    df = pl.read_parquet(src)
    filtered = df.filter(
        pl.col("timestamp") > timestamp_min,
        pl.col("played_ratio_pct") > played_ratio_min,
    )
    print(f"[prepare] flat events: {len(filtered):,}")

    filtered.write_parquet(flat_out)
    print(f"[prepare] wrote flat {flat_out}")

    seq_df = (
        filtered.sort(["uid", "timestamp"])
        .group_by("uid", maintain_order=True)
        .agg(
            pl.col("timestamp"),
            pl.col("item_id"),
            pl.col("is_organic"),
            pl.col("played_ratio_pct"),
            pl.col("track_length_seconds"),
        )
    )
    seq_df.write_parquet(seq_out)
    print(f"[prepare] wrote sequential {seq_out} ({len(seq_df):,} users)")

    return flat_out, seq_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Yambda dataset root (contains flat/ and sequential/).",
    )
    parser.add_argument("--size", default="500m", help="Dataset size split.")
    parser.add_argument(
        "--timestamp-min",
        type=int,
        default=TIMESTAMP_MIN,
        help="Keep events with timestamp strictly greater than this value.",
    )
    parser.add_argument(
        "--played-ratio-min",
        type=int,
        default=PLAYED_RATIO_MIN,
        help="Keep events with played_ratio_pct strictly greater than this value.",
    )
    args = parser.parse_args()
    build_listens_fltrd(
        args.data_root,
        size=args.size,
        timestamp_min=args.timestamp_min,
        played_ratio_min=args.played_ratio_min,
    )


if __name__ == "__main__":
    main()
