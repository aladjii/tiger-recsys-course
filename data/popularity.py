"""Item popularity counts from sequential interaction logs."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import torch


def load_item_popularity(
    dataset_folder: str,
    dataset_split: str,
    dataset_interaction: str,
    item_ids: torch.Tensor,
) -> torch.Tensor:
    """Return listen-count popularity aligned with ``item_ids`` corpus order."""
    seq_path = (
        Path(dataset_folder)
        / "sequential"
        / dataset_split
        / f"{dataset_interaction}.parquet"
    )
    counts_df = (
        pl.scan_parquet(seq_path)
        .select(pl.col("item_id").explode().alias("item_id"))
        .group_by("item_id")
        .len()
        .collect(engine="streaming")
    )
    count_map = dict(zip(counts_df["item_id"].to_list(), counts_df["len"].to_list()))

    popularity = torch.zeros(len(item_ids), dtype=torch.long)
    for idx, raw_id in enumerate(item_ids.tolist()):
        popularity[idx] = int(count_map.get(int(raw_id), 0))
    return popularity
