import math
from typing import Any, Generator

import polars as pl
import torch


# https://github.com/pola-rs/polars/issues/10683
def iter_slices(df: pl.LazyFrame, batch_size: int) -> Generator[pl.DataFrame, None, None]:
    assert isinstance(df, pl.LazyFrame), "df must be a LazyFrame"

    def get_batch(df: pl.LazyFrame, offset: int, batch_size: int) -> pl.DataFrame:
        batch = df.slice(offset, batch_size)
        batch = batch.collect()
        return batch

    batch = get_batch(df, 0, batch_size)

    yield batch

    offset = len(batch)
    if offset:
        while True:
            batch = get_batch(df, offset, batch_size)
            len_ = len(batch)
            if len_:
                offset += len_
                yield batch
            else:
                break


def iter_rows(df: pl.LazyFrame, batch_size: int) -> Generator[dict[str, Any], None, None]:
    for batch in iter_slices(df, batch_size):
        for row in batch.iter_rows(named=True):
            yield row


class YambdaIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, df: pl.LazyFrame | pl.DataFrame):
        super().__init__()

        self.start = 0
        self.end = df.lazy().select(pl.len()).collect().item()

        self.df = df.lazy()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        df = self.df

        start = self.start
        end = self.end

        if worker_info is not None:
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = self.start + worker_id * per_worker
            end = min(start + per_worker, self.end)

        df_slice = df.slice(start, end - start)

        for row in iter_rows(df_slice, 2048):
            yield row


def get_default_dataloader(dataset: YambdaIterableDataset, collator, batch_size: int):
    return torch.utils.data.DataLoader(
        dataset,
        multiprocessing_context="spawn",
        num_workers=3,
        prefetch_factor=10,
        batch_size=batch_size,
        collate_fn=collator,
        pin_memory=True,
        pin_memory_device="cuda",
        persistent_workers=True,
    )
