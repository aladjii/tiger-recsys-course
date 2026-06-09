import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List

import numpy as np
import polars as pl
import torch

from yambda.constants import Constants
from yambda.processing import timesplit


logger = logging.getLogger(__name__)


@dataclass
class Data:
    train: pl.LazyFrame
    validation: pl.LazyFrame | None
    test: pl.LazyFrame
    item_id_to_idx: dict[int, int]

    _train_user_ids: torch.Tensor | None = field(init=False, default=None)

    @property
    def num_items(self):
        return len(self.item_id_to_idx)

    @cached_property
    def num_train_users(self):
        return self.train.select(pl.len()).collect(engine="streaming").item()

    def train_user_ids(self, device):
        if self._train_user_ids is None or self._train_user_ids.device != device:
            self._train_user_ids = self.train.select('uid').collect(engine="streaming")['uid'].to_torch().to(device)
        return self._train_user_ids


def preprocess(df: pl.LazyFrame, interaction: str, val_size=Constants.VAL_SIZE, max_seq_len: int = 200) -> Data:
    """
    Preprocesses raw interaction data for recommendation system modeling.

    Args:
        df (pl.LazyFrame): Raw input data containing user interaction sequences
        interaction (str): Type of interaction to process. Must be either 'likes' or 'listens'.
        val_size (float): Proportion of data to use for validation (default: from Constants)

    Returns:
        Data: Named tuple containing:
            - train (pl.LazyFrame): Training data
            - val (pl.LazyFrame): Validation data
            - test (pl.LazyFrame): Test data
            - item_id_to_idx (dict): Mapping from original item IDs to model indices

    Note:
        - For 'listens' interactions, uses strict engagement threshold
        - Item indices start at 1 to reserve 0 for padding/masking
    """
    if interaction == 'listens':
        df = df.select(
            'uid',
            pl.col('item_id', 'timestamp').list.gather(
                pl.col('played_ratio_pct').list.eval(pl.arg_where(pl.element() >= Constants.TRACK_LISTEN_THRESHOLD))
            ),
        ).filter(pl.col('item_id').list.len() > 0)

    unique_item_ids = (
        df.select(pl.col("item_id").explode().unique().sort()).collect(engine="streaming")["item_id"].to_list()
    )

    item_id_to_idx = {int(item_id): i + 1 for i, item_id in enumerate(unique_item_ids)}

    train, val, test = timesplit.sequential_split_train_val_test(
        df, val_size=val_size, test_timestamp=Constants.TEST_TIMESTAMP, drop_non_train_items=False
    )

    def replace_strict(df):
        return (
            df.select(
                pl.col("item_id").list.eval(pl.element().replace_strict(item_id_to_idx)),
                pl.all().exclude("item_id"),
            )
            .collect(engine="streaming")
            .lazy()
        )

    # polars requires too much memory for replace strict if list is too big
    train = train.select('uid', pl.all().exclude('uid').list.slice(-max_seq_len, max_seq_len))
    train = replace_strict(train)

    if val is not None:
        val = replace_strict(val)

    test = replace_strict(test)

    return Data(train, val, test, item_id_to_idx)


class TrainDataset:
    def __init__(self, dataset: pl.DataFrame, num_items: int, max_seq_len: int):
        self._dataset = dataset
        self._num_items = num_items
        self._max_seq_len = max_seq_len

    @property
    def dataset(self) -> pl.DataFrame:
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, List[int] | int]:
        sample = self._dataset.row(index, named=True)

        item_sequence = sample['item_id'][:-1][-self._max_seq_len :]
        positive_sequence = sample['item_id'][1:][-self._max_seq_len :]
        negative_sequence = np.random.randint(1, self._num_items + 1, size=(len(item_sequence),)).tolist()

        return {
            'user.ids': [sample['uid']],
            'user.length': 1,
            'item.ids': item_sequence,
            'item.length': len(item_sequence),
            'positive.ids': positive_sequence,
            'positive.length': len(positive_sequence),
            'negative.ids': negative_sequence,
            'negative.length': len(negative_sequence),
        }


class EvalDataset:
    def __init__(self, dataset: pl.DataFrame, max_seq_len: int):
        self._dataset = dataset
        self._max_seq_len = max_seq_len

    @property
    def dataset(self) -> pl.DataFrame:
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Dict[str, List[int] | int]:
        sample = self._dataset.row(index, named=True)

        item_sequence = sample['item_id_train'][-self._max_seq_len :]
        next_items = sample['item_id_valid']

        return {
            'user.ids': [sample['uid']],
            'user.length': 1,
            'item.ids': item_sequence,
            'item.length': len(item_sequence),
            'labels.ids': next_items,
            'labels.length': len(next_items),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collates a batch of samples into batched tensors suitable for model input.

    This function processes a list of dictionaries, each containing keys like '{prefix}.ids'
    and '{prefix}.length' (the length of the sequence for that prefix). For each such prefix, it:
        - Concatenates all '{prefix}.ids' lists from the batch into a single flat list.
        - Collects all '{prefix}.length' values into a list.
        - Converts the resulting lists into torch.LongTensor objects.

    Args:
        batch (List[Dict]): List of sample dictionaries. Each sample must contain keys of the form
            '{prefix}.ids' (list of ints) and '{prefix}.length' (int).

    Returns:
        Dict[str, torch.Tensor]: Dictionary with keys '{prefix}.ids' and '{prefix}.length' for each prefix,
            where values are 1D torch.LongTensor objects suitable for model input.
    """
    processed_batch = {}
    for key in batch[0].keys():
        if key.endswith('.ids'):
            prefix = key.split('.')[0]
            assert '{}.length'.format(prefix) in batch[0]

            processed_batch[f'{prefix}.ids'] = []
            processed_batch[f'{prefix}.length'] = []

            for sample in batch:
                processed_batch[f'{prefix}.ids'].extend(sample[f'{prefix}.ids'])
                processed_batch[f'{prefix}.length'].append(sample[f'{prefix}.length'])

    for part, values in processed_batch.items():
        processed_batch[part] = torch.tensor(values, dtype=torch.long)

    return processed_batch
