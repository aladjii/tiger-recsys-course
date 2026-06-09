import gin
import os
import random
import torch

from data.schemas import SeqBatch
from data.yambda import YambdaItems
from data.yambda import YambdaSequences
from enum import Enum
from torch.utils.data import Dataset


@gin.constants_from_enum
class RecDataset(Enum):
    YAMBDA = 1


DATASET_ROOT = {
    RecDataset.YAMBDA: "dataset/yambda",
}

MAX_SEQ_LEN = {
    RecDataset.YAMBDA: 120,
}


class ItemData(Dataset):
    """Item corpus for RQ-VAE training (precomputed content embeddings)."""

    def __init__(
        self,
        root: str,
        dataset: RecDataset = RecDataset.YAMBDA,
        force_process: bool = False,
        train_test_split: str = "all",
        split: str = "500m",
        interaction: str = "likes",
        **kwargs,
    ) -> None:
        raw = YambdaItems(
            root=root,
            size=split,
            interaction=interaction,
            max_seq_len=MAX_SEQ_LEN[dataset],
        )
        if force_process or not os.path.exists(raw.processed_paths[0]):
            raw.process(force=force_process)

        stored = raw.data

        if train_test_split == "train":
            filt = stored["is_train"]
        elif train_test_split == "eval":
            filt = ~stored["is_train"]
        elif train_test_split == "all":
            filt = torch.ones(stored["x"].shape[0], dtype=torch.bool)
        else:
            raise ValueError(f"Unknown train_test_split: {train_test_split}")

        self.item_data = stored["x"][filt]
        self.item_ids = stored["item_id"][filt]
        self.embed_dim = stored["embed_dim"]

    def __len__(self):
        return self.item_data.shape[0]

    def __getitem__(self, idx):
        item_ids = (
            torch.tensor(idx).unsqueeze(0) if not isinstance(idx, torch.Tensor) else idx
        )
        x = self.item_data[idx]
        return SeqBatch(
            user_ids=-1 * torch.ones_like(item_ids.squeeze(0)),
            ids=item_ids,
            ids_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            x=x,
            x_fut=-1 * torch.ones_like(item_ids.squeeze(0)),
            seq_mask=torch.ones_like(item_ids, dtype=torch.bool),
        )


class SeqData(Dataset):
    """User interaction sequences for TIGER decoder."""

    def __init__(
        self,
        root: str,
        dataset: RecDataset = RecDataset.YAMBDA,
        is_train: bool = True,
        subsample: bool = False,
        force_process: bool = False,
        split: str = "500m",
        interaction: str = "listens_fltrd",
        **kwargs,
    ) -> None:
        assert (not subsample) or is_train, "Can only subsample on training split."

        max_seq_len = MAX_SEQ_LEN[dataset]
        raw = YambdaSequences(
            root=root,
            size=split,
            interaction=interaction,
            max_seq_len=max_seq_len,
        )
        if force_process or not os.path.exists(raw.processed_paths[0]):
            raw.process(force=force_process)

        seq_data = raw.data
        split_name = "train" if is_train else "test"
        self.sequence_data = seq_data[split_name]
        self.subsample = subsample
        self._max_seq_len = max_seq_len

        items = YambdaItems(
            root=root,
            size=split,
            interaction=interaction,
            max_seq_len=max_seq_len,
        )
        stored = items.data
        self.item_embeddings = stored["x"]

    @property
    def max_seq_len(self):
        return self._max_seq_len

    def __len__(self):
        return self.sequence_data["userId"].shape[0]

    def __getitem__(self, idx):
        user_ids = self.sequence_data["userId"][idx]

        if self.subsample:
            seq = self.sequence_data["itemId"][idx]
            start_idx = random.randint(0, max(0, len(seq) - 3))
            end_idx = random.randint(start_idx + 2, min(len(seq), start_idx + self.max_seq_len + 1))
            sample = seq[start_idx:end_idx]
            item_ids = torch.tensor(
                sample[:-1] + [-1] * (self.max_seq_len - len(sample[:-1])),
                dtype=torch.long,
            )
            item_ids_fut = torch.tensor([sample[-1]], dtype=torch.long)
        else:
            item_ids = self.sequence_data["itemId"][idx]
            item_ids_fut = self.sequence_data["itemId_fut"][idx].unsqueeze(0)

        x = self.item_embeddings[item_ids.clamp(min=0)]
        x[item_ids == -1] = -1

        x_fut = self.item_embeddings[item_ids_fut.clamp(min=0)]
        x_fut[item_ids_fut == -1] = -1

        return SeqBatch(
            user_ids=user_ids,
            ids=item_ids,
            ids_fut=item_ids_fut,
            x=x,
            x_fut=x_fut,
            seq_mask=(item_ids >= 0),
        )
