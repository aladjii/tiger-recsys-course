import dataclasses
from functools import cached_property

import numpy as np
import polars as pl
import torch
from tqdm import tqdm


@dataclasses.dataclass
class Embeddings:
    ids: torch.Tensor
    embeddings: torch.Tensor

    def __post_init__(self):
        assert self.ids.dim() == 1
        assert self.embeddings.dim() == 2
        assert self.ids.shape[0] == self.embeddings.shape[0]

        assert self.ids.device == self.embeddings.device

        if not torch.all(self.ids[:-1] <= self.ids[1:]):
            indexes = torch.argsort(self.ids, descending=False)
            self.embeddings = self.embeddings[indexes, :]
            self.ids = self.ids[indexes]

        assert torch.all(self.ids[:-1] < self.ids[1:]), "ids should be unique"

    @property
    def device(self):
        return self.ids.device

    def save(self, file_path: str):
        ids_np = self.ids.cpu().numpy()
        embeddings_np = self.embeddings.cpu().numpy()
        np.savez(file_path, ids=ids_np, embeddings=embeddings_np)

    @classmethod
    def load(cls, file_path: str, device: torch.device = torch.device('cpu')) -> 'Embeddings':
        with np.load(file_path) as data:
            ids_np = data['ids']
            embeddings_np = data['embeddings']

        ids = torch.from_numpy(ids_np).to(device)
        embeddings = torch.from_numpy(embeddings_np).to(device)

        return cls(ids=ids, embeddings=embeddings)


@dataclasses.dataclass
class Targets:
    user_ids: torch.Tensor
    item_ids: list[torch.Tensor]

    def __post_init__(self):
        assert len(self.item_ids) > 0
        assert self.user_ids.dim() == 1
        assert self.user_ids.shape[0] == len(self.item_ids)
        assert all(x.dim() == 1 for x in self.item_ids), "all ids should be 1D"

        assert all(x.device == self.item_ids[0].device for x in self.item_ids), "all ids should be on the same device"
        assert self.user_ids.device == self.item_ids[0].device

        if not torch.all(self.user_ids[:-1] <= self.user_ids[1:]):
            indexes = torch.argsort(self.user_ids, descending=False)
            self.item_ids = [self.item_ids[i] for i in indexes]
            self.user_ids = self.user_ids[indexes]

        assert torch.all(self.user_ids[:-1] < self.user_ids[1:]), "user_ids should be unique"

    @cached_property
    def lengths(self):
        return torch.tensor([ids.shape[0] for ids in self.item_ids], device=self.item_ids[0].device)

    def __len__(self):
        return len(self.item_ids)

    @property
    def device(self):
        return self.user_ids.device

    @classmethod
    def from_sequential(cls, df: pl.LazyFrame | pl.DataFrame, device: torch.device | str) -> 'Targets':
        df = df.lazy()
        return cls(
            df.select("uid").collect()["uid"].to_torch().to(device),
            [torch.tensor(x, device=device) for x in df.select("item_id").collect()["item_id"].to_list()],
        )


@dataclasses.dataclass
class Ranked:
    user_ids: torch.Tensor
    item_ids: torch.Tensor
    scores: torch.Tensor | None = None

    num_item_ids: int | None = None  # number of all items. Useful for coverage and etc.

    def __post_init__(self):
        if self.scores is None:
            self.scores = torch.arange(
                self.item_ids.shape[1], 0, -1, device=self.item_ids.device, dtype=torch.float32
            ).expand((self.user_ids.shape[0], self.item_ids.shape[1]))

        assert self.user_ids.dim() == 1
        assert self.scores.dim() == 2
        assert self.scores.shape == self.item_ids.shape
        assert self.user_ids.shape[0] == self.scores.shape[0]

        assert self.user_ids.device == self.scores.device == self.item_ids.device

        assert torch.all(self.scores[:, :-1] >= self.scores[:, 1:]), "scores should be sorted"

        if not torch.all(self.user_ids[:-1] <= self.user_ids[1:]):
            indexes = torch.argsort(self.user_ids, descending=False)
            self.item_ids = self.item_ids[indexes, :]
            self.scores = self.scores[indexes, :]
            self.user_ids = self.user_ids[indexes]

    @property
    def device(self):
        return self.user_ids.device


def rank_items(
    users: Embeddings,
    items: Embeddings,
    num_items: int,
    batch_size: int = 128,
    seen_item_ids: list[torch.Tensor] | None = None,
) -> Ranked:
    assert users.device == items.device

    num_users = users.ids.shape[0]

    scores = users.embeddings.new_empty((num_users, num_items))
    item_ids = users.embeddings.new_empty((num_users, num_items), dtype=torch.long)

    for batch_idx in tqdm(range((num_users + batch_size - 1) // batch_size), desc="Calc topk by batches"):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        batch_scores = users.embeddings[start_idx:end_idx, :] @ items.embeddings.T

        if seen_item_ids is not None:
            batch_seen = seen_item_ids[start_idx:end_idx]
            batch_scores = mask_seen_item_scores(batch_scores, batch_seen)

        sort_indices = batch_scores.topk(num_items, dim=-1).indices
        scores[start_idx:end_idx, :] = torch.gather(batch_scores, dim=-1, index=sort_indices)

        item_ids[start_idx:end_idx, :] = torch.gather(
            items.ids.expand(sort_indices.shape[0], items.ids.shape[0]), dim=-1, index=sort_indices
        )

    return Ranked(user_ids=users.ids, item_ids=item_ids, scores=scores, num_item_ids=items.ids.shape[0])


def mask_seen_item_scores(
    scores: torch.Tensor,
    seen_item_ids: list[torch.Tensor],
) -> torch.Tensor:
    """Zero out seen items by setting their scores to -inf before top-k."""
    masked = scores.clone()
    for row, seen in enumerate(seen_item_ids):
        if seen.numel() == 0:
            continue
        masked[row, seen.long()] = float("-inf")
    return masked
