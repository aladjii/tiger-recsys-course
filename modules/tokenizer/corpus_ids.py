"""Corpus-wide semantic ID assignment with dedup column."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import torch
from data.utils import batch_to

from modules.tokenizer.rrs import assign_semantic_ids_rrs

AssignmentPolicy = Literal["sequential", "popularity", "rrs"]


def sem_ids_as_batch(sem_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return semantic IDs as [batch_size, n_layers]."""
    if sem_ids.shape[0] == batch_size:
        return sem_ids
    if sem_ids.shape[1] == batch_size:
        return sem_ids.T
    raise ValueError(
        f"Cannot align sem_ids shape {tuple(sem_ids.shape)} with batch_size={batch_size}"
    )


def _compute_semantic_layers(model, item_dataset, device, batch_size: int = 2048) -> torch.Tensor:
    model.eval()
    n_items = len(item_dataset)
    probe = batch_to(item_dataset[0], device)
    probe_x = probe.x
    if probe_x.dim() == 1:
        probe_x = probe_x.unsqueeze(0)
    probe_sem = model.get_semantic_ids(probe_x).sem_ids
    n_layers = sem_ids_as_batch(probe_sem, probe_x.shape[0]).shape[1]

    cached = torch.empty((n_items, n_layers), dtype=torch.long)
    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        xs = torch.stack([item_dataset[i].x for i in range(start, end)]).to(device)
        sem = sem_ids_as_batch(model.get_semantic_ids(xs).sem_ids, xs.shape[0]).cpu()
        cached[start:end] = sem
    return cached


def _assign_dedup_sequential(semantic: torch.Tensor) -> torch.Tensor:
    n_items = semantic.shape[0]
    dedup = torch.zeros(n_items, dtype=torch.long)
    seen: dict[tuple[int, ...], int] = {}
    for idx in range(n_items):
        key = tuple(int(x) for x in semantic[idx].tolist())
        hit = seen.get(key, 0)
        seen[key] = hit + 1
        dedup[idx] = hit
    return dedup


def _assign_dedup_popularity(
    semantic: torch.Tensor,
    item_popularity: torch.Tensor,
) -> torch.Tensor:
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for idx in range(semantic.shape[0]):
        key = tuple(int(x) for x in semantic[idx].tolist())
        groups[key].append(idx)

    dedup = torch.zeros(semantic.shape[0], dtype=torch.long)
    for indices in groups.values():
        sorted_idx = sorted(indices, key=lambda i: int(item_popularity[i].item()), reverse=True)
        for rank, idx in enumerate(sorted_idx):
            dedup[idx] = rank
    return dedup


@torch.no_grad()
def precompute_corpus_semantic_ids(
    model,
    item_dataset,
    device,
    *,
    assignment: AssignmentPolicy = "sequential",
    item_popularity: torch.Tensor | None = None,
    rrs_k: int = 4,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Assign semantic IDs with collision dedup column (TIGER paper style)."""
    if assignment == "rrs":
        semantic = assign_semantic_ids_rrs(
            model, item_dataset, device, k=rrs_k, batch_size=batch_size
        ).cpu()
        dedup = _assign_dedup_sequential(semantic)
        n_rrs_unique = int((dedup == 0).sum())
        print(
            f"[corpus_ids] RRS: {n_rrs_unique}/{semantic.shape[0]} items "
            f"({n_rrs_unique / semantic.shape[0]:.1%}) assigned without dedup fallback",
            flush=True,
        )
    else:
        semantic = _compute_semantic_layers(model, item_dataset, device, batch_size)
        if assignment == "popularity":
            if item_popularity is None:
                raise ValueError("item_popularity required for assignment='popularity'")
            if item_popularity.shape[0] != semantic.shape[0]:
                raise ValueError(
                    f"item_popularity length {item_popularity.shape[0]} != "
                    f"corpus size {semantic.shape[0]}"
                )
            dedup = _assign_dedup_popularity(semantic, item_popularity)
        else:
            dedup = _assign_dedup_sequential(semantic)

    return torch.cat([semantic, dedup.unsqueeze(1)], dim=1)
