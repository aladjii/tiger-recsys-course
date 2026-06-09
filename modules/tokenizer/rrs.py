"""Recursive Residual Searching (RRS) for purely semantic ID assignment.

See: Purely Semantic Indexing (arXiv:2509.16446).
"""

from __future__ import annotations

import torch
from torch import Tensor


def _topk_rq_ids(layer, residual: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Return top-k centroid ids and embeddings for one RQ layer (single item)."""
    codebook = layer.codebook.weight
    x = residual.unsqueeze(0)
    dist = (
        (x**2).sum(dim=1, keepdim=True)
        + (codebook**2).sum(dim=1).unsqueeze(0)
        - 2 * x @ codebook.T
    )
    k = min(k, codebook.shape[0])
    _, top_ids = dist.squeeze(0).topk(k, largest=False)
    embs = codebook[top_ids]
    return top_ids, embs


def _topk_opq_subspace(opq_layer, sub: Tensor, m: int, k: int) -> tuple[Tensor, Tensor]:
    """Return top-k ids and embeddings for one OPQ subspace (single item)."""
    from modules.rq_opq import _orthogonalize

    rot = _orthogonalize(opq_layer.rotations[m])
    sub_r = sub.unsqueeze(0) @ rot
    codebook = opq_layer.codebooks[m]
    dist = (
        (sub_r**2).sum(dim=1, keepdim=True)
        + (codebook**2).sum(dim=1).unsqueeze(0)
        - 2 * sub_r @ codebook.T
    )
    k = min(k, codebook.shape[0])
    _, top_ids = dist.squeeze(0).topk(k, largest=False)
    emb_r = codebook[top_ids]
    emb = emb_r @ rot.T
    return top_ids, emb.squeeze(0)


def _nearest_semantic_ids(model, emb_i: Tensor) -> list[int]:
    res = emb_i.clone()
    ids: list[int] = []
    for layer in model.rq_levels:
        emb, cid = layer.assign(res.unsqueeze(0))
        ids.append(int(cid.item()))
        res = res - emb.squeeze(0)
    _, opq_ids, _ = model.opq(res.unsqueeze(0))
    for m in range(model.opq_subspaces):
        ids.append(int(opq_ids[0, m].item()))
    return ids


@torch.no_grad()
def assign_semantic_ids_rrs_opq(
    model,
    x: Tensor,
    k: int = 4,
    log_every: int = 50000,
) -> Tensor:
    """Assign unique 5-layer semantic IDs via RRS; dedup column appended separately."""
    model.eval()
    n_items = x.shape[0]
    z = model.project_in(x.to(model.project_in.weight.dtype))
    n_rq = len(model.rq_levels)
    n_opq = model.opq_subspaces
    n_layers = n_rq + n_opq

    assigned: set[tuple[int, ...]] = set()
    result = torch.empty((n_items, n_layers), dtype=torch.long)
    n_greedy = 0
    n_dfs = 0
    n_fallback = 0

    for idx in range(n_items):
        if log_every and idx > 0 and idx % log_every == 0:
            print(
                f"[rrs] {idx}/{n_items} items "
                f"(greedy={n_greedy} dfs={n_dfs} fallback={n_fallback})",
                flush=True,
            )

        emb_i = z[idx]
        nearest = tuple(_nearest_semantic_ids(model, emb_i))
        if nearest not in assigned:
            assigned.add(nearest)
            result[idx] = torch.tensor(nearest, dtype=torch.long)
            n_greedy += 1
            continue

        residual = emb_i.clone()
        prefix: list[int] = []
        found = False

        def dfs_rq(level: int, res: Tensor) -> bool:
            nonlocal prefix, found, residual
            if level < n_rq:
                top_ids, top_embs = _topk_rq_ids(model.rq_levels[level], res, k)
                for cid, emb in zip(top_ids.tolist(), top_embs):
                    prefix.append(int(cid))
                    new_res = res - emb
                    if dfs_rq(level + 1, new_res):
                        return True
                    prefix.pop()
                return False

            # OPQ levels: try all top-k combinations for subspaces
            return dfs_opq(0, res)

        def dfs_opq(sub_idx: int, res: Tensor) -> bool:
            nonlocal prefix, found
            if sub_idx < n_opq:
                sub = res[sub_idx * model.opq.sub_dim : (sub_idx + 1) * model.opq.sub_dim]
                top_ids, _ = _topk_opq_subspace(model.opq, sub, sub_idx, k)
                for cid in top_ids.tolist():
                    prefix.append(int(cid))
                    if dfs_opq(sub_idx + 1, res):
                        return True
                    prefix.pop()
                return False

            key = tuple(prefix)
            if key not in assigned:
                assigned.add(key)
                result[idx] = torch.tensor(prefix, dtype=torch.long)
                found = True
                return True
            return False

        if dfs_rq(0, residual):
            n_dfs += 1
        else:
            result[idx] = torch.tensor(nearest, dtype=torch.long)
            n_fallback += 1

    print(
        f"[rrs] done: greedy={n_greedy} dfs={n_dfs} fallback={n_fallback} "
        f"({n_greedy / n_items:.1%} unique via nearest)",
        flush=True,
    )
    return result


@torch.no_grad()
def assign_semantic_ids_rrs(
    model,
    item_dataset,
    device,
    k: int = 4,
    batch_size: int = 2048,
    log_every: int = 50000,
) -> Tensor:
    """Batch-load embeddings then run per-item RRS."""
    xs = []
    for start in range(0, len(item_dataset), batch_size):
        end = min(start + batch_size, len(item_dataset))
        batch = torch.stack([item_dataset[i].x for i in range(start, end)])
        xs.append(batch)
    x_all = torch.cat(xs, dim=0).to(device)
    print(f"[rrs] starting RRS on {x_all.shape[0]} items (k={k})", flush=True)
    return assign_semantic_ids_rrs_opq(model, x_all, k=k, log_every=log_every)
