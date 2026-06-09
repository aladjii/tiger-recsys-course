"""Hamming-guided Margin Repulsion (HaMR) from QuaSID (arXiv:2603.00632)."""

import torch
from torch import Tensor


def hamr_loss(
    embeddings: Tensor,
    sem_ids: Tensor,
    *,
    hamr_R: int = 2,
    m_full: float = 0.5,
    m_partial: float = 0.3,
    lambda_full: float = 1.0,
    lambda_partial: float = 0.5,
    eps: float = 1e-6,
) -> Tensor:
    """Compute HaMR loss for a batch of item encoder embeddings and semantic IDs.

    Args:
        embeddings: [B, D] continuous encoder vectors (before quantization).
        sem_ids: [B, L] discrete semantic token assignments.
    """
    b = embeddings.shape[0]
    if b < 2:
        return embeddings.new_zeros(())

    e = torch.nn.functional.normalize(embeddings, dim=-1)
    d_cos = 1.0 - e @ e.T

    eq = sem_ids.unsqueeze(1) == sem_ids.unsqueeze(0)
    hamming = (~eq).sum(dim=-1).float()

    eye = torch.eye(b, device=embeddings.device, dtype=torch.bool)
    valid = ~eye

    full_mask = valid & (hamming == 0)
    partial_mask = valid & (hamming > 0) & (hamming <= hamr_R)

    loss = embeddings.new_zeros(())
    n_full = full_mask.sum()
    if n_full > 0:
        full_hinge = torch.clamp(m_full - d_cos[full_mask], min=0.0)
        loss = loss + lambda_full * full_hinge.sum() / (n_full + eps)

    n_partial = partial_mask.sum()
    if n_partial > 0:
        partial_hinge = torch.clamp(m_partial - d_cos[partial_mask], min=0.0)
        loss = loss + lambda_partial * partial_hinge.sum() / (n_partial + eps)

    return loss
