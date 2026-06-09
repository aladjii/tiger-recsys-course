"""Mixed-radix encoding for semantic ID hierarchies (RQ layers + dedup column)."""

from torch import Tensor


def encode_prefix_columns(prefix: Tensor, codebook_size: int) -> Tensor:
    """Encode the first k semantic layers (no dedup column)."""
    prefix = prefix.long()
    enc = prefix[..., 0]
    for h in range(1, prefix.shape[-1]):
        enc = enc * codebook_size + prefix[..., h]
    return enc


def encode_sem_id_columns(codes: Tensor, codebook_size: int, dedup_vocab: int) -> Tensor:
    """Encode full semantic ID: n semantic layers + final dedup column."""
    codes = codes.long()
    n_sem = codes.shape[-1] - 1
    enc = codes[..., 0]
    for h in range(1, n_sem):
        enc = enc * codebook_size + codes[..., h]
    return enc * dedup_vocab + codes[..., -1]


def encode_sem_id_scalar(codes: Tensor, codebook_size: int, dedup_vocab: int) -> int:
    return int(encode_sem_id_columns(codes.unsqueeze(0), codebook_size, dedup_vocab).item())
