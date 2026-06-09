from typing import Dict, Tuple

import torch
import torch.nn as nn


def create_masked_tensor(data: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a batch of variable-length sequences into a padded tensor and corresponding mask.

    Args:
        data (torch.Tensor): Input tensor containing flattened sequences.
            - For indices: shape (total_elements,) of dtype long
            - For embeddings: shape (total_elements, embedding_dim)
        lengths (torch.Tensor): 1D tensor of sequence lengths, shape (batch_size,)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - padded_tensor: Padded tensor of shape:
                - (batch_size, max_seq_len) for indices
                - (batch_size, max_seq_len, embedding_dim) for embeddings
            - mask: Boolean mask of shape (batch_size, max_seq_len) where True indicates valid elements

    Note:
        - Zero-padding is added to the right of shorter sequences
    """
    batch_size = lengths.shape[0]
    max_sequence_length = int(lengths.max().item())

    if len(data.shape) == 1:  # indices
        padded_tensor = torch.zeros(
            batch_size, max_sequence_length, dtype=data.dtype, device=data.device
        )  # (batch_size, max_seq_len)
    else:
        assert len(data.shape) == 2  # embeddings
        padded_tensor = torch.zeros(
            batch_size, max_sequence_length, *data.shape[1:], dtype=data.dtype, device=data.device
        )  # (batch_size, max_seq_len, embedding_dim)

    mask = (
        torch.arange(end=max_sequence_length, device=lengths.device)[None] < lengths[:, None]
    )  # (batch_size, max_seq_len)

    padded_tensor[mask] = data

    return padded_tensor, mask


class SASRecEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_sequence_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU(),
        layer_norm_eps: float = 1e-9,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        self._num_items = num_items
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim

        self._item_embeddings = nn.Embedding(
            num_embeddings=num_items + 1,  # add zero id embedding
            embedding_dim=embedding_dim,
        )
        self._position_embeddings = nn.Embedding(num_embeddings=max_sequence_length, embedding_dim=embedding_dim)

        self._layernorm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self._dropout = nn.Dropout(dropout)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward or 4 * embedding_dim,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        self._encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers)

        self._init_weights(initializer_range)

    @property
    def item_embeddings(self) -> nn.Module:
        return self._item_embeddings

    @property
    def num_items(self) -> int:
        return self._num_items

    def _apply_sequential_encoder(self, events: torch.Tensor, lengths: torch.Tensor):
        """
        Processes variable-length event sequences through a transformer encoder with positional embeddings.

        Args:
            events (torch.Tensor): Flattened tensor of event indices, shape (total_events,)
            lengths (torch.Tensor): 1D tensor of sequence lengths, shape (batch_size,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - embeddings: Processed sequence embeddings, shape (batch_size, seq_len, embedding_dim)
                - mask: Boolean mask indicating valid elements, shape (batch_size, seq_len)

        Processing Steps:
            1. Embedding Lookup:
                - Converts event indices to dense embeddings
            2. Positional Encoding:
                - Generates reverse-order positions (newest event first)
                - Adds positional embeddings to item embeddings
            3. Transformer Processing:
                - Applies layer norm and dropout
                - Uses causal attention mask for autoregressive modeling
                - Uses padding mask to ignore invalid positions

        Note:
            - Position indices are generated in reverse chronological order (newest event = position 0)
        """
        embeddings = self._item_embeddings(events)  # (total_batch_events, embedding_dim)

        embeddings, mask = create_masked_tensor(
            data=embeddings, lengths=lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        positions = (
            torch.arange(start=seq_len - 1, end=-1, step=-1, device=mask.device)[None].tile([batch_size, 1]).long()
        )  # (batch_size, seq_len)
        positions_mask = positions < lengths[:, None]  # (batch_size, max_seq_len)

        positions = positions[positions_mask]  # (total_batch_events)
        position_embeddings = self._position_embeddings(positions)  # (total_batch_events, embedding_dim)
        position_embeddings, _ = create_masked_tensor(
            data=position_embeddings, lengths=lengths
        )  # (batch_size, seq_len, embedding_dim)

        embeddings = embeddings + position_embeddings  # (batch_size, seq_len, embedding_dim)
        embeddings = self._layernorm(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings = self._dropout(embeddings)  # (batch_size, seq_len, embedding_dim)
        embeddings[~mask] = 0

        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool().to(mask.device)  # (seq_len, seq_len)
        embeddings = self._encoder(
            src=embeddings, mask=~causal_mask, src_key_padding_mask=~mask
        )  # (batch_size, seq_len, embedding_dim)

        return embeddings, mask

    @torch.no_grad()
    def _init_weights(self, initializer_range: float) -> None:
        """
        Initialize all model parameters (weights and biases) in-place.

        For each parameter in the model:
            - If the parameter name contains 'weight':
                - If it also contains 'norm' (e.g., for normalization layers), initialize with ones.
                - Otherwise, initialize with a truncated normal distribution (mean=0, std=initializer_range)
                and values clipped to the range [-2 * initializer_range, 2 * initializer_range].
            - If the parameter name contains 'bias', initialize with zeros.
            - If the parameter name does not match either case, raise a ValueError.

        Args:
            initializer_range (float): Standard deviation for the truncated normal distribution
                used to initialize non-normalization weights.

        Note:
            This method should be called during model initialization to ensure all weights and biases
            are properly set. It runs in a no-grad context and does not track gradients.
        """
        for key, value in self.named_parameters():
            if 'weight' in key:
                if 'norm' in key:
                    nn.init.ones_(value.data)
                else:
                    nn.init.trunc_normal_(
                        value.data, std=initializer_range, a=-2 * initializer_range, b=2 * initializer_range
                    )
            else:
                assert 'bias' in key
                nn.init.zeros_(value.data)

    @staticmethod
    def _get_last_embedding(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Extracts the embedding of the last valid (non-padded) element from each sequence in a batch.

        Args:
            embeddings (torch.Tensor): Tensor of shape (batch_size, seq_len, embedding_dim)
                containing embeddings for each element in each sequence.
            mask (torch.Tensor): Boolean tensor of shape (batch_size, seq_len) indicating
                valid (True) and padded (False) positions in each sequence.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, embedding_dim) containing the embedding
                of the last valid element for each sequence in the batch.
        """
        flatten_embeddings = embeddings[mask]  # (total_batch_events, embedding_dim)
        lengths = torch.sum(mask, dim=-1)  # (batch_size)
        offsets = torch.cumsum(lengths, dim=0)  # (batch_size)
        last_embeddings = flatten_embeddings[offsets.long() - 1]  # (batch_size, embedding_dim)
        return last_embeddings

    def forward(self, inputs: Dict) -> torch.Tensor:
        """
        Forward pass of the model, handling both training and evaluation modes.

        Args:
            inputs (Dict): Input dictionary containing:
                - 'item.ids' (torch.LongTensor): Flattened tensor of item IDs for all sequences in the batch.
                    Shape: (total_batch_events,)
                - 'item.length' (torch.LongTensor): Sequence lengths for each sample in the batch.
                    Shape: (batch_size,)
                - 'positive.ids' (torch.LongTensor, training only): Positive sample IDs for contrastive learning.
                    Shape: (total_batch_events,)
                - 'negative.ids' (torch.LongTensor, training only): Negative sample IDs for contrastive learning.
                    Shape: (total_batch_events,)

        Returns:
            torch.Tensor:
                - During training: Binary cross-entropy loss between positive/negative sample scores.
                    Shape: (1,)
                - During evaluation: Embeddings of the last valid item in each sequence.
                    Shape: (batch_size, embedding_dim)
        """
        all_sample_events = inputs['item.ids']  # (total_batch_events)
        all_sample_lengths = inputs['item.length']  # (batch_size)

        embeddings, mask = self._apply_sequential_encoder(
            all_sample_events, all_sample_lengths
        )  # (batch_size, seq_len, embedding_dim), (batch_size, seq_len)

        if self.training:  # training mode
            # queries
            in_batch_queries_embeddings = embeddings[mask]  # (total_batch_events, embedding_dim)

            # positives
            in_batch_positive_events = inputs['positive.ids']  # (total_batch_events)
            in_batch_positive_embeddings = self._item_embeddings(
                in_batch_positive_events
            )  # (total_batch_events, embedding_dim)
            positive_scores = torch.einsum(
                'bd,bd->b', in_batch_queries_embeddings, in_batch_positive_embeddings
            )  # (total_batch_events)

            # negatives
            in_batch_negative_events = inputs['negative.ids']  # (total_batch_events)
            in_batch_negative_embeddings = self._item_embeddings(
                in_batch_negative_events
            )  # (total_batch_events, embedding_dim)
            negative_scores = torch.einsum(
                'bd,bd->b', in_batch_queries_embeddings, in_batch_negative_embeddings
            )  # (total_batch_events)

            loss = nn.functional.binary_cross_entropy_with_logits(
                torch.cat([positive_scores, negative_scores], dim=0),
                torch.cat([torch.ones_like(positive_scores), torch.zeros_like(negative_scores)]),
            )  # (1)

            return loss
        else:  # eval mode
            last_embeddings = self._get_last_embedding(embeddings, mask)  # (batch_size, embedding_dim)
            return last_embeddings
