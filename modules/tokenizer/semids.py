import torch

from data.processed import ItemData
from data.schemas import SeqBatch
from data.schemas import TokenizedSeqBatch
from einops import rearrange
from modules.tokenizer.corpus_ids import AssignmentPolicy
from modules.tokenizer.corpus_ids import precompute_corpus_semantic_ids
from modules.tokenizer.corpus_ids import sem_ids_as_batch as _normalize_sem_ids
from modules.tokenizer.load import load_rq_tokenizer
from modules.utils import eval_mode
from typing import List, Optional
from torch import Tensor, nn


class SemanticIdTokenizer(nn.Module):
    """Tokenize item features / sequences into semantic ID token sequences."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        n_layers: int = 3,
        n_cat_feats: int = 18,
        commitment_weight: float = 0.25,
        rqvae_weights_path: Optional[str] = None,
        rqvae_codebook_normalize: bool = False,
        rqvae_sim_vq: bool = False,
    ) -> None:
        super().__init__()

        self.codebook_size = codebook_size
        self.n_layers = n_layers
        self.tokenizer_type = "RQVAE"

        if rqvae_weights_path is not None:
            self.rq_model, meta = load_rq_tokenizer(rqvae_weights_path)
            self.tokenizer_type = meta["tokenizer_type"]
            self.n_layers = meta["n_sem_layers"]
            self.codebook_size = meta["codebook_size"]
            print(f"---Loaded {self.tokenizer_type} Iter {meta['iter']}---")
        else:
            from modules.rqvae import RqVae

            self.rq_model = RqVae(
                input_dim=input_dim,
                embed_dim=output_dim,
                hidden_dims=hidden_dims,
                codebook_size=codebook_size,
                codebook_kmeans_init=False,
                codebook_normalize=rqvae_codebook_normalize,
                codebook_sim_vq=rqvae_sim_vq,
                n_layers=n_layers,
                n_cat_features=n_cat_feats,
                commitment_weight=commitment_weight,
            )

        self.rq_model.eval()
        self.cached_ids = None

    @property
    def rq_vae(self):
        """Backward-compatible alias (decoder train may call push_to_hub)."""
        return self.rq_model

    def reset(self):
        self.cached_ids = None

    @property
    def sem_ids_dim(self):
        return self.n_layers + 1

    @torch.no_grad()
    @eval_mode
    def precompute_corpus_ids(
        self,
        movie_dataset: ItemData,
        *,
        assignment: AssignmentPolicy = "sequential",
        item_popularity: Tensor | None = None,
        rrs_k: int = 4,
    ) -> Tensor:
        device = next(self.rq_model.parameters()).device
        self.cached_ids = precompute_corpus_semantic_ids(
            self.rq_model,
            movie_dataset,
            device,
            assignment=assignment,
            item_popularity=item_popularity,
            rrs_k=rrs_k,
        ).to(device)
        return self.cached_ids

    def _tokenize_seq_batch_from_cached(self, ids: Tensor) -> Tensor:
        return rearrange(
            self.cached_ids[ids.flatten(), :], "(b n) d -> b (n d)", n=ids.shape[1]
        )

    @torch.no_grad()
    @eval_mode
    def forward(self, batch: SeqBatch) -> TokenizedSeqBatch:
        if self.cached_ids is None or batch.ids.max() >= self.cached_ids.shape[0]:
            B, N = batch.ids.shape
            x = batch.x
            if x.dim() == 1:
                x = x.unsqueeze(0)
            elif x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            sem_ids = _normalize_sem_ids(
                self.rq_model.get_semantic_ids(x).sem_ids, x.shape[0]
            )
            if sem_ids.shape[0] == B * N:
                D = sem_ids.shape[1]
                sem_ids = sem_ids.view(B, N * D)
            else:
                D = sem_ids.shape[1]
            seq_mask, sem_ids_fut = None, None
        else:
            B, N = batch.ids.shape
            _, D = self.cached_ids.shape
            sem_ids = self._tokenize_seq_batch_from_cached(batch.ids)
            seq_mask = batch.seq_mask.repeat_interleave(D, dim=1)
            sem_ids[~seq_mask] = -1

            sem_ids_fut = self._tokenize_seq_batch_from_cached(batch.ids_fut)

        token_type_ids = torch.arange(D, device=sem_ids.device).repeat(B, N)
        token_type_ids_fut = torch.arange(D, device=sem_ids.device).repeat(B, 1)
        return TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=sem_ids,
            sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask,
            token_type_ids=token_type_ids,
            token_type_ids_fut=token_type_ids_fut,
        )
