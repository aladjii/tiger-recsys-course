import torch

from data.schemas import SeqBatch
from einops import rearrange
from init.kmeans import kmeans_init_
from modules.loss import QuantizeLoss
from modules.normalize import l2norm
from typing import List, NamedTuple
from torch import Tensor, nn


class RqKmeansOutput(NamedTuple):
    embeddings: Tensor
    residuals: Tensor
    sem_ids: Tensor
    quantize_loss: Tensor


class RqKmeansLosses(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rq_loss: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor


class ResidualKMeansLayer(nn.Module):
    """Single RQ level with L2 nearest-centroid assignment and STE."""

    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.codebook = nn.Embedding(n_embed, embed_dim)
        self.quantize_loss = QuantizeLoss(commitment_weight)
        self.kmeans_initted = False
        nn.init.uniform_(self.codebook.weight, -1.0 / n_embed, 1.0 / n_embed)

    @property
    def weight(self) -> Tensor:
        return self.codebook.weight

    @torch.no_grad()
    def kmeans_init(self, x: Tensor) -> None:
        kmeans_init_(self.codebook.weight, x=x)
        self.kmeans_initted = True

    def assign(self, x: Tensor) -> tuple[Tensor, Tensor]:
        codebook = self.codebook.weight
        dist = (
            (x**2).sum(dim=1, keepdim=True)
            + (codebook**2).sum(dim=1).unsqueeze(0)
            - 2 * x @ codebook.T
        )
        ids = dist.argmin(dim=1)
        emb = self.codebook(ids)
        return emb, ids

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        emb, ids = self.assign(x)
        if self.training:
            emb_out = x + (emb - x).detach()
            loss = self.quantize_loss(query=x, value=emb)
        else:
            emb_out = emb
            loss = self.quantize_loss(query=x, value=emb)
        return emb_out, ids, loss


class RqKmeans(nn.Module):
    """Residual K-Means tokenizer on item embeddings (OneSearch / LETTER style).

    A lightweight linear map projects item vectors into a latent space; each RQ
    level assigns the current residual to the nearest codebook centroid (with
    STE during training). Reconstruction is the sum of selected codes mapped
    back to input space.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        codebook_size: int,
        n_layers: int = 3,
        commitment_weight: float = 0.25,
        n_cat_features: int = 0,
    ) -> None:
        self._config = locals()
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        self.n_layers = n_layers
        self.n_cat_feats = n_cat_features

        self.project_in = nn.Linear(input_dim, embed_dim, bias=False)
        self.project_out = nn.Linear(embed_dim, input_dim, bias=False)
        self.layers = nn.ModuleList(
            [
                ResidualKMeansLayer(
                    embed_dim=embed_dim,
                    n_embed=codebook_size,
                    commitment_weight=commitment_weight,
                )
                for _ in range(n_layers)
            ]
        )
        self.reconstruction_loss = nn.MSELoss()

    @property
    def config(self) -> dict:
        return self._config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state["model"])
        print(f"---Loaded RQ-KMeans Iter {state['iter']}---")

    @torch.no_grad()
    def kmeans_init_layers(self, x: Tensor) -> None:
        z = self.project_in(x)
        residual = z
        for layer in self.layers:
            layer.kmeans_init(residual)
            emb, _ = layer.assign(residual)
            residual = residual - emb

    def get_semantic_ids(self, x: Tensor) -> RqKmeansOutput:
        x = x.to(self.project_in.weight.dtype)
        z = self.project_in(x)
        residual = z

        quantize_loss = 0
        embs, residuals, sem_ids = [], [], []

        for layer in self.layers:
            residuals.append(residual)
            emb, ids, loss = layer(residual)
            quantize_loss = quantize_loss + loss
            residual = residual - emb
            embs.append(emb)
            sem_ids.append(ids)

        return RqKmeansOutput(
            embeddings=rearrange(torch.stack(embs, dim=0), "h b d -> h d b"),
            residuals=rearrange(torch.stack(residuals, dim=0), "h b d -> h d b"),
            sem_ids=torch.stack(sem_ids, dim=1).T,
            quantize_loss=quantize_loss,
        )

    def forward(self, batch: SeqBatch) -> RqKmeansLosses:
        x = batch.x
        quantized = self.get_semantic_ids(x)
        x_hat_z = quantized.embeddings.sum(dim=0).T
        x_hat = self.project_out(x_hat_z)
        if self.n_cat_feats:
            x_hat = torch.cat(
                [l2norm(x_hat[..., : -self.n_cat_feats]), x_hat[..., -self.n_cat_feats :]],
                dim=-1,
            )

        recon_loss = self.reconstruction_loss(x_hat, x)
        rq_loss = quantized.quantize_loss
        loss = (recon_loss + rq_loss).mean()

        with torch.no_grad():
            embs_norm = quantized.embeddings.norm(dim=1)
            sem_ids_t = quantized.sem_ids.T
            p_unique_ids = (
                ~torch.triu(
                    (
                        rearrange(sem_ids_t, "b d -> b 1 d")
                        == rearrange(sem_ids_t, "b d -> 1 b d")
                    ).all(axis=-1),
                    diagonal=1,
                )
            ).all(axis=1).sum() / sem_ids_t.shape[0]

        return RqKmeansLosses(
            loss=loss,
            reconstruction_loss=recon_loss.mean(),
            rq_loss=rq_loss.mean(),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
        )
