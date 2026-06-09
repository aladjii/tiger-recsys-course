import torch

from data.schemas import SeqBatch
from einops import rearrange
from init.kmeans import kmeans_init_
from modules.hamr_loss import hamr_loss
from modules.loss import QuantizeLoss
from modules.normalize import l2norm
from modules.rq_kmeans import ResidualKMeansLayer, RqKmeansOutput
from typing import NamedTuple
from torch import Tensor, nn


class RqOpqLosses(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rq_loss: Tensor
    hamr_loss: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor


def _orthogonalize(matrix: Tensor) -> Tensor:
    q, _ = torch.linalg.qr(matrix)
    return q


class OpqLayer(nn.Module):
    """Optimized Product Quantization on a residual (Faiss / OneSearch style).

    Splits the vector into ``n_subspaces`` sub-vectors, applies a learned
    orthogonal rotation per subspace, and assigns each to its own codebook.
    """

    def __init__(
        self,
        embed_dim: int,
        n_subspaces: int = 2,
        n_codes: int = 128,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if embed_dim % n_subspaces != 0:
            raise ValueError("embed_dim must be divisible by n_subspaces")
        self.embed_dim = embed_dim
        self.n_subspaces = n_subspaces
        self.n_codes = n_codes
        self.sub_dim = embed_dim // n_subspaces
        self.rotations = nn.ParameterList(
            [nn.Parameter(torch.eye(self.sub_dim)) for _ in range(n_subspaces)]
        )
        self.codebooks = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(n_codes, self.sub_dim).uniform_(
                        -1.0 / n_codes, 1.0 / n_codes
                    )
                )
                for _ in range(n_subspaces)
            ]
        )
        self.quantize_loss = QuantizeLoss(commitment_weight)
        self.kmeans_initted = False

    @torch.no_grad()
    def kmeans_init(self, x: Tensor) -> None:
        for m in range(self.n_subspaces):
            sub = x[:, m * self.sub_dim : (m + 1) * self.sub_dim]
            kmeans_init_(self.codebooks[m], x=sub)
        self.kmeans_initted = True

    def _assign_subspace(self, sub: Tensor, m: int) -> tuple[Tensor, Tensor]:
        rot = _orthogonalize(self.rotations[m])
        sub_r = sub @ rot
        codebook = self.codebooks[m]
        dist = (
            (sub_r**2).sum(dim=1, keepdim=True)
            + (codebook**2).sum(dim=1).unsqueeze(0)
            - 2 * sub_r @ codebook.T
        )
        ids = dist.argmin(dim=1)
        emb_r = codebook[ids]
        emb = emb_r @ rot.T
        return emb, ids

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        recon_parts = []
        ids_list = []
        loss = 0
        for m in range(self.n_subspaces):
            sub = x[:, m * self.sub_dim : (m + 1) * self.sub_dim]
            emb, ids = self._assign_subspace(sub, m)
            if self.training:
                emb_out = sub + (emb - sub).detach()
            else:
                emb_out = emb
            loss = loss + self.quantize_loss(query=sub, value=emb)
            recon_parts.append(emb_out)
            ids_list.append(ids)

        recon = torch.cat(recon_parts, dim=1)
        sem_ids = torch.stack(ids_list, dim=1)
        return recon, sem_ids, loss


class RqOpq(nn.Module):
    """Hybrid RQ-KMeans + OPQ tokenizer (Kuaishou OneSearch / COINS style).

    Three coarse RQ-KMeans layers capture shared hierarchical structure; two
    OPQ sub-codes quantize the remaining residual for fine-grained uniqueness.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        codebook_size: int = 128,
        rq_layers: int = 3,
        opq_subspaces: int = 2,
        opq_codes: int = 128,
        commitment_weight: float = 0.25,
        n_cat_features: int = 0,
    ) -> None:
        self._config = locals()
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        self.rq_layers = rq_layers
        self.opq_subspaces = opq_subspaces
        self.opq_codes = opq_codes
        self.n_semantic_layers = rq_layers + opq_subspaces
        self.n_cat_feats = n_cat_features

        self.project_in = nn.Linear(input_dim, embed_dim, bias=False)
        self.project_out = nn.Linear(embed_dim, input_dim, bias=False)
        self.rq_levels = nn.ModuleList(
            [
                ResidualKMeansLayer(
                    embed_dim=embed_dim,
                    n_embed=codebook_size,
                    commitment_weight=commitment_weight,
                )
                for _ in range(rq_layers)
            ]
        )
        self.opq = OpqLayer(
            embed_dim=embed_dim,
            n_subspaces=opq_subspaces,
            n_codes=opq_codes,
            commitment_weight=commitment_weight,
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
        print(f"---Loaded RQ-OPQ Iter {state['iter']}---")

    @torch.no_grad()
    def kmeans_init_layers(self, x: Tensor) -> None:
        z = self.project_in(x)
        residual = z
        for layer in self.rq_levels:
            layer.kmeans_init(residual)
            emb, _ = layer.assign(residual)
            residual = residual - emb
        self.opq.kmeans_init(residual)

    def get_semantic_ids(self, x: Tensor) -> RqKmeansOutput:
        x = x.to(self.project_in.weight.dtype)
        z = self.project_in(x)
        residual = z

        quantize_loss = 0
        embs, residuals, sem_ids = [], [], []

        for layer in self.rq_levels:
            residuals.append(residual)
            emb, ids, loss = layer(residual)
            quantize_loss = quantize_loss + loss
            residual = residual - emb
            embs.append(emb)
            sem_ids.append(ids)

        opq_recon, opq_ids, opq_loss = self.opq(residual)
        quantize_loss = quantize_loss + opq_loss
        embs.append(opq_recon)
        for m in range(self.opq_subspaces):
            sem_ids.append(opq_ids[:, m])

        return RqKmeansOutput(
            embeddings=rearrange(torch.stack(embs, dim=0), "h b d -> h d b"),
            residuals=rearrange(torch.stack(residuals, dim=0), "h b d -> h d b"),
            sem_ids=torch.stack(sem_ids, dim=1).T,
            quantize_loss=quantize_loss,
        )

    def forward(
        self,
        batch: SeqBatch,
        hamr_weight: float = 0.0,
        hamr_R: int = 2,
        hamr_m_full: float = 0.5,
        hamr_m_partial: float = 0.3,
        hamr_lambda_full: float = 1.0,
        hamr_lambda_partial: float = 0.5,
    ) -> RqOpqLosses:
        x = batch.x
        z = self.project_in(x.to(self.project_in.weight.dtype))
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
        hamr = hamr_loss(
            z,
            quantized.sem_ids.T,
            hamr_R=hamr_R,
            m_full=hamr_m_full,
            m_partial=hamr_m_partial,
            lambda_full=hamr_lambda_full,
            lambda_partial=hamr_lambda_partial,
        )
        loss = (recon_loss + rq_loss).mean()
        if hamr_weight > 0:
            loss = loss + hamr_weight * hamr

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

        return RqOpqLosses(
            loss=loss,
            reconstruction_loss=recon_loss.mean(),
            rq_loss=rq_loss.mean(),
            hamr_loss=hamr.detach(),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
        )
