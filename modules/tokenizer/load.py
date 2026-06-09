"""Load RQ-VAE / RQ-KMeans / RQ-OPQ tokenizer checkpoints."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from modules.rq_kmeans import RqKmeans
from modules.rq_opq import RqOpq
from modules.rqvae import RqVae


def load_rq_tokenizer(
    checkpoint_path: str,
    device: torch.device | str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a tokenizer checkpoint and return (model, metadata)."""
    map_loc = device if device is not None else "cpu"
    state = torch.load(checkpoint_path, map_location=map_loc, weights_only=False)
    cfg = state.get("model_config", {})
    kind = state.get("tokenizer_type", "RQVAE")

    if kind == "OPQ":
        model = RqOpq(
            input_dim=cfg["input_dim"],
            embed_dim=cfg["embed_dim"],
            codebook_size=cfg["codebook_size"],
            rq_layers=cfg["rq_layers"],
            opq_subspaces=cfg["opq_subspaces"],
            opq_codes=cfg["opq_codes"],
            commitment_weight=cfg.get("commitment_weight", 0.25),
            n_cat_features=cfg.get("n_cat_features", 0),
        )
        n_sem_layers = cfg["rq_layers"] + cfg["opq_subspaces"]
    elif kind == "KMEANS":
        model = RqKmeans(
            input_dim=cfg["input_dim"],
            embed_dim=cfg["embed_dim"],
            codebook_size=cfg["codebook_size"],
            n_layers=cfg["n_layers"],
            commitment_weight=cfg.get("commitment_weight", 0.25),
            n_cat_features=cfg.get("n_cat_features", 0),
        )
        n_sem_layers = cfg["n_layers"]
    else:
        model = RqVae(
            input_dim=cfg["input_dim"],
            embed_dim=cfg["embed_dim"],
            hidden_dims=cfg["hidden_dims"],
            codebook_size=cfg["codebook_size"],
            codebook_kmeans_init=False,
            codebook_normalize=cfg.get("codebook_normalize", False),
            codebook_sim_vq=cfg.get("codebook_sim_vq", False),
            n_layers=cfg["n_layers"],
            n_cat_features=cfg.get("n_cat_features", 0),
            commitment_weight=cfg.get("commitment_weight", 0.25),
        )
        n_sem_layers = cfg["n_layers"]

    model.load_state_dict(state["model"])
    if device is not None:
        model.to(device)
    model.eval()

    meta = {
        "tokenizer_type": kind,
        "iter": state.get("iter", -1),
        "codebook_size": cfg.get("codebook_size", 128),
        "n_sem_layers": n_sem_layers,
        "input_dim": cfg.get("input_dim"),
        "embed_dim": cfg.get("embed_dim"),
        "hidden_dims": cfg.get("hidden_dims"),
    }
    return model, meta
