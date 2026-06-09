import gin
import os
import time
import torch
import numpy as np
from accelerate import Accelerator
from data.processed import ItemData, RecDataset
from data.utils import batch_to, cycle, next_batch
from modules.rq_kmeans import RqKmeans
from modules.rq_opq import RqOpq
from modules.rq_tokenizer import RqTokenizerType
from modules.utils import parse_config
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler


def _log(msg: str) -> None:
    print(msg, flush=True)


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


from modules.tokenizer.corpus_ids import precompute_corpus_semantic_ids


@torch.no_grad()
def semantic_id_stats(corpus_ids: torch.Tensor, codebook_size: int) -> dict:
    n_sem_layers = corpus_ids.shape[1] - 1
    triplets = corpus_ids[:, : min(3, n_sem_layers)].long()
    enc = triplets[:, 0]
    for h in range(1, triplets.shape[1]):
        enc = enc * codebook_size + triplets[:, h]
    unique_triplets, triplet_counts = torch.unique(enc, return_counts=True)

    full = corpus_ids[:, :n_sem_layers].long()
    full_enc = full[:, 0]
    for h in range(1, n_sem_layers):
        full_enc = full_enc * codebook_size + full[:, h]
    unique_full = torch.unique(full_enc)

    stats = {
        "n_items": corpus_ids.shape[0],
        "n_layers": n_sem_layers,
        "unique_triplets": len(unique_triplets),
        "triplet_collision_rate": float((triplet_counts > 1).sum() / len(unique_triplets)),
        "items_in_colliding_triplets": float(
            triplet_counts[triplet_counts > 1].sum() / corpus_ids.shape[0]
        ),
        "unique_full_ids": len(unique_full),
        "full_id_unique_rate": float(len(unique_full) / corpus_ids.shape[0]),
        "max_dedup": int(corpus_ids[:, -1].max()),
        "mean_items_per_triplet": float(triplet_counts.float().mean()),
    }
    for h in range(min(3, n_sem_layers)):
        stats[f"codebook_usage_L{h}"] = (
            len(torch.unique(corpus_ids[:, h])) / codebook_size
        )
    return stats


@gin.configurable
def train(
    tokenizer_type=RqTokenizerType.KMEANS,
    iterations=100000,
    batch_size=1024,
    learning_rate=0.0001,
    weight_decay=0.0001,
    dataset_folder="dataset/yambda",
    dataset=RecDataset.YAMBDA,
    save_dir_root="out/",
    pretrained_path=None,
    split_batches=True,
    amp=False,
    mixed_precision_type="fp16",
    save_model_every=50000,
    eval_every=50000,
    kmeans_init_samples=20000,
    kmeans_refresh_every=10000,
    commitment_weight=0.25,
    vae_n_cat_feats=0,
    vae_input_dim=128,
    vae_embed_dim=32,
    vae_codebook_size=128,
    vae_n_layers=3,
    opq_subspaces=2,
    opq_codes=128,
    dataset_split="500m",
    dataset_interaction="listens_fltrd",
    log_every=100,
    force_dataset_process=False,
    hamr_weight=0.0,
    hamr_R=2,
    hamr_m_full=0.5,
    hamr_m_partial=0.3,
    hamr_lambda_full=1.0,
    hamr_lambda_partial=0.5,
):
    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )
    device = accelerator.device
    type_name = tokenizer_type.name

    _log(
        f"[{type_name}] dataset={dataset.name} split={dataset_split} "
        f"interaction={dataset_interaction} device={device} iters={iterations}"
    )

    train_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        train_test_split="train",
        split=dataset_split,
        interaction=dataset_interaction,
    )
    index_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=False,
        train_test_split="all",
        split=dataset_split,
        interaction=dataset_interaction,
    )
    train_dataloader = cycle(
        DataLoader(
            train_dataset,
            sampler=BatchSampler(RandomSampler(train_dataset), batch_size, False),
            batch_size=None,
            collate_fn=lambda batch: batch,
        )
    )

    if tokenizer_type == RqTokenizerType.KMEANS:
        model = RqKmeans(
            input_dim=vae_input_dim,
            embed_dim=vae_embed_dim,
            codebook_size=vae_codebook_size,
            n_layers=vae_n_layers,
            commitment_weight=commitment_weight,
            n_cat_features=vae_n_cat_feats,
        )
    elif tokenizer_type == RqTokenizerType.OPQ:
        model = RqOpq(
            input_dim=vae_input_dim,
            embed_dim=vae_embed_dim,
            codebook_size=vae_codebook_size,
            rq_layers=vae_n_layers,
            opq_subspaces=opq_subspaces,
            opq_codes=opq_codes,
            commitment_weight=commitment_weight,
            n_cat_features=vae_n_cat_feats,
        )
    else:
        raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start_iter = 0
    if pretrained_path is not None:
        state = torch.load(pretrained_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iter"] + 1
        _log(f"[{type_name}] resumed from {pretrained_path} @ iter {start_iter}")

    model, optimizer = accelerator.prepare(model, optimizer)

    _log(
        f"[{type_name}] corpus train={len(train_dataset)} index={len(index_dataset)} "
        f"embed={vae_embed_dim} codebook={vae_codebook_size}"
    )

    init_idx = torch.randperm(len(train_dataset))[:kmeans_init_samples]
    init_x = torch.stack([train_dataset[int(i)].x for i in init_idx]).to(device)
    with torch.no_grad():
        _unwrap(model).kmeans_init_layers(init_x)
    _log(f"[{type_name}] k-means init done on {kmeans_init_samples} samples")

    t_start = time.time()
    losses = [[], [], [], []]

    for iter in range(start_iter, iterations):
        model.train()
        optimizer.zero_grad()

        data = next_batch(train_dataloader, device)
        with accelerator.autocast():
            if tokenizer_type == RqTokenizerType.OPQ and hamr_weight > 0:
                out = _unwrap(model)(
                    data,
                    hamr_weight=hamr_weight,
                    hamr_R=hamr_R,
                    hamr_m_full=hamr_m_full,
                    hamr_m_partial=hamr_m_partial,
                    hamr_lambda_full=hamr_lambda_full,
                    hamr_lambda_partial=hamr_lambda_partial,
                )
            else:
                out = model(data)
            loss = out.loss

        accelerator.backward(loss)
        optimizer.step()

        losses[0].append(loss.detach().cpu().item())
        losses[1].append(out.reconstruction_loss.detach().cpu().item())
        losses[2].append(out.rq_loss.detach().cpu().item())
        hamr_val = getattr(out, "hamr_loss", None)
        losses[3].append(
            hamr_val.detach().cpu().item() if hamr_val is not None else 0.0
        )
        for bucket in losses:
            del bucket[:-1000]

        if kmeans_refresh_every and iter > 0 and iter % kmeans_refresh_every == 0:
            refresh_idx = torch.randperm(len(train_dataset))[:kmeans_init_samples]
            refresh_x = torch.stack(
                [train_dataset[int(i)].x for i in refresh_idx]
            ).to(device)
            with torch.no_grad():
                _unwrap(model).kmeans_init_layers(refresh_x)
            if accelerator.is_main_process:
                _log(f"[{type_name}] k-means refresh @ iter {iter}")

        if accelerator.is_main_process:
            if (iter + 1) % save_model_every == 0 or iter + 1 == iterations:
                os.makedirs(save_dir_root, exist_ok=True)
                ckpt = os.path.join(save_dir_root, f"checkpoint_{iter}.pt")
                torch.save(
                    {
                        "iter": iter,
                        "model": _unwrap(model).state_dict(),
                        "model_config": _unwrap(model).config,
                        "tokenizer_type": type_name,
                        "optimizer": optimizer.state_dict(),
                    },
                    ckpt,
                )
                _log(f"[{type_name}] saved {ckpt}")

            if (iter + 1) % eval_every == 0 or iter + 1 == iterations:
                corpus_ids = precompute_corpus_semantic_ids(
                    _unwrap(model), index_dataset, device
                )
                stats = semantic_id_stats(corpus_ids, vae_codebook_size)
                _log(
                    f"[{type_name}] iter {iter + 1} stats: "
                    f"unique_triplets={stats['unique_triplets']}/{stats['n_items']} "
                    f"({stats['unique_triplets']/stats['n_items']:.1%}) "
                    f"full_unique={stats['full_id_unique_rate']:.1%} "
                    f"colliding_items={stats['items_in_colliding_triplets']:.1%} "
                    f"max_dedup={stats['max_dedup']} "
                    f"recon={np.mean(losses[1]):.4f} rq={np.mean(losses[2]):.4f}"
                )
                for h in range(min(3, stats["n_layers"])):
                    _log(
                        f"[{type_name}]   L{h} usage="
                        f"{stats[f'codebook_usage_L{h}']:.1%}"
                    )

            if iter % log_every == 0 or iter + 1 == iterations:
                elapsed = time.time() - t_start
                eta = (elapsed / max(iter + 1, 1)) * (iterations - iter - 1)
                hamr_str = (
                    f" hamr={np.mean(losses[3]):.4f}" if hamr_weight > 0 else ""
                )
                _log(
                    f"[{type_name}] iter {iter + 1}/{iterations} "
                    f"loss={np.mean(losses[0]):.4f} recon={np.mean(losses[1]):.4f} "
                    f"rq={np.mean(losses[2]):.4f}{hamr_str} "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
                )

    if accelerator.is_main_process:
        _log(f"[{type_name}] training finished")


if __name__ == "__main__":
    parse_config()
    train()
