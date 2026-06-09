import gin
import os
import sys
import time
import torch
import numpy as np
from accelerate import Accelerator
from data.processed import ItemData
from data.processed import RecDataset
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import parse_config
from torch.optim import AdamW
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler


def _log(msg: str) -> None:
    print(msg, flush=True)


@gin.configurable
def train(
    iterations=50000,
    batch_size=64,
    learning_rate=0.0001,
    weight_decay=0.01,
    dataset_folder="dataset/yambda",
    dataset=RecDataset.YAMBDA,
    pretrained_rqvae_path=None,
    save_dir_root="out/",
    use_kmeans_init=True,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    do_eval=True,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    eval_every=50000,
    commitment_weight=0.25,
    vae_n_cat_feats=18,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_codebook_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
    vae_sim_vq=False,
    vae_n_layers=3,
    dataset_split="500m",
    dataset_interaction="likes",
    log_every=100,
):
    if wandb_logging:
        params = locals()

    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        os.environ.setdefault("ACCELERATE_USE_MPS_DEVICE", "True")
        if amp:
            print(
                "Warning: MPS does not support mixed precision training. Disabling amp."
            )
            amp = False

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )

    device = accelerator.device

    _log(
        f"[rqvae] dataset={dataset.name} split={dataset_split} interaction={dataset_interaction} "
        f"device={device} iterations={iterations}"
    )

    train_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        train_test_split="train" if do_eval else "all",
        split=dataset_split,
        interaction=dataset_interaction,
    )
    train_sampler = BatchSampler(RandomSampler(train_dataset), batch_size, False)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=None,
        collate_fn=lambda batch: batch,
    )
    train_dataloader = cycle(train_dataloader)

    if do_eval:
        eval_dataset = ItemData(
            root=dataset_folder,
            dataset=dataset,
            force_process=False,
            train_test_split="eval",
            split=dataset_split,
            interaction=dataset_interaction,
        )
        eval_sampler = BatchSampler(RandomSampler(eval_dataset), batch_size, False)
        eval_dataloader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=None,
            collate_fn=lambda batch: batch,
        )

    index_dataset = (
        ItemData(
            root=dataset_folder,
            dataset=dataset,
            force_process=False,
            train_test_split="all",
            split=dataset_split,
            interaction=dataset_interaction,
        )
        if do_eval
        else train_dataset
    )

    _log(
        f"[rqvae] corpus: train={len(train_dataset)} "
        f"eval={len(eval_dataset) if do_eval else 0} "
        f"index={len(index_dataset)} embed_dim={train_dataset.embed_dim}"
    )
    _log(
        f"[rqvae] model: input={vae_input_dim} hidden={vae_hidden_dims} "
        f"latent={vae_embed_dim} codebook={vae_codebook_size} layers={vae_n_layers}"
    )

    # train_dataloader = accelerator.prepare(train_dataloader)
    # TODO: Investigate bug with prepare eval_dataloader

    model = RqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        codebook_size=vae_codebook_size,
        codebook_kmeans_init=use_kmeans_init and pretrained_rqvae_path is None,
        codebook_normalize=vae_codebook_normalize,
        codebook_sim_vq=vae_sim_vq,
        codebook_mode=vae_codebook_mode,
        n_layers=vae_n_layers,
        n_cat_features=vae_n_cat_feats,
        commitment_weight=commitment_weight,
    )

    optimizer = AdamW(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    if wandb_logging and accelerator.is_main_process:
        import wandb

        wandb.login()
        run = wandb.init(project="rq-vae-training", config=params)

    start_iter = 0
    if pretrained_rqvae_path is not None:
        model.load_pretrained(pretrained_rqvae_path)
        state = torch.load(
            pretrained_rqvae_path, map_location=device, weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        start_iter = state["iter"] + 1

    model, optimizer = accelerator.prepare(model, optimizer)

    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
        rqvae_codebook_normalize=vae_codebook_normalize,
        rqvae_sim_vq=vae_sim_vq,
    )
    tokenizer.rq_vae = model

    if accelerator.is_main_process:
        _log(f"[rqvae] training {start_iter} -> {start_iter + iterations} (log every {log_every})")
        t_start = time.time()

    losses = [[], [], []]
    for iter in range(start_iter, start_iter + 1 + iterations):
            model.train()
            total_loss = 0
            t = 0.2
            if iter == 0 and use_kmeans_init:
                if accelerator.is_main_process:
                    _log("[rqvae] kmeans codebook init...")
                kmeans_init_data = batch_to(
                    train_dataset[torch.arange(min(20000, len(train_dataset)))], device
                )
                with accelerator.autocast():
                    model(kmeans_init_data, t)
                if accelerator.is_main_process:
                    _log("[rqvae] kmeans init done")

            optimizer.zero_grad()
            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)

                with accelerator.autocast():
                    model_output = model(data, gumbel_t=t)
                    loss = model_output.loss
                    loss = loss / gradient_accumulate_every
                    total_loss += loss

            accelerator.backward(total_loss)

            losses[0].append(total_loss.cpu().item())
            losses[1].append(model_output.reconstruction_loss.cpu().item())
            losses[2].append(model_output.rqvae_loss.cpu().item())
            losses[0] = losses[0][-1000:]
            losses[1] = losses[1][-1000:]
            losses[2] = losses[2][-1000:]

            accelerator.wait_for_everyone()

            optimizer.step()

            accelerator.wait_for_everyone()

            id_diversity_log = {}
            if accelerator.is_main_process and wandb_logging:
                # Compute logs depending on training model_output here to avoid cuda graph overwrite from eval graph.
                emb_norms_avg = model_output.embs_norm.mean(axis=0)
                emb_norms_avg_log = {
                    f"emb_avg_norm_{i}": emb_norms_avg[i].cpu().item()
                    for i in range(vae_n_layers)
                }
                train_log = {
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss.cpu().item(),
                    "reconstruction_loss": model_output.reconstruction_loss.cpu().item(),
                    "rqvae_loss": model_output.rqvae_loss.cpu().item(),
                    "temperature": t,
                    "p_unique_ids": model_output.p_unique_ids.cpu().item(),
                    **emb_norms_avg_log,
                }

            if do_eval and ((iter + 1) % eval_every == 0 or iter + 1 == iterations):
                if accelerator.is_main_process:
                    _log(f"[rqvae] eval recon loss @ iter {iter + 1}...")
                model.eval()
                eval_losses = [[], [], []]
                for batch in eval_dataloader:
                    data = batch_to(batch, device)
                    with torch.no_grad():
                        eval_model_output = model(data, gumbel_t=t)

                    eval_losses[0].append(eval_model_output.loss.cpu().item())
                    eval_losses[1].append(
                        eval_model_output.reconstruction_loss.cpu().item()
                    )
                    eval_losses[2].append(eval_model_output.rqvae_loss.cpu().item())

                eval_losses = np.array(eval_losses).mean(axis=-1)
                id_diversity_log["eval_total_loss"] = eval_losses[0]
                id_diversity_log["eval_reconstruction_loss"] = eval_losses[1]
                id_diversity_log["eval_rqvae_loss"] = eval_losses[2]

            if accelerator.is_main_process:
                if (iter + 1) % save_model_every == 0 or iter + 1 == iterations:
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "model_config": model.config,
                        "optimizer": optimizer.state_dict(),
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    ckpt = save_dir_root + f"checkpoint_{iter}.pt"
                    torch.save(state, ckpt)
                    _log(f"[rqvae] saved {ckpt}")

                if (iter + 1) % eval_every == 0 or iter + 1 == iterations:
                    _log(f"[rqvae] computing semantic IDs @ iter {iter + 1}...")
                    tokenizer.reset()
                    model.eval()

                    corpus_ids = tokenizer.precompute_corpus_ids(index_dataset)
                    max_duplicates = corpus_ids[:, -1].max() / corpus_ids.shape[0]

                    _, counts = torch.unique(
                        corpus_ids[:, :-1], dim=0, return_counts=True
                    )
                    p = counts / corpus_ids.shape[0]
                    rqvae_entropy = -(p * torch.log(p)).sum()

                    for cid in range(vae_n_layers):
                        _, counts = torch.unique(corpus_ids[:, cid], return_counts=True)
                        id_diversity_log[f"codebook_usage_{cid}"] = (
                            len(counts) / vae_codebook_size
                        )

                    id_diversity_log["rqvae_entropy"] = rqvae_entropy.cpu().item()
                    id_diversity_log["max_id_duplicates"] = max_duplicates.cpu().item()
                    usage = [
                        id_diversity_log[f"codebook_usage_{cid}"]
                        for cid in range(vae_n_layers)
                    ]
                    _log(
                        f"[rqvae] iter {iter + 1} eval_loss={eval_losses[0]:.4f} "
                        f"codebook_usage={[f'{u:.2%}' for u in usage]} "
                        f"entropy={id_diversity_log['rqvae_entropy']:.3f} "
                        f"max_dup={id_diversity_log['max_id_duplicates']:.4f}"
                    )

                if iter % log_every == 0 or iter + 1 == iterations:
                    elapsed = time.time() - t_start
                    _log(
                        f"[rqvae] iter {iter}/{start_iter + iterations} "
                        f"loss={np.mean(losses[0]):.4f} recon={np.mean(losses[1]):.4f} "
                        f"vq={np.mean(losses[2]):.4f} elapsed={elapsed:.0f}s"
                    )

                if wandb_logging:
                    import wandb

                    wandb.log({**train_log, **id_diversity_log})

    if accelerator.is_main_process:
        _log("[rqvae] training finished")

    if wandb_logging:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
