import os
import sys
import time
from collections import deque
import gin
import torch

from accelerate import Accelerator
from data.popularity import load_item_popularity
from data.processed import ItemData
from data.processed import RecDataset
from data.processed import SeqData
from data.utils import batch_to
from data.utils import cycle
from data.utils import next_batch
from evaluate.metrics import TopKAccumulator
from modules.model import EncoderDecoderRetrievalModel
from modules.scheduler.inv_sqrt import InverseSquareRootScheduler
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.utils import compute_debug_metrics
from modules.utils import parse_config
from huggingface_hub import login
from torch.optim import AdamW
from torch.utils.data import DataLoader
@gin.configurable
def train(
    iterations=500000,
    batch_size=64,
    learning_rate=0.001,
    weight_decay=0.01,
    dataset_folder="dataset/yambda",
    save_dir_root="out/",
    dataset=RecDataset.YAMBDA,
    pretrained_rqvae_path=None,
    pretrained_decoder_path=None,
    split_batches=True,
    amp=False,
    wandb_logging=False,
    force_dataset_process=False,
    mixed_precision_type="fp16",
    gradient_accumulate_every=1,
    save_model_every=1000000,
    partial_eval_every=1000,
    full_eval_every=10000,
    vae_input_dim=18,
    vae_embed_dim=16,
    vae_hidden_dims=[18, 18],
    vae_codebook_size=32,
    vae_codebook_normalize=False,
    vae_sim_vq=False,
    vae_n_cat_feats=18,
    vae_n_layers=3,
    dataset_split="500m",
    dataset_interaction="listens_fltrd",
    push_vae_to_hf=False,
    train_data_subsample=True,
    vae_hf_model_name="",
    max_grad_norm=None,
    t5_d_model=128,
    t5_num_heads=2,
    t5_d_ff=512,
    t5_num_layers=2,
    top_k_for_generation=10,
    should_add_sep_token=True,
    num_user_bins=None,
    top_k_eval_list=[1, 5, 10],
    log_every=100,
    loss_window=50,
    eval_gen_batch_size=32,
    corpus_assignment="sequential",
    rrs_k=4,
):
    if dataset != RecDataset.YAMBDA:
        raise Exception(f"Dataset currently not supported: {dataset}.")

    if wandb_logging:
        params = locals()

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=mixed_precision_type if amp else "no",
    )

    device = accelerator.device

    if wandb_logging and accelerator.is_main_process:
        import wandb

        wandb.login()
        run = wandb.init(project="gen-retrieval-decoder-training", config=params)

    item_dataset = ItemData(
        root=dataset_folder,
        dataset=dataset,
        force_process=force_dataset_process,
        split=dataset_split,
        interaction=dataset_interaction,
    )
    train_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        is_train=True,
        subsample=train_data_subsample,
        force_process=force_dataset_process,
        split=dataset_split,
        interaction=dataset_interaction,
    )
    eval_dataset = SeqData(
        root=dataset_folder,
        dataset=dataset,
        is_train=False,
        subsample=False,
        force_process=force_dataset_process,
        split=dataset_split,
        interaction=dataset_interaction,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = cycle(train_dataloader)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)

    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )

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
    tokenizer = accelerator.prepare(tokenizer)
    item_popularity = None
    if corpus_assignment == "popularity":
        item_popularity = load_item_popularity(
            dataset_folder, dataset_split, dataset_interaction, item_dataset.item_ids
        )
        if accelerator.is_main_process:
            print(
                f"[tiger] loaded popularity for {len(item_popularity)} items "
                f"(max={int(item_popularity.max())})",
                flush=True,
            )

    if accelerator.is_main_process:
        print(
            f"[tiger] precomputing semantic IDs for {len(item_dataset)} items "
            f"(assignment={corpus_assignment})...",
            flush=True,
        )
    tokenizer.precompute_corpus_ids(
        item_dataset,
        assignment=corpus_assignment,
        item_popularity=item_popularity,
        rrs_k=rrs_k,
    )
    if accelerator.is_main_process:
        print(f"[tiger] semantic IDs ready, corpus shape={tokenizer.cached_ids.shape}", flush=True)

    if push_vae_to_hf:
        login()
        tokenizer.rq_vae.push_to_hub(vae_hf_model_name)

    codebooks = tokenizer.cached_ids.cpu()
    if accelerator.is_main_process:
        print(
            f"[tiger] decoder hierarchies={codebooks.shape[1]} "
            f"(semantic_layers={codebooks.shape[1] - 1}, dedup_vocab={int(codebooks[:, -1].max()) + 1})",
            flush=True,
        )

    model = EncoderDecoderRetrievalModel(
        codebooks=codebooks,
        rqvae_codebook_size=vae_codebook_size,
        t5_d_model=t5_d_model,
        t5_num_heads=t5_num_heads,
        t5_d_ff=t5_d_ff,
        t5_num_layers=t5_num_layers,
        top_k_for_generation=top_k_for_generation,
        should_add_sep_token=should_add_sep_token,
        num_user_bins=num_user_bins,
    )
    optimizer = AdamW(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    lr_scheduler = InverseSquareRootScheduler(optimizer=optimizer, warmup_steps=10000)

    start_iter = 0
    if pretrained_decoder_path is not None:
        checkpoint = torch.load(
            pretrained_decoder_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        start_iter = checkpoint["iter"] + 1

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    metrics_accumulator = TopKAccumulator(ks=top_k_eval_list)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}, Num Parameters: {num_params}", flush=True)
    print(
        f"[tiger] train users={len(train_dataset)} batch_size={batch_size} "
        f"iters={iterations} log_every={log_every}",
        flush=True,
    )

    t_start = time.time()
    loss_window_buf: deque[float] = deque(maxlen=loss_window)
    for iter in range(start_iter, start_iter + iterations):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad()
            train_debug_metrics = {}

            for _ in range(gradient_accumulate_every):
                data = next_batch(train_dataloader, device)
                tokenized_data = tokenizer(data)

                with accelerator.autocast():
                    model_output = model(tokenized_data)
                    loss = model_output.loss / gradient_accumulate_every

                total_loss += loss.detach().item()

                if wandb_logging and accelerator.is_main_process:
                    train_debug_metrics = compute_debug_metrics(tokenized_data)

                accelerator.backward(loss)

            assert model.item_sid_embedding_table.weight.grad is not None

            loss_window_buf.append(total_loss)
            done = iter + 1 - start_iter
            elapsed = time.time() - t_start
            eta_s = (elapsed / max(done, 1)) * (iterations - done)

            accelerator.wait_for_everyone()

            if max_grad_norm is not None:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()

            accelerator.wait_for_everyone()

            if (iter + 1) % partial_eval_every == 0:
                model.eval()
                eval_losses = []
                for batch in eval_dataloader:
                    data = batch_to(batch, device)
                    tokenized_data = tokenizer(data)
                    with torch.no_grad():
                        eval_losses.append(model(tokenized_data).loss.item())
                eval_loss = sum(eval_losses) / max(len(eval_losses), 1)
                if accelerator.is_main_process:
                    print(
                        f"[tiger] partial_eval @ iter {iter + 1}: eval_loss={eval_loss:.4f}",
                        flush=True,
                    )
                if wandb_logging and accelerator.is_main_process:
                    import wandb

                    wandb.log({"eval_loss": eval_loss})

            if (iter + 1) % full_eval_every == 0:
                model.eval()
                if accelerator.is_main_process:
                    print(f"[tiger] full_eval start @ iter {iter + 1}...", flush=True)
                eval_gen_loader = DataLoader(
                    eval_dataset, batch_size=eval_gen_batch_size, shuffle=False
                )
                for batch in eval_gen_loader:
                    data = batch_to(batch, device)
                    tokenized_data = tokenizer(data)

                    with torch.no_grad():
                        generated = model.generate_next_sem_id(
                            tokenized_data, top_k=True, temperature=1
                        )

                    actual = tokenized_data.sem_ids_fut
                    metrics_accumulator.accumulate(
                        actual=actual, top_k=generated.sem_ids
                    )

                eval_metrics = metrics_accumulator.reduce()
                if accelerator.is_main_process:
                    print(f"[tiger] full_eval @ iter {iter + 1}: {eval_metrics}", flush=True)
                if accelerator.is_main_process and wandb_logging:
                    import wandb

                    wandb.log(eval_metrics)
                metrics_accumulator.reset()

            if accelerator.is_main_process:
                if (iter + 1) % log_every == 0 or iter + 1 == start_iter + iterations:
                    mean_loss = sum(loss_window_buf) / len(loss_window_buf)
                    print(
                        f"[tiger] iter {iter + 1}/{start_iter + iterations} "
                        f"loss_mean@{len(loss_window_buf)}={mean_loss:.4f} "
                        f"elapsed={elapsed:.0f}s eta={eta_s:.0f}s",
                        flush=True,
                    )

                if (iter + 1) % save_model_every == 0 or iter + 1 == start_iter + iterations:
                    state = {
                        "iter": iter,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": lr_scheduler.state_dict(),
                    }

                    if not os.path.exists(save_dir_root):
                        os.makedirs(save_dir_root)

                    ckpt_path = save_dir_root + f"checkpoint_{iter}.pt"
                    torch.save(state, ckpt_path)
                    print(f"[tiger] saved {ckpt_path}", flush=True)

                if wandb_logging:
                    import wandb

                    wandb.log(
                        {
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "total_loss": total_loss,
                            **train_debug_metrics,
                        }
                    )

    if accelerator.is_main_process:
        print("[tiger] training finished", flush=True)

    if wandb_logging:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
