import importlib.util
import logging
import sys
import time
from pathlib import Path

import click
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.popularity import load_item_popularity
from data.processed import ItemData
from data.processed import RecDataset
from data.processed import SeqData
from data.utils import batch_to
from modules.model import EncoderDecoderRetrievalModel
from modules.semantic_id_encoding import encode_prefix_columns
from modules.semantic_id_encoding import encode_sem_id_columns
from modules.tokenizer.semids import SemanticIdTokenizer

ROOT = Path(__file__).resolve().parent
BENCHMARKS = ROOT / "benchmarks"
SASREC_DIR = ROOT / "benchmarks" / "models" / "sasrec"

sys.path.insert(0, str(BENCHMARKS))

from yambda.evaluation.metrics import calc_metrics  # noqa: E402
from yambda.evaluation.ranking import Ranked, Targets, mask_seen_item_scores  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def _load_sasrec_data_module():
    spec = importlib.util.spec_from_file_location("sasrec_data", SASREC_DIR / "data.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def encode_sem_id(codes: torch.Tensor, codebook_size: int, dedup_vocab: int) -> int:
    return int(
        encode_sem_id_columns(codes.unsqueeze(0), codebook_size, dedup_vocab).item()
    )


def build_encoded_to_sasrec_items(
    codebooks: torch.Tensor,
    corpus_raw_ids: torch.Tensor,
    item_id_to_idx: dict[int, int],
    codebook_size: int,
    dedup_vocab: int,
) -> dict[int, int]:
    encoded_to_item: dict[int, int] = {}
    codes = codebooks.long()
    for corpus_idx in range(codes.shape[0]):
        enc = encode_sem_id(codes[corpus_idx], codebook_size, dedup_vocab)
        raw_id = int(corpus_raw_ids[corpus_idx].item())
        sasrec_idx = item_id_to_idx.get(raw_id)
        if sasrec_idx is not None:
            if enc in encoded_to_item and encoded_to_item[enc] != sasrec_idx:
                raise RuntimeError(
                    f"Duplicate full semantic ID {enc} maps to multiple SASRec items"
                )
            encoded_to_item[enc] = sasrec_idx
    return encoded_to_item


def build_triplet_to_sasrec_items(
    codebooks: torch.Tensor,
    corpus_raw_ids: torch.Tensor,
    item_id_to_idx: dict[int, int],
    codebook_size: int,
    n_triplet_layers: int = 3,
) -> dict[int, list[int]]:
    triplet_to_items: dict[int, list[int]] = {}
    codes = codebooks.long()
    for corpus_idx in range(codes.shape[0]):
        enc = int(
            encode_prefix_columns(
                codes[corpus_idx, :n_triplet_layers].unsqueeze(0), codebook_size
            ).item()
        )
        raw_id = int(corpus_raw_ids[corpus_idx].item())
        sasrec_idx = item_id_to_idx.get(raw_id)
        if sasrec_idx is not None:
            triplet_to_items.setdefault(enc, []).append(sasrec_idx)
    return triplet_to_items


def batch_sem_id_recommendations(
    sem_ids: torch.Tensor,
    log_probas: torch.Tensor,
    encoded_to_item: dict[int, int],
    num_catalog: int,
    num_items: int,
    codebook_size: int,
    dedup_vocab: int,
    device: torch.device,
    seen_item_ids: list[torch.Tensor] | None = None,
    triplet_to_items: dict[int, list[int]] | None = None,
    triplet_layers: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    batch_size, num_beams, num_h = sem_ids.shape
    encoded = encode_sem_id_columns(
        sem_ids.reshape(-1, num_h), codebook_size, dedup_vocab
    ).reshape(batch_size, num_beams)

    batch_indices: list[torch.Tensor] = []
    item_indices: list[torch.Tensor] = []
    score_values: list[torch.Tensor] = []

    stats = {"full_hits": 0, "triplet_hits": 0, "misses": 0}

    for b in range(batch_size):
        for ki in range(num_beams):
            score = log_probas[b, ki]
            enc = int(encoded[b, ki].item())
            item_idx = encoded_to_item.get(enc)
            if item_idx is not None:
                stats["full_hits"] += 1
                batch_indices.append(torch.tensor([b], device=device, dtype=torch.long))
                item_indices.append(torch.tensor([item_idx], device=device, dtype=torch.long))
                score_values.append(score.unsqueeze(0))
                continue

            if triplet_to_items is not None and num_h >= triplet_layers:
                triplet_enc = int(
                    encode_prefix_columns(
                        sem_ids[b, ki, :triplet_layers].unsqueeze(0), codebook_size
                    ).item()
                )
                candidates = triplet_to_items.get(triplet_enc)
                if candidates:
                    stats["triplet_hits"] += 1
                    for item_idx in candidates:
                        batch_indices.append(torch.tensor([b], device=device, dtype=torch.long))
                        item_indices.append(
                            torch.tensor([item_idx], device=device, dtype=torch.long)
                        )
                        score_values.append(score.unsqueeze(0))
                    continue

            stats["misses"] += 1

    item_scores = torch.full(
        (batch_size, num_catalog + 1),
        float("-inf"),
        device=device,
        dtype=torch.float32,
    )

    if batch_indices:
        b_idx = torch.cat(batch_indices)
        i_idx = torch.cat(item_indices)
        s_val = torch.cat(score_values)
        linear = b_idx * (num_catalog + 1) + i_idx
        flat = item_scores.view(-1)
        flat.scatter_reduce_(0, linear, s_val, reduce="amax", include_self=True)
        item_scores = flat.view(batch_size, num_catalog + 1)

    if seen_item_ids is not None:
        item_scores = mask_seen_item_scores(item_scores, seen_item_ids)

    top_scores, top_ids = item_scores.topk(num_items, dim=1)
    return top_ids, top_scores, stats


@click.command()
@click.option("--checkpoint", required=True, type=str)
@click.option(
    "--pretrained_rqvae_path",
    required=True,
    type=str,
    default="out/rqvae/listens_fltrd/checkpoint_99999.pt",
    show_default=True,
)
@click.option("--dataset_folder", default="dataset/yambda", show_default=True)
@click.option("--dataset_split", default="500m", show_default=True)
@click.option("--dataset_interaction", default="listens_fltrd", show_default=True)
@click.option("--batch_size", default=32, show_default=True)
@click.option("--num_ranked_items", default=100, show_default=True)
@click.option("--top_k_semantic", default=50, show_default=True)
@click.option("--max_seq_len", default=120, show_default=True)
@click.option("--filter_seen", is_flag=True, default=False, help="Mask train-history items before ranking.")
@click.option("--triplet_fallback", is_flag=True, default=False, help="Expand beams with no full-ID hit to all items sharing the triplet prefix.")
@click.option(
    "--corpus_assignment",
    default="sequential",
    show_default=True,
    type=click.Choice(["sequential", "popularity", "rrs"]),
    help="Must match the assignment used during decoder training.",
)
@click.option("--rrs_k", default=4, show_default=True, help="Top-k for RRS assignment (only if corpus_assignment=rrs).")
@click.option("--device", default="cuda:0", show_default=True)
def main(
    checkpoint: str,
    pretrained_rqvae_path: str,
    dataset_folder: str,
    dataset_split: str,
    dataset_interaction: str,
    batch_size: int,
    num_ranked_items: int,
    top_k_semantic: int,
    max_seq_len: int,
    filter_seen: bool,
    triplet_fallback: bool,
    corpus_assignment: str,
    rrs_k: int,
    device: str,
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if not filter_seen:
        raise click.UsageError("--filter_seen is required for all eval runs.")

    torch.set_float32_matmul_precision("high")
    device_t = torch.device(device)

    vae_input_dim = 128
    vae_hidden_dims = [256, 128]
    vae_embed_dim = 32
    vae_codebook_size = 128
    vae_n_layers = 3
    vae_n_cat_feats = 0
    t5_d_model = 128
    t5_num_heads = 2
    t5_d_ff = 512
    t5_num_layers = 2
    should_add_sep_token = True

    sasrec_data = _load_sasrec_data_module()
    seq_path = Path(dataset_folder) / "sequential" / dataset_split / f"{dataset_interaction}.parquet"
    logger.info("Loading SASRec-compatible eval split from %s", seq_path)
    df = pl.scan_parquet(seq_path)
    data = sasrec_data.preprocess(df, dataset_interaction, val_size=0, max_seq_len=max_seq_len)
    train_df = data.train.collect(engine="streaming")
    test_df = data.test.collect(engine="streaming")
    eval_df = train_df.join(test_df, on="uid", how="inner", suffix="_valid").select(
        pl.col("uid"),
        pl.col("item_id").alias("item_id_train"),
        pl.col("item_id_valid"),
    )
    logger.info(
        "Preprocessing done: %d train users, %d eval users, %d catalog items",
        len(train_df),
        len(eval_df),
        data.num_items,
    )

    item_id_to_idx = data.item_id_to_idx

    logger.info("Loading item corpus and test sequences...")
    item_dataset = ItemData(
        root=dataset_folder,
        dataset=RecDataset.YAMBDA,
        split=dataset_split,
        interaction=dataset_interaction,
    )
    eval_dataset = SeqData(
        root=dataset_folder,
        dataset=RecDataset.YAMBDA,
        is_train=False,
        subsample=False,
        split=dataset_split,
        interaction=dataset_interaction,
    )
    corpus = torch.load(
        Path(dataset_folder) / "processed" / f"items_{dataset_interaction}.pt",
        weights_only=False,
    )
    corpus_raw_ids = corpus["item_id"]

    uid_to_eval_idx = {
        int(uid): idx for idx, uid in enumerate(eval_dataset.sequence_data["userId"].tolist())
    }
    missing_users = 0
    eval_indices = []
    target_item_ids: list[torch.Tensor] = []
    for row in eval_df.iter_rows(named=True):
        uid = int(row["uid"])
        eval_idx = uid_to_eval_idx.get(uid)
        if eval_idx is None:
            missing_users += 1
            continue
        eval_indices.append(eval_idx)
        valid = [int(x) for x in row["item_id_valid"]]
        target_item_ids.append(torch.tensor(valid, dtype=torch.long, device=device_t))

    if missing_users:
        logger.warning("Skipped %d SASRec eval users missing from TIGER test split", missing_users)

    uid_to_seen: dict[int, torch.Tensor] = {}
    if filter_seen:
        for row in eval_df.iter_rows(named=True):
            uid_to_seen[int(row["uid"])] = torch.tensor(
                [int(x) for x in row["item_id_train"]],
                dtype=torch.long,
                device=device_t,
            )
        logger.info("Seen-item filtering enabled (%d users with history)", len(uid_to_seen))

    logger.info("Building eval subset: %d users", len(eval_indices))
    subset = torch.utils.data.Subset(eval_dataset, eval_indices)
    eval_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    logger.info("Loading tokenizer from %s", pretrained_rqvae_path)
    tokenizer = SemanticIdTokenizer(
        input_dim=vae_input_dim,
        hidden_dims=vae_hidden_dims,
        output_dim=vae_embed_dim,
        codebook_size=vae_codebook_size,
        n_layers=vae_n_layers,
        n_cat_feats=vae_n_cat_feats,
        rqvae_weights_path=pretrained_rqvae_path,
    ).to(device_t)
    item_popularity = None
    if corpus_assignment == "popularity":
        item_popularity = load_item_popularity(
            dataset_folder, dataset_split, dataset_interaction, item_dataset.item_ids
        )
        logger.info(
            "Loaded popularity for %d items (max=%d)",
            len(item_popularity),
            int(item_popularity.max()),
        )

    logger.info(
        "Precomputing semantic IDs for %d corpus items (assignment=%s)...",
        len(item_dataset),
        corpus_assignment,
    )
    t0 = time.time()
    tokenizer.precompute_corpus_ids(
        item_dataset,
        assignment=corpus_assignment,
        item_popularity=item_popularity,
        rrs_k=rrs_k,
    )
    logger.info(
        "Semantic IDs ready in %.1fs, corpus shape=%s",
        time.time() - t0,
        tuple(tokenizer.cached_ids.shape),
    )
    full_codebooks = tokenizer.cached_ids.cpu()
    dedup_vocab = int(full_codebooks[:, -1].max()) + 1
    sem_to_item = build_encoded_to_sasrec_items(
        full_codebooks,
        corpus_raw_ids,
        item_id_to_idx,
        vae_codebook_size,
        dedup_vocab,
    )
    logger.info(
        "Built unique semantic->item lookup for %d full IDs (dedup_vocab=%d)",
        len(sem_to_item),
        dedup_vocab,
    )
    triplet_lookup = None
    if triplet_fallback:
        triplet_lookup = build_triplet_to_sasrec_items(
            full_codebooks, corpus_raw_ids, item_id_to_idx, vae_codebook_size
        )
        logger.info("Triplet fallback enabled: %d unique triplets", len(triplet_lookup))

    model = EncoderDecoderRetrievalModel(
        codebooks=full_codebooks,
        rqvae_codebook_size=vae_codebook_size,
        t5_d_model=t5_d_model,
        t5_num_heads=t5_num_heads,
        t5_d_ff=t5_d_ff,
        t5_num_layers=t5_num_layers,
        top_k_for_generation=top_k_semantic,
        should_add_sep_token=should_add_sep_token,
    ).to(device_t)

    logger.info("Loading decoder checkpoint: %s", checkpoint)
    ckpt = torch.load(checkpoint, map_location=device_t, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.rebuild_prefix_index()
    model.eval()
    logger.info("Loaded checkpoint from iter %d", ckpt.get("iter", -1))

    ranked_user_ids = []
    ranked_item_ids = []
    ranked_scores = []
    lookup_stats = {"full_hits": 0, "triplet_hits": 0, "misses": 0}

    logger.info(
        "Running inference: batch_size=%d, semantic_top_k=%d, item_top_k=%d, filter_seen=%s, triplet_fallback=%s",
        batch_size,
        top_k_semantic,
        num_ranked_items,
        filter_seen,
        triplet_fallback,
    )
    t_infer = time.time()
    with torch.inference_mode():
        for batch in tqdm(eval_loader, desc="Infer users", file=sys.stdout):
            data_b = batch_to(batch, device_t)
            tokenized = tokenizer(data_b)
            generated = model.generate_next_sem_id(tokenized, top_k=True, temperature=1)

            seen_batch = None
            if filter_seen:
                seen_batch = [uid_to_seen[int(uid)] for uid in data_b.user_ids.tolist()]

            batch_item_ids, batch_item_scores, batch_stats = batch_sem_id_recommendations(
                sem_ids=generated.sem_ids,
                log_probas=generated.log_probas,
                encoded_to_item=sem_to_item,
                num_catalog=data.num_items,
                num_items=num_ranked_items,
                codebook_size=vae_codebook_size,
                dedup_vocab=dedup_vocab,
                device=device_t,
                seen_item_ids=seen_batch,
                triplet_to_items=triplet_lookup,
            )
            for key in lookup_stats:
                lookup_stats[key] += batch_stats[key]
            ranked_user_ids.extend(data_b.user_ids.tolist())
            ranked_item_ids.append(batch_item_ids)
            ranked_scores.append(batch_item_scores)

    infer_s = time.time() - t_infer
    logger.info("Inference finished in %.1fs (%.2f users/s)", infer_s, len(ranked_user_ids) / max(infer_s, 1e-6))

    ranked = Ranked(
        user_ids=torch.tensor(ranked_user_ids, dtype=torch.long, device=device_t),
        item_ids=torch.cat(ranked_item_ids, dim=0),
        scores=torch.cat(ranked_scores, dim=0),
        num_item_ids=data.num_items,
    )
    targets = Targets(
        user_ids=torch.tensor(ranked_user_ids, dtype=torch.long, device=device_t),
        item_ids=target_item_ids,
    )

    metric_names = [
        f"{name}@{k}" for name in ["recall", "ndcg", "coverage"] for k in [10, 50, 100]
    ]
    logger.info("Computing metrics: %s", metric_names)
    metrics = calc_metrics(ranked, targets, metrics=metric_names)
    logger.info("Lookup stats: %s", lookup_stats)
    logger.info("Metrics: %s", metrics)
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
