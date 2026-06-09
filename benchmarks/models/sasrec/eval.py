import logging
import pathlib as Path
import random
import sys

import click
import numpy as np
import polars as pl
import torch
from model import SASRecEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Data, EvalDataset, collate_fn, preprocess
from yambda.evaluation.metrics import calc_metrics
from yambda.evaluation.ranking import Embeddings, Targets, rank_items


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def infer_users(eval_dataloader: DataLoader, model: torch.nn.Module, device: str):
    user_ids = []
    user_embeddings = []

    model.eval()
    for batch in tqdm(eval_dataloader, desc='Infer users', file=sys.stdout):
        for key in batch.keys():
            batch[key] = batch[key].to(device)

        user_ids.append(batch['user.ids'])  # (batch_size)
        user_embeddings.append(model(batch))  # (batch_size, embedding_dim)

    return torch.cat(user_ids, dim=0), torch.cat(user_embeddings, dim=0)


def infer_items(model: SASRecEncoder):
    return model.item_embeddings.weight.data


@click.command()
@click.option('--exp_name', required=True, type=str)
@click.option('--data_dir', required=True, type=str, default='../../data/', show_default=True)
@click.option(
    '--size',
    required=True,
    type=click.Choice(['50m', '500m', '5b']),
    default='50m',
    show_default=True,
)
@click.option(
    '--interaction',
    required=True,
    type=click.Choice(['likes', 'listens', 'listens_fltrd']),
    default='likes',
    show_default=True,
)
@click.option('--batch_size', required=True, type=int, default=256, show_default=True)
@click.option('--max_seq_len', required=False, type=int, default=200, show_default=True)
@click.option('--seed', required=False, type=int, default=42, show_default=True)
@click.option('--filter_seen', is_flag=True, default=False, help='Mask train-history items before ranking.')
@click.option('--device', required=True, type=str, default='cuda:0', show_default=True)
def main(
    exp_name: str,
    data_dir: str,
    size: str,
    interaction: str,
    batch_size: int,
    max_seq_len: int,
    seed: int,
    filter_seen: bool,
    device: str,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision('high')

    path = Path.Path(data_dir) / 'sequential' / size / interaction
    df = pl.scan_parquet(path.with_suffix('.parquet'))

    logger.info('Preprocessing data...')
    data: Data = preprocess(df, interaction, val_size=0, max_seq_len=max_seq_len)
    train_df = data.train.collect(engine="streaming")
    eval_df = data.test.collect(engine="streaming")
    logger.info('Preprocessing done: %d train users, %d test users', len(train_df), len(eval_df))

    eval_df = train_df.join(eval_df, on='uid', how='inner', suffix='_valid').select(
        pl.col('uid'), pl.col('item_id').alias('item_id_train'), pl.col('item_id_valid')
    )
    eval_dataset = EvalDataset(dataset=eval_df, max_seq_len=max_seq_len)

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        shuffle=True,
    )

    logger.info('Loading checkpoint: ./checkpoints/%s_best_state.pth', exp_name)
    model = torch.load(f'./checkpoints/{exp_name}_best_state.pth', weights_only=False).to(device)
    model.eval()
    with torch.inference_mode():
        user_ids, user_embeddings = infer_users(eval_dataloader=eval_dataloader, model=model, device=device)
        item_embeddings = infer_items(model=model)

    logger.info('Ranking items (top-100)...')
    item_embeddings = Embeddings(
        ids=torch.arange(start=0, end=item_embeddings.shape[0], device=device), embeddings=item_embeddings
    )
    user_embeddings = Embeddings(ids=user_ids, embeddings=user_embeddings)

    uid_to_seen = {
        int(row['uid']): torch.tensor(row['item_id_train'], dtype=torch.long, device=device)
        for row in eval_df.iter_rows(named=True)
    }
    seen_item_ids = None
    if filter_seen:
        seen_item_ids = [uid_to_seen[int(uid)] for uid in user_embeddings.ids.tolist()]
        logger.info('Seen-item filtering enabled (%d users)', len(seen_item_ids))

    df_user_ids = torch.tensor(eval_df['uid'].to_list(), dtype=torch.long, device=device)
    df_target_ids = [
        torch.tensor(item_ids, dtype=torch.long, device=device) for item_ids in eval_df['item_id_valid'].to_list()
    ]
    targets = Targets(user_ids=df_user_ids, item_ids=df_target_ids)
    with torch.no_grad():
        ranked = rank_items(
            users=user_embeddings,
            items=item_embeddings,
            num_items=100,
            seen_item_ids=seen_item_ids,
        )

    logger.info('Computing metrics...')
    metric_names = [f'{name}@{k}' for name in ["recall", "ndcg", "coverage"] for k in [10, 50, 100]]
    metrics = calc_metrics(ranked, targets, metrics=metric_names)
    logger.info('Metrics: %s', metrics)
    print(metrics, flush=True)


if __name__ == '__main__':
    main()
