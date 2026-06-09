import logging
import os
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

from data import Data, TrainDataset, collate_fn, preprocess


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def train(
    train_dataloader: DataLoader,
    model: SASRecEncoder,
    optimizer: torch.optim.Optimizer,
    device: str = 'cpu',
    num_epochs: int = 100,
):
    logger.info('Start training (%d epochs, %d batches/epoch)', num_epochs, len(train_dataloader))

    model.train()

    for epoch_num in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(
            train_dataloader,
            desc=f'Epoch {epoch_num + 1}/{num_epochs}',
            leave=False,
            file=sys.stdout,
        )
        for batch in pbar:
            for key in batch.keys():
                batch[key] = batch[key].to(device)

            loss = model(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            num_batches += 1
            pbar.set_postfix(loss=f'{batch_loss:.4f}')

        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info('Epoch %d/%d — avg_loss=%.6f', epoch_num + 1, num_epochs, avg_loss)
        sys.stdout.flush()

    logger.info('Training finished')
    return model.state_dict()


@click.command()
@click.option('--exp_name', required=True, type=str)
@click.option('--data_dir', required=True, type=str, default='../../data/', show_default=True)
@click.option('--checkpoint_dir', required=True, type=str, default='./checkpoints/', show_default=True)
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
@click.option('--embedding_dim', required=False, type=int, default=64, show_default=True)
@click.option('--num_heads', required=False, type=int, default=2, show_default=True)
@click.option('--num_layers', required=False, type=int, default=2, show_default=True)
@click.option('--learning_rate', required=False, type=float, default=1e-3, show_default=True)
@click.option('--dropout', required=False, type=float, default=0.0, show_default=True)
@click.option('--seed', required=False, type=int, default=42, show_default=True)
@click.option('--device', required=True, type=str, default='cuda:0', show_default=True)
@click.option('--num_epochs', required=True, type=int, default=100, show_default=True)
def main(
    exp_name: str,
    data_dir: str,
    checkpoint_dir: str,
    size: str,
    interaction: str,
    batch_size: int,
    max_seq_len: int,
    embedding_dim: int,
    num_heads: int,
    num_layers: int,
    learning_rate: float,
    dropout: float,
    seed: int,
    device: str,
    num_epochs: int,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision('high')

    data_path = Path.Path(data_dir) / 'sequential' / size / interaction
    df = pl.scan_parquet(data_path.with_suffix('.parquet'))

    checkpoint_path = Path.Path(checkpoint_dir) / f'{exp_name}_best_state.pth'
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info('Preprocessing data...')
    data: Data = preprocess(df, interaction, val_size=0, max_seq_len=max_seq_len)
    train_df = data.train.collect(engine="streaming")
    logger.info('Preprocessing done: %d users, %d items', len(train_df), data.num_items)

    train_dataset = TrainDataset(dataset=train_df, num_items=data.num_items, max_seq_len=max_seq_len)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=True,
        num_workers=3,
        prefetch_factor=10,
        pin_memory_device="cuda",
        pin_memory=True,
    )

    model = SASRecEncoder(
        num_items=data.num_items,
        max_sequence_length=max_seq_len,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_checkpoint = train(
        train_dataloader=train_dataloader, model=model, optimizer=optimizer, device=device, num_epochs=num_epochs
    )

    logger.info('Saving model...')

    os.makedirs(checkpoint_dir, exist_ok=True)

    model.load_state_dict(best_checkpoint)
    torch.save(model, checkpoint_path)
    logger.info('Saved model: %s', checkpoint_path)


if __name__ == '__main__':
    main()
