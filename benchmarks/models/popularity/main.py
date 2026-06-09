import json
from pathlib import Path
from typing import Any

import click
import polars as pl
import torch

from yambda.constants import Constants
from yambda.evaluation.metrics import calc_metrics
from yambda.evaluation.ranking import Embeddings, Ranked, Targets
from yambda.processing import timesplit
from yambda.utils import argmax


@click.command()
@click.option(
    '--data_dir', required=True, type=str, default="../../data/flat", show_default=True, help="Expects flat data"
)
@click.option(
    '--size',
    required=True,
    type=click.Choice(['50m', '500m', "5b"]),
    default=["50m"],
    multiple=True,
    show_default=True,
)
@click.option(
    '--interaction',
    required=True,
    type=click.Choice(['likes', 'listens', 'listens_fltrd']),
    default=["likes"],
    multiple=True,
    show_default=True,
)
@click.option(
    '--hours',
    required=True,
    type=float,
    default=[0.5, 1, 2, 3, 6, 12, 24],
    multiple=True,
    show_default=True,
    help="Hyperparameter",
)
@click.option('--validation_metric', required=True, type=str, default="ndcg@100", show_default=True)
@click.option('--report_metrics', required=True, type=str, default=Constants.METRICS, multiple=True, show_default=True)
@click.option('--device', required=True, type=str, default="cuda:0", show_default=True)
def main(
    data_dir: str,
    size: list[str],
    interaction: list[str],
    hours: list[float],
    validation_metric: str,
    report_metrics: list[str],
    device: str,
):
    print(f"REPORT METRICS: {report_metrics}")
    for s in size:
        for i in interaction:
            print(f"SIZE {s}, INTERACTION {i}")
            result = popularity(
                data_dir,
                s,
                i,
                device,
                hours=hours,
                validation_metric=validation_metric,
                report_metrics=report_metrics,
            )
            print(json.dumps(result, indent=2))


def scan(path: str, dataset_size: str, dataset_name: str) -> pl.LazyFrame:
    path: Path = Path(path) / dataset_size / dataset_name
    df = pl.scan_parquet(path.with_suffix(".parquet"))
    return df


def preprocess(
    df: pl.LazyFrame, interaction: str, val_size: int
) -> tuple[pl.LazyFrame, pl.LazyFrame | None, pl.LazyFrame]:
    if interaction == "listens":
        df = df.filter(pl.col("played_ratio_pct") >= Constants.TRACK_LISTEN_THRESHOLD)

    train, val, test = timesplit.flat_split_train_val_test(
        df, val_size=val_size, test_timestamp=Constants.TEST_TIMESTAMP
    )

    return (
        train,
        val.collect(engine="streaming").lazy() if val is not None else None,
        test.collect(engine="streaming").lazy(),
    )


def training(hour: float, train: pl.LazyFrame, max_timestamp: float, device: str, decay: float = 0.9) -> Embeddings:
    if hour == 0:
        embeddings = train.group_by("item_id").agg(pl.count().alias("item_embedding")).collect(engine="streaming")
    else:
        tau = decay ** (1 / Constants.DAY_SECONDS / (hour / 24))

        embeddings = (
            train.select(
                "item_id",
                (tau ** (max_timestamp - pl.col("timestamp"))).alias("value"),
            )
            .group_by("item_id")
            .agg(pl.col("value").sum().alias("item_embedding"))
            .collect(engine="streaming")
        )

    item_ids = embeddings["item_id"].to_torch().to(device)

    item_embeddings = embeddings["item_embedding"].to_torch().to(device)[:, None]

    return Embeddings(item_ids, item_embeddings)


def evaluation(
    train: pl.LazyFrame, val: pl.LazyFrame, device: str, hours: list[float], metrics: list[str]
) -> list[dict[str, Any]]:
    num_ranked_items = max([int(x.split("@")[1]) for x in metrics])

    max_timestamp = train.select(pl.col("timestamp").max()).collect(engine="streaming").item()
    user_ids = train.select("uid").unique().collect(engine="streaming")["uid"].to_torch().to(device)

    targets = Targets.from_sequential(
        val.group_by('uid', maintain_order=True).agg("item_id"),
        device,
    )

    hour2metrics = []
    for hour in hours:
        item_embeddings = training(
            hour=hour,
            train=train,
            max_timestamp=max_timestamp,
            device=device,
        )

        ranked = Ranked(
            user_ids=user_ids,
            item_ids=item_embeddings.ids[torch.topk(item_embeddings.embeddings, num_ranked_items, dim=0).indices]
            .ravel()
            .expand((user_ids.shape[0], num_ranked_items)),
            num_item_ids=item_embeddings.ids.shape[0],
        )

        hour2metrics.append(calc_metrics(ranked, targets, metrics))

    return hour2metrics


def popularity(
    data_dir: str,
    size: str,
    interaction: str,
    device: str,
    hours: list[float],
    validation_metric: str,
    report_metrics: list[str],
) -> dict[str, Any]:
    df = scan(data_dir, size, interaction)

    # hyperopt by validation
    train, val, _ = preprocess(df, interaction, val_size=Constants.VAL_SIZE)

    results = evaluation(train, val, device, hours, [validation_metric])

    metric_name, k = validation_metric.split('@')

    best_hour = hours[argmax(results, lambda x: x[metric_name][int(k)])]

    print(f"FINAL HYPERPARAMS {best_hour=}")

    # train final model
    train, _, test = preprocess(df, interaction, val_size=0)

    return evaluation(train, test, device, [best_hour], report_metrics)[0]


if __name__ == "__main__":
    main()
