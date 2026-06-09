from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Iterable

import torch
from tqdm import tqdm

from .ranking import Ranked, Targets


def cut_off_ranked(ranked: Ranked, targets: Targets) -> Ranked:
    mask = torch.isin(ranked.user_ids, targets.user_ids)

    assert ranked.scores is not None

    ranked = Ranked(
        user_ids=ranked.user_ids[mask],
        scores=ranked.scores[mask, :],
        item_ids=ranked.item_ids[mask, :],
        num_item_ids=ranked.num_item_ids,
    )

    assert ranked.item_ids.shape[0] == len(targets), "Ranked doesn't contain all targets.user_ids"

    return ranked


class Metric(ABC):
    @abstractmethod
    def __call__(
        self, ranked: Ranked | None, targets: Targets | None, target_mask: torch.Tensor | None, ks: Iterable[int]
    ) -> dict[int, float]:
        pass


class Recall(Metric):
    def __call__(
        self, ranked: Ranked | None, targets: Targets, target_mask: torch.Tensor, ks: Iterable[int]
    ) -> dict[int, float]:
        assert all(0 < k <= target_mask.shape[1] for k in ks)

        values = {}

        for k in ks:
            num_positives = targets.lengths.to(torch.float32)
            num_positives[num_positives == 0] = torch.inf

            # there was a bug: we divided by num_positives instead of max(num_positives, k)
            # this may have slightly affected the absolute metric values,
            # but as far as we can judge it didn't change the ranking of the models reported in the paper.
            values[k] = target_mask[:, :k].to(torch.float32).sum(dim=-1) / torch.clamp(num_positives, max=k)

            values[k] = torch.mean(values[k]).item()

        return values


class Precision(Metric):
    def __call__(
        self, ranked: Ranked | None, targets: Targets | None, target_mask: torch.Tensor, ks: Iterable[int]
    ) -> dict[int, float]:
        assert all(0 < k <= target_mask.shape[1] for k in ks)

        values = {}

        for k in ks:
            values[k] = target_mask[:, :k].to(torch.float32).sum(dim=-1) / k
            values[k] = torch.mean(values[k]).item()

        return values


class Coverage(Metric):
    def __init__(self, cut_off_ranked: bool = False):
        self.cut_off_ranked = cut_off_ranked

    def __call__(
        self, ranked: Ranked, targets: Targets | None, target_mask: torch.Tensor | None, ks: Iterable[int]
    ) -> dict[int, float]:
        if self.cut_off_ranked:
            assert targets is not None
            ranked = cut_off_ranked(ranked, targets)

        assert all(0 < k <= ranked.item_ids.shape[1] for k in ks)

        assert ranked.num_item_ids is not None

        values = {}
        for k in ks:
            values[k] = ranked.item_ids[:, :k].flatten().unique().shape[0] / ranked.num_item_ids

        return values


class MRR(Metric):
    def __call__(
        self, ranked: Ranked | None, targets: Targets, target_mask: torch.Tensor, ks: Iterable[int]
    ) -> dict[int, float]:
        assert all(0 < k <= target_mask.shape[1] for k in ks)

        values = {}
        for k in ks:
            num_positives = targets.lengths.to(torch.float32)

            indexes = torch.argmax(target_mask[:, :k].to(torch.float32), dim=-1).to(torch.float32) + 1
            indexes[num_positives == 0] = torch.inf

            rr = 1 / indexes

            values[k] = torch.mean(rr).item()

        return values


class DCG(Metric):
    def __call__(
        self, ranked: Ranked | None, targets: Targets | None, target_mask: torch.Tensor, ks: Iterable[int]
    ) -> dict[int, float]:
        assert all(0 < k <= target_mask.shape[1] for k in ks)

        values = {}

        discounts = 1.0 / torch.log2(
            torch.arange(2, target_mask.shape[1] + 2, device=target_mask.device, dtype=torch.float32)
        )

        for k in ks:
            dcg_k = torch.sum(target_mask[:, :k] * discounts[:k], dim=1)
            values[k] = torch.mean(dcg_k).item()

        return values


class NDCG(Metric):
    def __call__(
        self, ranked: Ranked | None, targets: Targets, target_mask: torch.Tensor, ks: Iterable[int]
    ) -> dict[int, float]:

        # there was a bug: we computed (dcg_1 + ... + dcg_n) / (idcg_1 + ... + idcg_n)
        # instead of (1 / n) * (dcg_1 / idcg_1 + ... + dcg_n / idcg_n)
        # this may have affected the absolute metric values,
        # but as far as we can judge it didn't change the ranking of the models reported in the paper.

        assert all(0 < k <= target_mask.shape[1] for k in ks)

        def calc_dcg(target_mask: torch.Tensor) -> dict[int, torch.Tensor]:
            values = {}

            discounts = 1.0 / torch.log2(
                torch.arange(2, target_mask.shape[1] + 2, device=target_mask.device, dtype=torch.float32)
            )

            for k in ks:
                dcg_k = torch.sum(target_mask[:, :k] * discounts[:k], dim=1)
                values[k] = dcg_k
            return values

        actual_dcg = calc_dcg(target_mask)

        ideal_target_mask = (
            torch.arange(target_mask.shape[1], device=targets.device)[None, :] < targets.lengths[:, None]
        ).to(torch.float32)
        assert target_mask.shape == ideal_target_mask.shape

        ideal_dcg = calc_dcg(target_mask)

        def divide(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            assert x.shape == y.shape
            assert x.shape[0] == target_mask.shape[0]
            return torch.where(y == 0, 0, x / y).mean()

        ndcg_values = {k: divide(actual_dcg[k], ideal_dcg[k]).item() for k in ks}

        return ndcg_values


REGISTERED_METRIC_FN = {
    "recall": Recall(),
    "precision": Precision(),
    "mrr": MRR(),
    "dcg": DCG(),
    "ndcg": NDCG(),
    "coverage": Coverage(cut_off_ranked=False),
}


def _parse_metrics(metric_names: list[str]) -> dict[str, list[int]]:
    parsed_metrics = []

    for metric in metric_names:
        parts = metric.split('@')
        name = parts[0]

        assert len(parts) > 1, f"Invalid metric: {metric}, specify @k"

        value = int(parts[1])
        parsed_metrics.append((name, value))

    metrics = defaultdict(list)
    for m in parsed_metrics:
        metrics[m[0]].append(m[1])

    return metrics


def create_target_mask(ranked: Ranked, targets: Targets) -> torch.Tensor:
    ranked = cut_off_ranked(ranked, targets)

    assert ranked.device == targets.device
    assert ranked.item_ids.shape[0] == len(targets)

    target_mask = ranked.item_ids.new_zeros(ranked.item_ids.shape, dtype=torch.float32)

    for i, target in enumerate(tqdm(targets.item_ids, desc="Making target mask")):
        target_mask[i, torch.isin(ranked.item_ids[i], target)] = 1.0

    return target_mask


def calc_metrics(ranked: Ranked, targets: Targets, metrics: list[str]) -> dict[str, Any]:
    grouped_metrics = _parse_metrics(metrics)

    result = {}

    target_mask = create_target_mask(ranked, targets)

    for name, ks in grouped_metrics.items():
        result[name] = REGISTERED_METRIC_FN[name](ranked, targets, target_mask, ks=ks)

    return result