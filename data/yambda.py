import os
from pathlib import Path

import numpy as np
import polars as pl
import torch

from yambda.constants import Constants
from yambda.processing import timesplit


PROCESSED_DIR = "processed"


def items_file(interaction: str) -> str:
    return "items.pt" if interaction == "likes" else f"items_{interaction}.pt"


def seq_file(interaction: str) -> str:
    return f"seq_{interaction}.pt"


class YambdaItems:
    """Build RQ-VAE item corpus from precomputed audio embeddings (train items only)."""

    def __init__(
        self,
        root: str,
        size: str = "500m",
        interaction: str = "likes",
        eval_holdout_frac: float = 0.1,
        seed: int = 42,
        max_seq_len: int = 120,
    ) -> None:
        self.root = Path(root)
        self.size = size
        self.interaction = interaction
        self.eval_holdout_frac = eval_holdout_frac
        self.seed = seed
        self.max_seq_len = max_seq_len
        self.processed_paths = [str(self.root / PROCESSED_DIR / items_file(interaction))]

    def _train_item_ids(self) -> pl.DataFrame:
        print(
            f"[yambda] temporal split (test_ts={Constants.TEST_TIMESTAMP}, gap={Constants.GAP_SIZE})...",
            flush=True,
        )
        if self.interaction == "likes":
            print(f"[yambda] loading flat {self.interaction} ({self.size})...", flush=True)
            flat_path = self.root / "flat" / self.size / f"{self.interaction}.parquet"
            df = pl.scan_parquet(flat_path)
            train, _, _ = timesplit.flat_split_train_val_test(
                df,
                val_size=0,
                test_timestamp=Constants.TEST_TIMESTAMP,
            )
            return train.select("item_id").unique().sort("item_id").collect(engine="streaming")

        print(
            f"[yambda] loading sequential {self.interaction} ({self.size}), "
            f"max_seq_len={self.max_seq_len}...",
            flush=True,
        )
        seq_path = self.root / "sequential" / self.size / f"{self.interaction}.parquet"
        df = pl.scan_parquet(seq_path)
        train, _, _ = timesplit.sequential_split_train_val_test(
            df,
            val_size=0,
            test_timestamp=Constants.TEST_TIMESTAMP,
            drop_non_train_items=False,
        )
        train = train.select(
            "uid",
            pl.all().exclude("uid").list.slice(-self.max_seq_len, self.max_seq_len),
        ).filter(pl.col("item_id").list.len() > 0)
        return (
            train.select(pl.col("item_id").explode().alias("item_id"))
            .unique()
            .sort("item_id")
            .collect(engine="streaming")
        )

    def process(self, force: bool = False) -> None:
        out_path = Path(self.processed_paths[0])
        if out_path.exists() and not force:
            return

        out_path.parent.mkdir(parents=True, exist_ok=True)

        train_item_ids = self._train_item_ids()
        print(f"[yambda] train-sequence unique items: {len(train_item_ids)}", flush=True)

        print("[yambda] loading embeddings...", flush=True)
        emb_path = self.root / "embeddings.parquet"
        embeddings = (
            pl.scan_parquet(emb_path)
            .select("item_id", pl.col("normalized_embed").alias("embed"))
            .collect(engine="streaming")
        )

        merged = train_item_ids.join(embeddings, on="item_id", how="inner").sort("item_id")
        n_before = len(train_item_ids)
        n_after = len(merged)
        print(
            f"[yambda] items with embeddings: {n_after} "
            f"(dropped {n_before - n_after} train items without embedding)",
            flush=True,
        )

        embed_dim = len(merged["embed"][0])
        item_ids = merged["item_id"].to_numpy()
        item_x = np.stack(merged["embed"].to_list()).astype(np.float32)

        rng = np.random.default_rng(self.seed)
        is_train = rng.random(n_after) >= self.eval_holdout_frac

        data = {
            "item_id": torch.from_numpy(item_ids.astype(np.int64)),
            "x": torch.from_numpy(item_x),
            "is_train": torch.from_numpy(is_train),
            "embed_dim": embed_dim,
            "size": self.size,
            "interaction": self.interaction,
            "n_train_items": int(is_train.sum()),
            "n_eval_items": int((~is_train).sum()),
        }
        torch.save(data, out_path)
        print(
            f"[yambda] saved {out_path} — {data['n_train_items']} train, "
            f"{data['n_eval_items']} eval holdout, dim={embed_dim}",
            flush=True,
        )

    @property
    def data(self) -> dict:
        if not os.path.exists(self.processed_paths[0]):
            self.process()
        return torch.load(self.processed_paths[0], weights_only=False)


class YambdaSequences:
    """User sequences for TIGER decoder (corpus indices, GTS split)."""

    def __init__(
        self,
        root: str,
        size: str = "500m",
        interaction: str = "listens_fltrd",
        max_seq_len: int = 120,
    ) -> None:
        self.root = Path(root)
        self.size = size
        self.interaction = interaction
        self.max_seq_len = max_seq_len
        self.processed_paths = [str(self.root / PROCESSED_DIR / seq_file(interaction))]

    def process(self, force: bool = False) -> None:
        out_path = Path(self.processed_paths[0])
        if out_path.exists() and not force:
            return

        items_path = self.root / PROCESSED_DIR / items_file(self.interaction)
        if not items_path.exists():
            raise FileNotFoundError(
                f"Item corpus missing ({items_path}). Run RQ-VAE item processing first."
            )

        corpus = torch.load(items_path, weights_only=False)
        raw_to_corpus = {
            int(item_id): idx for idx, item_id in enumerate(corpus["item_id"].tolist())
        }

        def map_items(raw_ids: list[int]) -> list[int]:
            return [raw_to_corpus[int(i)] for i in raw_ids if int(i) in raw_to_corpus]

        print(f"[yambda] building sequences for {self.interaction}...", flush=True)
        seq_path = self.root / "sequential" / self.size / f"{self.interaction}.parquet"
        df = pl.scan_parquet(seq_path)
        train, _, test = timesplit.sequential_split_train_val_test(
            df,
            val_size=0,
            test_timestamp=Constants.TEST_TIMESTAMP,
            drop_non_train_items=False,
        )

        train_df = train.collect(engine="streaming")
        test_df = test.collect(engine="streaming")

        train_split = {"userId": [], "itemId": [], "itemId_fut": []}
        for row in train_df.iter_rows(named=True):
            mapped = map_items(row["item_id"])
            if len(mapped) < 2:
                continue
            train_split["userId"].append(int(row["uid"]))
            train_split["itemId"].append(mapped)
            train_split["itemId_fut"].append(mapped[-1])

        train_hist = {
            int(row["uid"]): map_items(row["item_id"])
            for row in train_df.iter_rows(named=True)
        }

        test_user_ids = []
        test_item_ids = []
        test_item_ids_fut = []
        for row in test_df.iter_rows(named=True):
            uid = int(row["uid"])
            hist = train_hist.get(uid)
            if not hist:
                continue
            fut = map_items(row["item_id"])
            if not fut:
                continue
            window = hist[-self.max_seq_len :]
            padded = window + [-1] * (self.max_seq_len - len(window))
            test_user_ids.append(uid)
            test_item_ids.append(padded)
            test_item_ids_fut.append(fut[0])

        data = {
            "train": {
                "userId": torch.tensor(train_split["userId"], dtype=torch.long),
                "itemId": train_split["itemId"],
                "itemId_fut": torch.tensor(train_split["itemId_fut"], dtype=torch.long),
            },
            "test": {
                "userId": torch.tensor(test_user_ids, dtype=torch.long),
                "itemId": torch.tensor(test_item_ids, dtype=torch.long),
                "itemId_fut": torch.tensor(test_item_ids_fut, dtype=torch.long),
            },
            "max_seq_len": self.max_seq_len,
            "interaction": self.interaction,
            "size": self.size,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out_path)
        print(
            f"[yambda] saved {out_path} — train_users={len(train_split['userId'])} "
            f"test_users={len(test_user_ids)} max_seq_len={self.max_seq_len}",
            flush=True,
        )

    @property
    def data(self) -> dict:
        if not os.path.exists(self.processed_paths[0]):
            self.process()
        return torch.load(self.processed_paths[0], weights_only=False)
