"""
检查 regression 任务在 val/test 上 label 的分布。

用法:
    python scripts/check_regression_label_dist.py
    python scripts/check_regression_label_dist.py --db rel-amazon
    python scripts/check_regression_label_dist.py --db rel-amazon --table user-ltv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rt.data import RelationalDataset
from rt.tasks import forecast_reg_tasks
from torch.utils.data import DataLoader


REGR_TABLES = [
    "item-sales",
    "user-ltv",
    "item-ltv",
    "post-votes",
    "site-success",
    "study-adverse",
    "user-attendance",
    "driver-position",
    "ad-ctr",
]


def collect_labels(
    dataset: RelationalDataset,
    max_batches: int = -1,
) -> np.ndarray:
    """从 dataset 中收集所有 regression target labels。"""
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        shuffle=False,
    )
    labels_list: list[np.ndarray] = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:
            print("total batches:", len(loader))
        if max_batches > 0 and batch_idx >= max_batches:
            break
        tbs = batch["true_batch_size"]
        true_batch_size = int(tbs.item() if hasattr(tbs, "item") else tbs)
        is_targets = batch["is_targets"][:true_batch_size]  # (B, S)
        number_values = batch["number_values"][:true_batch_size]  # (B, S, 1)
        y = number_values[is_targets].float().numpy().flatten().astype(np.float64)
        labels_list.append(y)
    return np.concatenate(labels_list, axis=0) if labels_list else np.array([])


def summarize(arr: np.ndarray) -> dict:
    """计算分布摘要统计。"""
    if arr.size == 0:
        return {}
    arr = arr[~np.isnan(arr) & np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        **{f"p{p}": float(np.percentile(arr, p)) for p in percentiles},
    }


def main(
    db_filter: str | None = None,
    table_filter: str | None = None,
    batch_size: int = 64,
    seq_len: int = 1024,
    max_bfs_width: int = 256,
    num_temporal_neighbors: int = 32,
    embedding_model: str = "all-MiniLM-L12-v2",
    d_text: int = 384,
    max_batches: int = -1,
    seed: int = 0,
) -> None:
    reg_tasks = [
        (db, table, target, cols)
        for db, table, target, cols in forecast_reg_tasks
        if table in REGR_TABLES
        and (db_filter is None or db == db_filter)
        and (table_filter is None or table == table_filter)
    ]
    if not reg_tasks:
        print(f"No regression tasks match db={db_filter}, table={table_filter}")
        return

    for db_name, table_name, target_column, columns_to_drop in reg_tasks:
        print(f"\n{'='*70}")
        print(f"{db_name}/{table_name} (target: {target_column})")
        print("=" * 70)

        for split in ("train","val", "test"):
            print(f"  [{split}] initializing dataset...", flush=True)
            dataset = RelationalDataset(
                tasks=[(db_name, table_name, target_column, split, columns_to_drop)],
                batch_size=batch_size,
                seq_len=seq_len,
                rank=0,
                world_size=1,
                max_bfs_width=max_bfs_width,
                num_temporal_neighbors=num_temporal_neighbors,
                embedding_model=embedding_model,
                d_text=d_text,
                seed=seed,
            )
            print(f"  [{split}] dataset ready, shuffling...", flush=True)
            dataset.sampler.shuffle_py(0)
            print(f"  [{split}] collecting labels...", flush=True)
            labels = collect_labels(dataset, max_batches=max_batches)
            print(f"  [{split}] done.", flush=True)
            s = summarize(labels)
            if s.get("n", 0) == 0:
                print(f"  {split}: (no samples)")
                continue
            print(f"\n  {split}: n={s['n']}")
            print(f"    mean={s['mean']:.4f}, std={s['std']:.4f}")
            print(f"    min={s['min']:.4f}, max={s['max']:.4f}, median={s['median']:.4f}")
            print(f"    percentiles: p1={s['p1']:.2f}, p5={s['p5']:.2f}, p50={s['p50']:.2f}, p75={s['p75']:.2f}, p95={s['p95']:.2f}, p99={s['p99']:.2f}")

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check regression label distributions")
    parser.add_argument("--db", type=str, default=None, help="Filter by database (e.g. rel-amazon)")
    parser.add_argument("--table", type=str, default=None, help="Filter by table (e.g. user-ltv)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--max_bfs_width", type=int, default=256)
    parser.add_argument("--num_temporal_neighbors", type=int, default=32)
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L12-v2")
    parser.add_argument("--d_text", type=int, default=384)
    parser.add_argument("--max_batches", type=int, default=-1, help="Limit batches per split (-1 = all)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    main(
        db_filter=args.db,
        table_filter=args.table,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_bfs_width=args.max_bfs_width,
        num_temporal_neighbors=args.num_temporal_neighbors,
        embedding_model=args.embedding_model,
        d_text=args.d_text,
        max_batches=args.max_batches,
        seed=args.seed,
    )
