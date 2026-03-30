"""Continued pretraining with leave-one-out over ``dag_rdb_*`` under ``$HOME/scratch/pre``.

For each holdout dataset *H*, trains on every other ``dag_rdb_*`` (one primary task per DB,
inferred from ``table_info.json`` / ``column_index.json``) and evaluates on *H*'s task — same
pattern as training on all rel-amazon tasks except ``user-churn`` while evaluating on
``user-churn``.

Override preprocessed root with env ``RT_PRE_ROOT`` (default: ``$HOME/scratch/pre``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from rt.main import main

log = logging.getLogger(__name__)

# Checkpoint to continue from (must match model hyperparameters below).
LOAD_CKPT_PATH = "/data/caijunyu/relational-transformer/ckpts/leave_rel-amazon/tn32/steps=9000.pt"
SAVE_CKPT_BASE = "/data/caijunyu/scratch/rt_ckpts/contd_pretrain_dag_loo"

# Holdout loop slice: ``holdout_names = all_dags[HOLDOUT_START:HOLDOUT_END]`` (end exclusive).
# ``None`` = run leave-one-out for every dataset; an int = only the first N holdouts (e.g. 24 runs).
HOLDOUT_START = 0
HOLDOUT_END: int | None = 24

EMBEDDING_MODEL = "all-MiniLM-L12-v2"

# Files the Rust sampler (rustler/src/fly.rs) requires per dataset.
_REQUIRED_FILES = [
    "table_info.json",
    "column_index.json",
    "nodes.rkyv",
    "offsets.rkyv",
    "p2f_adj.rkyv",
    f"text_emb_{EMBEDDING_MODEL}.bin",
]


def _pre_root() -> Path:
    """Return directory containing ``dag_rdb_*/`` preprocessed trees."""
    override = os.environ.get("RT_PRE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(os.environ.get("HOME", ".")).expanduser() / "scratch" / "pre"


def _natural_dag_key(name: str) -> tuple[int, int] | tuple[int, str]:
    """Sort key so ``dag_rdb_2`` precedes ``dag_rdb_10``."""
    m = re.fullmatch(r"dag_rdb_(\d+)", name)
    if m:
        return (0, int(m.group(1)))
    return (1, name)


def _has_train_task(table_info: dict[str, Any]) -> bool:
    """Return True if *table_info* contains at least one ``*_task:Train`` entry."""
    return any(
        k.endswith(":Train") and "_task" in k.split(":", 1)[0]
        for k in table_info
    )


def list_dag_rdb_datasets(pre_root: Path) -> list[str]:
    """List ``dag_rdb_*`` names under *pre_root* that have usable task metadata.

    A dataset is considered usable when:
    - It is a directory containing both ``table_info.json`` and ``column_index.json``.
    - ``table_info.json`` has at least one ``*_task:Train`` key (i.e. a trainable task).

    Args:
        pre_root: Root path (e.g. ``~/scratch/pre``).

    Returns:
        Sorted dataset directory basenames.
    """
    if not pre_root.is_dir():
        raise FileNotFoundError(f"Pre root is not a directory: {pre_root}")

    names: list[str] = []
    skipped: list[str] = []
    for child in pre_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("dag_rdb_"):
            continue
        missing = [f for f in _REQUIRED_FILES if not (child / f).is_file()]
        if missing:
            skipped.append(child.name)
            log.warning("Skipping %s: missing files %s", child.name, missing)
            continue
        with (child / "table_info.json").open(encoding="utf-8") as f:
            table_info = json.load(f)
        if not _has_train_task(table_info):
            skipped.append(child.name)
            continue
        names.append(child.name)

    if skipped:
        log.warning("Skipped %d incomplete dag_rdb_* dirs: %s", len(skipped), skipped[:10])

    return sorted(names, key=_natural_dag_key)


def _target_column_hint_from_task_table(table_name: str) -> str | None:
    """Map ``..._feature_<n>_task`` to ``feature_<n>`` (prediction column in the naming scheme).

    Example:
        ``table_3_feature_11_task`` -> ``feature_11``.
    """
    m = re.search(r"_feature_(\d+)_task\Z", table_name)
    if m:
        return f"feature_{m.group(1)}"
    return None


def infer_task_tuple(
    pre_root: Path, db_name: str
) -> tuple[str, str, str, list[str]] | None:
    """Infer ``(db_name, task_table, target_column, leakage_cols)`` for one DAG RDB dataset.

    Table: prefers ``table_1_feature_1_task`` if it has a ``:Train`` split, otherwise the
    lexicographically smallest ``*_task`` table with ``:Train``.

    Target column: task tables are named ``table_*_feature_<n>_task``; the intended label is
    ``feature_<n>`` when that key exists in ``column_index.json``. If the name does not parse or
    ``feature_<n>`` is missing, falls back to the largest ``feature_<k>`` for that table.

    Args:
        pre_root: Root containing ``{db_name}/table_info.json`` etc.
        db_name: Dataset basename (e.g. ``dag_rdb_0``).

    Returns:
        A task quadruple, or ``None`` if the dataset cannot produce a valid task (logged as
        warning).
    """
    table_path = pre_root / db_name / "table_info.json"
    col_path = pre_root / db_name / "column_index.json"
    with table_path.open(encoding="utf-8") as f:
        table_info: dict[str, Any] = json.load(f)
    with col_path.open(encoding="utf-8") as f:
        column_index: dict[str, int] = json.load(f)

    train_tables: list[str] = []
    for key in table_info:
        if not key.endswith(":Train"):
            continue
        table_name = key.split(":", 1)[0]
        if "_task" not in table_name:
            continue
        train_tables.append(table_name)

    if not train_tables:
        log.warning("Skipping %s: no *_task:Train in %s", db_name, table_path)
        return None

    preferred = "table_1_feature_1_task"
    if preferred in train_tables:
        table_name = preferred
    else:
        table_name = sorted(train_tables)[0]

    suffix = f" of {table_name}"
    candidates: list[str] = []
    for k in column_index:
        if not k.startswith("feature_") or not k.endswith(suffix):
            continue
        candidates.append(k[: -len(suffix)])

    if not candidates:
        log.warning("Skipping %s: no feature_* column for %s in %s", db_name, table_name, col_path)
        return None

    name_hint = _target_column_hint_from_task_table(table_name)
    if name_hint is not None and name_hint in candidates:
        target = name_hint
    else:
        def feature_rank(col: str) -> int:
            body = col.removeprefix("feature_")
            return int(body) if body.isdigit() else -1

        target = max(candidates, key=feature_rank)

    leakage: list[str] = []
    return (db_name, table_name, target, leakage)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    pre_root = _pre_root()
    all_dags = list_dag_rdb_datasets(pre_root)
    if not all_dags:
        raise SystemExit(f"No dag_rdb_* datasets found under {pre_root}")

    end = HOLDOUT_END if HOLDOUT_END is not None else len(all_dags)
    holdout_names = all_dags[HOLDOUT_START:end]
    log.info(
        "Holdout slice [%s:%s] → %d holdouts (of %d total dag_rdb_*)",
        HOLDOUT_START,
        end,
        len(holdout_names),
        len(all_dags),
    )

    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")

    # Build task cache; skip datasets whose metadata is incomplete.
    task_cache: dict[str, tuple[str, str, str, list[str]]] = {}
    for db in all_dags:
        tup = infer_task_tuple(pre_root, db)
        if tup is not None:
            task_cache[db] = tup

    usable_dbs = [db for db in all_dags if db in task_cache]
    log.info("Usable datasets: %d / %d", len(usable_dbs), len(all_dags))

    run_summaries: list[dict[str, Any]] = []

    for holdout in holdout_names:
        if holdout not in task_cache:
            log.warning("Holdout %s has no valid task — skipping.", holdout)
            continue
        train_tasks = [task_cache[db] for db in usable_dbs if db != holdout]
        if not train_tasks:
            log.warning("Holdout %s: no other usable datasets for training — skipping.", holdout)
            continue
        eval_tasks = [task_cache[holdout]]

        start_time = time.time()
        run_summary = main(
            project=f"rt_contd_dag_loo_{holdout}",
            eval_splits=["val", "test"],
            eval_freq=1_000,
            eval_pow2=False,
            max_eval_steps=40,
            load_ckpt_path=LOAD_CKPT_PATH,
            save_ckpt_dir=f"{SAVE_CKPT_BASE}/{holdout}",
            compile_=True,
            seed=0,
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            batch_size=64,
            num_workers=8,
            max_bfs_width=256,
            num_temporal_neighbors=0,
            # optimization — effective_batch = 64 * 2_GPUs * 2_accum = 256
            lr=1e-3,
            wd=0.1,
            lr_schedule=False,
            max_grad_norm=1.0,
            max_steps=2**12+1,
            grad_accum_steps=2,
            embedding_model=EMBEDDING_MODEL,
            d_text=384,
            seq_len=1024,
            num_blocks=12,
            d_model=256,
            num_heads=8,
            d_ff=1024,
            use_full_attention=False,
        )
        elapsed_minutes = (time.time() - start_time) / 60.0
        if run_summary is not None:
            run_summary["holdout_db"] = holdout
            run_summary["elapsed_minutes"] = elapsed_minutes
            run_summary["best_val_metrics"] = [
                [list(k), v] for k, v in run_summary["best_val_metrics"].items()
            ]
            run_summary["best_test_metrics"] = [
                [list(k), v] for k, v in run_summary["best_test_metrics"].items()
            ]
            run_summaries.append(run_summary)

    if ddp:
        dist.destroy_process_group()

    is_rank0 = ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0)
    if is_rank0 and run_summaries:
        summary_dir = Path(SAVE_CKPT_BASE)
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "contd_pretrain_dag_loo_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(run_summaries, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to {summary_path}")
