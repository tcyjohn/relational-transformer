#!/usr/bin/env python3
"""Lay out a flat RDB dataset tree for relational-transformer / RelBench-style preprocessing.

Creates ``db/`` and ``tasks/`` under *dataset_root* and moves:

- ``table_<n>.parquet`` at the root → ``db/table_<n>.parquet``
- directories named ``*_task`` → ``tasks/<same_name>/``

Other files and directories (e.g. ``metadata.yaml``, ``csv_data/``) are left untouched.

Example::

    python scripts/layout_rdb_for_relbench.py \\
        /path/to/plurel_scale_raw/dag_rdb_1

After this, symlink or copy the dataset to ``$HOME/scratch/relbench/<db_name>/``.
Parquet key-value metadata (``pkey_col``, etc.) is **not** added; the Rust preprocessor
still requires those unless you extend it separately.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# Base tables: table_0.parquet, table_11.parquet, etc. (root only)
_BASE_TABLE_PATTERN = re.compile(r"^table_\d+\.parquet$")


def collect_moves(dataset_root: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Return (file_moves, warnings) for items directly under *dataset_root*."""
    moves: list[tuple[Path, Path]] = []
    warnings: list[str] = []

    db_dir = dataset_root / "db"
    tasks_dir = dataset_root / "tasks"

    for child in sorted(dataset_root.iterdir(), key=lambda p: p.name):
        if child.name in ("db", "tasks", "csv_data"):
            continue
        if child.is_file() and child.suffix == ".parquet":
            if _BASE_TABLE_PATTERN.match(child.name):
                dest = db_dir / child.name
                moves.append((child, dest))
            else:
                warnings.append(f"skip parquet (not table_<n>.parquet): {child.name}")
        elif child.is_dir() and child.name.endswith("_task"):
            dest = tasks_dir / child.name
            moves.append((child, dest))
        # else: leave as-is

    return moves, warnings


def apply_moves(
    moves: Sequence[tuple[Path, Path]],
    *,
    dry_run: bool,
) -> None:
    """Move paths; refuses to overwrite existing destinations."""
    for src, dst in moves:
        if dst.exists():
            raise FileExistsError(f"refusing to overwrite existing: {dst}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        print(f"{'[dry-run] ' if dry_run else ''}{src} -> {dst}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI and reorganize one dataset directory."""
    parser = argparse.ArgumentParser(
        description="Move base parquets into db/ and *_task dirs into tasks/."
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Dataset directory (e.g. .../plurel_scale_raw/dag_rdb_1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without moving files.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = args.dataset_root.resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 1

    moves, warnings = collect_moves(root)
    for w in warnings:
        print(f"note: {w}", file=sys.stderr)

    if not moves:
        print("nothing to move (already laid out or no matching entries).")
        return 0

    try:
        apply_moves(moves, dry_run=args.dry_run)
    except FileExistsError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
