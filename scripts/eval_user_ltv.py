"""
加载 checkpoint 对 user-ltv 做 eval，并报告 R² 及预测值分布（回答占比）。

用法:
    python scripts/eval_user_ltv.py
    python scripts/eval_user_ltv.py --ckpt ckpts/leave_rel-amazon/tn32/steps=6000.pt
    python scripts/eval_user_ltv.py --max_batches 10  # 限制 batch 数用于快速测试
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rt.data import RelationalDataset
from rt.model import RelationalTransformer


# 与 example_pretrain 一致的模型配置
DEFAULT_CKPT = "ckpts/leave_rel-amazon/tn32/steps=5000.pt"
NUM_TEMPORAL_NEIGHBORS = 32
BATCH_SIZE = 64
SEQ_LEN = 1024
MAX_BFS_WIDTH = 256
EMBEDDING_MODEL = "all-MiniLM-L12-v2"
D_TEXT = 384
NUM_BLOCKS = 12
D_MODEL = 256
NUM_HEADS = 8
D_FF = 1024
USE_FULL_ATTENTION = False


def report_pred_distribution(
    preds: np.ndarray,
    labels: np.ndarray,
    split: str,
    n_bins: int = 10,
) -> None:
    """报告预测值分布（回答占比）及与 label 的对比。

    区间基于 pred 与 label 的 Union 范围 [min, max] 等宽分桶，
    每个区间内同时统计 pred 与 label 的占比。
    注：pred/label 均为预处理后的标准化值（pre.rs: log1p + z-score）。
    """
    valid = ~np.isnan(preds) & np.isfinite(preds) & ~np.isnan(labels) & np.isfinite(labels)
    preds = preds[valid]
    labels = labels[valid]
    if preds.size == 0:
        print(f"  [{split}] 无有效预测")
        return

    # 用 pred ∪ label 的并集范围建桶：等宽区间 [min, max]，避免分位数重复导致退化区间
    v_min = min(preds.min(), labels.min())
    v_max = max(preds.max(), labels.max())
    if v_max <= v_min:
        v_max = v_min + 1e-6  # 全同值时的退化保护
    edges = np.linspace(v_min, v_max, n_bins + 1)
    edges[-1] += 1e-6  # 确保右闭端被包含
    pred_bins = np.digitize(preds, edges[1:-1])
    pred_bins = np.clip(pred_bins, 0, n_bins - 1)
    label_bins = np.digitize(labels, edges[1:-1])
    label_bins = np.clip(label_bins, 0, n_bins - 1)

    print(f"\n  [{split}] 预测值分布（回答占比，基于 pred∪label 等宽区间）:")
    print(f"  {'区间':<24} {'预测占比':>10} {'label占比':>10} {'样本数':>8}")
    print("  " + "-" * 56)

    for i in range(n_bins):
        pred_pct = 100 * np.mean(pred_bins == i)
        label_pct = 100 * np.mean(label_bins == i)
        n_pred = int(np.sum(pred_bins == i))
        low_p, high_p = edges[i], edges[i + 1]
        interval = f"[{low_p:.2f}, {high_p:.2f})"
        print(f"  {interval:<24} {pred_pct:>9.1f}% {label_pct:>9.1f}% {n_pred:>8}")

    print(f"\n  [{split}] 预测值统计: mean={np.mean(preds):.4f}, std={np.std(preds):.4f}, "
          f"min={np.min(preds):.4f}, max={np.max(preds):.4f}")
    print(f"  [{split}] 真实值统计: mean={np.mean(labels):.4f}, std={np.std(labels):.4f}, "
          f"min={np.min(labels):.4f}, max={np.max(labels):.4f}")


def run_eval(
    ckpt_path: str,
    max_batches: int = -1,
    eval_splits: tuple[str, ...] = ("val", "test"),
    compile_model: bool = False,
    seed: int = 0,
) -> None:
    """加载 checkpoint，在 user-ltv 上评估并报告指标与预测分布。"""
    ckpt_path = Path(ckpt_path).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {ckpt_path}")

    net = RelationalTransformer(
        num_blocks=NUM_BLOCKS,
        d_model=D_MODEL,
        d_text=D_TEXT,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        use_full_attention=USE_FULL_ATTENTION,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(state_dict)
    net = net.to(device)
    net = net.to(torch.bfloat16)
    net.eval()

    if compile_model:
        net = torch.compile(net, dynamic=False)

    task = ("rel-amazon", "user-ltv", "ltv", [])

    for split in eval_splits:
        print(f"\n{'='*70}")
        print(f"rel-amazon/user-ltv [{split}]")
        print("=" * 70)

        dataset = RelationalDataset(
            tasks=[(task[0], task[1], task[2], split, task[3])],
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            rank=0,
            world_size=1,
            max_bfs_width=MAX_BFS_WIDTH,
            num_temporal_neighbors=NUM_TEMPORAL_NEIGHBORS,
            embedding_model=EMBEDDING_MODEL,
            d_text=D_TEXT,
            seed=seed if split == "train" else 0,
        )
        dataset.sampler.shuffle_py(0)

        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
            shuffle=False,
            in_order=True,
        )

        preds_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []

        total_steps = min(max_batches, len(loader)) if max_batches > 0 else len(loader)
        pbar = tqdm(total=total_steps, desc=f"eval {split}")

        with torch.inference_mode():
            for batch_idx, batch in enumerate(loader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break

                true_batch_size = batch.pop("true_batch_size")
                for k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)

                batch["masks"][true_batch_size:, :] = False
                batch["is_targets"][true_batch_size:, :] = False
                batch["is_padding"][true_batch_size:, :] = True

                _, yhat_dict = net(batch)
                yhat = yhat_dict["number"][batch["is_targets"]]
                y = batch["number_values"][batch["is_targets"]].flatten()

                preds_list.append(yhat.flatten().float().cpu())
                labels_list.append(y.float().cpu())
                pbar.update(1)

        pbar.close()

        preds = torch.cat(preds_list, dim=0).numpy()
        labels = torch.cat(labels_list, dim=0).numpy()

        r2 = r2_score(labels, preds)
        print(f"\n  [{split}] R² = {r2:.6f} (n={len(preds)})")

        report_pred_distribution(preds, labels, split)

    print("\n" + "=" * 70)
    print("Eval 完成")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval user-ltv with checkpoint, report R² and pred distribution")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=DEFAULT_CKPT,
        help=f"Checkpoint path (default: {DEFAULT_CKPT})",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=-1,
        help="Limit eval batches per split (-1 = all)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["val", "test"],
        help="Eval splits (default: val test)",
    )
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for model")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_eval(
        ckpt_path=args.ckpt,
        max_batches=args.max_batches,
        eval_splits=tuple(args.splits),
        compile_model=args.compile,
        seed=args.seed,
    )
