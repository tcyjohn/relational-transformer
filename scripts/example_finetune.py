import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from rt.main import main
from rt.tasks import all_tasks, forecast_tasks

if __name__ == "__main__":
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")

    temporal_neighbors_list = [0,32]
    run_summaries = []
    save_ckpt_base = "ckpts/rel-amazon_user-churn_ft"
    for num_temporal_neighbors in temporal_neighbors_list:
        start_time = time.time()
        run_summary = main(
            # misc
            project=f"rt_finetune_tn{num_temporal_neighbors}",
            eval_splits=["val", "test"],
            eval_freq=1_000,
            eval_pow2=False,
            max_eval_steps=40,
            # Point to a pretrained checkpoint whose d_model / architecture matches below.
            load_ckpt_path="ckpts/rel-amazon_user-churn/rel-amazon_user-churn_best.pt",
            save_ckpt_dir=f"{save_ckpt_base}/tn{num_temporal_neighbors}",
            compile_=True,
            seed=0,
            # data
            train_tasks=[("rel-amazon", "user-churn", "churn", [])],
            eval_tasks=[("rel-amazon", "user-churn", "churn", [])],
            batch_size=64,
            num_workers=8,
            max_bfs_width=256,
            num_temporal_neighbors=num_temporal_neighbors,
            # optimization — effective_batch = 64 * 2_GPUs * 2_accum = 256
            lr=1e-4,
            wd=0.0,
            lr_schedule=False,
            max_grad_norm=1.0,
            max_steps=32_679,
            grad_accum_steps=2,
            # model
            embedding_model="all-MiniLM-L12-v2",
            d_text=384,
            seq_len=1024,
            num_blocks=12,
            d_model=256,
            num_heads=8,
            d_ff=1024,
            use_full_attention=True,
        )
        elapsed_minutes = (time.time() - start_time) / 60.0
        if run_summary is not None:
            run_summary["num_temporal_neighbors"] = num_temporal_neighbors
            run_summary["elapsed_minutes"] = elapsed_minutes
            # Convert tuple keys to lists for JSON serialization
            run_summary["best_val_metrics"] = [
                [list(k), v] for k, v in run_summary["best_val_metrics"].items()
            ]
            run_summary["best_test_metrics"] = [
                [list(k), v] for k, v in run_summary["best_test_metrics"].items()
            ]
            run_summaries.append(run_summary)

    if ddp:
        dist.destroy_process_group()

    # Only rank0 writes experiment summary when launched with torchrun.
    is_rank0 = ("LOCAL_RANK" not in os.environ) or (int(os.environ["LOCAL_RANK"]) == 0)
    if is_rank0 and run_summaries:
        summary_dir = Path(save_ckpt_base)
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "finetune_tn_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(run_summaries, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to {summary_path}")
