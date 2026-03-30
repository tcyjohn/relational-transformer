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

    temporal_neighbors_list = [0]
    run_summaries = []
    for num_temporal_neighbors in temporal_neighbors_list:
        start_time = time.time()
        run_summary = main(
            # misc
            project=f"rt_pretrain_tn{num_temporal_neighbors}",
            eval_splits=["val", "test"],
            eval_freq=1_000,
            eval_pow2=False,
            max_eval_steps=25,
            load_ckpt_path=None,
            save_ckpt_dir=f"ckpts/leave_rel-amazon/tn{num_temporal_neighbors}",
            compile_=True,
            seed=0,
            # data
            train_tasks=[t for t in all_tasks if t[0] != "rel-amazon"],
            eval_tasks=[t for t in forecast_tasks if t[0] == "rel-amazon"],
            batch_size=64,
            num_workers=8,
            max_bfs_width=256,
            num_temporal_neighbors=num_temporal_neighbors,
            # optimization — effective_batch = 64 * 2_GPUs * 2_accum = 256
            lr=1e-3,
            wd=0.1,
            lr_schedule=True,
            max_grad_norm=1.0,
            max_steps=50_001,
            grad_accum_steps=2,
            # model
            embedding_model="all-MiniLM-L12-v2",
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
        summary_dir = Path("ckpts/leave_rel-amazon")
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "pretrain_tn_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(run_summaries, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to {summary_path}")
