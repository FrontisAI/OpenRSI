from argparse import Namespace
from copy import deepcopy
from typing import Callable

import psutil
import torch

from slime.utils import wandb_utils
from slime.utils.timer import Timer


def get_memory_stats():
    """Get current GPU and CPU memory usage statistics."""
    memory_stats = {}

    # GPU memory stats
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device)
        memory_stats.update(
            {
                "gpu_allocated_gb": round(torch.cuda.memory_allocated(device) / (1024**3), 2),
                "gpu_reserved_gb": round(torch.cuda.memory_reserved(device) / (1024**3), 2),
                "gpu_free_gb": round(free / (1024**3), 2),
                "gpu_total_gb": round(total / (1024**3), 2),
                "gpu_used_gb": round((total - free) / (1024**3), 2),
                "gpu_utilization": round((total - free) / total * 100, 2),
            }
        )

    # CPU memory stats
    mem = psutil.virtual_memory()
    memory_stats.update(
        {
            "cpu_used_gb": round(mem.used / (1024**3), 2),
            "cpu_available_gb": round(mem.available / (1024**3), 2),
            "cpu_total_gb": round(mem.total / (1024**3), 2),
            "cpu_percent": round(mem.percent, 2),
        }
    )

    return memory_stats


def log_perf_data_raw(
    rollout_id: int, args: Namespace, is_primary_rank: bool, compute_total_fwd_flops: Callable
) -> None:
    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    timer_instance.reset()

    if not is_primary_rank:
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}

    # Add memory statistics if enabled
    memory_stats = get_memory_stats()
    for key, val in memory_stats.items():
        log_dict[f"memory/{key}"] = val

    if ("perf/actor_train_time" in log_dict) and (compute_total_fwd_flops is not None):
        total_fwd_flops = compute_total_fwd_flops(seq_lens=timer_instance.seq_lens)

        if "perf/log_probs_time" in log_dict:
            log_dict["perf/log_probs_tflops"] = total_fwd_flops / log_dict["perf/log_probs_time"]

        if "perf/ref_log_probs_time" in log_dict:
            log_dict["perf/ref_log_probs_tflops"] = total_fwd_flops / log_dict["perf/ref_log_probs_time"]

        if log_dict["perf/actor_train_time"] > 0:
            log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["perf/actor_train_time"]
            log_dict["perf/actor_train_tok_per_s"] = sum(timer_instance.seq_lens) / log_dict["perf/actor_train_time"]

    if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
        total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
        if total_time > 0:
            log_dict["perf/step_time"] = total_time
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time

    print(f"perf {rollout_id}: {log_dict}")

    step = (
        rollout_id
        if not args.wandb_always_use_train_step
        else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    )
    if args.use_wandb:
        log_dict["rollout/step"] = step
        wandb_utils.log_metrics(log_dict)

    if args.use_tensorboard:
        from slime.utils.tensorboard_utils import _TensorboardAdapter

        tb = _TensorboardAdapter(args)
        tb.log(data=log_dict, step=step)
