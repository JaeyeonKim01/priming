import functools
import logging
import os
import random
import re
import signal
import socket

from datetime import timedelta

import torch
import torch.distributed as dist

logger = logging.getLogger("FD_loss")


def is_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    return dist.get_rank() if is_enabled() else 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return dist.get_world_size() if is_enabled() else 1


def is_main_process() -> bool:
    return get_global_rank() == 0


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        t = x.clone().detach().cuda() if isinstance(x, torch.Tensor) else torch.tensor(x).cuda()
        dist.all_reduce(t)
        return (t.float() / world_size).item()
    return x


def concat_all_gather(tensor, gather_dim=0) -> torch.Tensor:
    if get_world_size() == 1:
        return tensor
    return torch.cat(dist.nn.functional.all_gather(tensor), dim=gather_dim)


def broadcast_module_params(module, src=0):
    """Broadcast all parameters and buffers of a module from src rank."""
    if not is_enabled():
        return
    for p in module.parameters():
        dist.broadcast(p.data, src=src)
    for b in module.buffers():
        dist.broadcast(b.data, src=src)


def broadcast_scalar(value: float, device: str = "cuda") -> float:
    if not is_enabled():
        return value
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.broadcast(t, src=0)
    return t.item()


def broadcast_bool(value: bool, device: str = "cuda") -> bool:
    return broadcast_scalar(1.0 if value else 0.0, device) > 0.5


def _parse_slurm_node_list(s: str) -> list[str]:
    nodes = []
    for m in re.finditer(r"(([^\[]+)(?:\[([^\]]+)\])?),?", s):
        prefix, suffixes = s[m.start(2) : m.end(2)], s[m.start(3) : m.end(3)]
        for suffix in suffixes.split(","):
            span = suffix.split("-")
            if len(span) == 1:
                nodes.append(prefix + suffix)
            else:
                w = len(span[0])
                lo, hi = int(span[0]), int(span[1]) + 1
                nodes.extend(f"{prefix}{i:0{w}}" for i in range(lo, hi))
    return nodes


def _get_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@functools.lru_cache
def enable_distributed():
    env = os.environ
    if "TORCHELASTIC_RUN_ID" in env:
        pass  # torchrun already configured
    elif "SLURM_JOB_ID" in env:
        env["MASTER_ADDR"] = _parse_slurm_node_list(env["SLURM_JOB_NODELIST"])[0]
        env["MASTER_PORT"] = str(random.Random(env["SLURM_JOB_ID"]).randint(20_000, 60_000))
        env["RANK"] = env["SLURM_PROCID"]
        env["WORLD_SIZE"] = env["SLURM_NTASKS"]
        env["LOCAL_RANK"] = env["SLURM_LOCALID"]
        env["LOCAL_WORLD_SIZE"] = str(int(env["WORLD_SIZE"]) // int(env["SLURM_JOB_NUM_NODES"]))
    elif "MASTER_ADDR" not in env:
        env.update(
            MASTER_ADDR="127.0.0.1", 
            MASTER_PORT=str(_get_available_port()),
            RANK="0",
            WORLD_SIZE="1",
            LOCAL_RANK="0",
            LOCAL_WORLD_SIZE="1",
        )
    torch.cuda.set_device(int(env["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    dist.barrier(device_ids=[int(env["LOCAL_RANK"])])


# ---------------------------------------------------------------------------
# Preemption: save checkpoint before the job is killed
# ---------------------------------------------------------------------------

_preempt_requested = False
_preempt_flag_file = os.environ.get("PREEMPT_FLAG_FILE", "")


def preempt_requested() -> bool:
    if _preempt_requested:
        return True
    if _preempt_flag_file and os.path.exists(_preempt_flag_file):
        return True
    return False


def _preempt_signal_handler(signum, frame):
    global _preempt_requested
    _preempt_requested = True
    logger.info(
        f"Received signal {signum} (SIGUSR1) — will save checkpoint and exit "
        "after current step."
    )


def register_preempt_handler():
    signal.signal(signal.SIGUSR1, _preempt_signal_handler)
    if _preempt_flag_file:
        logger.info(
            f"Registered SIGUSR1 handler; preemption flag file: {_preempt_flag_file}"
        )
    else:
        logger.info("Registered SIGUSR1 handler (no PREEMPT_FLAG_FILE set).")
