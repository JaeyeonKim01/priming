import argparse
import datetime
import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from typing import Any

import torch
import torch.distributed
import wandb
from rich.logging import RichHandler
from typing_extensions import override

from .distributed_util import get_global_rank, is_enabled, is_main_process

logger = logging.getLogger("FD_loss")


def move_to_device(obj: Any, device: torch.device) -> Any:
    """recursively move tensors to *device*."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(o, device) for o in obj)
    return obj


# ---------------------------------------------------------------------------
# smoothed value / metric logger
# ---------------------------------------------------------------------------

class SmoothedValue:
    """track a series of values and provide smoothed statistics."""

    def __init__(self, window_size: int = 20, fmt: str | None = None):
        self.deque: deque[float] = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value: float, num: int = 1) -> None:
        self.deque.append(value)
        self.count += num
        self.total += value * num

    def synchronize_between_processes(self) -> None:
        if not is_enabled():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        self.count, self.total = int(t[0].item()), t[1].item()

    @property
    def median(self) -> float:
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self) -> float:
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / self.count

    @property
    def max(self) -> float:
        return max(self.deque)

    @property
    def value(self) -> float:
        return self.deque[-1]

    @override
    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg,
            max=self.max, value=self.value,
        )


class MetricLogger:
    def __init__(self, delimiter: str = "\t", output_file: str | None = None, prefetch: bool = False):
        self.meters: defaultdict[str, SmoothedValue] = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.output_file = output_file
        self.prefetch = prefetch
        logger.info(f"MetricLogger: output_file={output_file}, prefetch={prefetch}")

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr: str):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")

    @override
    def __str__(self) -> str:
        return self.delimiter.join(f"{k}: {v}" for k, v in self.meters.items())

    def synchronize_between_processes(self) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def dump_in_output_file(self, iteration: int, iter_time: float, data_time: float) -> None:
        if self.output_file is None or not is_main_process():
            return
        row = dict(iteration=iteration, iter_time=iter_time, data_time=data_time)
        row.update({k: v.median for k, v in self.meters.items()})
        with open(self.output_file, "a") as f:
            f.write(json.dumps(row) + "\n")

    def log_every(
        self,
        iterable,
        print_freq: int,
        header: str | None = None,
        n_iterations: int | None = None,
        start_iteration: int = 0,
    ):
        i = start_iteration
        header = header or ""
        start_time = end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")

        if n_iterations is None:
            if not hasattr(iterable, "__len__"):
                raise ValueError("n_iterations must be provided for iterables without __len__")
            n_iterations = len(iterable)

        w = len(str(n_iterations))
        parts = [header, f"[{{0:{w}d}}/{n_iterations}]",
                 "eta: {eta}", "elapsed: {elapsed}", "{meters}",
                 "time: {time}", "data: {data}"]
        log_msg = self.delimiter.join(parts)

        for obj in iterable:
            if self.prefetch:
                obj = move_to_device(obj, torch.device("cuda"))
            data_time.update(time.time() - end)
            yield (i, obj)
            iter_time.update(time.time() - end)

            if i % print_freq == 0 or i == n_iterations - 1:
                self.dump_in_output_file(i, iter_time.avg, data_time.avg)
                eta = str(datetime.timedelta(seconds=int(iter_time.global_avg * (n_iterations - i))))
                elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                fmt_kw = dict(eta=eta, elapsed=elapsed, meters=str(self),
                              time=str(iter_time), data=str(data_time))
                logger.info(log_msg.format(i, n_iterations, **fmt_kw))

            i += 1
            end = time.time()
            if i >= n_iterations:
                break

        total = time.time() - start_time
        logger.info(f"{header} Total time: {datetime.timedelta(seconds=int(total))} "
                     f"({total / n_iterations:.6f} s / it)")


# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------

class WandbLogger:
    def __init__(
        self,
        config: argparse.Namespace,
        entity: str,
        project: str,
        name: str,
        log_dir: str,
        run_id: str | None = None,
    ):
        self.run = wandb.init(config=config, entity=entity, project=project,
                              name=name, dir=log_dir, resume="allow", id=run_id)
        self.run_id = self.run.id
        self.step = 0
        self.run.log_code(".")

    def update(self, metrics, step: int | None = None) -> None:
        log_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in metrics.items() if v is not None}
        try:
            wandb.log(log_dict, step=step or self.step)
        except Exception as e:
            logger.error(f"wandb logging failed: {e}")
        if step is not None:
            self.step = step

    def finish(self) -> None:
        try:
            wandb.finish()
        except Exception as e:
            logger.error(f"wandb finish failed: {e}")


# ---------------------------------------------------------------------------
# setup helpers
# ---------------------------------------------------------------------------

def setup_logging(output: str, name: str = "FD_loss", rank0_log_only: bool = True) -> None:
    """configure file + console logging."""
    logging.captureWarnings(True)

    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.propagate = False

    fmt = "%(levelname).1s%(asctime)s %(filename)s:%(lineno)s] %(message)s"
    datefmt = "%y%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # console handler (main process only)
    if is_main_process():
        if sys.stdout.isatty():
            ch = RichHandler(markup=True, show_time=False, show_level=False, show_path=False)
        else:
            ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        log.addHandler(ch)

    # file handler
    if output:
        if os.path.splitext(output)[-1] in (".txt", ".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if not is_main_process() and not rank0_log_only:
            rank = get_global_rank()
            base, ext = os.path.splitext(filename)
            filename = f"{base}.rank{rank}{ext}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fh = logging.StreamHandler(open(filename, "a"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)


def setup_wandb(args: argparse.Namespace, entity: str, project: str,
                name: str, log_dir: str) -> WandbLogger:
    """setup wandb with resume capability."""
    id_path = os.path.join(log_dir, "wandb_run_id.txt")
    run_id = None
    if os.path.exists(id_path):
        with open(id_path) as f:
            run_id = f.readlines()[-1].strip()

    wandb_logger = WandbLogger(config=args, entity=entity, project=project,
                     name=name, log_dir=log_dir, run_id=run_id)
    if run_id is None:
        with open(id_path, "a") as f:
            f.write(wandb_logger.run_id + "\n")
    return wandb_logger
