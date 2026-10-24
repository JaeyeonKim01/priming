"""Local-only OST-style smoothing, progress logging and JSON-lines metrics."""

from collections import deque
from datetime import datetime, timedelta
import json
import math
from pathlib import Path


class SmoothedValue:
    def __init__(self, window_size=20):
        self.values = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.values.append(value)
        self.total += value
        self.count += 1

    @property
    def median(self):
        # Match torch.median (lower middle for an even population), as in OST.
        return sorted(self.values)[(len(self.values) - 1) // 2]

    @property
    def avg(self):
        return sum(self.values) / len(self.values)

    @property
    def global_avg(self):
        return self.total / self.count


class LocalMetricLogger:
    """Update every step; emit raw snapshots and OST-compatible smoothed rows.

    Windows match OST: 20 for ordinary metrics/time, log_every for throughput,
    samples seen and device memory, and one for LR. No remote logger imports.
    """
    def __init__(self, output_dir, log_every):
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.meters = {}
        self.last = {}

    def info(self, message):
        line = f"I{datetime.now():%y%m%d %H:%M:%S} distmatch] {message}"
        print(line, flush=True)
        with (self.output / "log.txt").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def update(self, values):
        converted = {key: float(value) for key, value in values.items()}
        if not all(math.isfinite(value) for value in converted.values()):
            raise ValueError("non-finite training metric")
        self.last = converted
        for key, value in converted.items():
            window = (1 if key == "lr" else self.log_every if key in {
                "samples/s/device", "samples/s", "samples_seen(M)", "device_mem(GB)"} else 20)
            self.meters.setdefault(key, SmoothedValue(window)).update(value)

    def log(self, step, total_steps, *, elapsed_seconds, eta_seconds, seconds_since_log, world_size):
        progress = {"elapsed_seconds": elapsed_seconds, "eta_seconds": eta_seconds,
                    "elapsed_device_hours": elapsed_seconds * world_size / 3600,
                    "eta_device_hours": eta_seconds * world_size / 3600}
        raw = {**self.last, **progress, "step": step, "seconds_since_log": seconds_since_log}
        smoothed = {key: meter.median for key, meter in self.meters.items()}
        smoothed.update(iteration=step - 1, step=step, iter_time=self.meters["iter_time"].avg,
                        data_time=self.meters["data_time"].avg, **progress)
        for filename, record in (("metrics.jsonl", raw), ("training_metrics.json", smoothed)):
            with (self.output / filename).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        parts = [f"Train: [{step:{len(str(total_steps))}d}/{total_steps}]",
                 f"eta: {timedelta(seconds=int(eta_seconds))}",
                 f"elapsed: {timedelta(seconds=int(elapsed_seconds))}"]
        for key in ("loss", "grad_norm"):
            meter = self.meters[key]
            parts.append(f"{key}: {meter.median:.4g} ({meter.global_avg:.4g})")
        parts.append(f"lr: {self.last['lr']:.3e}")
        for key in ("samples/s/device", "samples/s"):
            parts.append(f"{key}: {self.meters[key].avg:.2f}")
        for key in ("samples_seen(M)", "device_mem(GB)"):
            parts.append(f"{key}: {self.last[key]:.3f}")
        parts.extend(f"{key}: {self.meters[key].avg:.4f}" for key in ("iter_time", "data_time"))
        parts.extend(f"{key}: {self.meters[key].median:.4g}" for key in sorted(self.meters)
                     if key.endswith(("/loss", "/scale", "/cross_kernel_mean", "/self_kernel_mean")))
        self.info("  ".join(parts))

