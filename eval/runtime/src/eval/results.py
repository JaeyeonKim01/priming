"""Evaluation result persistence helpers."""

from __future__ import annotations

import csv
import datetime
import logging
import os
from pathlib import Path
from typing import Sequence


logger = logging.getLogger("FD_loss")


def _ensure_eval_csv_schema(path: Path, fields: Sequence[str]) -> None:
    """Add newly introduced metric columns to an existing summary in place."""
    if not path.exists():
        return
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        current_fields = tuple(reader.fieldnames or ())
        if current_fields == tuple(fields):
            return
        if not current_fields or not set(current_fields).issubset(fields):
            raise RuntimeError(
                f"Cannot upgrade {path}: unexpected CSV columns {current_fields}"
            )
        rows = list(reader)

    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)
    logger.info("Upgraded %s with Inception Score and P&R columns", path)


def _append_eval_csv(
    args, ema_label, cfg, result, fdr, fdr6, inception_metrics
) -> None:
    path = Path(args.log_dir) / "final_eval_summary.csv"
    fields = (
        "model",
        "fd",
        "fdr",
        "fdr6",
        "is",
        "precision",
        "recall",
        "cmmd",
        "n",
        "ema_label",
        "cfg",
        "step",
        "interval_min",
        "interval_max",
        "num_sampling_steps",
        "timestamp",
    )
    _ensure_eval_csv_schema(path, fields)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for name, value in result.fd.items():
            is_inception = name == "FID(ADM)"
            writer.writerow(
                {
                    "model": name,
                    "fd": value,
                    "fdr": fdr[name],
                    "fdr6": fdr6,
                    "is": inception_metrics["is"] if is_inception else None,
                    "precision": (
                        inception_metrics["precision"] if is_inception else None
                    ),
                    "recall": inception_metrics["recall"] if is_inception else None,
                    "cmmd": None,
                    "n": result.num_samples,
                    "ema_label": ema_label,
                    "cfg": cfg,
                    "step": args.current_step,
                    "interval_min": args.interval_min,
                    "interval_max": args.interval_max,
                    "num_sampling_steps": args.num_sampling_steps,
                    "timestamp": datetime.datetime.now().isoformat(timespec="minutes"),
                }
            )
