#!/usr/bin/env python3
"""Evaluate one FD-training checkpoint without entering the training path.

This small host-side wrapper intentionally reuses the evaluator from
``scripts/main_fd_and_eval.py`` so deferred checkpoint evaluations use the
same FDr-6, Inception Score, precision, and recall implementation as the
training job's online evaluations.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.main_fd_and_eval import (  # noqa: E402
    _shutdown,
    parse_config,
    run_fdr6_evaluation,
)
from src.utils.builders import create_generation_model, create_tokenizer  # noqa: E402
from src.utils.checkpoint_util import ckpt_resume  # noqa: E402
from src.utils.distributed_util import get_global_rank  # noqa: E402
from src.utils.setup_util import setup  # noqa: E402


def evaluate_checkpoint(argv: list[str] | None = None) -> int:
    args = parse_config(argv)
    if not args.resume_from:
        raise ValueError(
            "checkpointing.resume_from must name the checkpoint to evaluate"
        )
    if not args.eval_ema_labels:
        raise ValueError(
            "optimizer.eval_ema_labels must explicitly select the evaluation weights"
        )

    wandb_logger = setup(args)
    try:
        tokenizer = create_tokenizer(args)
        model, ema_model = create_generation_model(args)
        ckpt_resume(args, model, optimizer=None, model_ema=ema_model)

        requested_labels = list(args.eval_ema_labels)
        missing = [label for label in requested_labels if label not in ema_model.labels]
        if missing:
            raise RuntimeError(
                f"checkpoint EMA labels {ema_model.labels} do not include {missing}"
            )

        rows = run_fdr6_evaluation(
            args,
            model,
            ema_model,
            tokenizer,
            wandb_logger=wandb_logger,
        )
        expected_rows = len(requested_labels) * len(args.cfg_list or [args.cfg])
        if len(rows) != expected_rows:
            raise RuntimeError(
                f"expected {expected_rows} evaluation rows, produced {len(rows)}"
            )
        if any(row["num_samples"] != args.num_images for row in rows):
            raise RuntimeError("an evaluation row has an incomplete sample count")

        if get_global_rank() == 0:
            result_path = Path(args.log_dir) / "final_eval_summary.csv"
            if not result_path.is_file():
                raise RuntimeError(f"evaluation CSV was not written: {result_path}")
            (Path(args.log_dir) / "evaluation_complete").touch()
        return 0
    finally:
        _shutdown(wandb_logger)


if __name__ == "__main__":
    raise SystemExit(evaluate_checkpoint())
