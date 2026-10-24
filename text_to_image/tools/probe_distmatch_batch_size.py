"""Find the no-replay single-GPU batch limit with fresh-process training probes.

The production train loop is used unchanged. Only checkpoint serialization is
intercepted inside a probe process; model, EMA, Adam state, data and SIM remain
resident as in training. No source checkpoint or main configuration is changed.
"""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def find_boundary(fits, start=64, ceiling=1024):
    """Return adjacent (largest fitting, first failing) integers under monotonic capacity."""
    if not 1 <= start <= ceiling:
        raise ValueError("expected 1 <= start <= ceiling")
    low, high = 0, start
    while fits(high):
        low = high
        if high == ceiling:
            raise RuntimeError(f"no failing upper bound found through {ceiling}")
        high = min(high * 2, ceiling)
    while high - low > 1:
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low, high


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def worker(args):
    import torch
    import yaml
    from unittest.mock import patch
    from mini_t2i.distmatch_config import DistMatchTrainConfig
    import mini_t2i.train_distmatch as training

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    result = {"n_c": args.n_c, "requested_steps": args.steps, "world_size": 1,
              "status": "error", "completed_steps": 0, "checkpoint_writes": 0}
    started = time.monotonic()
    try:
        if "RANK" in os.environ or int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("batch probes must be standalone single-GPU processes")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"expected exactly one visible GPU, got {torch.cuda.device_count()}")
        torch.cuda.set_device(0)
        prop = torch.cuda.get_device_properties(0)
        result.update(gpu=prop.name, gpu_capacity_gib=prop.total_memory / 2**30)
        cfg = DistMatchTrainConfig.from_yaml(args.config)
        if cfg.use_replay or cfg.b != 1 or cfg.n_pos != 1 or cfg.mode != "marginal":
            raise ValueError("probe requires the requested no-replay, B=1, N_pos=1 marginal recipe")
        cfg = replace(cfg, n_c=args.n_c, num_steps=args.steps, resume_from=None,
                      output_dir=str(output / "run"), vis_every=0, vis_at_start=False,
                      ckpt_every=args.steps, log_every=1).validate()
        effective_config = output / "probe_config.yml"
        with effective_config.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False)

        def inspect_instead_of_save(*call_args, **kwargs):
            optimizer = call_args[4]
            if not optimizer.state:
                raise RuntimeError("optimizer state was not allocated")
            result["optimizer_state_entries"] = len(optimizer.state)
            result["checkpoint_skipped_at_step"] = call_args[5]

        # This interception affects only this disposable worker. All forward,
        # backward, gradient checks, Adam/EMA updates and metric logging are real.
        with patch.object(training, "save_checkpoint", side_effect=inspect_instead_of_save):
            training.main(["--config", str(effective_config)])
        records = [json.loads(line) for line in (output / "run/metrics.jsonl").read_text().splitlines()]
        if [r["step"] for r in records] != list(range(1, args.steps + 1)):
            raise RuntimeError("missing completed optimizer updates")
        if not all(r["grad_norm"] > 0 for r in records):
            raise RuntimeError("zero parameter gradients in probe")
        result.update(status="pass", completed_steps=len(records),
                      last_grad_norm=records[-1]["grad_norm"],
                      last_loss=records[-1]["loss"],
                      last_cross_kernel_means={k: v for k, v in records[-1].items() if k.endswith("/cross_kernel_mean")})
    except torch.OutOfMemoryError as exc:
        result.update(status="oom", error=str(exc))
        traceback.print_exc()
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        metrics = output / "run/metrics.jsonl"
        if metrics.exists():
            result["completed_steps"] = len(metrics.read_text().splitlines())
        if torch.cuda.is_initialized():
            result.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        result["wall_seconds"] = time.monotonic() - started
        write_json(output / "result.json", result)
        print("[probe-result] " + json.dumps(result), flush=True)
    return 0 if result["status"] in {"pass", "oom"} else 1


def controller(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "input_config.yml"
    shutil.copyfile(args.config, snapshot)
    results = []
    environment = os.environ.copy()
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    def probe(n_c, steps, label):
        trial = output / f"nc_{n_c:04d}_{label}"
        log = output / f"nc_{n_c:04d}_{label}.log"
        print(f"[batch-search] probing n_c={n_c}, steps={steps}, log={log}", flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config", str(snapshot),
                   "--output-dir", str(trial), "--n-c", str(n_c), "--steps", str(steps)]
        with log.open("x", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle,
                                       stderr=subprocess.STDOUT, timeout=args.probe_timeout)
        path = trial / "result.json"
        if not path.exists():
            raise RuntimeError(f"probe terminated without result (exit={completed.returncode}); inspect {log}")
        result = json.loads(path.read_text())
        result["log"] = str(log)
        results.append(result)
        with (output / "results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, allow_nan=False) + "\n")
        print(f"[batch-search] n_c={n_c}: {result['status']}; completed={result['completed_steps']}; "
              f"peak_allocated_gib={result.get('peak_allocated_gib')}", flush=True)
        if completed.returncode != 0 or result["status"] not in {"pass", "oom"}:
            raise RuntimeError(f"non-memory probe failure; inspect {log}: {result.get('error')}")
        return result

    try:
        low, high = find_boundary(lambda n: probe(n, args.steps, "search")["status"] == "pass",
                                  start=args.start, ceiling=args.ceiling)
        search_limit = low
        # Longer, fresh-process verification at the boundary. If extra steps
        # expose OOM, lower the candidate until one completes all verify steps.
        while low > 0:
            confirmed = probe(low, args.verify_steps, "verify")
            if confirmed["status"] == "pass":
                break
            high, low = low, low - 1
        if low == 0:
            raise RuntimeError("no batch size completed the full training probe")
        summary = {"status": "complete", "max_verified_n_c": low, "first_failing_n_c": high,
                   "short_probe_limit": search_limit, "probe_steps": args.steps,
                   "verification_steps": args.verify_steps, "world_size": 1, "use_replay": False,
                   "verification": confirmed, "results": results,
                   "scope": "single-GPU empirical limit; no DDP buffers/collectives; no checkpoint writes or previews",
                   "main_config_modified": False, "main_job_resubmitted": False}
        write_json(output / "summary.json", summary)
        print("[batch-search] COMPLETE " + json.dumps({k: v for k, v in summary.items() if k not in {"results", "verification"}}), flush=True)
        return 0
    except Exception as exc:
        write_json(output / "summary.json", {"status": "error", "error": str(exc), "results": results,
                                             "main_config_modified": False, "main_job_resubmitted": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/finetune_distmatch.yml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--n-c", type=int)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--verify-steps", type=int, default=10)
    parser.add_argument("--start", type=int, default=64)
    parser.add_argument("--ceiling", type=int, default=1024)
    parser.add_argument("--probe-timeout", type=int, default=600)
    args = parser.parse_args()
    if args.steps < 2 or args.verify_steps < args.steps or (args.worker and (args.n_c is None or args.n_c < 1)):
        parser.error("require steps >= 2, verify_steps >= steps and positive worker n_c")
    return worker(args) if args.worker else controller(args)


if __name__ == "__main__":
    sys.exit(main())
