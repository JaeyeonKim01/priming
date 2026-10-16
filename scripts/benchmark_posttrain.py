"""Bounded post-training benchmark using the production runner, with per-rank telemetry.

Example: torchrun ... scripts/benchmark_posttrain.py --steps 130 -- \
    --config configs/posttrain/b160_fast.yaml --output outputs/benchmark/b160
The separator forwards the remaining arguments to posttrain.py. No production
run is allowed: this wrapper always supplies a bounded --smoke-steps count.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import posttrain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=130)
    parser.add_argument('--capture-gradient', action='store_true',
                        help='Save the first synchronized parameter gradient on rank zero')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.steps <= 200:
        parser.error('Benchmarks must be bounded to 1–200 updates per invocation')
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    if any(arg.split('=')[0] == '--smoke-steps' for arg in arguments):
        parser.error('Use --steps to bound the benchmark')
    records = []
    current = {}
    original_step, original_update, original_run = posttrain.train_step, posttrain.optimizer_step, posttrain.run_training

    def train_step(*values, **kwargs):
        import torch
        # Fingerprint the main noise stream before drawing; no random numbers consumed.
        current['rng'] = hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
        torch.cuda.synchronize()
        current['started'] = time.perf_counter()
        result = original_step(*values, **kwargs)
        current['loss'] = result[0]
        return result

    def optimizer_step(model, optimizer, ema, step, **kwargs):
        import torch
        if args.capture_gradient and not records and int(os.environ['RANK']) == 0:
            torch.save({name: p.grad.detach().cpu() for name, p in model.named_parameters()
                        if p.grad is not None}, current['output']/'first_gradient.pt')
        norm, lr = original_update(model, optimizer, ema, step, **kwargs)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - current['started']
        records.append(dict(step=step+1, seconds=seconds, loss=float(current['loss']),
            grad_norm=float(norm), lr=lr, noise_rng_sha256=current['rng'],
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        return norm, lr

    def run_training(config, model, optimizer, ema, matcher, reference, stream, output, **kwargs):
        import torch
        current['output'] = Path(output)
        restored = None
        if kwargs.get('checkpoint') is not None:
            from utils.posttrain_checkpoint import snapshot_rng
            import numpy as np
            state = torch.load(kwargs['checkpoint'], mmap=True, map_location='cpu', weights_only=False)
            def equal(actual, expected):
                if isinstance(expected, torch.Tensor):
                    return isinstance(actual, torch.Tensor) and torch.equal(actual.cpu(), expected.cpu())
                if isinstance(expected, np.ndarray):
                    return np.array_equal(actual, expected)
                if isinstance(expected, dict):
                    return actual.keys() == expected.keys() and all(equal(actual[k], v) for k, v in expected.items())
                if isinstance(expected, (list, tuple)):
                    return len(actual) == len(expected) and all(equal(a, b) for a, b in zip(actual, expected))
                return actual == expected
            restored = dict(model=equal(model.state_dict(), state['model']),
                optimizer=equal(optimizer.state_dict(), state['optimizer']),
                ema=equal(ema.state_dict(), state['model_ema']),
                rng=equal(snapshot_rng(), state['mmd_rng_state']['rank_states'][stream.rank]),
                stream=equal(stream.state_dict(), state['sinkhorn_ot_stream_state']['rank_states'][stream.rank]))
            if not all(restored.values()):
                raise AssertionError(f'Restored state differs on rank {stream.rank}: {restored}')
            del state
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        passed = False
        try:
            result = original_run(config, model, optimizer, ema, matcher, reference, stream, output, **kwargs)
            passed = True
            return result
        finally:
            target = Path(output)/'benchmark'
            target.mkdir(parents=True, exist_ok=True)
            completed = kwargs.get('completed', 0)
            payload = dict(passed=passed, rank=stream.rank, world_size=stream.world_size,
                gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                job_id=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename,
                omp_num_threads=os.environ.get('OMP_NUM_THREADS'),
                restored_exactly=restored, elapsed_seconds=time.perf_counter()-started, records=records)
            (target/f'rank_{stream.rank}_from_{completed:06d}.json').write_text(json.dumps(payload, indent=2)+'\n')

    posttrain.train_step, posttrain.optimizer_step, posttrain.run_training = train_step, optimizer_step, run_training
    try:
        return posttrain.main([*arguments, '--smoke-steps', str(args.steps)])
    finally:
        posttrain.train_step, posttrain.optimizer_step, posttrain.run_training = original_step, original_update, original_run


if __name__ == '__main__':
    main()
