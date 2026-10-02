"""Exact checkpoint restoration and Adam/EMA replay; report GPU repeatability."""
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.distributed as dist
from scripts import main_mmd_and_eval as driver
from src.utils.checkpoint_util import _to_cpu


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.detach().cpu(), b.detach().cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) == type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def diagnostic(args, model, ema, optimizer, tokenizer, objective, reference,
               xm_objective, xm_stream, cross, self_ot, stream, intervention,
               *, wandb_logger, rng):
    assert args.current_step == 130 and args.world_size == 8
    source = torch.load(args.resume_from, map_location='cpu', mmap=True, weights_only=False)
    checks = {}
    for name, value in [('model', model.state_dict()), ('optimizer', optimizer.state_dict()), ('model_ema', ema.state_dict())]:
        checks['initial_' + name] = equal(value, source[name])
    checks['initial_rng'] = equal(rng.snapshot(), source[driver.MMD_RNG_STATE_KEY]['rank_states'][args.rank])
    checks['initial_stream'] = equal(stream.state_dict(), source[driver.OT_STREAM_STATE_KEY]['rank_states'][args.rank])
    checks['initial_manifest'] = equal(objective.running_state_dict(), source[driver.MMD_QUEUE_STATE_KEY])
    assert all(checks.values()), checks
    del source
    initial_rng, initial_stream = rng.snapshot(), stream.state_dict()
    initial_step, initial_samples = args.current_step, args.samples_seen
    driver._save(args, initial_step-1, model, optimizer, ema, objective, time.time(), rng, ot_stream=stream)
    roundtrip = str(Path(args.ckpt_dir) / f'step_{initial_step-1:07d}.pth')
    sampling = dict(t_min=args.interval_min, t_max=args.interval_max, cfg=args.cfg, num_steps=args.num_sampling_steps)

    def backward():
        optimizer.zero_grad(set_to_none=True)
        model.train()
        driver.adjust_learning_rate(optimizer, initial_step, args)
        loss, _, _ = driver.train_step(args, model, tokenizer, sampling, objective, reference,
            xm_objective, xm_stream, cross, self_ot, stream, intervention)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        assert torch.isfinite(norm) and norm > 0 and torch.isfinite(loss)
        return float(loss), float(norm)

    loss, norm = backward()
    gradients = [None if p.grad is None else p.grad.detach().cpu().clone() for p in model.parameters()]
    # Repeat the same backward without an update. Unlike state restoration,
    # this statistic includes nondeterministic CUDA reductions in the loss.
    rng.load(initial_rng)
    stream.load_state_dict(initial_stream)
    repeated_loss, repeated_norm = backward()
    delta2 = left2 = right2 = dot = 0.0
    for parameter, gradient in zip(model.parameters(), gradients):
        if gradient is None:
            assert parameter.grad is None
            continue
        other = parameter.grad.detach().cpu().double()
        value = gradient.double()
        delta2 += float((value-other).square().sum())
        left2 += float(value.square().sum())
        right2 += float(other.square().sum())
        dot += float((value*other).sum())
    repeatability = dict(loss=loss, repeated_loss=repeated_loss,
        gradient_norm=norm, repeated_gradient_norm=repeated_norm,
        clipped_gradient_relative_l2=(delta2/max(left2, 1e-30))**.5,
        clipped_gradient_cosine=dot/max((left2*right2)**.5, 1e-30))
    for parameter, gradient in zip(model.parameters(), gradients):
        parameter.grad = None if gradient is None else gradient.to(parameter.device)
    optimizer.step()
    ema.step(model)
    expected = copy.deepcopy({name: _to_cpu(value) for name, value in
        [('model', model.state_dict()), ('optimizer', optimizer.state_dict()), ('model_ema', ema.state_dict())]})
    optimizer.zero_grad(set_to_none=True)
    args.resume_from = roundtrip
    extra = driver.ckpt_resume(args, model, optimizer, ema,
        extra_keys=[driver.MMD_QUEUE_STATE_KEY, driver.MMD_RNG_STATE_KEY, driver.OT_STREAM_STATE_KEY])
    objective.load_running_state_dict(extra[driver.MMD_QUEUE_STATE_KEY], strict=True)
    assert driver._restore_checkpoint_rng_state(args, rng, extra)
    assert driver._restore_checkpoint_ot_stream_state(args, stream, extra, require_negative_noise_state=True)
    checks['roundtrip_rng'] = equal(rng.snapshot(), initial_rng)
    checks['roundtrip_stream'] = equal(stream.state_dict(), initial_stream)
    checks['roundtrip_counters'] = args.current_step == initial_step and args.samples_seen == initial_samples
    for parameter, gradient in zip(model.parameters(), gradients):
        parameter.grad = None if gradient is None else gradient.to(parameter.device)
    optimizer.step()
    ema.step(model)
    for name, value in [('model', model.state_dict()), ('optimizer', optimizer.state_dict()), ('model_ema', ema.state_dict())]:
        checks['same_gradient_' + name] = equal(value, expected[name])
    assert all(checks.values()), checks
    report = dict(passed=True, rank=args.rank, checks=checks, gpu_repeatability=repeatability,
        scope='Exact restoration and exact identical-gradient Adam/EMA replay. Repeated CUDA backward statistics are observational; no bitwise trajectory claim.')
    Path(args.log_dir, f'resume_state_rank{args.rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    gathered = [None] * args.world_size if args.rank == 0 else None
    dist.gather_object(report, gathered, dst=0)
    if args.rank == 0:
        result = dict(passed=all(row['passed'] for row in gathered), ranks=gathered)
        Path(args.log_dir, 'resume_state_audit.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
    dist.barrier()
    return False


if __name__ == '__main__':
    driver.run_training = diagnostic
    raise SystemExit(driver.train_and_evaluate(driver.parse_config()))
