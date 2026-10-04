"""Private compiler caches and read-only initialization/memory instrumentation."""
import os
import hashlib
from pathlib import Path
import runpy
import tempfile

identity = hashlib.sha256(os.environ['JOB_DIR'].encode()).hexdigest()[:16]
root = Path(os.environ.get('PRIMING_LOCAL_TMP', '/tmp')) / f'priming-{os.getuid()}'
root = root / 'posttraining' / identity / os.environ.get('SLURM_JOB_ID', str(os.getppid())) / f"rank_{os.environ['RANK']}"
for variable, folder in (('TMPDIR', 'tmp'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                         ('TRITON_CACHE_DIR', 'triton'), ('CUDA_CACHE_PATH', 'cuda')):
    path = root / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(path)
tempfile.tempdir = os.environ['TMPDIR']

if os.environ.get('RANK_MODE') == 'restore':
    runpy.run_path(str(Path(__file__).with_name('resume_state.py')), run_name='__main__')
else:
    import json
    import torch
    import torch.distributed as dist
    from scripts import main_mmd_and_eval as driver
    from resume_state import equal
    original = driver.run_training

    def instrumented(args, model, ema, optimizer, *positional, **kwargs):
        if args.resume_from and os.environ.get('MODE') == 'train':
            checkpoint = torch.load(args.resume_from, map_location='cpu', mmap=True, weights_only=False)
            checks = dict(
                online=equal(model.state_dict(), checkpoint['model']),
                ema=equal(ema.state_dict(), checkpoint['model_ema']),
                optimizer=equal(optimizer.state_dict(), checkpoint['optimizer']),
                main_rng=equal(kwargs['rng'].snapshot(), checkpoint[driver.MMD_RNG_STATE_KEY]['rank_states'][args.rank]),
                ot_rng=equal(positional[7].state_dict(), checkpoint[driver.OT_STREAM_STATE_KEY]['rank_states'][args.rank]),
                feature_contract=equal(positional[1].running_state_dict(), checkpoint[driver.MMD_QUEUE_STATE_KEY]),
                counters=args.current_step == checkpoint['current_step'] and args.samples_seen == checkpoint['samples_seen'])
            assert all(checks.values()), checks
            report = dict(passed=True, rank=args.rank, restored_updates=args.current_step,
                          checkpoint=args.resume_from, checks=checks)
            destination = Path(os.environ['JOB_DIR']) / 'reports' / f"recovery_restore_{os.environ.get('SLURM_RESTART_COUNT', '0')}_rank{args.rank}.json"
            destination.write_text(json.dumps(report, indent=2) + '\n')
            del checkpoint
        if args.current_step == 0:
            source = torch.load(args.load_from, map_location='cpu', mmap=True, weights_only=False)
            manifest = json.loads((Path(os.environ['JOB_DIR']) / 'config.manifest.json').read_text())
            initialization = manifest.get('initialization')
            checks = dict(online_weights=equal(model.state_dict(), source['model']),
                source_ema=equal(ema.shadows['0.999'], source['model_ema']['shadows']['0.9999']),
                fresh_optimizer=not optimizer.state, samples_zero=args.samples_seen == 0,
                source_ema_counter=ema.step_count == source['model_ema']['step_count'],
                source_steps=(source['model_ema']['step_count'] == initialization['ema_step_count']
                    if initialization else source['model_ema']['step_count'] == 200160),
                perceptual_source=(source['contract']['loss']['aux']['enabled'] is True
                    if args.model == 'OST_B' else True),
                source_model=(source['contract']['model']['model'] == args.model
                    if args.model == 'OST_B' else args.model == 'OST_L'))
            assert all(checks.values()), checks
            Path(args.log_dir, f'initialization_rank{args.rank}.json').write_text(json.dumps(checks, indent=2)+'\n')
            del source
        elif os.environ.get('MODE') == 'train' and args.current_step % 1250 == 0 and args.current_step < args.total_steps:
            # A node failure can leave a saved milestone with only part of its
            # evaluation. Complete that draw before the next training update.
            from src.eval.dashboard_data import evaluation_groups
            summary = Path(args.log_dir) / 'final_eval_summary.csv'
            missing = [False]
            if args.rank == 0:
                missing[0] = not summary.is_file() or any(
                    not any(row['step'] == args.current_step for row in evaluation_groups(summary, weights=label))
                    for label in ['online', '0.999'])
            dist.broadcast_object_list(missing, src=0)
            if missing[0]:
                rng = kwargs['rng']
                snapshot = rng.snapshot()
                try:
                    driver.run_fdr6_evaluation(args, model, ema, positional[0],
                        wandb_logger=kwargs['wandb_logger'])
                finally:
                    rng.load(snapshot)
        completed = original(args, model, ema, optimizer, *positional, **kwargs)
        report = dict(rank=args.rank, completed_updates=args.current_step,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            device=torch.cuda.get_device_name(), total_gib=torch.cuda.get_device_properties(0).total_memory/2**30)
        gathered = [None] * args.world_size if args.rank == 0 else None
        dist.gather_object(report, gathered, dst=0)
        if args.rank == 0:
            Path(args.log_dir, 'memory_audit.json').write_text(json.dumps(gathered, indent=2)+'\n')
        return completed

    driver.run_training = instrumented
    raise SystemExit(driver.train_and_evaluate(driver.parse_config()))
