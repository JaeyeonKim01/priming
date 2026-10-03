"""Render an allocation configuration without changing scientific settings."""
import argparse
import hashlib
import json
from pathlib import Path
from paths import load_config
from recipes import INITIALIZATION


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True)
    p.add_argument('--arm', choices=INITIALIZATION, required=True)
    p.add_argument('--mode', choices=['probe', 'train'], required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--run-name', required=True)
    p.add_argument('--resume')
    p.add_argument('--updates', type=int, default=130)
    a = p.parse_args()
    cfg = load_config(a.arm, a.task)
    cfg['logging']['exp_name'] = a.run_name
    cfg['checkpointing']['resume_from'] = a.resume
    cfg['checkpointing']['auto_resume'] = False
    if a.mode == 'probe':
        cfg['train'].update(epochs=a.updates, steps_per_epoch=1)
        cfg['optimizer']['warmup_epochs'] = 125
        cfg['logging'].update(save_freq=130, vis_freq=0, disable_vis=True, print_freq=10)
        cfg['evaluation'].update(online_eval=False, skip_final_eval=True)
    horizon = cfg['train']['epochs'] * cfg['train']['steps_per_epoch']
    run = Path(cfg['logging']['output_dir']) / cfg['logging']['project'] / a.run_name
    for key in ('load_from', 'resume_from'):
        source = cfg['checkpointing'][key]
        if source and not Path(source).is_file():
            raise FileNotFoundError(source)
    out = Path(a.out)
    out.write_text(json.dumps(cfg, indent=2)+'\n')
    manifest = dict(arm=a.arm, mode=a.mode, start_update=0, horizon=horizon,
        run_dir=str(run), config_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
