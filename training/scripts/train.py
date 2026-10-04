"""Run a prepared post-training snapshot on eight torchrun ranks."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
from paths import ROOT, resolve, site_config
from verify import verify, sha

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--resume', help='Checkpoint from this prepared job')
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', '0')) != 8:
        raise ValueError('Production recipes require exactly eight torchrun ranks.')
    job = resolve(args.job)
    verify(job)
    manifest = json.loads((job / 'config.manifest.json').read_text())
    if sha(job / 'config.json') != manifest['config_sha256']:
        raise ValueError('Prepared config changed; prepare a new job.')
    config = json.loads((job / 'config.json').read_text())
    if 'initialization' in manifest:
        from initialization import verify_prepared_source
        verify_prepared_source(manifest['initialization'], config,
                               check_hash=not args.resume and int(os.environ['RANK']) == 0)
    site = site_config(job)
    os.environ.update(PRIMING_ROOT=str(ROOT), JOB_DIR=str(job), MODE='train',
        HF_HOME=site['hf_home'], HF_HUB_CACHE=str(Path(site['hf_home']) / 'hub'),
        TORCH_HOME=site['torch_home'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path[:0] = [str(job / 'runtime'), str(job / 'runtime/packages/distmatch/src'), str(job / 'scripts')]
    os.chdir(job / 'runtime')
    entry = job / 'scripts/rank_entry.py'
    sys.argv = [str(entry), '--config', str(job / 'config.json')]
    if args.resume:
        checkpoint = resolve(args.resume)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        sys.argv += [f'checkpointing.resume_from={checkpoint}', 'checkpointing.auto_resume=false']
    elif int(os.environ['RANK']) == 0:
        log = config['logging']
        if (Path(log['output_dir']) / log['project'] / log['exp_name']).exists():
            raise FileExistsError('Output exists; use --resume or prepare a new run.')
    runpy.run_path(str(entry), run_name='__main__')
