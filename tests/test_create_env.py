"""Environment creation must validate the destination selected by the caller."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('flags,target', [
    ([], ['-n', 'priming']),
    (['--prefix', '/scratch/path with spaces/env'], ['--prefix', '/scratch/path with spaces/env']),
    (['--prefix=/scratch/env'], ['-p', '/scratch/env']),
    (['--name', 'custom'], ['--name', 'custom']),
])
def test_checks_target_created_environment(tmp_path, flags, target):
    mock = tmp_path / 'conda'
    mock.write_text('#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ["CALL_LOG"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    mock.chmod(0o755)
    log = tmp_path / 'calls.jsonl'
    subprocess.run(['bash', str(ROOT / 'scripts/create_env.sh'), *flags], check=True,
                   env={**os.environ, 'CONDA_EXE': str(mock), 'CALL_LOG': str(log)})
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0] == ['env', 'create', '--file', 'environment.yml', *flags]
    assert all(call[2:4] == target for call in calls[1:])
