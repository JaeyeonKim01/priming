"""Environment creation validates the destination selected by the caller."""
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CreateEnvironmentTests(unittest.TestCase):
    def test_checks_the_environment_destination(self):
        for flags, target in (([], ['-n', 'priming']),
                              (['--prefix', '/scratch/path with spaces/env'], ['--prefix', '/scratch/path with spaces/env']),
                              (['--prefix=/scratch/env'], ['-p', '/scratch/env']),
                              (['--name', 'custom'], ['--name', 'custom'])):
            with self.subTest(flags=flags), TemporaryDirectory() as folder:
                root = Path(folder)
                mock = root/'conda'
                mock.write_text('#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ["CALL_LOG"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
                mock.chmod(0o755)
                log = root/'calls.jsonl'
                subprocess.run(['bash', str(ROOT/'scripts/create_env.sh'), *flags], check=True,
                    env={**os.environ, 'CONDA_EXE': str(mock), 'CALL_LOG': str(log)})
                calls = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual(calls[0], ['env', 'create', '--file', 'environment.yml', *flags])
                self.assertTrue(all(call[2:4] == target for call in calls[1:]))


if __name__ == '__main__':
    unittest.main()
