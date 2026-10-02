"""Run the retained CPU regression suite and record a fresh launch prerequisite."""
import os
from pathlib import Path
import subprocess
import sys
from verify import TASK, verify, write

if __name__ == '__main__':
    verify()
    env = {**os.environ, 'PYTHONNOUSERSITE': '1', 'OMP_NUM_THREADS': '2',
           'PYTHONPATH': os.pathsep.join(str(p) for p in
               (TASK / 'runtime', TASK / 'runtime/packages/distmatch/src', TASK / 'scripts'))}
    subprocess.run([sys.executable, '-c', 'from scripts import main_mmd_and_eval'],
                   cwd=TASK, env=env, check=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', str(TASK / 'tests')],
                   cwd=TASK, env=env, check=True)
    write(TASK / 'state/cpu_checks.json', dict(passed=True,
          checks=['isolated driver import', 'retained CPU regression suite'], python=sys.version))
