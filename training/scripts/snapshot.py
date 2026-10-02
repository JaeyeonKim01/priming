"""Create an allocation-local copy of release code and a private asset view."""
import argparse
import json
from pathlib import Path
import shutil
from verify import TASK, verify


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True, type=Path)
    a = p.parse_args()
    verify()
    assert not a.job.exists(), a.job
    a.job.mkdir(parents=True)
    for name in ['runtime', 'scripts', 'configs', 'provenance', 'reports']:
        shutil.copytree(TASK / name, a.job / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.lock'))
    site = json.loads((a.job / 'configs/site.json').read_text())
    data = a.job / 'runtime/data'
    data.mkdir()
    # Only the precision/recall reference and frozen FD statistics are needed.
    (data / 'fid_stats').symlink_to(site['fid_stats'])
    (data / 'prc').symlink_to(Path(site['data_root']) / 'prc')
    (TASK / 'results' / a.job.name).symlink_to(a.job)
    print(f'Frozen allocation source: {a.job}', flush=True)


if __name__ == '__main__':
    main()
