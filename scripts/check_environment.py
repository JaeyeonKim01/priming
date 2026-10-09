"""Check the pinned environment and optionally require a working CUDA device."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(require_cuda=False):
    mismatches = {}
    versions = {}
    for line in (ROOT / 'old/environment/requirements-linux-cu124.txt').read_text().splitlines():
        if '==' not in line or line.startswith(('#', '-')):
            continue
        name, expected = line.split('==')
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        versions[name] = actual
        if actual != expected:
            mismatches[name] = dict(expected=expected, actual=actual)
    if mismatches:
        raise RuntimeError(f'Environment differs from lock: {mismatches}')
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('The production runtime requires Python 3.11.')
    import torch
    import torchvision
    import muon
    import transformers
    available = torch.cuda.is_available()
    if require_cuda and not available:
        raise RuntimeError('CUDA is unavailable; check NVIDIA driver/GPU access.')
    return dict(passed=True, python=platform.python_version(), platform=platform.platform(),
                torch=torch.__version__, cuda_build=torch.version.cuda, cuda_available=available,
                devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                packages=versions, scope='Dependency/import checks; no training reproduction.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = check(args.cuda)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
