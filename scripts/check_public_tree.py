"""Reject private infrastructure identifiers and historical logs in public Git files."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = [
    re.compile(r'/n/(?:home\d*|netscratch)/'),
    re.compile('kemp' + 'ner', re.I),
    re.compile('holy' + r'gpu\w+', re.I),
    re.compile(r'\b(?:sham|sitanc)' + '_lab\\b'),
    re.compile('jaeyeon' + '_kim-harvard-university'),
    re.compile(r'~/' + 'woobin/'),
    re.compile(r'^#SBATCH\s+--mail-user', re.M),
]


def check():
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT
    ).decode().split('\0')
    failures = []
    count = 0
    for name in sorted(set(names) - {''}):
        path = ROOT / name
        if not path.is_file() or path.is_symlink():
            continue
        try:
            contents = path.read_text()
        except UnicodeDecodeError:
            continue
        count += 1
        if any(pattern.search(contents) for pattern in PATTERNS):
            failures.append(name)
    if failures:
        raise RuntimeError('Private information or old training reports: ' + ', '.join(sorted(set(failures))))
    print(f'PASS public-tree scan: {count} text files; Git history and binary metadata are outside this check.')


if __name__ == '__main__':
    check()
