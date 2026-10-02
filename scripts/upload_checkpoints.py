"""Plan or explicitly upload the eight manifest-identified checkpoints to Hugging Face."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_checkpoints import records, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True, help='Hugging Face USER_OR_ORG/priming')
    parser.add_argument('--upload', action='store_true', help='Create/upload to the specified model repository')
    parser.add_argument('--public', action='store_true', help='Create a public repository; new repos are private by default')
    args = parser.parse_args()
    files = []
    for key, item in records():
        source = ROOT / item['checkpoint']
        files.append(dict(path=item['checkpoint'], bytes=item['checkpoint_bytes'], sha256=item['checkpoint_sha256']))
        if args.upload:
            if not source.is_file() or source.stat().st_size != item['checkpoint_bytes']:
                raise ValueError(f'Missing/wrong-size checkpoint: {source}')
            if digest(source) != item['checkpoint_sha256']:
                raise ValueError(f'Checkpoint checksum differs: {source}')
    print(json.dumps(dict(repo_id=args.repo_id, upload=args.upload, files=files,
                         total_bytes=sum(row['bytes'] for row in files)), indent=2))
    if not args.upload:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, repo_type='model', private=not args.public, exist_ok=True)
    for item in files:
        api.upload_file(path_or_fileobj=str(ROOT / item['path']), path_in_repo=item['path'],
                        repo_id=args.repo_id, repo_type='model', commit_message=f'Add {item["path"]}')
    metadata = [ROOT / 'checkpoints/manifest.json', ROOT / 'checkpoints/pretraining_manifest.json']
    metadata += [ROOT / item['config'] for _, item in records('posttraining')]
    for source in metadata:
        api.upload_file(path_or_fileobj=str(source), path_in_repo=str(source.relative_to(ROOT)),
                        repo_id=args.repo_id, repo_type='model')
    api.upload_file(path_or_fileobj=str(ROOT / 'docs/huggingface-model-card.md'), path_in_repo='README.md',
                    repo_id=args.repo_id, repo_type='model')
    print(f'https://huggingface.co/{args.repo_id}')


if __name__ == '__main__':
    main()
