"""Stage the exact packed training data without copying it; optionally publish to HF.

The stage contains only an explicit file allowlist. Run on storage that holds the
dataset; upload caches and manifests remain in the supplied staging directory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def validate_arrow(path):
    import pyarrow as pa
    import pyarrow.compute as pc
    import numpy as np
    from PIL import Image
    import io
    labels = []
    with pa.memory_map(str(path), 'r') as source:
        reader = pa.ipc.open_stream(source)
        if reader.schema.names != ['id', 'image', 'label']:
            raise ValueError(f'Unexpected dataset schema: {path.name}')
        metadata = json.dumps({k.decode(): v.decode() for k, v in (reader.schema.metadata or {}).items()})
        if any(s in metadata for s in ('/n/home', '/n/netscratch', '/home/', '/scratch/')):
            raise ValueError('Private path in Arrow metadata')
        for batch in reader:
            images = batch.column('image')
            if images.field('path').null_count != len(images):
                raise ValueError('Image paths must be null; publish embedded images only')
            if images.null_count or images.field('bytes').null_count:
                raise ValueError('Missing embedded image bytes')
            label = batch.column('label')
            if label.null_count or pc.min(label).as_py() < 0 or pc.max(label).as_py() >= 1000:
                raise ValueError('Invalid ImageNet labels')
            labels.append(label.to_numpy().copy())
            image = Image.open(io.BytesIO(images.field('bytes')[0].as_py()))
            if image.size != (256, 256) or image.mode != 'RGB':
                raise ValueError('Expected RGB 256x256 images')
    return np.concatenate(labels)


def prepare(dataset, representations, stage, card, workers, repo_id='jaykim1102/priming'):
    import numpy as np
    if stage.exists():
        raise FileExistsError('Use a new stage, or --upload-existing to resume its upload')
    stage.mkdir(parents=True)
    sources = []
    split_labels = {}
    for split, expected in (('', 1281167), ('val', 50000)):
        folder = dataset / split
        state = json.loads((folder / 'state.json').read_text())
        names = [row['filename'] for row in state['_data_files']]
        if len(names) != len(set(names)) or any(Path(n).name != n or not n.endswith('.arrow') for n in names):
            raise ValueError('Invalid shard paths')
        def inspect(name):
            path = folder / name
            labels = validate_arrow(path)
            row = dict(path=str(Path('data/imagenet/imagenet-latents-images') / split / name),
                       bytes=path.stat().st_size, sha256=digest(path))
            return path, row, labels
        labels = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, (path, row, values) in enumerate(pool.map(inspect, names)):
                sources.append((path, row)); labels.append(values)
                if (i + 1) % 25 == 0:
                    print(f'Checked {split or "train"}: {i + 1}/{len(names)} shards', flush=True)
        split_labels[split] = np.concatenate(labels)
        if len(split_labels[split]) != expected or len(np.unique(split_labels[split])) != 1000:
            raise ValueError('Dataset split population differs from release')
        for name in ('dataset_info.json', 'state.json'):
            path = folder / name
            text = path.read_text()
            if any(s in text for s in ('/n/home', '/n/netscratch', '/home/', '/scratch/')):
                raise ValueError('Private path in dataset metadata')
            sources.append((path, dict(path=str(Path('data/imagenet/imagenet-latents-images') / split / name),
                                      bytes=path.stat().st_size, sha256=digest(path))))
    labels = split_labels[''].astype(np.int64)
    for name, expected in (
        ('dataset_labels.npy', labels),
        ('class_order.npy', np.argsort(labels, kind='stable')),
        ('class_counts.npy', np.bincount(labels, minlength=1000)),
    ):
        path = representations / name
        if not np.array_equal(np.load(path, allow_pickle=False), expected):
            raise ValueError(f'Index does not match packed dataset: {name}')
        sources.append((path, dict(path=f'assets/ot/representations/{name}',
                                  bytes=path.stat().st_size, sha256=digest(path))))
    for name in ('downloads.json', 'converted_manifest.json', 'calibration.json'):
        obj = json.loads((representations / name).read_text())
        if name == 'downloads.json':
            for row in obj:
                row['path'] = f"sources/{row['family']}/{row['filename']}"
        elif name == 'converted_manifest.json':
            for family, row in obj['conversions'].items():
                row['path'] = f'assets/ot/representations/{family}_encoder.pt'
        else:
            obj.pop('elapsed_seconds', None)
        target = stage / 'assets/ot/representations' / name
        write_json(target, obj)
        sources.append((target, dict(path=str(target.relative_to(stage)),
                                    bytes=target.stat().st_size, sha256=digest(target))))
    for source, row in sources:
        target = stage / row['path']
        if source != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source.resolve())
    (stage / 'README.md').write_text(card.read_text())
    manifest = dict(format='priming.dataset.v1', repo_id=repo_id, repo_type='dataset',
                    train_images=1281167, validation_images=50000,
                    image_size=[256, 256], ordering='original packed Arrow row order',
                    files=[row for _, row in sources])
    write_json(stage / 'dataset_manifest.json', manifest)
    print(json.dumps(dict(files=len(sources), bytes=sum(row['bytes'] for _, row in sources))), flush=True)


def upload(stage, repo_id, workers, batch_files=8):
    from huggingface_hub import HfApi, CommitOperationAdd
    manifest = json.loads((stage / 'dataset_manifest.json').read_text())
    if manifest['repo_id'] != repo_id:
        raise ValueError('Repo ID differs from staged manifest')
    api = HfApi()
    api.create_repo(repo_id, repo_type='dataset', private=True, exist_ok=True)
    # Install access terms before visibility. Public gated storage avoids using
    # the author's private-storage allowance for the full image release.
    api.update_repo_settings(repo_id, repo_type='dataset', gated='auto')
    api.upload_file(repo_id=repo_id, repo_type='dataset', path_in_repo='README.md',
                    path_or_fileobj=(stage / 'README.md').read_bytes(),
                    commit_message='Document ImageNet access terms and dataset layout')
    api.update_repo_settings(repo_id, repo_type='dataset', gated='auto', private=False)
    remote = {row.rfilename: row for row in api.repo_info(
        repo_id, repo_type='dataset', files_metadata=True).siblings}
    pending = []
    for row in manifest['files']:
        found = remote.get(row['path'])
        if found and found.size == row['bytes'] and found.lfs and found.lfs.sha256 == row['sha256']:
            continue
        pending.append(row['path'])
    pending += ['README.md', 'dataset_manifest.json']
    print(f'{len(pending)} files to commit; at most {batch_files} per upload batch', flush=True)
    # Keep batches bounded: large concurrent Xet batches can exceed a login
    # node's memory limit. Each successful commit is independently resumable.
    for start in range(0, len(pending), batch_files):
        names = pending[start:start + batch_files]
        api.create_commit(repo_id, repo_type='dataset',
                          operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=stage / name)
                                      for name in names],
                          num_threads=workers,
                          commit_message=f'Add processed ImageNet release files ({start + 1}-{start + len(names)})')
        print(f'Committed {start + len(names)}/{len(pending)} pending files', flush=True)
    info = api.repo_info(repo_id, repo_type='dataset', files_metadata=True)
    remote = {row.rfilename: row for row in info.siblings}
    for row in manifest['files']:
        found = remote[row['path']]
        if found.size != row['bytes']:
            raise ValueError(f"Remote size differs: {row['path']}")
        if found.lfs and found.lfs.sha256 != row['sha256']:
            raise ValueError(f"Remote checksum differs: {row['path']}")
        if not found.lfs:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(repo_id, row['path'], repo_type='dataset', revision=info.sha,
                                    local_dir=stage / '.verification')
            if digest(local) != row['sha256']:
                raise ValueError(f"Remote checksum differs: {row['path']}")
    api.update_repo_settings(repo_id, repo_type='dataset', gated='auto', private=False)
    final = api.repo_info(repo_id, repo_type='dataset')
    if final.private or final.gated != 'auto':
        raise ValueError('Dataset visibility/gating differs from release settings')
    write_json(stage / 'publication.json', dict(repo_id=repo_id, revision=info.sha, gated='auto'))
    print(f'Published https://huggingface.co/datasets/{repo_id} revision {info.sha}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', type=Path)
    p.add_argument('--representations-dir', type=Path)
    p.add_argument('--stage-dir', type=Path, required=True)
    p.add_argument('--repo-id', default='jaykim1102/priming')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-files', type=int, default=8, help='Bound upload memory and checkpoint transfer progress')
    p.add_argument('--upload', action='store_true')
    p.add_argument('--upload-existing', action='store_true')
    a = p.parse_args()
    if a.workers < 1 or a.batch_files < 1:
        p.error('--workers and --batch-files must be positive')
    if not a.upload_existing:
        if a.dataset_dir is None or a.representations_dir is None:
            p.error('Preparation requires --dataset-dir and --representations-dir')
        prepare(a.dataset_dir, a.representations_dir, a.stage_dir,
                Path(__file__).resolve().parents[1] / 'docs/huggingface-dataset-card.md', a.workers, a.repo_id)
    if a.upload or a.upload_existing:
        upload(a.stage_dir, a.repo_id, a.workers, a.batch_files)


if __name__ == '__main__':
    main()
