"""Read and verify the existing offline post-training encoder assets."""
import json
import os
from pathlib import Path

from metrics.specs import SPECS, PAPER_SPECS
from .files import sha256

ROOT = Path(__file__).resolve().parents[1]


def verify_file(path, record):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing post-training asset: {path}. See docs/assets.md.')
    identities = [(record['bytes'], record['sha256'])]
    if 'published_sha256' in record:
        identities.append((record['published_bytes'], record['published_sha256']))
    size = path.stat().st_size
    if size not in {item[0] for item in identities} or (size, sha256(path)) not in identities:
        raise ValueError(f'Post-training asset differs from the release: {path}')
    return path


def verify_references(cache, *, full=False):
    records = json.loads((ROOT/'old/releases/posttrain_encoders.json').read_text())['files']
    names = {'class_order.npy', 'dataset_labels.npy', 'class_counts.npy'}
    if full:
        names.update({'downloads.json', 'converted_manifest.json', 'calibration.json',
                      'mae_encoder.pt', 'convnext_encoder.pt'})
    records = {row['path']:row for row in records}
    if not names <= set(records):
        raise ValueError('Incomplete post-training asset manifest')
    return {name:verify_file(Path(cache)/name,records[name]) for name in sorted(names)}


def sim_weights(*, hf_home=None, torch_home=None):
    hf_home = Path(hf_home or os.environ.get('HF_HOME', ROOT/'assets/huggingface')).expanduser()
    torch_home = Path(torch_home or os.environ.get('TORCH_HOME', ROOT/'assets/torch')).expanduser()
    records = json.loads((ROOT/'old/releases/encoders.json').read_text())['files']
    models = {'siglip': 'timm/vit_so400m_patch16_siglip_256.v2_webli',
              'mae': 'timm/vit_large_patch16_224.mae'}
    result = {}
    for name, repo_id in models.items():
        row = next(row for row in records if row.get('repo_id') == repo_id and row.get('filename') == 'model.safetensors')
        directory = 'models--'+repo_id.replace('/', '--')
        path = hf_home/'hub'/directory/'snapshots'/row['revision']/row['filename']
        result[name] = verify_file(path, row)
    row = next(row for row in records if Path(row['path']).name == 'weights-inception-2015-12-05-6726825d.pth')
    result['inception'] = verify_file(torch_home/'hub/checkpoints'/Path(row['path']).name, row)
    return result


def evaluation_assets(config, *, hf_home, torch_home):
    """Resolve the six evaluation encoders and their pinned real-data references."""
    records = json.loads((ROOT/'old/releases/encoders.json').read_text())['files']
    weights = encoder_weights(SPECS, records, hf_home=hf_home, torch_home=torch_home)
    for _, _, _, _, filename in SPECS:
        row = next(row for row in records if row['path'] == 'assets/fid_stats/'+filename)
        verify_file(Path(config['stats_dir'])/filename, row)
    row = next(row for row in records if row['path'] == 'data/prc/VIRTUAL_imagenet256_labeled_inception.pt')
    verify_file(config['prc_reference'], row)
    return weights


def encoder_weights(specs, records, *, hf_home, torch_home):
    weights = {}
    for name, model, _, _, _ in specs:
        if model == 'inception':
            row = next(row for row in records if Path(row['path']).name == 'weights-inception-2015-12-05-6726825d.pth')
            path = Path(torch_home)/'hub/checkpoints'/Path(row['path']).name
        else:
            row = next(row for row in records if row.get('repo_id') == 'timm/'+model
                       and row.get('filename') in {'model.safetensors', 'pytorch_model.bin'})
            path = Path(hf_home)/'hub'/('models--'+row['repo_id'].replace('/', '--'))/'snapshots'/row['revision']/row['filename']
        weights[name] = verify_file(path, row)
    return weights


def paper_evaluation_assets(args):
    """The five FDr5 references and distinct official FID reference, all local."""
    from .paths import resolve
    records = json.loads((ROOT/'old/releases/encoders.json').read_text())['files']
    weights = encoder_weights(PAPER_SPECS, records, hf_home=os.environ['HF_HOME'], torch_home=os.environ['TORCH_HOME'])
    stats = resolve(args.stats_dir or os.environ.get('PRIMING_FDR5_STATS', 'assets/fid_stats'))
    for _, _, _, _, filename in PAPER_SPECS:
        row = next(row for row in records if row['path'] == 'assets/fid_stats/'+filename)
        verify_file(stats/filename, row)
    row = next(row for row in records if Path(row['path']).name == 'jit_in256_stats.npz')
    reference = verify_file(resolve(args.fid_reference or os.environ.get('PRIMING_FID_REFERENCE', stats/'jit_in256_stats.npz')), row)
    row = next(row for row in records if Path(row['path']).name == 'weights-inception-2015-12-05-6726825d.pth')
    inception = verify_file(resolve(args.inception_weights or os.environ.get('PRIMING_INCEPTION_WEIGHTS', weights['FID(ADM)'])), row)
    return dict(weights={name:str(path) for name,path in weights.items()}, stats_dir=str(stats),
                fid_reference=str(reference), inception_weights=str(inception))
