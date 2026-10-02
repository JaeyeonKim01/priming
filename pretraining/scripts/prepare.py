"""Resolve a production recipe and freeze a fresh launch bundle; never launch training."""
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from common import ROOT, RECIPES, sha, verify_runtime, write_json


def data_identity(path, validation):
    """Metadata hashes and shard stat identities, matching the B production guard."""
    metadata, arrows = {}, {}
    for folder in [path] + ([path / 'val'] if validation else []):
        for name in ('dataset_info.json', 'state.json'):
            file = folder / name
            metadata[str(file.relative_to(path))] = sha(file)
        shards = sorted(folder.glob('*.arrow'))
        if not shards:
            raise FileNotFoundError(f'No packed ImageNet Arrow shards in {folder}')
        for file in shards:
            stat = file.stat()
            arrows[str(file.relative_to(path))] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return dict(data_metadata_sha256=metadata, arrow_files=arrows,
                data_identity_scope='Metadata hashes and Arrow filenames/sizes/mtime; image bytes are not fully hashed.')


def prepare(recipe, bundle, output, site):
    if recipe not in RECIPES:
        raise ValueError(recipe)
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    if bundle.exists() or output.exists():
        raise FileExistsError('Use new bundle and training output directories; reuse an existing bundle for recovery.')
    if bundle == output or bundle in output.parents or output in bundle.parents:
        raise ValueError('Bundle and training output must be separate, non-nested directories.')
    family = recipe[0]
    runtime = verify_runtime(family)
    source = ROOT / f'configs/{recipe}.yaml'
    cfg = OmegaConf.load(source)
    cfg.data.data_path = str(Path(site['data_path']).resolve())
    cfg.train.output_dir = str(output.parent)
    cfg.train.project = '.'
    cfg.train.exp_name = output.name
    cfg.train.enable_wandb = False
    identity = data_identity(Path(cfg.data.data_path) / 'imagenet-latents-images', family == 'b')
    assets = {}
    if cfg.loss.aux.enabled:
        lpips = Path(site['lpips_weights']).resolve()
        if hashlib_md5(lpips) != 'd507d7349b931f0638a25a48a722f98a':
            raise ValueError('LPIPS calibration checksum differs from production')
        assets['lpips'] = dict(path=str(lpips), sha256=sha(lpips))
        vgg = Path(site['torch_home']).resolve() / 'hub/checkpoints/vgg16-397923af.pth'
        assets['vgg16'] = dict(path=str(vgg), sha256=sha(vgg))
        hub = Path(site['hf_home']).resolve() / 'hub/models--facebook--convnextv2-base-22k-224'
        revision = (hub / 'refs/main').read_text().strip()
        snapshot = hub / 'snapshots' / revision
        files = sorted(file for file in snapshot.rglob('*') if file.is_file())
        if not (snapshot / 'config.json').is_file() or not any(file.suffix in ('.bin', '.safetensors') for file in files):
            raise FileNotFoundError(f'Incomplete ConvNeXt cache: {snapshot}')
        # The original loader resolves the model name through refs/main. Guard
        # the ref as well as the snapshot bytes so it cannot silently select a
        # newer cached revision after preparation.
        assets['convnext'] = dict(revision=revision, files={
            str(hub / 'refs/main'): sha(hub / 'refs/main'),
            **{str(f): sha(f) for f in files}})
    bundle.mkdir(parents=True)
    work = bundle / 'work'; work.mkdir()
    if cfg.loss.aux.enabled:
        (work / 'checkpoints/lpips').mkdir(parents=True)
        (work / 'checkpoints/lpips/vgg.pth').symlink_to(assets['lpips']['path'])
    config = bundle / 'config.yaml'
    OmegaConf.save(cfg, config)
    manifest = dict(version=1, runtime=runtime, configs={
        'perceptual' if cfg.loss.aux.enabled else 'no_perceptual': sha(config)}, **identity)
    write_json(bundle / 'manifest.json', manifest)
    launch = dict(recipe=recipe, family=family, output=str(output), world_size=8,
                  config_sha256=sha(config), manifest_sha256=sha(bundle / 'manifest.json'),
                  recipe_sha256=sha(source), assets=assets,
                  environment=dict(HF_HOME=str(Path(site['hf_home']).resolve()),
                                   HF_HUB_CACHE=str(Path(site['hf_home']).resolve() / 'hub'),
                                   TORCH_HOME=str(Path(site['torch_home']).resolve()),
                                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
    write_json(bundle / 'launch.json', launch)
    return launch


def hashlib_md5(path):
    import hashlib
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', choices=RECIPES, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--site', type=Path, default=ROOT / 'configs/site.json')
    for name in ('data-path', 'hf-home', 'torch-home', 'lpips-weights'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    site = json.loads(args.site.read_text())
    for key in ('data_path', 'hf_home', 'torch_home', 'lpips_weights'):
        if getattr(args, key):
            site[key] = getattr(args, key)
    print(json.dumps(prepare(args.recipe, args.bundle, args.output, site), indent=2))
