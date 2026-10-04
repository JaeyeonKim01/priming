"""Validate and freeze an explicit pretraining source without an epoch restriction."""
import hashlib
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def expected_shapes(config):
    """Construct the selected architecture on meta; no GPU or weight allocation."""
    import torch

    runtime = Path(__file__).resolve().parents[1] / 'runtime'
    sys.path[:0] = [str(runtime), str(runtime / 'packages/distmatch/src')]
    from src.models import OSTDenoiser_models

    model = config['model']
    kwargs = {key: model[key] for key in (
        'img_size', 'patch_size', 'label_drop_prob', 'class_tokens',
        'guidance_tokens', 'interval_tokens', 'rope_2d', 'learned_pe',
        'norm_eps', 'norm_p')}
    kwargs.update(in_channels=config['tokenizer']['token_channels'],
                  tokenizer_patch_size=config['tokenizer']['tokenizer_patch_size'],
                  num_classes=config['data']['num_classes'],
                  noise_scale=config['train']['noise_scale'])
    with torch.device('meta'):
        net = OSTDenoiser_models[model['model']](**kwargs)
    return {name: tuple(value.shape) for name, value in net.state_dict().items()}


def validate_state(state, config, shapes):
    """Require complete, finite online/EMA weights in the packaged pretraining format."""
    import torch

    if not isinstance(state, dict):
        raise ValueError('Expected a full pretraining checkpoint dictionary')
    ema = state.get('model_ema')
    if not isinstance(ema, dict) or ema.get('schedules') != [('0.9999', 0.9999)]:
        raise ValueError('Expected pretraining EMA 0.9999; use --resume for post-training recovery')
    if ema.get('schedule') != 'const_schedule' or set(ema.get('shadows', {})) != {'0.9999'}:
        raise ValueError('Expected one constant pretraining EMA shadow')
    step = state.get('step')
    if type(step) is not int or step < 0:
        raise ValueError('Expected a nonnegative pretraining step')
    if state.get('current_step') != step + 1 or ema.get('step_count') != step:
        raise ValueError('Pretraining step/current_step/EMA counters disagree')

    # B includes a scientific contract. Historical L checkpoints do not, so
    # both families also undergo the exact tensor schema check below.
    contract = state.get('contract')
    if config['model']['model'] == 'OST_B' and not isinstance(contract, dict):
        raise ValueError('Expected the B pretraining scientific contract')
    if contract is not None:
        for section, keys in (
            ('model', ('model', 'patch_size', 'class_tokens', 'guidance_tokens',
                       'interval_tokens', 'label_drop_prob', 'rope_2d', 'learned_pe')),
            ('data', ('num_classes',)),
        ):
            for key in keys:
                if contract.get(section, {}).get(key) != config[section][key]:
                    raise ValueError(f'Pretraining contract differs: {section}.{key}')
        if contract.get('loss', {}).get('aux', {}).get('enabled') is not True:
            raise ValueError('These post-training recipes require perceptual pretraining')

    for label, weights in (('online', state.get('model')), ('EMA', ema['shadows']['0.9999'])):
        if not isinstance(weights, dict) or set(weights) != set(shapes):
            raise ValueError(f'{label} tensor names differ from the selected model')
        for name, value in weights.items():
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shapes[name]:
                raise ValueError(f'{label} tensor shape differs: {name}')
            if not value.is_floating_point() or not torch.isfinite(value).all().item():
                raise ValueError(f'{label} tensor is not finite floating-point data: {name}')
    return dict(model=config['model']['model'], source_step=step,
                source_current_step=state['current_step'], ema_label='0.9999',
                ema_step_count=ema['step_count'])


def inspect_checkpoint(path, config):
    import torch

    path = Path(path).expanduser().resolve(strict=True)
    before = path.stat()
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    metadata = validate_state(state, config, expected_shapes(config))
    digest = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Initialization checkpoint changed during preparation')
    return dict(kind='custom_pretraining', path=str(path), bytes=after.st_size,
                sha256=digest, **metadata)


def verify_prepared_source(record, config, *, check_hash=True):
    """Bind the rendered config to the frozen source and detect replaced weights."""
    if record['kind'] != 'custom_pretraining':
        raise ValueError('Unknown initialization record')
    if record['model'] != config['model']['model']:
        raise ValueError('Initialization model differs from prepared config')
    if Path(record['path']) != Path(config['checkpointing']['load_from']):
        raise ValueError('Initialization path differs from prepared config')
    if check_hash:
        path = Path(record['path'])
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise ValueError('Initialization checkpoint changed; prepare a new job')
