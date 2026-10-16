"""Load calibrated frozen encoders independently of sampling and objectives."""
import hashlib
import json
from pathlib import Path

import torch

from utils.posttrain_assets import verify_references

ROOT = Path(__file__).resolve().parents[2]


def full_source_hash():
    directory = ROOT/'models/encoders/full'
    paths = [directory/'encoders.py', *sorted((directory/'vendor').glob('*.py'))]
    return hashlib.sha256(b''.join(path.read_bytes() for path in paths)).hexdigest()


def configure_full_execution(encoders, *, mae_bf16_channels_last=False, compile_encoders=False):
    """Optimize execution after verifying/loading the original calibrated weights."""
    if mae_bf16_channels_last:
        for module in encoders.mae.modules():
            if isinstance(module, torch.nn.Conv2d):
                weight = module.weight.detach().to(dtype=torch.bfloat16,
                    memory_format=torch.channels_last)
                module.weight = torch.nn.Parameter(weight, requires_grad=False)
    if compile_encoders:
        from torch.utils._python_dispatch import is_in_torch_dispatch_mode
        eager = encoders.encode_family
        compiled = torch.compile(eager, fullgraph=True, dynamic=False,
                                 options={'layout_optimization': False})

        def encode_family(images, family):
            # A FLOP/profiling dispatch mode must not mark the compiled code eager.
            if is_in_torch_dispatch_mode(include_infra_modes=False):
                return eager(images, family)
            return compiled(images, family)

        encoders.encode_family = encode_family


class FrozenEncoders(torch.nn.Module):
    def __init__(self, cache, *, stack='convnext_full_mae', device='cuda',
                 microbatch=8, hf_home=None, torch_home=None,
                 mae_bf16_channels_last=False, compile_encoders=False):
        super().__init__()
        if stack not in {'convnext_full_mae', 'sim'}:
            raise ValueError('Expected the full MAE/ConvNeXt or SIM feature stack')
        if int(microbatch) < 1:
            raise ValueError('Encoder microbatch must be positive')
        if not isinstance(mae_bf16_channels_last, bool) or not isinstance(compile_encoders, bool):
            raise ValueError('Encoder execution switches must be booleans')
        if stack != 'convnext_full_mae' and (mae_bf16_channels_last or compile_encoders):
            raise ValueError('Encoder execution optimizations require the full feature stack')
        self.execution = dict(microbatch=int(microbatch),
            mae_bf16_channels_last=mae_bf16_channels_last, compile_encoders=compile_encoders)
        self.stack, self.cache, self.device = stack, Path(cache), torch.device(device)
        verify_references(self.cache, full=stack == 'convnext_full_mae')
        if stack == 'convnext_full_mae':
            from models.encoders.full.encoders import PixelDriftingEncoders
            self.calibration = json.loads((self.cache/'calibration.json').read_text())
            if full_source_hash() != self.calibration['encoder_source_sha256']:
                raise ValueError('Encoder implementation differs from calibration source')
            self.bandwidths, self.weights = self.calibration['bandwidths'], self.calibration['weights']
            self.names = tuple(sorted(self.bandwidths))
            if len(self.names) != 97 or set(self.names) != set(self.weights):
                raise ValueError('Expected the 97 calibrated MAE/ConvNeXt outputs')
            self.encoders = PixelDriftingEncoders(self.cache, device=self.device,
                microbatch=microbatch, mixed_precision=True, checkpoint_inputs=False)
            self.encoders.family_names = ('mae', 'convnext')
            configure_full_execution(self.encoders,
                mae_bf16_channels_last=mae_bf16_channels_last, compile_encoders=compile_encoders)
            # Preserve the historical full-feature metadata digest.
            paths = [self.cache/name for name in
                     ('downloads.json','converted_manifest.json','calibration.json')]
        else:
            from models.encoders.sim import SIMEncoders
            calibration_path = ROOT/'models/encoders/sim_calibration.json'
            self.calibration = json.loads(calibration_path.read_text())
            self.encoders = SIMEncoders(device=self.device, microbatch=microbatch,
                                       hf_home=hf_home, torch_home=torch_home)
            self.names = self.encoders.family_names
            self.bandwidths = {name: self.calibration[name]['bandwidth'] for name in self.names}
            self.weights = {name: 1./3. for name in self.names}
            for name, spec in self.encoders.specs.items():
                expected = self.calibration[name]
                if (spec.feature_id, spec.feature_dim) != (expected['feature_id'], expected['feature_dim']):
                    raise ValueError(f'Historical SIM feature identity mismatch: {name}')
            paths = [ROOT/path for path in ('models/encoders/base.py', 'models/encoders/timm_features.py',
                'models/encoders/inception.py', 'models/encoders/sim.py',
                'models/encoders/sim_calibration.json', 'models/encoders/load.py', 'losses/matching.py',
                'utils/posttrain_assets.py', 'data/imagenet.py', 'old/releases/encoders.json')]
            paths += [self.cache/name for name in ('class_order.npy','dataset_labels.npy','class_counts.npy')]
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.read_bytes())
        if stack == 'sim':
            digest.update(json.dumps(dict(microbatch=int(microbatch), precision='bf16',
                real_references='online', preprocessing='historical_SIM_unclipped',
                weights=self.weights), sort_keys=True).encode())
        self.manifest_digest = digest.hexdigest()
        self.eval().requires_grad_(False)

    @property
    def family_names(self):
        return self.encoders.family_names

    @property
    def microbatch(self):
        return self.encoders.microbatch

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, images):
        rows = self.encoders(images)
        if set(rows) != set(self.names):
            raise RuntimeError('Loss representation names changed')
        return rows

    def encode_family(self, images, family):
        return self.encoders.encode_family(images, family)

    def checkpoint_metadata(self):
        metadata = dict(feature_stack=self.stack, manifest_sha256=self.manifest_digest)
        # Preserve the baseline metadata; optimized checkpoints bind execution too.
        if self.execution != dict(microbatch=8, mae_bf16_channels_last=False, compile_encoders=False):
            metadata['encoder_execution'] = dict(version=1, **self.execution)
        return metadata
