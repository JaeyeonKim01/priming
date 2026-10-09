"""Frozen historical SigLIP, ViT-MAE, and Inception features for SIM OT."""
import torch
from torch import nn

from .inception import InceptionExtractor
from .timm_features import ModelFamilyExtractor
from utils.posttrain_assets import sim_weights

MODELS = {
    'siglip': 'vit_so400m_patch16_siglip_256.v2_webli',
    'mae': 'vit_large_patch16_224.mae',
}


class SIMEncoders(nn.Module):
    family_names = ('siglip', 'mae', 'inception')

    def __init__(self, device='cuda', *, microbatch=8, mixed_precision=True,
                 hf_home=None, torch_home=None):
        super().__init__()
        self.microbatch = int(microbatch)
        if self.microbatch < 1:
            raise ValueError('Encoder microbatch must be positive')
        self.mixed_precision = bool(mixed_precision)
        paths = sim_weights(hf_home=hf_home, torch_home=torch_home)
        device = torch.device(device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        # Pretrained timm constructors consume RNG before loading their weights.
        with torch.random.fork_rng(devices=devices):
            self.extractors = nn.ModuleDict()
            for family, name in MODELS.items():
                self.extractors[family] = ModelFamilyExtractor(
                    family, name, pool='cls', target_size=224, weights_path=paths[family])
                self.extractors[family].to(device).eval().requires_grad_(False)
            self.extractors['inception'] = InceptionExtractor(
                model='inception', metadata={}, weights_path=paths['inception'])
            self.extractors['inception'].to(device).eval().requires_grad_(False)
        self.specs = {name: module.spec for name, module in self.extractors.items()}
        self.eval().requires_grad_(False)

    def encode_family(self, images, family):
        with torch.autocast(images.device.type, dtype=torch.bfloat16,
                            enabled=self.mixed_precision):
            output = self.extractors[family].forward(images.float())
        return {family: output}

    def forward(self, images):
        return {
            family: torch.cat([self.encode_family(part, family)[family]
                               for part in images.split(self.microbatch)])
            for family in self.family_names
        }
