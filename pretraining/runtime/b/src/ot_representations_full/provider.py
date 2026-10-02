"""Opt-in feature provider and deterministic full-population references."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from torchvision.transforms.functional import pil_to_tensor
from .encoders import PixelDriftingEncoders


class RealImages:
    def __init__(self, cache, data_path, workers=4):
        cache = Path(cache)
        self.order = np.load(cache/'class_order.npy')
        self.original_labels = np.load(cache/'dataset_labels.npy')
        self.class_counts = torch.from_numpy(np.load(cache/'class_counts.npy'))
        self.class_starts = torch.cat([self.class_counts.new_zeros(1), self.class_counts.cumsum(0)[:-1]])
        self.labels = torch.from_numpy(self.original_labels[self.order])
        self.dataset = load_from_disk(str(Path(data_path)/'imagenet-latents-images'))
        assert len(self.dataset) == len(self.order) == 1281167
        assert self.class_counts.numel() == 1000 and self.class_counts.min() >= 128
        self.pool = ThreadPoolExecutor(max_workers=int(workers))

    def _image(self, dataset_index):
        item = self.dataset[int(dataset_index)]
        assert item['label'] == self.original_labels[int(dataset_index)]
        image = pil_to_tensor(item['image'].convert('RGB')).float().div_(255)
        assert image.shape == (3, 256, 256)
        return image

    def fetch(self, indices, labels=None):
        indices = indices.detach().cpu().numpy()
        original_ids = self.order[indices]
        if labels is not None:
            assert np.array_equal(self.original_labels[original_ids], labels.detach().cpu().numpy())
        return torch.stack(list(self.pool.map(self._image, original_ids)))


class RepresentationProvider:
    repulsion_queue_size = 0
    kernel = 'ot_only'
    repulsion_queue_kernel = 'disabled'
    repulsion_queue_bandwidths = {}
    repulsion_queue_weight = 0.
    local_generated_scoring = True
    include_real_constant = False
    representation_swap = True

    def __init__(self, args, device):
        self.cache = Path(args.ot_repr_cache)
        self.calibration = json.loads((self.cache/'calibration.json').read_text())
        source_dir = Path(__file__).resolve().parent
        source_files = [source_dir/'encoders.py', *sorted((source_dir/'vendor').glob('*.py'))]
        source_hash = hashlib.sha256(b''.join(p.read_bytes() for p in source_files)).hexdigest()
        if source_hash != self.calibration['encoder_source_sha256']:
            raise ValueError('Encoder implementation differs from calibration source')
        self.sigmas = self.calibration['bandwidths']
        self.weights = self.calibration['weights']
        self.names = tuple(sorted(self.sigmas))
        assert len(self.names) == 97 and set(self.names) == set(self.weights)
        self.encoders = PixelDriftingEncoders(self.cache, device=device,
            microbatch=args.ot_encoder_microbatch, mixed_precision=True,
            checkpoint_inputs=args.ot_encoder_checkpointing)
        self.reference = RealImages(self.cache, args.data_path, args.ot_reference_workers)
        selected = np.asarray(self.calibration['sorted_row_indices'])
        if not np.array_equal(self.reference.order[selected], self.calibration['dataset_indices']):
            raise ValueError('Real dataset order differs from calibration manifest')
        self.device = device
        manifest_bytes = b''.join((self.cache/name).read_bytes() for name in
            ['downloads.json','converted_manifest.json','calibration.json'])
        self.manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()

    @property
    def repulsion_queue_fill_stats(self):
        return dict(total_rows=0,populated_classes=0,min_rows_per_populated_class=0,
                    max_rows_per_class=0)

    def extract_features(self, images, *, compiled=True):
        rows = self.encoders(images)
        if set(rows) != set(self.names):
            raise RuntimeError('Loss representation names changed')
        return rows

    @torch.no_grad()
    def target_features(self, batch):
        images = self.reference.fetch(batch.target_indices,batch.target_labels)
        images = images.to(self.device,non_blocking=True)
        return self.extract_features(images,compiled=False)

    def compile(self, **kwargs):
        # Checkpointed encoders initially remain eager; compilation is an
        # execution-only optimization to benchmark after functional validation.
        pass

    def input_vjp(self, images, feature_gradients):
        """Replay one encoder microbatch at a time after the full OT solve.

        All population coupling happens before this function. This is the
        chain rule through fixed encoders, not per-microbatch transport.
        """
        result = torch.zeros_like(images)
        microbatch = self.encoders.microbatch
        for family in ['mae', 'convnext']:
            for start in range(0, len(images), microbatch):
                stop = min(start + microbatch, len(images))
                part = images[start:stop].detach().requires_grad_(True)
                outputs = self.encoders.encode_family(part, family)
                names = tuple(outputs)
                cotangents = tuple(feature_gradients[k][start:stop].to(outputs[k].dtype)
                                   for k in names)
                gradient, = torch.autograd.grad(tuple(outputs[k] for k in names),
                                               part, grad_outputs=cotangents)
                result[start:stop].add_(gradient.detach())
                del outputs, cotangents, gradient, part
        return result

    def running_state_dict(self):
        return dict(format='ost.ot_representation_swap',version=1,
                    manifest_sha256=self.manifest_digest)

    def load_running_state_dict(self, state, strict=True):
        if state != self.running_state_dict():
            raise ValueError('Resume representation/calibration manifest mismatch')
