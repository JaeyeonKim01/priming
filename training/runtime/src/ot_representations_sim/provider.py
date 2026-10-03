"""Frozen historical SIM encoders with online references and feature replay.

The parent supplies the production real-image lookup, detached target encoding,
microbatch input VJP, and checkpoint contract. No cached real feature rows are
loaded. The historical bank supplies only frozen bandwidths and feature IDs.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from distmatch import Feature
from distmatch.extractors import FeatureResolver
from src.ot_representations.provider import RepresentationProvider as BaseProvider
from src.ot_representations.provider import RealImages


FEATURES = (
    Feature(name="siglip", model="vit_so400m_patch16_siglip_256.v2_webli", pool="cls", target_size=224),
    Feature(name="mae", model="vit_large_patch16_224.mae", pool="cls", target_size=224),
    Feature(name="inception", model="inception", pool="cls"),
)


class SIMEncoders(nn.Module):
    def __init__(self, device, *, microbatch=8, mixed_precision=True):
        super().__init__()
        self.microbatch = int(microbatch)
        if self.microbatch < 1:
            raise ValueError("encoder microbatch must be positive")
        self.mixed_precision = bool(mixed_precision)
        self.family_names = tuple(feature.name for feature in FEATURES)
        device = torch.device(device)
        # timm initializes modules before loading pretrained weights. Keep that
        # construction out of the generator's live-noise random stream.
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            resolved = FeatureResolver(device).resolve_all(FEATURES)
        self.extractors = nn.ModuleDict({f.name: f.extractor for f in resolved})
        self.specs = {f.name: f.spec for f in resolved}
        self.eval().requires_grad_(False)

    def encode_family(self, images, family):
        # Explicit context is needed on the no-grad feature pass AND replay:
        # the production driver invokes these outside generator autocast.
        with torch.autocast(images.device.type, dtype=torch.bfloat16,
                            enabled=self.mixed_precision):
            output = self.extractors[family].forward(images.float())
        return {family: output}

    def forward(self, images):
        return {
            family: torch.cat([
                self.encode_family(part, family)[family]
                for part in images.split(self.microbatch)
            ])
            for family in self.family_names
        }


class RepresentationProvider(BaseProvider):
    def __init__(self, args, device):
        if args.ot_feature_stack != "sim" or args.discrepancy_kind != "featurewise_ot":
            raise ValueError("This arm requires SIM feature-wise OT")
        if args.ot_encoder_checkpointing or args.ot_compile_encoders:
            raise ValueError("This arm preserves eager encoders and input-gradient replay")
        if not args.ot_feature_gradient_replay:
            raise ValueError("SIM online references require production feature replay")
        self.stack = "sim"
        self.device = torch.device(device)
        self.cache = Path(args.ot_repr_cache)
        calibration_path = Path(__file__).with_name("calibration.json")
        self.calibration = json.loads(calibration_path.read_text())
        metadata = self.calibration["features"]
        self.names = tuple(feature.name for feature in FEATURES)
        self.sigmas = {name: float(metadata[name]["bandwidth"]) for name in self.names}
        self.weights = {name: 1.0 / 3.0 for name in self.names}
        self.encoders = SIMEncoders(device, microbatch=args.ot_encoder_microbatch)
        for name, spec in self.encoders.specs.items():
            if spec.feature_id != metadata[name]["feature_id"] or spec.feature_dim != metadata[name]["feature_dim"]:
                raise ValueError(f"Historical SIM feature identity mismatch: {name}")
        self.reference = RealImages(self.cache, args.data_path, args.ot_reference_workers)
        self.loss_contract = dict(
            kind=args.discrepancy_kind, coefficient=float(args.lambda_ot),
            energy_delta=float(args.energy_delta),
            epsilon_mixture=dict(args.ot_epsilon_mixture),
            iterations=int(args.ot_num_iters), tolerance=float(args.ot_tolerance),
            self_weight=1.0,
        )
        import distmatch.extractors as extractors
        extractor_root = Path(extractors.__file__).parent
        sources = [Path(__file__), calibration_path,
                   Path(__file__).parents[1] / "ot_representations/provider.py",
                   *sorted(extractor_root.glob("*.py"))]
        references = [self.cache / name for name in
                      ("class_order.npy", "dataset_labels.npy", "class_counts.npy")]
        digest = hashlib.sha256()
        for path in sources + references:
            digest.update(path.read_bytes())
        digest.update(json.dumps(dict(microbatch=args.ot_encoder_microbatch,
                                     precision="bf16", real_references="online",
                                     preprocessing="historical_SIM_unclipped",
                                     weights=self.weights), sort_keys=True).encode())
        self.manifest_digest = digest.hexdigest()
