"""Export verified released post-training EMA weights; run as a module."""
import argparse
import json
from pathlib import Path

import torch

from . import PrimingPipeline, PrimingScheduler, PrimingTransformer2DModel


def convert(model_key, output, *, checkpoint=None):
    from utils.checkpoint import load_released_model

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    native, args = load_released_model(model_key, checkpoint=checkpoint, terminal=True)
    net = native.net
    config = dict(
        sample_size=net.input_size, patch_size=net.patch_size,
        in_channels=net.in_channels, hidden_size=net.hidden_size,
        depth=len(net.blocks), num_attention_heads=net.num_heads,
        mlp_ratio=net.mlp_ratio, num_classes=net.num_classes,
        num_class_tokens=net.num_class_tokens, num_cfg_tokens=net.num_cfg_tokens,
        num_interval_tokens=net.num_interval_tokens, noise_scale=args['noise_scale'],
        guidance_scale=1.0, guidance_interval_min=args['interval_min'],
        guidance_interval_max=args['interval_max'], rope_2d=args['rope_2d'],
        learned_pe=args['learned_pe'],
    )
    with torch.device('meta'):
        transformer = PrimingTransformer2DModel(**config)
    transformer.load_state_dict(net.state_dict(), strict=True, assign=True)
    transformer.eval().requires_grad_(False)
    pipeline = PrimingPipeline(transformer, PrimingScheduler())
    pipeline.save_pretrained(output, safe_serialization=True)
    root = Path(__file__).resolve().parents[2]
    record = json.loads((root/'old/checkpoints/manifest.json').read_text())[model_key]
    provenance = dict(model=model_key, stage='posttraining', weights='ema', ema_decay='0.999',
                      checkpoint_sha256=record['checkpoint_sha256'],
                      config_sha256=record['config_sha256'],
                      tensor_mapping='Remove net. prefix; no dtype conversion',
                      dtype=str(transformer.dtype))
    (output/'conversion.json').write_text(json.dumps(provenance, indent=2)+'\n')
    return pipeline, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=[f'priming_{m}_{e}' for m in ['b','l'] for e in [80,160]])
    parser.add_argument('--checkpoint', help='Alternative path to the same SHA-verified released checkpoint')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    _, record = convert(args.model, args.output, checkpoint=args.checkpoint)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
