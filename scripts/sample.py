"""Generate ImageNet-256 PNG samples from one released Priming checkpoint."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from priming_paths import resolve
from verify_checkpoints import digest


def load_generator(key, stage, weights, checkpoint, device):
    """Load only the pixel generator; no dataset, metric encoders, or VAE needed."""
    import torch
    runtime = ROOT / 'eval/runtime'
    sys.path[:0] = [str(runtime), str(runtime / 'packages/distmatch/src')]
    from src.models import OSTDenoiser_models

    manifest = 'pretraining_manifest.json' if stage == 'pretraining' else 'manifest.json'
    item = json.loads((ROOT / 'checkpoints' / manifest).read_text())[key]
    path = resolve(checkpoint or item['checkpoint'])
    if not path.is_file():
        raise FileNotFoundError(f'{path}\nDownload it with: hf download jaykim1102/priming {item["checkpoint"]} --local-dir {ROOT}')
    if path.stat().st_size != item['checkpoint_bytes'] or digest(path) != item['checkpoint_sha256']:
        raise ValueError(f'Checkpoint does not match the released {stage}/{key}: {path}')
    config_path = ROOT / 'checkpoints' / f'{key}.json'
    config_identity = json.loads((ROOT / 'checkpoints/manifest.json').read_text())[key]
    if digest(config_path) != config_identity['config_sha256']:
        raise ValueError(f'Inference config changed: {config_path}')
    args = SimpleNamespace(**json.loads(config_path.read_text())['sampling_args'])
    net = OSTDenoiser_models[args.model](
        img_size=args.img_size, patch_size=args.patch_size, in_channels=args.token_channels,
        tokenizer_patch_size=1, num_classes=args.num_classes, label_drop_prob=args.label_drop_prob,
        corruption='denoising', t_dist='lognorm', t_min=0., t_max=1.,
        P_mean=args.P_mean, P_std=args.P_std, cfg_beta=args.cfg_beta,
        cfg_omega_max=args.cfg_omega_max, class_tokens=args.class_tokens,
        guidance_tokens=args.guidance_tokens, interval_tokens=args.interval_tokens,
        rope_2d=args.rope_2d, learned_pe=args.learned_pe, noise_scale=args.noise_scale,
        norm_eps=args.norm_eps, norm_p=args.norm_p, perceptual_threshold=args.perceptual_threshold)
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    ema = '0.9999' if stage == 'pretraining' else '0.999'
    selected = state['model'] if weights == 'online' else state['model_ema']['shadows'][ema]
    net.load_state_dict(selected, strict=True, assign=True)
    net.to(device).eval().requires_grad_(False)
    return net, args, dict(checkpoint=item['checkpoint'], checkpoint_sha256=item['checkpoint_sha256'],
                           stage=stage, weights=weights, ema=ema if weights == 'ema' else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['priming_b_80', 'priming_b_160', 'priming_l_80', 'priming_l_160'], default='priming_b_160')
    parser.add_argument('--stage', choices=['pretraining', 'posttraining'], default='posttraining')
    parser.add_argument('--weights', choices=['ema', 'online'], default='ema')
    parser.add_argument('--checkpoint', help='Optional path to the selected released checkpoint')
    parser.add_argument('--num-images', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--class-ids', type=int, nargs='+', help='ImageNet IDs 0–999, repeated in the supplied order; default cycles through 0–999')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--output-dir', required=True, help='New directory, relative to the repository')
    options = parser.parse_args()
    if options.num_images < 1 or options.batch_size < 1:
        parser.error('--num-images and --batch-size must be positive')
    if options.class_ids and any(label < 0 or label >= 1000 for label in options.class_ids):
        parser.error('--class-ids must be between 0 and 999')
    output = resolve(options.output_dir)
    if output.exists():
        parser.error(f'Output already exists; choose a new directory: {output}')

    import torch
    from PIL import Image
    if options.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Run on a GPU node, or use --device cpu for a small FP32 smoke test.')
    torch.set_num_threads(4)
    net, args, metadata = load_generator(options.model, options.stage, options.weights,
                                          options.checkpoint, options.device)
    # Seed after construction so model initialization cannot consume the sample stream.
    torch.manual_seed(options.seed)
    if options.device == 'cuda':
        torch.cuda.manual_seed_all(options.seed)
    labels = options.class_ids or list(range(1000))
    labels = [labels[i % len(labels)] for i in range(options.num_images)]
    output.mkdir(parents=True, exist_ok=False)
    images = []
    with torch.inference_mode():
        for start in range(0, len(labels), options.batch_size):
            batch_labels = torch.tensor(labels[start:start + options.batch_size], device=options.device)
            amp = torch.autocast('cuda', dtype=torch.bfloat16) if options.device == 'cuda' else contextlib.nullcontext()
            with amp:
                batch = net.generate(len(batch_labels), labels=batch_labels, cfg=1., args=args, verbose=False)
            if not torch.isfinite(batch).all():
                raise ValueError('Generator returned non-finite pixels')
            batch = ((batch.float() + 1) / 2).clamp(0, 1)
            batch = (batch * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for offset, pixels in enumerate(batch):
                index = start + offset
                filename = f'{index:06d}_class{labels[index]:04d}.png'
                Image.fromarray(pixels).save(output / filename)
                images.append(dict(file=filename, class_id=labels[index]))
    metadata.update(model=options.model, seed=options.seed, batch_size=options.batch_size,
                    num_images=len(images), device=options.device,
                    precision='BF16' if options.device == 'cuda' else 'FP32',
                    sampling_steps=1, cfg=1., images=images,
                    scope='Single-process samples; use eval/evaluate.py for the paper metric protocol.')
    (output / 'samples.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved {len(images)} PNGs and samples.json to {output}')


if __name__ == '__main__':
    main()
