"""Compare local exports against released native EMA weights on one CUDA GPU."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import diffusers
from safetensors import safe_open
import torch

from . import PrimingPipeline
from .convert import convert
from utils.checkpoint import load_released_model


def errors(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    return dict(max_abs=difference.max().item(), mean_abs=difference.mean().item(),
                exact=torch.equal(actual, expected))


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Run full B/L validation on a CUDA allocation')
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(torch=torch.__version__, diffusers=diffusers.__version__,
                  gpu=torch.cuda.get_device_name(), seed=123, labels=[0,207,999],
                  scope='Fixed-input inference parity and serialization; not a FID reproduction', models={})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((root/'old/checkpoints/manifest.json').read_text())
    for variant in ['b','l']:
        community = Path(args.community_weights)/f'Priming-{variant.upper()}.safetensors'
        matches = []
        candidate_checks = {}
        with safe_open(community, framework='pt', device='cpu') as weights:
            # Exact whole-state comparisons, including dtype. Compare both 80/160.
            for epochs in [80,160]:
                key = f'priming_{variant}_{epochs}'
                path = root/manifest[key]['checkpoint']
                state = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
                ema = state['model_ema']['shadows']['0.999']
                same_keys = set(weights.keys()) == {k.removeprefix('net.') for k in ema}
                equal = same_keys
                compared = 0
                for name, tensor in ema.items():
                    if not equal:
                        break
                    other = weights.get_tensor(name.removeprefix('net.'))
                    equal = tensor.dtype == other.dtype and torch.equal(tensor, other)
                    compared += 1
                candidate_checks[key] = dict(equal=equal, tensors_compared=compared)
                if equal:
                    matches.append(key)
                del state, ema
        if len(matches) != 1:
            raise AssertionError(f'Expected one exact released EMA match: {candidate_checks}')
        key = matches[0]
        print(f'{variant}: exact community match {key}', flush=True)
        pipeline, provenance = convert(key, output/key)
        native, _ = load_released_model(key, device='cuda', terminal=True)
        pipeline.to('cuda')
        pipeline.set_progress_bar_config(disable=True)
        generator = torch.Generator(device='cuda').manual_seed(123)
        noise = torch.randn(3,3,256,256, generator=generator, device='cuda')
        labels = torch.tensor([0,207,999], device='cuda')
        checks = {}
        references = {}
        for mode in ['fp32', 'bf16_autocast']:
            context = torch.autocast('cuda', dtype=torch.bfloat16) if mode == 'bf16_autocast' else nullcontext()
            with torch.inference_mode(), context:
                expected = native._forward_generator(noise, labels)
                actual = pipeline(labels.tolist(), latents=noise, output_type='latent').images
            torch.testing.assert_close(actual, expected.float(), rtol=0, atol=0)
            checks[mode] = errors(actual, expected)
            references[mode] = expected.float().cpu()
        # Compare the community README's weight-cast mode separately.
        native.to(torch.bfloat16)
        pipeline.to(dtype=torch.bfloat16)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            expected = native._forward_generator(noise, labels)
            actual = pipeline(labels.tolist(), latents=noise, output_type='latent').images
        torch.testing.assert_close(actual, expected.float(), rtol=0, atol=0)
        checks['bf16_weights'] = errors(actual, expected)
        checks['bf16_weights_vs_fp32_weights_autocast'] = errors(actual.cpu(), references['bf16_autocast'])
        del native, pipeline
        torch.cuda.empty_cache()
        loaded = PrimingPipeline.from_pretrained(output/key, local_files_only=True).to('cuda')
        loaded.set_progress_bar_config(disable=True)
        with torch.inference_mode():
            actual = loaded(labels.tolist(), latents=noise, output_type='latent').images.cpu()
        torch.testing.assert_close(actual, references['fp32'], rtol=0, atol=0)
        checks['saved_reloaded_fp32'] = errors(actual, references['fp32'])
        record = dict(provenance=provenance, community_sha256=file_hash(community),
                      candidate_checks=candidate_checks, parity=checks, passed=True)
        report['models'][variant] = record
        (output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(record, indent=2), flush=True)
        del loaded
        torch.cuda.empty_cache()
    report['passed'] = True
    (output/'validation.json').write_text(json.dumps(report, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--community-weights', required=True)
    parser.add_argument('--output', required=True)
    validate(parser.parse_args())


if __name__ == '__main__':
    main()
