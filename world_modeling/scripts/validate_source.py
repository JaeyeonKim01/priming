"""Compare migrated models to an extracted, unchanged archive on CUDA."""
import argparse
import gc
import importlib
import json
from pathlib import Path
import sys
import types
import torch
from world_modeling.utils.config import load_config, parser
from world_modeling.models.denoiser import Denoiser


def source_denoiser(reference, dataset):
    # The archive chooses its architecture through the top-level `unet` name.
    saved = {k: v for k, v in sys.modules.items() if k == 'unet' or k.startswith('unet.')}
    for key in saved: del sys.modules[key]
    package = types.ModuleType('unet')
    package.__path__ = [str(reference / f'{dataset}_driftworld/unet')]
    sys.modules['unet'] = package
    name = f'_archive_world_{dataset}'
    package = types.ModuleType(name)
    package.__path__ = [str(reference / 'rt1_driftworld')]
    sys.modules[name] = package
    try:
        return importlib.import_module(name + '.flow_matching_denoiser').Denoiser
    finally:
        for key in list(sys.modules):
            if key == 'unet' or key.startswith('unet.'): del sys.modules[key]
        sys.modules.update(saved)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--storage-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    results = {}
    for dataset, step in [('rt1', 84608), ('bridge', 53184)]:
        cfg = load_config(parser('pretrain').parse_args([
            '--config', f'world_modeling/configs/pretrain/{dataset}.yaml',
            '--storage-root', str(args.storage_root)]), 'pretrain')
        old = source_denoiser(args.reference, dataset)(cfg).cuda()
        new = Denoiser(cfg).cuda()
        for stage, checkpoint in [
            ('posttrain', args.storage_root / f'outputs/{dataset}_flow_distmatch_sinkhorn/ckpt_save/ckpt-step4000.pth'),
            ('pretrain', args.storage_root / f'outputs/{dataset}_flow_xpred/ckpt_save/ckpt-step{step}.pth')]:
            state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)['model']
            old.load_state_dict(state, strict=True); new.load_state_dict(state, strict=True)
            del state
            torch.manual_seed(2026)
            z = torch.randn(2, 16, 32, 32, device='cuda')
            h = torch.randn(2, 64, 32, 32, device='cuda')
            a = torch.randn(2, 4, 10 if dataset == 'rt1' else 7, device='cuda')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                expected = old._run_model(old.ema_model, z, h, a)
                actual = new._run_model(new.ema_model, z, h, a)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            results[f'{dataset}/{stage}/ema_prediction_max_error'] = float((actual-expected).abs().max())
        batch = dict(video_latents=torch.randn(2, 5, 16, 32, 32, device='cuda'), action=a)
        losses = []
        for model in (old, new):
            torch.manual_seed(456)
            with torch.autocast('cuda', dtype=torch.bfloat16): loss, _ = model(batch, 'cuda')
            loss.backward(); losses.append(loss.detach())
        torch.testing.assert_close(*losses, rtol=0, atol=0)
        maximum = 0.
        for (key, x), (other, y) in zip(old.named_parameters(), new.named_parameters()):
            assert key == other
            if x.grad is None: assert y.grad is None
            else:
                torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0, msg=key)
                maximum = max(maximum, float((x.grad-y.grad).abs().max()))
        results[f'{dataset}/pretrain/loss'] = float(losses[0])
        results[f'{dataset}/pretrain/gradient_max_error'] = maximum
        for model in (old, new):
            parameters = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=cfg.opt.lr, betas=(.9, cfg.opt.beta2), weight_decay=cfg.opt.weight_decay)
            torch.nn.utils.clip_grad_norm_(parameters, cfg.opt.grad_clip)
            optimizer.step(); model.update_ema()
            del optimizer
        for key, x in old.state_dict().items():
            torch.testing.assert_close(x, new.state_dict()[key], atol=0, rtol=0, msg=key)
        results[f'{dataset}/pretrain/adamw_and_ema_exact'] = True
        print(json.dumps(results), flush=True)
        del old, new, z, h, a, batch, loss, actual, expected
        gc.collect(); torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__': main()
