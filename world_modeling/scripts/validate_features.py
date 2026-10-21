"""Check archived versus migrated SD3/DINO feature values and input gradients."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import types
import torch
from world_modeling.utils.config import configure_assets, load_config, parser
from world_modeling.losses.features import LatentDinoFeatureBranches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--storage-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = load_config(parser('posttrain').parse_args([
        '--config', 'world_modeling/configs/posttrain/rt1.yaml',
        '--storage-root', str(args.storage_root)]), 'posttrain')
    configure_assets(cfg)
    saved = {k: v for k, v in sys.modules.items() if k.split('.')[0] in ('features', 'encoders')}
    for key in saved: del sys.modules[key]
    for name in ('features', 'encoders'):
        package = types.ModuleType(name)
        package.__path__ = [str(args.reference / 'rt1_driftworld' / name)]
        sys.modules[name] = package
    try:
        reference_type = importlib.import_module('features.distmatch_latent_features').LatentDinoFeatureBranches
    finally:
        for key in list(sys.modules):
            if key.split('.')[0] in ('features', 'encoders'): del sys.modules[key]
        sys.modules.update(saved)
    torch.manual_seed(81)
    source = reference_type(cfg).cuda().eval()
    migrated = LatentDinoFeatureBranches(cfg).cuda().eval()
    latent = torch.randn(2,16,32,32,device='cuda')
    result = {}
    for precision in ('fp32', 'bf16'):
        outputs, gradients = [], []
        for model in (source, migrated):
            x = latent.clone().requires_grad_()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
                branches = model(x)
                loss = sum(value.float().square().mean() for value in branches.values())
            gradients.append(torch.autograd.grad(loss,x)[0])
            outputs.append({k:v.detach() for k,v in branches.items()})
        entry = {}
        for key in outputs[0]:
            torch.testing.assert_close(outputs[0][key],outputs[1][key],rtol=0,atol=0)
            entry[key] = {'shape':list(outputs[0][key].shape), 'max_error':float((outputs[0][key]-outputs[1][key]).abs().max())}
        difference = gradients[0] - gradients[1]
        entry['input_gradient_max_error'] = float(difference.abs().max())
        entry['input_gradient_norm'] = float(gradients[0].norm())
        entry['input_gradient_relative_l2_error'] = float(difference.norm()/gradients[0].norm())
        entry['input_gradient_cosine'] = float(torch.nn.functional.cosine_similarity(gradients[0].flatten(),gradients[1].flatten(),dim=0))
        result[precision] = entry
        print(json.dumps(result,indent=2),flush=True)
        assert entry['input_gradient_norm'] > 0
        if precision == 'fp32':
            torch.testing.assert_close(*gradients,rtol=1e-5,atol=1e-6)
        else:
            # Recomputed BF16 activations can change accumulation order. Require
            # near-identical gradient direction and <1% relative L2 discrepancy.
            assert entry['input_gradient_relative_l2_error'] < .01
            assert entry['input_gradient_cosine'] > .9999
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
