"""Package a local conversion with self-contained Diffusers custom components."""
import argparse
import json
from pathlib import Path
import shutil


def export_hub(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f'Output directory must be empty: {destination}')
    package = Path(__file__).resolve().parent
    # Copy only inference assets; never optimizer/training state or local caches.
    files = ['model_index.json', 'conversion.json', 'transformer/config.json',
             'transformer/diffusion_pytorch_model.safetensors', 'scheduler/scheduler_config.json']
    for name in files:
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/name, target)
    for name in ['pipeline.py', 'transformer/transformer_priming.py',
                 'scheduler/scheduling_priming.py', 'community-source.json', 'validation.json']:
        shutil.copyfile(package/name, destination/name)
    shutil.copyfile(package.parents[1]/'LICENSE', destination/'LICENSE')
    notice = (package/'NOTICE.md').read_text().replace('(../../LICENSE)', '(LICENSE)')
    (destination/'NOTICE.md').write_text(notice)
    index = json.loads((destination/'model_index.json').read_text())
    index['_class_name'] = ['pipeline', 'PrimingPipeline']
    index['transformer'] = ['transformer_priming', 'PrimingTransformer2DModel']
    index['scheduler'] = ['scheduling_priming', 'PrimingScheduler']
    (destination/'model_index.json').write_text(json.dumps(index, indent=2)+'\n')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(export_hub(args.source, args.output))


if __name__ == '__main__':
    main()
