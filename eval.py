"""Standalone FID/FDr5 and scheduled post-training FDr6 evaluation."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from metrics.frechet import covariance_sqrt, frechet_distance
from metrics.inception import compute_isc, compute_precision_recall
from metrics.moments import ExactMoments
from metrics.specs import SPECS, BASELINES, PAPER_SPECS
from metrics.statistics import load_feature_stats
from utils.collectives import DistributedContext
from utils.files import sha256, write_json

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)


def make_extractors(weights, device, specs=SPECS):
    from models.encoders.inception import InceptionExtractor
    from models.encoders.timm_features import ModelFamilyExtractor
    extractors = {}
    for name, model, family, size, _ in specs:
        if model == 'inception':
            encoder = InceptionExtractor(model='inception', metadata={}, weights_path=weights[name])
        else:
            encoder = ModelFamilyExtractor(family, model, pool='cls', target_size=size,
                                           weights_path=weights[name])
        extractors[name] = encoder.to(device).eval().requires_grad_(False)
    return extractors


class FrechetEvaluator:
    """Sample-in, metric-out accumulation using the original FP64 moments."""
    def __init__(self, extractors, references, *, device, distributed='auto'):
        self.extractors, self.device = extractors, torch.device(device)
        self.context = DistributedContext(distributed)
        self.context.validate_same([(name, module.spec.feature_id) for name, module in extractors.items()],
                                   label='evaluation feature configuration')
        self.references, self.reference_sqrt = {}, {}
        for name, module in extractors.items():
            reference = load_feature_stats(references[name], expected_spec=module.spec)
            mean, cov = reference.tensors(device=self.device, dtype=torch.float64)
            self.references[name] = mean, cov
            self.reference_sqrt[name] = covariance_sqrt(cov)
        self.reset()

    def reset(self):
        self.moments = {name: ExactMoments.empty(module.spec.feature_dim, device=self.device)
                        for name, module in self.extractors.items()}

    @torch.inference_mode()
    def update(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or not torch.isfinite(images).all():
            raise ValueError('Evaluation expects finite RGB image batches')
        for name, encoder in self.extractors.items():
            rows = encoder(images)
            if not torch.isfinite(rows).all():
                raise FloatingPointError(f'Nonfinite evaluation features: {name}')
            self.moments[name].update(rows)

    @torch.inference_mode()
    def compute(self):
        moments = {name: value.globalized(self.context) for name, value in self.moments.items()}
        counts = {value.count for value in moments.values()}
        if len(counts) != 1:
            raise RuntimeError('Evaluation encoders observed different sample counts')
        fd = {}
        for name, value in moments.items():
            mean, cov = value.mean_covariance()
            ref_mean, ref_cov = self.references[name]
            fd[name] = float(frechet_distance(mean, cov, ref_mean, ref_cov,
                                             reference_sqrt=self.reference_sqrt[name]).item())
        return fd, counts.pop()


def partition(total, world, rank):
    base, remainder = divmod(total, world)
    start = rank*base+min(rank, remainder)
    return start, start+base+int(rank < remainder)


def gather_rows(rows, context):
    """Gather uneven rank shards in rank order, preserving the original IS ordering."""
    if context.world_size == 1:
        return rows
    count = torch.tensor([len(rows)], dtype=torch.long, device=rows.device)
    counts = [torch.zeros_like(count) for _ in range(context.world_size)]
    dist.all_gather(counts, count, group=context.group)
    maximum = max(int(value.item()) for value in counts)
    if len(rows) < maximum:
        rows = torch.cat([rows, rows.new_zeros(maximum-len(rows), rows.shape[1])])
    gathered = [torch.zeros_like(rows) for _ in counts] if context.rank == 0 else None
    dist.gather(rows, gather_list=gathered, dst=0, group=context.group)
    if context.rank == 0:
        return torch.cat([value[:int(count.item())] for value, count in zip(gathered, counts)])
    return None


@torch.inference_mode()
def inception_metrics(feature_batches, logit_batches, reference, context, *, expected, device=None, k=3, batch_size=10000):
    device = torch.device(device or feature_batches[0].device)
    local_count = sum(len(value) for value in feature_batches)
    if local_count != sum(len(value) for value in logit_batches):
        raise RuntimeError('Inception feature/logit counts differ')
    if context.sum_int(local_count, device=device) != expected:
        raise RuntimeError('Inception sample count differs from evaluation protocol')
    features = gather_rows(torch.cat(feature_batches).to(device), context)
    logits = gather_rows(torch.cat(logit_batches).to(device), context)
    feature_batches.clear()
    logit_batches.clear()
    # The historical distributed scalar broadcasts round native metrics to FP32.
    dtype = torch.float32 if context.enabled else torch.float64
    values = torch.zeros(3, dtype=dtype, device=device)
    if context.rank == 0:
        score, _ = compute_isc(logits.cpu())
        precision, recall = compute_precision_recall(reference.to(device=device, dtype=torch.float32),
            features.float(), k=k, batch_size=batch_size)
        values[:] = torch.tensor([score, precision, recall], dtype=values.dtype, device=device)
    if context.world_size > 1:
        dist.broadcast(values, src=0, group=context.group)
    return dict(zip(('is', 'precision', 'recall'), map(float, values.tolist())))


@torch.inference_mode()
def evaluate_samples(model, ema, evaluator, inception, reference, config, sampling, *, num_classes=1000):
    """Use identical images for FDr-6 (BF16) and the native IS/P&R (FP32) path."""
    device = next(model.parameters()).device
    context = evaluator.context
    total = config['num_images']
    start, end = partition(total, context.world_size, context.rank)
    classes = torch.arange(start, end, dtype=torch.long).remainder(num_classes)
    args = SimpleNamespace(interval_min=sampling['t_min'], interval_max=sampling['t_max'], same_noise=False)
    rows = []
    for label in ('online', ema.label):
        evaluator.reset()
        feature_batches, logit_batches = [], []
        model.eval()
        with nullcontext() if label == 'online' else ema.swap(model):
            for labels in classes.split(config['batch_size']):
                labels = labels.to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    images = model.generate(len(labels), labels, cfg=sampling['cfg'], args=args,
                                            verbose=False)
                    images = ((images+1)/2).clamp(0, 1)
                features, logits = inception(images)
                feature_batches.append(features.float().cpu())
                logit_batches.append(logits.cpu())
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    evaluator.update(images)
        fd, count = evaluator.compute()
        if count != total:
            raise RuntimeError('Fréchet sample count differs from evaluation protocol')
        native = inception_metrics(feature_batches, logit_batches, reference, context,
            expected=total, device=device, k=config['prc_k'], batch_size=config['prc_batch_size'])
        fdr = {name: fd[name]/baseline for name, baseline in BASELINES.items()}
        rows.append(dict(ema_label=label, cfg=sampling['cfg'], fd=fd, fdr=fdr,
                         fdr6=sum(fdr.values())/len(fdr), num_samples=count, **native))
    return rows


def evaluate(model, ema, config, sampling, weights):
    from models.encoders.adm import load_inception
    device = next(model.parameters()).device
    # The old reference loader resolves its encoders before the evaluator resolves
    # them again. Retain that constructor order and its CPU RNG consumption.
    preliminary = make_extractors(weights, device)
    references = {name: load_feature_stats(Path(config['stats_dir'])/filename,
                   expected_spec=preliminary[name].spec) for name, _, _, _, filename in SPECS}
    del preliminary
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    evaluator = FrechetEvaluator(make_extractors(weights, device), references, device=device)
    inception = load_inception(weights['FID(ADM)'], device=device)
    reference = torch.load(config['prc_reference'], map_location='cpu', weights_only=True) if evaluator.context.rank == 0 else None
    if reference is not None and (reference.ndim != 2 or reference.shape[1] != 2048 or not torch.isfinite(reference).all()):
        raise ValueError('Expected finite native Inception reference features [N, 2048]')
    return evaluate_samples(model, ema, evaluator, inception, reference, config, sampling)


def paper_batches(total, batch, world, rank, *, num_classes=1000, device='cuda'):
    """Original balanced class blocks, interleaved rank batches, and padded tail."""
    if total < num_classes or total % num_classes or batch < 1:
        raise ValueError('Image count must be a positive multiple of the class count; batch must be positive')
    for start in range(rank*batch, total, world*batch):
        indices = torch.arange(start, start+batch, device=device)
        labels = (indices//(total//num_classes)).clamp(max=num_classes-1).long()
        yield start, min(batch, total-start), labels


@torch.inference_mode()
def sample_images(model, labels, sampling):
    device = labels.device
    args = SimpleNamespace(**sampling)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        images = model.generate(len(labels), labels=labels, cfg=1., args=args, verbose=False)
        return ((images+1)/2).clamp(0, 1)


@torch.inference_mode()
def evaluate_paper_seed(sample, evaluator, official, output, *, total, batch, seed, num_classes=1000, amp=True):
    """Paired float-image FDr5 and rounded-uint8 official FID, with the old sampling order."""
    import numpy as np
    device = evaluator.device
    rank, world = evaluator.context.rank, evaluator.context.world_size
    torch.manual_seed(seed*1000003+rank)
    np.random.seed(seed*1000003+rank)
    evaluator.reset()
    parts, ids = [], []
    digest = hashlib.sha256()
    for start, valid, labels in paper_batches(total, batch, world, rank, num_classes=num_classes, device=device):
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        images = sample(labels, seed, start)[:valid].float().clamp(0, 1)
        if images.shape != (valid, 3, 256, 256) or not torch.isfinite(images).all():
            raise ValueError('Generator must return finite RGB 256x256 images')
        pixels = images.mul(255).round().to(torch.uint8)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            evaluator.update(images)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device.type, enabled=False):
            features = official(pixels)
            if not torch.isfinite(features).all():
                raise FloatingPointError('Nonfinite official FID features')
            parts.append(features.cpu())
        array = pixels.cpu().numpy()
        digest.update(array.tobytes())
        ids.extend(range(start, start+valid))
        if len(ids) <= batch:
            np.save(output/f'first_bytes_rank{rank}.npy', array)
    dimension = 2048 if not parts else parts[0].shape[1]
    features = torch.cat(parts).numpy() if parts else np.empty((0, dimension), dtype=np.float32)
    np.save(output/f'features_rank{rank}.npy', features)
    np.save(output/f'ids_rank{rank}.npy', np.asarray(ids, dtype=np.int64))
    write_json(output/f'rank{rank}.json', dict(num_images=len(ids), image_bytes_sha256=digest.hexdigest()))
    fd, count = evaluator.compute()
    if count != total:
        raise RuntimeError('FDr5 sample count differs from the requested image count')
    return fd


def finalize_paper_seed(output, *, total, world, fd, reference):
    import numpy as np
    from metrics.fid import feature_statistics, fid_distance
    ids = np.concatenate([np.load(output/f'ids_rank{rank}.npy') for rank in range(world)])
    if not np.array_equal(np.sort(ids), np.arange(total)):
        raise ValueError('Evaluation image indices are missing or duplicated')
    features = np.concatenate([np.load(output/f'features_rank{rank}.npy') for rank in range(world)])[np.argsort(ids)]
    statistics = feature_statistics(torch.from_numpy(features))
    np.savez(output/'official_generated_stats.npz', **statistics)
    ratios = {name: value/BASELINES[name] for name, value in fd.items()}
    if set(ratios) != {row[0] for row in PAPER_SPECS}:
        raise ValueError('FDr5 requires its five historical representations')
    return dict(fid=fid_distance(statistics, reference), fdr5=sum(ratios.values())/5,
                fd_legacy=fd, fdr_components=ratios)


def main(argv=None):
    from utils.paths import resolve, configure_caches
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['priming_b_80', 'priming_b_160', 'priming_l_80', 'priming_l_160'], default='priming_b_160')
    parser.add_argument('--checkpoint', help='Your own terminal post-training checkpoint, or a pretraining checkpoint with --stage pretraining')
    parser.add_argument('--stage', choices=['pretraining', 'posttraining'], default='posttraining')
    parser.add_argument('--weights', choices=['ema', 'online'], default='ema')
    parser.add_argument('--num-images', type=int, default=50000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--hf-home')
    parser.add_argument('--torch-home')
    parser.add_argument('--stats-dir')
    parser.add_argument('--fid-reference')
    parser.add_argument('--inception-weights')
    args = parser.parse_args(argv)
    if args.num_images < 1000 or args.num_images % 1000 or args.batch < 1:
        parser.error('--num-images must be a positive multiple of 1,000 and --batch must be positive')
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed*1000003+int(os.environ.get('RANK', 0)) >= 2**32 for seed in args.seeds):
        parser.error('Seeds must be unique nonnegative integers within the NumPy rank-seed range')
    if not torch.cuda.is_available():
        parser.error('CUDA is required for the paper evaluation protocol')
    configure_caches(hf_home=args.hf_home, torch_home=args.torch_home)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    initialized = 'RANK' in os.environ
    if initialized:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group('nccl')
    context = DistributedContext('auto')
    rank, world = context.rank, context.world_size
    output_root = resolve(args.output_dir)
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING, format='%(asctime)s %(message)s')
    try:
        import numpy as np
        from models.encoders.fid import FIDInception
        from utils.checkpoint import load_released_model
        from utils.posttrain_assets import paper_evaluation_assets
        from utils.files import prepare_output_directory
        message = [None]
        if rank == 0:
            try:
                assets = paper_evaluation_assets(args)
                manifest_name = 'pretraining_manifest.json' if args.stage == 'pretraining' else 'manifest.json'
                item = json.loads((ROOT/'old/checkpoints'/manifest_name).read_text())[args.model]
                checkpoint = resolve(args.checkpoint or item['checkpoint'])
                identity = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
                if args.checkpoint is None and (checkpoint.stat().st_size, identity['checkpoint_sha256']) != (item['checkpoint_bytes'], item['checkpoint_sha256']):
                    raise ValueError('Checkpoint differs from the pinned release')
                message[0] = dict(assets=assets, identity=identity)
            except Exception as exc:
                message[0] = dict(error=repr(exc))
        if initialized:
            dist.broadcast_object_list(message, src=0)
        if 'error' in message[0]:
            raise RuntimeError(message[0]['error'])
        assets, metadata = message[0]['assets'], message[0]['identity']
        extractors = make_extractors(assets['weights'], 'cuda', PAPER_SPECS)
        references = {name: Path(assets['stats_dir'])/filename for name, _, _, _, filename in PAPER_SPECS}
        evaluator = FrechetEvaluator(extractors, references, device='cuda')
        official = FIDInception(assets['inception_weights']).cuda()
        model, sampling = load_released_model(args.model, checkpoint=metadata['checkpoint'], stage=args.stage,
            weights=args.weights, device='cuda', verify_release=False, terminal=args.stage == 'posttraining')
        source = ROOT/'old/checkpoints'/f'{args.model}.json'
        metadata.update(stage=args.stage, weights=args.weights, cfg=1.,
            ema=('0.9999' if args.stage == 'pretraining' else '0.999') if args.weights == 'ema' else None,
            args_source=str(source), args_sha256=sha256(source), official_reference=assets['fid_reference'],
            official_reference_sha256=sha256(assets['fid_reference']), script_sha256=sha256(__file__),
            protocol='FID: LTH14 FP32 rounded uint8, JiT reference. FDr5: frozen BF16 float images, FP64 moments, excludes ConvNeXt; legacy ADM FID component retained.')
        source_files = ['eval.py', 'models/priming.py', 'models/layers.py', 'utils/checkpoint.py',
            'utils/collectives.py', 'models/encoders/base.py', 'models/encoders/timm_features.py',
            'models/encoders/inception.py', 'models/encoders/fid.py', 'metrics/fid.py',
            'metrics/frechet.py', 'metrics/moments.py', 'metrics/statistics.py']
        metadata.update(source_sha256={name:sha256(ROOT/name) for name in source_files},
            feature_ids={name:module.spec.feature_id for name,module in extractors.items()},
            asset_manifest_sha256=sha256(ROOT/'old/releases/encoders.json'))
        with np.load(assets['fid_reference'], allow_pickle=False) as archive:
            reference = {key: archive[key] for key in ('mu', 'sigma')}
        for seed in args.seeds:
            output = output_root/args.model/f'n{args.num_images}_seed{seed}'
            prepare_output_directory(output, rank, dist if initialized else None)
            fd = evaluate_paper_seed(lambda labels, *_: sample_images(model, labels, sampling), evaluator, official,
                output, total=args.num_images, batch=args.batch, seed=seed)
            if initialized:
                dist.barrier()
            message = [None]
            if rank == 0:
                try:
                    metrics = finalize_paper_seed(output, total=args.num_images, world=world, fd=fd, reference=reference)
                    report = dict(model=args.model, seed=seed, num_images=args.num_images, batch=args.batch,
                        world_size=world, rng_rank_seed=seed*1000003, **metrics, **metadata)
                    write_json(output/'results.json', report)
                    LOG.info('Seed %s: FID=%.6f FDr5=%.6f', seed, metrics['fid'], metrics['fdr5'])
                    message[0] = dict(ok=True)
                except Exception as exc:
                    message[0] = dict(error=repr(exc))
            if initialized:
                dist.broadcast_object_list(message, src=0)
            if 'error' in message[0]:
                raise RuntimeError(message[0]['error'])
        return 0
    finally:
        if initialized:
            dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
