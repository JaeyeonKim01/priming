"""Pixel OST-B reconstruction pair with stateless inputs and all-rank resume."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'packages/distmatch/src'))

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.flop_counter import FlopCounterMode
from torchvision import transforms
from torchvision.utils import save_image

from scripts.train_ost import build_aux_loss
from src.models.denoiser_ost import OSTDenoiser, OSTDenoiser_models
from src.utils.checkpoint_util import _to_cpu
from src.utils.data_util import center_crop_arr
from src.utils.distributed_util import enable_distributed, broadcast_module_params, preempt_requested, register_preempt_handler
from src.utils.ema_util import EMAModel
from src.utils.grad_util import get_grad_norm
from src.utils.optimizer_util import create_optimizer

LOG = logging.getLogger('FD_loss')
TOTAL = 200160
EPOCH_STEPS = 1251
STREAM_VERSION = 'pixel_pair_stateless_v1'


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def mixed_seed(seed, *coordinates):
    mask = (1 << 64) - 1
    value = int(seed)
    for coordinate in coordinates:
        value = (value + int(coordinate) + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
    return value & ((1 << 63) - 1)


def flip_image(seed, epoch, index):
    return bool(mixed_seed(seed, 11, epoch, index) & 1)


class Images(Dataset):
    def __init__(self, path, seed=1, hflip=True):
        from datasets import load_from_disk
        self.data = load_from_disk(path)
        self.seed, self.hflip = seed, hflip
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        epoch, index = item
        row = self.data[int(index)]
        im = row['image'].convert('RGB')
        if im.size != (256, 256):
            raise ValueError(f'Expected packed 256x256 RGB, got {im.size}')
        flipped = self.hflip and flip_image(self.seed, epoch, index)
        if flipped:
            im = transforms.functional.hflip(im)
        label = int(row['label'])
        if not 0 <= label < 1000:
            raise ValueError(f'Invalid label: {label}')
        return self.to_tensor(im).mul(2).sub(1), label, torch.tensor([index, int(flipped)])


class ResumeSampler(DistributedSampler):
    offset = 0

    def __iter__(self):
        indices = list(super().__iter__())
        return iter((self.epoch, index) for index in indices[self.offset:])

    def __len__(self):
        return self.num_samples - self.offset


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


def tensor_hash(value):
    return hashlib.sha256(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def optimizer_layout(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [dict(use_muon=bool(group['use_muon']), parameters=[names[id(p)] for p in group['params']])
            for group in optimizer.param_groups]


def validate_optimizer_shard(optimizer, step, rank, world):
    expected = set()
    muon_count = adam_count = 0
    for group in optimizer.param_groups:
        for index, parameter in enumerate(group['params']):
            owned = not group['use_muon'] or index % world == rank
            if step and owned:
                expected.add(parameter)
                state = optimizer.state.get(parameter, {})
                if group['use_muon']:
                    if set(state) != {'momentum_buffer'} or state['momentum_buffer'].shape != parameter.shape:
                        raise ValueError('Missing or invalid rank-owned Muon momentum')
                    muon_count += 1
                else:
                    if int(state.get('step', -1)) != step or not {'exp_avg', 'exp_avg_sq'} <= set(state):
                        raise ValueError('Auxiliary Adam state/counter differs')
                    adam_count += 1
    if set(optimizer.state) != expected:
        raise ValueError('Optimizer contains another rank\'s shards or lacks owned state')
    return dict(muon_tensors=muon_count, adam_tensors=adam_count)


def scientific_contract(cfg, manifest):
    return dict(version=1, stream=STREAM_VERSION, world_size=8, global_batch=1024,
        data=OmegaConf.to_container(cfg.data, resolve=True), model=OmegaConf.to_container(cfg.model, resolve=True),
        loss=OmegaConf.to_container(cfg.loss, resolve=True), optimizer=OmegaConf.to_container(cfg.optimizer, resolve=True),
        train={key: OmegaConf.to_container(cfg.train, resolve=True)[key] for key in
               ['epochs', 'steps_per_epoch', 'micro_batch_size', 'gradient_accumulation_steps', 'seed', 'dtype', 'ema_rates', 'compile']},
        numerical_backend=dict(cudnn_benchmark=False, cudnn_deterministic=True, tf32=True, sdpa="math", deterministic_algorithms=True, perceptual_crop_backward="native_cpu_bicubic"),
        manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())


def save_state(out, step, model, optimizer, ema, contract, manifest, elapsed, loader_generator):
    rank, world = dist.get_rank(), dist.get_world_size()
    shard = validate_optimizer_shard(optimizer, step, rank, world)
    local = dict(rank=rank, rng=rng_state(), optimizer=_to_cpu(optimizer.state_dict()),
        loader_rng=loader_generator.get_state(), shard=shard,
        next_data=dict(epoch=step // EPOCH_STEPS, offset=(step % EPOCH_STEPS) * 128),
        streams=dict(version=STREAM_VERSION, next_step=step + 1, seed=contract['train']['seed'], rank=rank))
    rank_states = [None] * world if rank == 0 else None
    dist.gather_object(local, rank_states, dst=0)
    error = [None]
    if rank == 0:
        try:
            directory = out / 'checkpoints'; directory.mkdir(exist_ok=True)
            path = directory / f'step_{step:07d}.pth'
            payload = dict(model=_to_cpu(model.state_dict()), model_ema=_to_cpu(ema.state_dict()),
                optimizer=rank_states[0]['optimizer'], rank_states=rank_states, world_size=world,
                optimizer_layout=optimizer_layout(model, optimizer), step=step, current_step=step + 1,
                samples_seen=step * 1024, contract=contract, manifest=manifest, last_elapsed_time=elapsed)
            temporary = path.with_suffix('.pth.tmp')
            torch.save(payload, temporary); os.replace(temporary, path)
            link = directory / 'latest.tmp'; link.unlink(missing_ok=True); link.symlink_to(path.name)
            os.replace(link, directory / 'latest.pth')
            for old in sorted(directory.glob('step_*.pth'))[:-3]:
                old_step = int(old.stem.split('_')[-1])
                if old_step % (20 * EPOCH_STEPS) and old_step != 250:
                    old.unlink()
            write_json(out / 'checkpoint_status.json', dict(step=step, checkpoint=str(path), all_rank_shards=world))
            LOG.info('Saved all-rank checkpoint %s', path)
        except Exception as exc:
            error[0] = repr(exc)
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])
    dist.barrier()


def load_state(path, model, optimizer, ema, contract, manifest, loader_generator):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    rank, world = dist.get_rank(), dist.get_world_size()
    if checkpoint['contract'] != contract or checkpoint['manifest'] != manifest:
        raise ValueError('Scientific config or frozen manifest changed')
    if checkpoint.get('world_size') != world or len(checkpoint.get('rank_states', [])) != world:
        raise ValueError('Checkpoint does not contain this complete rank topology')
    if checkpoint['optimizer_layout'] != optimizer_layout(model, optimizer):
        raise ValueError('Muon ownership/parameter ordering differs')
    local = checkpoint['rank_states'][rank]
    if local['rank'] != rank:
        raise ValueError('Rank identity differs')
    step = int(checkpoint['step'])
    if checkpoint['current_step'] != step + 1 or checkpoint['samples_seen'] != step * 1024:
        raise ValueError('Checkpoint counters disagree')
    if local['next_data'] != dict(epoch=step // EPOCH_STEPS, offset=(step % EPOCH_STEPS) * 128):
        raise ValueError('Data cursor differs')
    if local['streams'] != dict(version=STREAM_VERSION, next_step=step + 1, seed=contract['train']['seed'], rank=rank):
        raise ValueError('Corruption/crop stream identity differs')
    model.load_state_dict(checkpoint['model'], strict=True)
    ema.load_state_dict(checkpoint['model_ema']); ema.to('cuda')
    optimizer.load_state_dict(local['optimizer'])
    validate_optimizer_shard(optimizer, step, rank, world)
    if ema.step_count != step:
        raise ValueError('EMA counter differs')
    loader_generator.set_state(local['loader_rng'])
    restore_rng(local['rng'])
    # Exact payload loading is checked before any compilation/bucket warmup.
    assert state_hash(model.state_dict()) == state_hash(checkpoint['model'])
    exact_optimizer(optimizer.state_dict(), local['optimizer'])
    exact_optimizer(ema.state_dict(), checkpoint['model_ema'])
    exact_optimizer(rng_state(), local['rng'])
    return step, float(checkpoint['last_elapsed_time'])


def exact_optimizer(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert torch.equal(actual.detach().cpu(), expected.detach().cpu())
    elif isinstance(actual, np.ndarray):
        assert np.array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            exact_optimizer(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            exact_optimizer(a, b)
    else:
        assert actual == expected


class AuxStream:
    def __init__(self, fn, seed):
        self.fn, self.seed, self.call_seed = fn, seed, seed

    def __call__(self, pred, target, mask):
        # Fork outside compiled auxiliary function; preserve corruption RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.cuda.manual_seed(self.call_seed)
            return self.fn(pred, target, mask)


def train_step(cfg, model, batches, aux, step, audit=None):
    loss_total = torch.zeros((), device='cuda'); values = {}
    rank = dist.get_rank()
    for micro, (images, labels, ids) in enumerate(batches):
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        noise_seed = mixed_seed(cfg.train.seed, 101, rank, step, micro)
        torch.cuda.manual_seed(noise_seed)
        if aux is not None:
            aux.call_seed = mixed_seed(cfg.train.seed, 202, rank, step, micro)
        if audit is not None:
            audit.append(dict(step=step, micro=micro, ids_and_flips=tensor_hash(ids),
                images=tensor_hash(images), labels=tensor_hash(labels), noise_seed=noise_seed))
        sync = nullcontext() if micro + 1 == cfg.train.gradient_accumulation_steps or not hasattr(model, 'no_sync') else model.no_sync()
        with sync, torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            loss, metrics = model(images, labels, aux_loss_fn=aux)
            (loss / cfg.train.gradient_accumulation_steps).backward()
        loss_total += loss.detach() / cfg.train.gradient_accumulation_steps
        for name, value in metrics.items():
            values[name] = values.get(name, torch.zeros((), device='cuda')) + value.detach() / cfg.train.gradient_accumulation_steps
    return loss_total, values


def make_model(cfg):
    kwargs = dict(cfg.model); name = kwargs.pop('model')
    kwargs['in_channels'] = kwargs.pop('token_channels')
    kwargs.update({k: v for k, v in cfg.loss.items() if k != 'aux'})
    kwargs.update(img_size=cfg.data.img_size, num_classes=cfg.data.num_classes)
    kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(OSTDenoiser).parameters}
    model = OSTDenoiser_models[name](**kwargs).cuda()
    assert len(model.net.blocks) == 16 and sum(p.numel() for p in model.parameters()) == 115063936
    return model


@torch.no_grad()
def diagnostics(out, step, model, ema, validation_rows):
    saved = rng_state(); was_training = model.training
    try:
        model.eval()
        images = torch.stack([x for x, _ in validation_rows]).cuda()
        labels = torch.tensor([y for _, y in validation_rows], device='cuda')
        generator = torch.Generator(device='cuda').manual_seed(20260912)
        noise = torch.randn(images.shape, device='cuda', generator=generator)
        panels = [images]; report = dict(step=step, source='ImageNet validation', rows=['real'])
        for label in ['online', ema.default_label]:
            with ema.swap(model, label), torch.autocast('cuda', dtype=torch.bfloat16):
                denoised = model.net(.2 * images + .8 * noise, labels).float()
                generated = model.net(noise, labels).float()
            assert torch.isfinite(denoised).all() and torch.isfinite(generated).all()
            report[label + '_mse'] = (denoised - images).square().mean().item()
            report['rows'].extend([label + ' denoising', label + ' pure noise'])
            panels.extend([denoised, generated])
        directory = out / 'visualization'; directory.mkdir(exist_ok=True)
        save_image((torch.cat(panels) * .5 + .5).clamp(0, 1), directory / f'step_{step:07d}.png', nrow=len(images))
        write_json(directory / f'step_{step:07d}.json', report)
    finally:
        model.train(was_training); restore_rng(saved)


def train(cfg, cli):
    enable_distributed(); rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8 and cfg.train.micro_batch_size == 64 and cfg.train.gradient_accumulation_steps == 2
    assert cfg.train.epochs == 160 and cfg.train.steps_per_epoch == EPOCH_STEPS
    assert cfg.model.label_drop_prob == 0 and not cfg.endpoint_ot.pre_ot_enabled
    assert cfg.optimizer.lr_sched == 'constant' and cfg.optimizer.warmup_epochs == 0 and cfg.optimizer.grad_clip == 0
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Flash attention backward is not deterministic across fresh processes.
    # Math attention keeps the same SDPA operation and differentiable objective.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)
    seed = int(cfg.train.seed) + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    out = Path(cli.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler()]
    if rank == 0:
        handlers.append(logging.FileHandler(out / 'log.txt'))
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
        format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)
    register_preempt_handler()
    manifest = json.loads(Path(cli.manifest).read_text())
    for name, digest in manifest['runtime'].items():
        if file_hash(ROOT / name) != digest:
            raise ValueError(f'Runtime hash differs: {name}')
    arm = 'perceptual' if cfg.loss.aux.enabled else 'no_perceptual'
    if file_hash(cli.config) != manifest['configs'][arm]:
        raise ValueError('Config hash differs')
    data_path = Path(cfg.data.data_path) / 'imagenet-latents-images'
    for name, digest in manifest['data_metadata_sha256'].items():
        if file_hash(data_path / name) != digest:
            raise ValueError(f'Data metadata differs: {name}')
    for name, identity in manifest['arrow_files'].items():
        stat = (data_path / name).stat()
        if dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns) != identity:
            raise ValueError(f'Arrow shard identity differs: {name}')
    dataset = Images(data_path, cfg.train.seed, cfg.data.random_horizontal_flip)
    assert len(dataset) == 1281167
    sampler = ResumeSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=cfg.train.seed, drop_last=True)
    loader_generator = torch.Generator().manual_seed(mixed_seed(cfg.train.seed, 303, rank))
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=cfg.data.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=cfg.data.num_workers > 0, generator=loader_generator)
    model = make_model(cfg); broadcast_module_params(model)
    initial_hash = state_hash(model.state_dict())
    ema = EMAModel(model, values=[0.9999], batch_size=1024)
    optargs = argparse.Namespace(**OmegaConf.to_container(cfg.optimizer), global_bsz=1024,
        batch_size=128, world_size=world, log_dir=str(out))
    optimizer = create_optimizer(optargs, model, print_trainable_params=rank == 0)
    saved_rng = rng_state(); raw_aux = build_aux_loss(cfg); restore_rng(saved_rng)
    aux = AuxStream(raw_aux, cfg.train.seed) if raw_aux is not None else None
    contract = scientific_contract(cfg, manifest)
    aux_module = next((cell.cell_contents for cell in raw_aux.__closure__ if isinstance(cell.cell_contents, torch.nn.Module)), None) if raw_aux else None
    auxiliary_identity = [state_hash(aux_module.state_dict()) if rank == 0 and aux_module is not None else None]
    dist.broadcast_object_list(auxiliary_identity, src=0)
    contract['auxiliary_state_sha256'] = auxiliary_identity[0]
    assert (aux_module is not None) == bool(cfg.loss.aux.enabled)
    if aux_module is not None:
        assert not aux_module.training and all(not p.requires_grad for p in aux_module.parameters())
    frozen_versions = tuple(p._version for p in aux_module.parameters()) if aux_module is not None else ()
    resume = Path(cli.resume) if cli.resume else out / 'checkpoints/latest.pth'
    decision = [dict(resume_exists=resume.exists(), config_exists=(out / 'config.yaml').exists()) if rank == 0 else None]
    dist.broadcast_object_list(decision, src=0)
    resume_exists = decision[0]['resume_exists']
    step = 0; elapsed_before = 0.
    if resume_exists:
        step, elapsed_before = load_state(resume, model, optimizer, ema, contract, manifest, loader_generator)
        write_json(out / f'restore_rank_{rank}.json', dict(passed=True, rank=rank, step=step, all_state_exact=True))
    elif cli.resume:
        raise FileNotFoundError(resume)
    elif decision[0]['config_exists']:
        raise RuntimeError('Existing output without resumable checkpoint; use a fresh output directory')
    if rank == 0:
        OmegaConf.save(cfg, out / 'config.yaml')
        write_json(out / 'initialization.json', dict(seed=cfg.train.seed, initial_model_sha256=initial_hash,
            parameters=115063936, depth=16, auxiliary_enabled=aux is not None, contract=contract))
    if not resume_exists:
        save_state(out, 0, model, optimizer, ema, contract, manifest, 0., loader_generator)
    # Capture actual inputs to corruption only at predeclared audit updates.
    stream_records = []; capture = {'enabled': False}
    original_corrupt = model.corrupt_inputs
    def corrupt(x, t, noise):
        if capture['enabled']:
            stream_records[-1].update(noise=tensor_hash(noise), times=tensor_hash(t))
        return original_corrupt(x, t, noise)
    model.corrupt_inputs = corrupt
    dummy_batches = [(torch.zeros(64, 3, 256, 256, device='cuda'), torch.arange(64, device='cuda'), torch.zeros(64, 2, dtype=torch.long)) for _ in range(2)]
    if step == 0:
        saved_rng = rng_state()
        if rank == 0:
            probe = dict(auxiliary_enabled=aux is not None, gradients={})
            if aux is not None:
                pred = torch.randn(2, 3, 256, 256, device='cuda', requires_grad=True)
                target = torch.randn_like(pred)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    _, lpips_value, convnext_value = aux(pred, target, torch.ones(2, device='cuda', dtype=torch.bool))
                for name, value in [('lpips', lpips_value), ('convnext', convnext_value)]:
                    gradient, = torch.autograd.grad(value.sum(), pred, retain_graph=True)
                    norm = gradient.float().norm().item()
                    assert torch.isfinite(gradient).all() and norm > 0
                    probe['gradients'][name] = dict(value=value.mean().item(), norm=norm)
                del pred, target, lpips_value, convnext_value, gradient
            write_json(out / 'auxiliary_gradient_probe.json', probe)
        restore_rng(saved_rng)
        dist.barrier()
        with FlopCounterMode(display=False) as counter:
            train_step(cfg, model, dummy_batches, aux, 1)
        local_flops = counter.get_total_flops(); optimizer.zero_grad(set_to_none=True); restore_rng(saved_rng)
        global_flops = torch.tensor(local_flops, dtype=torch.int64, device='cuda'); dist.all_reduce(global_flops)
        if rank == 0:
            write_json(out / 'training_flops.json', dict(global_step_flops=global_flops.item(), planned_updates=TOTAL,
                planned_eflops=global_flops.item() * TOTAL / 1e18, auxiliary_enabled=aux is not None,
                scope='Explicit accumulated forward/backward; supported matmul/conv/SDPA; excludes optimizer/EMA/communication/evaluation/unsupported operations'))
    if cfg.train.compile:
        model.net.compile(fullgraph=True)
        if aux is not None:
            aux.fn = torch.compile(raw_aux, fullgraph=True)
    training_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ['LOCAL_RANK'])],
        broadcast_buffers=False, gradient_as_bucket_view=True, find_unused_parameters=False)
    saved_rng = rng_state()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True); train_step(cfg, training_model, dummy_batches, aux, max(step, 1))
    optimizer.zero_grad(set_to_none=True); restore_rng(saved_rng); del dummy_batches
    validation_rows = []
    if rank == 0:
        validation_dataset = Images(data_path / 'val', cfg.train.seed, False)
        assert len(validation_dataset) == 50000
        validation_identity = []
        for index in range(0, 50000, 6250):
            tensor, label, _ = validation_dataset[(0, index)]
            validation_rows.append((tensor, label))
            validation_identity.append(dict(index=index, label=label, image_sha256=tensor_hash(tensor)))
        write_json(out / 'validation_inputs.json', dict(source=str(data_path / 'val'), rows=validation_identity))
        del validation_dataset
        diagnostics(out, step, model, ema, validation_rows)
    dist.barrier()
    start_step = step; end_step = min(TOTAL, cli.until or TOTAL)
    times = []; start = time.perf_counter(); torch.cuda.reset_peak_memory_stats(); stopping = False
    LOG.info('Training starts: completed=%d target=%d arm=%s', step, end_step, 'perceptual' if aux else 'no_perceptual')
    while step < end_step:
        epoch = step // EPOCH_STEPS; sampler.set_epoch(epoch); sampler.offset = (step % EPOCH_STEPS) * 128
        iterator = iter(loader)
        while step < min((epoch + 1) * EPOCH_STEPS, end_step):
            tick = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
            capture['enabled'] = step + 1 in {1, 2, 249, 250, 251, 252} or (step + 1) % EPOCH_STEPS == 0
            batches = [next(iterator) for _ in range(2)]
            loss, metrics = train_step(cfg, training_model, batches, aux, step + 1, stream_records if capture['enabled'] else None)
            capture['enabled'] = False
            norm = get_grad_norm(model.parameters())
            finite = torch.stack([loss, norm, *metrics.values()]).isfinite().all().to(torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(f'Nonfinite loss/gradient at update {step + 1}')
            replay_grads = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()] if cli.preflight and step == 250 else None
            rng_after_backward = rng_state() if replay_grads is not None else None
            optimizer.step(); ema.step(model); step += 1
            if replay_grads is not None:
                expected_model = _to_cpu(model.state_dict()); expected_opt = _to_cpu(optimizer.state_dict()); expected_ema = _to_cpu(ema.state_dict())
                load_state(out / 'checkpoints/step_0000250.pth', model, optimizer, ema, contract, manifest, loader_generator)
                for parameter, gradient in zip(model.parameters(), replay_grads):
                    parameter.grad = gradient
                optimizer.step(); ema.step(model)
                exact_optimizer(model.state_dict(), expected_model); exact_optimizer(optimizer.state_dict(), expected_opt); exact_optimizer(ema.state_dict(), expected_ema)
                restore_rng(rng_after_backward)
                write_json(out / f'identical_gradient_replay_rank_{rank}.json', dict(passed=True, rank=rank, step=step))
                del expected_model, expected_opt, expected_ema, replay_grads
            optimizer.zero_grad(set_to_none=True)
            values = torch.stack([loss, norm, *metrics.values()]); dist.all_reduce(values); values /= world
            torch.cuda.synchronize()
            seconds = torch.tensor(time.perf_counter() - tick, device='cuda'); dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
            seconds = seconds.item()
            if step - start_step > 40 and step <= 248:
                times.append(seconds)
            memory = torch.tensor(torch.cuda.max_memory_reserved()/2**30, device='cuda'); dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            if rank == 0:
                row = dict(step=step, epoch=step/EPOCH_STEPS, loss=values[0].item(), grad_norm=values[1].item(),
                    **{name: value.item() for name, value in zip(metrics, values[2:])}, seconds=seconds, peak_reserved_gib=memory.item())
                with (out/'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(row)+'\n')
                if step <= 2 or step % 50 == 0:
                    LOG.info('step=%d/%d loss=%.6f mse=%.6f sec=%.3f memory=%.2fGiB', step, end_step, row['loss'], row['loss_mse'], seconds, memory.item())
                    write_json(out/'status.json', dict(status='running', step=step, total_steps=TOTAL, slurm_job_id=os.environ.get('SLURM_JOB_ID'), metrics=row))
            stop = torch.tensor(int(preempt_requested()), device='cuda'); dist.all_reduce(stop, op=dist.ReduceOp.MAX); stopping = bool(stop.item())
            save = stopping or step == end_step or step % EPOCH_STEPS == 0 or (cli.preflight and step == 250)
            if save:
                if aux_module is not None and (frozen_versions != tuple(p._version for p in aux_module.parameters()) or any(p.grad is not None for p in aux_module.parameters())):
                    raise RuntimeError('Frozen perceptual encoders changed')
                save_state(out, step, model, optimizer, ema, contract, manifest, elapsed_before+time.perf_counter()-start, loader_generator)
            if save and (step % (20 * EPOCH_STEPS) == 0 or step == end_step):
                if rank == 0:
                    diagnostics(out, step, model, ema, validation_rows)
                dist.barrier()
            if stream_records and (save or step <= 2):
                with (out/f'stream_rank_{rank}.jsonl').open('a') as handle:
                    for row in stream_records:
                        handle.write(json.dumps(row)+'\n')
                stream_records.clear()
            if stopping:
                break
        if stopping:
            break
    if rank == 0:
        report = dict(status='preempted' if stopping else 'completed', step=step, total_steps=TOTAL,
            session_updates=step-start_step, benchmark_updates=len(times), median_seconds=statistics.median(times) if times else None,
            p95_seconds=float(np.quantile(times,.95)) if times else None, peak_reserved_gib=memory.item() if step > start_step else None,
            elapsed_seconds=time.perf_counter()-start, successful_optimizer_updates=step, slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        write_json(out/'report.json',report); write_json(out/'status.json',report)
    dist.barrier(); dist.destroy_process_group()
    return 75 if stopping else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--manifest', required=True); parser.add_argument('--resume')
    parser.add_argument('--until', type=int); parser.add_argument('--preflight', action='store_true')
    cli = parser.parse_args()
    sys.exit(train(OmegaConf.load(cli.config), cli))
