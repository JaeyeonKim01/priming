import argparse
from dataclasses import dataclass
from contextlib import nullcontext
import datetime
import logging
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "train_ost.yaml")

sys.path.insert(0, REPO_ROOT)

import torch
import torch.distributed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.flop_counter import FlopCounterMode
from torchvision import datasets, transforms

from src.utils.builders import create_generation_model
from src.utils.checkpoint_util import ckpt_resume, save_checkpoint
from src.utils.data_util import center_crop_arr
from src.utils.distributed_util import (
    enable_distributed,
    get_world_size,
    is_enabled,
    preempt_requested,
    register_preempt_handler,
)
from src.utils.grad_util import get_grad_norm
from src.utils.logging_util import MetricLogger, SmoothedValue
from src.utils.loss_perceptual import PerceptualLoss
from src.utils.optimizer_util import create_optimizer
from src.utils.schedule_util import adjust_learning_rate
from src.utils.setup_util import setup


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

logger = logging.getLogger("FD_loss")

UTIL_ARG_SECTIONS = (
    "data",
    "train",
    "model",
    "loss",
    "endpoint_ot",
    "optimizer",
)

PRETRAIN_OT_STREAM_STATE_KEY = "pretraining_endpoint_ot_stream_state"


@dataclass(frozen=True)
class EndpointOTContext:
    feature_objective: object
    ot_objective: object
    self_ot_objective: object | None
    objective_name: str
    real_features: dict[str, torch.Tensor]
    stream: object
    noise_generator: torch.Generator


class HFImageDataset(Dataset):
    """ImageNet images stored by iREPA as a Hugging Face Arrow dataset."""

    def __init__(self, dataset_path, transform, num_classes):
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise ImportError(
                "Loading iREPA's packed ImageNet data requires the `datasets` package"
            ) from exc

        self.dataset = load_from_disk(dataset_path)
        self.transform = transform
        self.classes = range(num_classes)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image = sample["image"].convert("RGB")
        return self.transform(image), int(sample["label"])

    def __len__(self):
        return len(self.dataset)


def build_dataset(cfg):
    data = cfg.data
    transform_steps = []
    if bool(data.get("random_horizontal_flip", True)):
        transform_steps.append(transforms.RandomHorizontalFlip())
    transform_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform = transforms.Compose(transform_steps)

    hf_path = os.path.join(data.data_path, "imagenet-latents-images")
    if os.path.isfile(os.path.join(hf_path, "dataset_info.json")):
        logger.info(f"Loading packed ImageNet images from {hf_path}")
        return HFImageDataset(hf_path, transform=transform, num_classes=data.num_classes)

    imagefolder_transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, data.img_size)),
        transform,
    ])
    return datasets.ImageFolder(data.data_path, transform=imagefolder_transform)


def infer_steps_per_epoch(cfg, dataset):
    accum_steps = int(cfg.train.get("gradient_accumulation_steps", 1))
    if accum_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    samples_per_rank = len(dataset) // get_world_size()
    micro_steps = samples_per_rank // cfg.train.micro_batch_size
    natural_steps = micro_steps // accum_steps
    configured_steps = cfg.train.get("steps_per_epoch", None)
    steps = natural_steps if configured_steps is None else int(configured_steps)
    if steps <= 0 or steps > natural_steps:
        raise ValueError(
            f"invalid steps_per_epoch={steps}; available={natural_steps} from "
            f"dataset={len(dataset)}, world_size={get_world_size()}, "
            f"micro_batch_size={cfg.train.micro_batch_size}, "
            f"gradient_accumulation_steps={accum_steps}"
        )
    return steps


def build_dataloader(cfg, args, dataset):
    data = cfg.data
    train_cfg = cfg.train
    sampler = DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=cfg.train.seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.micro_batch_size,
        sampler=sampler,
        num_workers=data.num_workers,
        pin_memory=data.pin_mem,
        drop_last=True,
        persistent_workers=data.num_workers > 0,
    )
    logger.info(
        f"ImageNet dataset: {len(dataset):,} images, {len(dataset.classes):,} classes, "
        f"micro_steps/epoch/device={len(loader):,}"
    )
    return sampler, loader


def build_aux_loss(cfg):
    aux_cfg = cfg.loss.aux
    if not aux_cfg.enabled:
        return None

    aux = PerceptualLoss(
        lpips_weight=aux_cfg.lpips_weight,
        convnext_weight=aux_cfg.convnext_weight,
        norm_eps=aux_cfg.norm_eps,
        norm_p=aux_cfg.norm_p,
        random_crop=aux_cfg.random_crop,
        crop_scale=tuple(aux_cfg.crop_scale),
    ).cuda().eval()

    def aux_loss_fn(pred, target, mask):
        pred = (pred * 0.5 + 0.5).clamp(0, 1)
        target = (target * 0.5 + 0.5).clamp(0, 1)
        return aux(pred, target, mask)

    return aux_loss_fn


def compile_training_modules(cfg, model, aux_loss_fn, endpoint_ot=None):
    if not bool(cfg.train.get("compile", True)):
        logger.info("Model compilation disabled")
        return aux_loss_fn

    logger.info("Compiling full OST backbone")
    model.net.compile(fullgraph=True)
    if aux_loss_fn is not None:
        logger.info("Compiling auxiliary perceptual loss")
        aux_loss_fn = torch.compile(aux_loss_fn, fullgraph=True)
    if endpoint_ot is not None:
        logger.info("Compiling endpoint-OT feature extractors")
        endpoint_ot.feature_objective.compile(fullgraph=True)
    return aux_loss_fn


def wrap_distributed_model(cfg, args, model):
    if get_world_size() == 1:
        return model

    logger.info(
        "Wrapping model with DDP: gradient_as_bucket_view=True, "
        "find_unused_parameters=False"
    )
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        find_unused_parameters=False,
    )


def reduce_scalar_metrics(metrics):
    """Average scalar GPU metrics across ranks using one collective."""
    names = list(metrics)
    values = torch.stack([
        value.detach().to(device="cuda", dtype=torch.float32).reshape(())
        if isinstance(value, torch.Tensor)
        else torch.tensor(value, device="cuda", dtype=torch.float32)
        for value in metrics.values()
    ])
    if is_enabled():
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        values.div_(get_world_size())
    return dict(zip(names, values.tolist()))


def _checkpoint_endpoint_ot_stream(args, endpoint_ot):
    if endpoint_ot is None:
        return None
    local_state = {
        "stream": endpoint_ot.stream.state_dict(),
        "noise_generator_state": endpoint_ot.noise_generator.get_state().cpu(),
    }
    if is_enabled():
        rank_states = [None] * args.world_size if args.rank == 0 else None
        torch.distributed.gather_object(local_state, rank_states, dst=0)
    else:
        rank_states = [local_state]
    if args.rank != 0:
        return None
    return {"version": 2, "rank_states": rank_states}


def _restore_endpoint_ot_stream(args, endpoint_ot, checkpoint_extra):
    if endpoint_ot is None or not checkpoint_extra:
        return False
    payload = checkpoint_extra.get(PRETRAIN_OT_STREAM_STATE_KEY)
    if payload is None:
        return False
    version = payload.get("version")
    if version not in {1, 2}:
        raise ValueError("unsupported pretraining endpoint-OT stream state")
    rank_states = payload.get("rank_states")
    if not isinstance(rank_states, list) or len(rank_states) != args.world_size:
        raise ValueError("endpoint-OT checkpoint world size differs")
    local_state = rank_states[args.rank]
    if version == 1:
        # Backward compatibility for the initial one-step smoke checkpoints,
        # which predate the private endpoint-noise generator.
        endpoint_ot.stream.load_state_dict(local_state)
    else:
        if not isinstance(local_state, dict):
            raise TypeError("endpoint-OT rank state must be a mapping")
        endpoint_ot.stream.load_state_dict(local_state.get("stream"))
        noise_state = local_state.get("noise_generator_state")
        if not isinstance(noise_state, torch.Tensor):
            raise TypeError("endpoint-OT noise generator state must be a tensor")
        endpoint_ot.noise_generator.set_state(noise_state.detach().cpu())
    return True


def save_training_checkpoint(
    args, step, model, optimizer, ema_model, elapsed, endpoint_ot=None
):
    stream_state = _checkpoint_endpoint_ot_stream(args, endpoint_ot)
    extra = None
    if args.rank == 0 and stream_state is not None:
        extra = {PRETRAIN_OT_STREAM_STATE_KEY: stream_state}
    save_checkpoint(
        args,
        step,
        model,
        optimizer,
        ema_model,
        elapsed,
        saver=None,
        extra=extra,
    )
    if is_enabled():
        torch.distributed.barrier()


def sync_config_snapshot(args, cfg):
    args.config = OmegaConf.to_container(cfg, resolve=True)


def accumulation_windows(loader, steps_per_epoch, accum_steps):
    iterator = iter(loader)
    for _ in range(steps_per_epoch):
        yield [next(iterator) for _ in range(accum_steps)]


def move_batch_to_cuda(batch):
    images, labels = batch
    return images.cuda(non_blocking=True), labels.cuda(non_blocking=True)


def apply_gradient_accumulation_config(args):
    args.micro_batch_size = int(args.micro_batch_size)
    args.gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    if args.micro_batch_size < 1:
        raise ValueError("micro_batch_size must be >= 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    args.batch_size = args.micro_batch_size * args.gradient_accumulation_steps
    args.global_micro_bsz = args.micro_batch_size * get_world_size()


def _endpoint_ot_weight(args, step):
    if not bool(getattr(args, "pre_ot_enabled", False)):
        return 0.0
    start = int(args.pre_ot_start_step)
    if int(step) < start:
        return 0.0
    target = float(args.pre_ot_weight)
    warmup = int(args.pre_ot_warmup_steps)
    if warmup <= 0:
        return target
    progress = min(1.0, (int(step) - start + 1) / warmup)
    return target * progress


def _backward_endpoint_ot(args, model, endpoint_ot, *, step):
    weight = _endpoint_ot_weight(args, step)
    if endpoint_ot is None or weight == 0.0:
        return None, {}
    batch = endpoint_ot.stream.next_batch()
    base_model = model.module if hasattr(model, "module") else model
    device = next(base_model.parameters()).device
    noise = torch.randn(
        (
            endpoint_ot.stream.local_generated_count,
            int(base_model.in_channels),
            int(base_model.input_size),
            int(base_model.input_size),
        ),
        device=device,
        dtype=next(base_model.parameters()).dtype,
        generator=endpoint_ot.noise_generator,
    )
    noise.mul_(float(base_model.noise_scale))
    with torch.autocast(
        "cuda",
        dtype=args.amp_dtype,
        enabled=args.enable_amp,
        cache_enabled=False,
    ):
        generated = model(
            noise,
            batch.generated_labels,
            endpoint_generation=True,
        )
        generated_features = endpoint_ot.feature_objective.extract_features(
            generated * 0.5 + 0.5
        )
        target_features = {
            name: rows.index_select(0, batch.target_indices.to(rows.device))
            for name, rows in endpoint_ot.real_features.items()
        }
        cross_result = endpoint_ot.ot_objective.score_features(
            generated_features,
            target_features,
            generated_labels=batch.generated_labels,
            target_labels=batch.target_labels,
        )
        self_result = None
        endpoint_loss = cross_result.loss
        if endpoint_ot.self_ot_objective is not None:
            # The independent, stop-gradient q' population makes
            # grad_q OT(q, p) - grad_q OT(q, q') a two-batch stochastic
            # approximation to the generator-dependent part of Sinkhorn
            # divergence.  The omitted OT(p, p) term is constant in theta.
            negative_noise = torch.randn(
                noise.shape,
                device=noise.device,
                dtype=noise.dtype,
                generator=endpoint_ot.noise_generator,
            )
            negative_noise.mul_(float(base_model.noise_scale))
            with torch.no_grad():
                negative_generated = base_model(
                    negative_noise,
                    batch.generated_labels,
                    endpoint_generation=True,
                )
                negative_features = endpoint_ot.feature_objective.extract_features(
                    negative_generated * 0.5 + 0.5
                )
            self_result = endpoint_ot.self_ot_objective.score_features(
                generated_features,
                negative_features,
                generated_labels=batch.generated_labels,
                target_labels=batch.generated_labels,
            )
            endpoint_loss = cross_result.loss - self_result.loss
        weighted_loss = weight * endpoint_loss
    # The endpoint objective is a mean over this rank's equal-sized class
    # shard. DDP's parameter-gradient AVG therefore produces the mean over all
    # global classes, matching the configured coefficient at any world size.
    weighted_loss.backward()
    metrics = {
        "pre_ot_weight": weight,
        "pre_ot_loss": endpoint_loss.detach(),
        "pre_ot_weighted_loss": weighted_loss.detach(),
        "pre_ot_cross_loss": cross_result.loss.detach(),
        **{
            f"pre_ot_cross_{name}": value
            for name, value in cross_result.metrics.items()
        },
    }
    if self_result is not None:
        metrics.update(
            {
                "pre_ot_self_loss": self_result.loss.detach(),
                **{
                    f"pre_ot_self_{name}": value
                    for name, value in self_result.metrics.items()
                },
            }
        )
    return weighted_loss.detach(), metrics


def _snapshot_parameter_gradients(parameters):
    """Copy current gradients for an opt-in branch-decomposition probe."""
    return [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]


def _endpoint_gradient_metrics(parameters, reconstruction_gradients, weight):
    """Decompose reconstruction and endpoint gradients after both backwards.

    ``parameter.grad`` contains reconstruction + weighted endpoint gradients;
    subtracting the saved reconstruction gradient gives the exact endpoint
    contribution seen by the optimizer, including DDP averaging.
    """
    device = next(parameter.device for parameter in parameters)
    dtype = torch.float64
    reconstruction_sq = torch.zeros((), device=device, dtype=dtype)
    endpoint_scaled_sq = torch.zeros((), device=device, dtype=dtype)
    combined_sq = torch.zeros((), device=device, dtype=dtype)
    dot = torch.zeros((), device=device, dtype=dtype)
    for parameter, reconstruction in zip(parameters, reconstruction_gradients):
        combined = parameter.grad
        if reconstruction is None:
            if combined is None:
                continue
            reconstruction = torch.zeros_like(combined)
        elif combined is None:
            combined = torch.zeros_like(reconstruction)
        endpoint_scaled = combined.detach() - reconstruction
        reconstruction64 = reconstruction.double()
        endpoint_scaled64 = endpoint_scaled.double()
        combined64 = combined.detach().double()
        reconstruction_sq += reconstruction64.square().sum()
        endpoint_scaled_sq += endpoint_scaled64.square().sum()
        combined_sq += combined64.square().sum()
        dot += (reconstruction64 * endpoint_scaled64).sum()

    reconstruction_norm = reconstruction_sq.sqrt()
    endpoint_scaled_norm = endpoint_scaled_sq.sqrt()
    combined_norm = combined_sq.sqrt()
    eps = torch.finfo(dtype).tiny
    cosine = dot / (reconstruction_norm * endpoint_scaled_norm).clamp_min(eps)
    weighted_ratio = endpoint_scaled_norm / reconstruction_norm.clamp_min(eps)
    cancellation = combined_norm / (
        reconstruction_norm + endpoint_scaled_norm
    ).clamp_min(eps)
    unscaled_norm = endpoint_scaled_norm / float(weight)
    return {
        "pre_ot_reconstruction_grad_norm": reconstruction_norm.float(),
        "pre_ot_unscaled_grad_norm": unscaled_norm.float(),
        "pre_ot_scaled_grad_norm": endpoint_scaled_norm.float(),
        "pre_ot_unscaled_to_reconstruction_grad_ratio": (
            unscaled_norm / reconstruction_norm.clamp_min(eps)
        ).float(),
        "pre_ot_weighted_to_reconstruction_grad_ratio": weighted_ratio.float(),
        "pre_ot_reconstruction_grad_cosine": cosine.float(),
        "pre_ot_combined_grad_norm": combined_norm.float(),
        "pre_ot_gradient_cancellation_ratio": cancellation.float(),
    }


def train_step(args, model, batches, aux_loss_fn, endpoint_ot=None, *, step=0):
    """Run one accumulated forward/backward training step."""
    model.train()
    step_loss = torch.zeros((), device="cuda")
    step_metrics = {}

    for micro_idx, batch in enumerate(batches):
        images, labels = move_batch_to_cuda(batch)
        should_sync = micro_idx + 1 == args.gradient_accumulation_steps
        sync_context = (
            nullcontext()
            if should_sync or not hasattr(model, "no_sync")
            else model.no_sync()
        )
        with sync_context:
            with torch.autocast(
                "cuda",
                dtype=args.amp_dtype,
                enabled=args.enable_amp,
                cache_enabled=False,
            ):
                loss, metrics = model(images, labels, aux_loss_fn=aux_loss_fn)
            (loss / args.gradient_accumulation_steps).backward()

        step_loss += loss.detach()
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().item()
            step_metrics[name] = step_metrics.get(name, 0.0) + float(value)

    scale = args.gradient_accumulation_steps
    reconstruction_loss = step_loss / scale
    endpoint_weight = _endpoint_ot_weight(args, step)
    log_endpoint_gradients = bool(
        getattr(args, "pre_ot_log_grad_norms", False)
    ) and endpoint_ot is not None and endpoint_weight > 0.0
    parameters = list(model.parameters())
    reconstruction_gradients = (
        _snapshot_parameter_gradients(parameters)
        if log_endpoint_gradients
        else None
    )
    endpoint_loss, endpoint_metrics = _backward_endpoint_ot(
        args, model, endpoint_ot, step=step
    )
    if reconstruction_gradients is not None:
        endpoint_metrics.update(
            _endpoint_gradient_metrics(
                parameters,
                reconstruction_gradients,
                endpoint_weight,
            )
        )
    total_loss = reconstruction_loss
    if endpoint_loss is not None:
        total_loss = total_loss + endpoint_loss
    return total_loss, {
        name: value / scale for name, value in step_metrics.items()
    } | endpoint_metrics


def synthetic_batches(args):
    for _ in range(args.gradient_accumulation_steps):
        images = torch.empty(
            args.micro_batch_size,
            *args.input_size,
            device="cuda",
            dtype=torch.float32,
        ).uniform_(-1.0, 1.0)
        labels = torch.randint(
            args.num_classes,
            (args.micro_batch_size,),
            device="cuda",
        )
        yield images, labels


def _validate_endpoint_ot(args):
    if not bool(getattr(args, "pre_ot_enabled", False)):
        return
    for name in (
        "pre_ot_num_classes",
        "pre_ot_samples_per_class",
        "pre_ot_num_real_refs",
        "pre_ot_num_iters",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("pre_ot_start_step", "pre_ot_warmup_steps"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name in ("pre_ot_weight", "pre_ot_epsilon", "pre_ot_tolerance"):
        value = float(getattr(args, name))
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"{name} must be finite")
    if float(args.pre_ot_weight) <= 0:
        raise ValueError("pre_ot_weight must be positive when endpoint OT is enabled")
    if float(args.pre_ot_epsilon) <= 0 or float(args.pre_ot_tolerance) < 0:
        raise ValueError("endpoint OT epsilon/tolerance are invalid")
    if int(args.pre_ot_num_classes) % int(args.world_size):
        raise ValueError("pre_ot_num_classes must be divisible by world size")
    if int(args.pre_ot_num_classes) > int(args.num_classes):
        raise ValueError("pre_ot_num_classes exceeds the dataset class count")
    if args.pre_ot_objective not in {"cross", "sinkhorn_divergence"}:
        raise ValueError(
            "pre_ot_objective must be cross or sinkhorn_divergence"
        )


def build_endpoint_ot(args):
    """Build the optional pure-noise distribution branch from existing banks."""
    if not bool(getattr(args, "pre_ot_enabled", False)):
        return None
    _validate_endpoint_ot(args)
    # Import lazily so the original, disabled pretraining path retains its
    # historical dependencies and startup behavior.
    from scripts import main_mmd_and_eval as mmd_main
    from distmatch import BalancedSinkhornOT, MMD
    from src.utils.ot_util import ClasswiseOTStream

    bank_args = argparse.Namespace(
        mmd_feature_bank_dir=args.pre_ot_feature_bank_dir,
        mmd_feature_bank_device=args.pre_ot_feature_bank_device,
        mmd_sigmas=args.pre_ot_sigmas,
        mmd_repr_weights=args.pre_ot_repr_weights,
        num_classes=args.num_classes,
        rank=args.rank,
    )
    features = mmd_main._features(mmd_main.TRAINING_SPECS)
    sigmas = mmd_main._mmd_sigmas(bank_args, features)
    feature_objective = MMD(
        features=features,
        bandwidths=sigmas,
        distributed=False,
        device="cuda",
    )
    banks = mmd_main._load_feature_banks(bank_args, feature_objective)
    real = mmd_main._build_full_class_attraction(bank_args, banks)
    local_classes = int(args.pre_ot_num_classes) // int(args.world_size)
    # The pretraining model uses DDP, which averages rank-local gradients.
    # A local class mean on each rank therefore becomes the desired global
    # class mean without the manual gradient SUM used by post-training.
    objective = BalancedSinkhornOT(
        bandwidths=sigmas,
        weights=None,
        global_classes_per_step=local_classes,
        samples_per_class=args.pre_ot_samples_per_class,
        real_references_per_class=args.pre_ot_num_real_refs,
        epsilon=args.pre_ot_epsilon,
        num_iterations=args.pre_ot_num_iters,
        tolerance=args.pre_ot_tolerance,
        distributed=False,
    )
    self_objective = None
    if args.pre_ot_objective == "sinkhorn_divergence":
        self_objective = BalancedSinkhornOT(
            bandwidths=sigmas,
            weights=None,
            global_classes_per_step=local_classes,
            samples_per_class=args.pre_ot_samples_per_class,
            real_references_per_class=args.pre_ot_samples_per_class,
            epsilon=args.pre_ot_epsilon,
            num_iterations=args.pre_ot_num_iters,
            tolerance=args.pre_ot_tolerance,
            distributed=False,
        )
    stream = ClasswiseOTStream(
        class_counts=real.class_counts,
        global_classes_per_step=args.pre_ot_num_classes,
        samples_per_class=args.pre_ot_samples_per_class,
        real_references_per_class=args.pre_ot_num_real_refs,
        rank=args.rank,
        world_size=args.world_size,
        seed=args.pre_ot_seed,
        device="cuda",
    )
    noise_generator = torch.Generator(device="cuda")
    noise_generator.manual_seed(
        int(args.pre_ot_seed) + 2_000_033 * int(args.rank) + 1_000_003
    )
    logger.info(
        "Coverage-aware OST endpoint branch: objective=%s "
        "weight=%s start_step=%s warmup_steps=%s global_classes=%s "
        "local_classes=%s generated_per_class=%s real_refs_per_class=%s "
        "epsilon=%s iterations=%s; reconstruction dataloader unchanged",
        args.pre_ot_objective,
        args.pre_ot_weight,
        args.pre_ot_start_step,
        args.pre_ot_warmup_steps,
        args.pre_ot_num_classes,
        local_classes,
        args.pre_ot_samples_per_class,
        args.pre_ot_num_real_refs,
        args.pre_ot_epsilon,
        args.pre_ot_num_iters,
    )
    return EndpointOTContext(
        feature_objective=feature_objective,
        ot_objective=objective,
        self_ot_objective=self_objective,
        objective_name=args.pre_ot_objective,
        real_features=dict(real.features),
        stream=stream,
        noise_generator=noise_generator,
    )


def measure_training_flops(args, model, aux_loss_fn):
    """Measure one accumulated forward/backward step without updating weights."""
    if args.current_step != 1:
        logger.info("Skipping FLOPs measurement when resuming training")
        return

    logger.info(
        "Measuring one training step's forward/backward FLOPs "
        f"({args.gradient_accumulation_steps} microbatches per device)"
    )
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()
    buffer_states = {
        name: buffer.detach().clone() for name, buffer in model.named_buffers()
    }
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)

    try:
        with FlopCounterMode(display=False) as counter:
            train_step(args, model, synthetic_batches(args), aux_loss_fn)

        local_step_flops = counter.get_total_flops()
        grad_norm = get_grad_norm(model.parameters())
        if not torch.isfinite(grad_norm) or not grad_norm.item() > 0.0:
            raise RuntimeError(
                f"eager FLOP measurement produced invalid grad_norm={grad_norm.item()}"
            )
        logger.info(
            f"Eager FLOP measurement grad_norm={grad_norm.item():.6f}"
        )
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        with torch.no_grad():
            current_buffers = dict(model.named_buffers())
            for name, saved_buffer in buffer_states.items():
                current_buffers[name].copy_(saved_buffer)
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)

    global_step_flops = torch.tensor(local_step_flops, dtype=torch.int64, device="cuda")
    if is_enabled():
        torch.distributed.all_reduce(global_step_flops, op=torch.distributed.ReduceOp.SUM)
    global_step_flops = int(global_step_flops.item())

    if args.rank == 0:
        average_device_flops = global_step_flops / get_world_size()
        per_sample_flops = global_step_flops / args.global_bsz
        total_flops = global_step_flops * args.total_steps
        logger.info(
            "Training FLOPs estimate (supported matmul/conv/SDPA operations; "
            "optimizer, EMA, DDP communication, and unsupported operations excluded): "
            f"device_step={average_device_flops / 1e12:.3f} TFLOPs, "
            f"global_step={global_step_flops / 1e12:.3f} TFLOPs, "
            f"sample={per_sample_flops / 1e9:.3f} GFLOPs, "
            f"planned_total={total_flops / 1e18:.3f} EFLOPs, "
            f"planned_total={total_flops / (1e15 * 86400):.3f} PF-days"
        )


def train(cfg, args):
    # Stage 1: initialize distributed state and infer dataset-dependent schedule.
    enable_distributed()
    apply_gradient_accumulation_config(args)
    dataset = build_dataset(cfg)
    args.steps_per_epoch = infer_steps_per_epoch(cfg, dataset)
    cfg.train.steps_per_epoch = args.steps_per_epoch
    sync_config_snapshot(args, cfg)

    # Stage 2: create output/logging directories and install preemption handling.
    wandb_logger = setup(args)
    logger.info(
        f"gradient accumulation: micro_global_bsz={args.global_micro_bsz}, "
        f"accum_steps={args.gradient_accumulation_steps}, global_bsz={args.global_bsz}"
    )
    if args.rank == 0:
        OmegaConf.save(cfg, os.path.join(args.log_dir, "config.yaml"))
    register_preempt_handler()

    # Stage 3: build dataloader, model, optimizer, auxiliary loss, and resume state.
    sampler, loader = build_dataloader(cfg, args, dataset)
    model, ema_model = create_generation_model(args)
    optimizer = create_optimizer(args, model, print_trainable_params=True)
    aux_loss_fn = build_aux_loss(cfg)
    if bool(getattr(args, "pre_ot_enabled", False)) and bool(
        getattr(args, "pre_ot_preserve_build_rng", True)
    ):
        # Frozen encoder constructors may consume the process-global RNG even
        # though all endpoint populations use private generators.  A delayed
        # auxiliary branch must not perturb the historical reconstruction
        # corruption/label-drop sequence merely by being configured.
        endpoint_cpu_rng_state = torch.get_rng_state()
        endpoint_cuda_rng_state = torch.cuda.get_rng_state()
        try:
            endpoint_ot = build_endpoint_ot(args)
        finally:
            torch.set_rng_state(endpoint_cpu_rng_state)
            torch.cuda.set_rng_state(endpoint_cuda_rng_state)
    else:
        endpoint_ot = build_endpoint_ot(args)
    checkpoint_extra = ckpt_resume(
        args,
        model,
        optimizer,
        ema_model,
        extra_keys=[PRETRAIN_OT_STREAM_STATE_KEY] if endpoint_ot is not None else None,
    )
    if _restore_endpoint_ot_stream(args, endpoint_ot, checkpoint_extra):
        logger.info("Restored rank-local pretraining endpoint-OT stream state")
    elif endpoint_ot is not None and args.current_step > 1 and args.resume_from:
        # Historical reconstruction-only checkpoints legitimately have no OT
        # stream state if the endpoint branch starts at or after the first
        # resumed step. Once the branch could have consumed a batch, missing
        # state would make an exact continuation impossible and remains fatal.
        if int(args.current_step) <= int(args.pre_ot_start_step):
            logger.info(
                "Initializing a fresh endpoint-OT stream at historical "
                "reconstruction checkpoint step %s (activation step %s)",
                args.current_step - 1,
                args.pre_ot_start_step,
            )
        else:
            raise RuntimeError("resumed endpoint-OT checkpoint has no stream state")
    measure_flops = bool(cfg.train.get("measure_flops", False))
    if measure_flops and endpoint_ot is not None:
        logger.warning(
            "FLOPs measurement covers original reconstruction only, not endpoint OT"
        )
    if measure_flops:
        measure_training_flops(args, model, aux_loss_fn)
    aux_loss_fn = compile_training_modules(cfg, model, aux_loss_fn, endpoint_ot)
    train_model = wrap_distributed_model(cfg, args, model)

    # Stage 4: initialize metric meters and step timers.
    metric_logger = MetricLogger(
        delimiter="  ",
        output_file=os.path.join(args.log_dir, "training_metrics.json"),
        prefetch=False,
    )
    for name, window, fmt in [
        ("lr", 1, "{value:.6f}"),
        ("samples/s/device", args.print_freq, "{avg:.2f}"),
        ("samples/s", args.print_freq, "{avg:.2f}"),
        ("samples_seen(M)", args.print_freq, "{value:.2f}"),
        ("device_mem(GB)", args.print_freq, "{value:.2f}"),
    ]:
        metric_logger.add_meter(name, SmoothedValue(window, fmt))

    optimizer.zero_grad(set_to_none=True)
    session_start = time.time()
    step_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    logger.info(
        f"training from step {args.current_step:,} -> {args.total_steps:,} "
        f"({(args.current_step - 1) // args.steps_per_epoch} -> {args.epochs} epochs)"
    )
    stop_after_steps = getattr(args, "stop_after_steps", None)
    if stop_after_steps is not None:
        stop_after_steps = int(stop_after_steps)
        if stop_after_steps < 1:
            raise ValueError("stop_after_steps must be positive or null")
        session_final_step = min(
            int(args.total_steps),
            int(args.current_step) + stop_after_steps - 1,
        )
        logger.info(
            "Bounded continuation: at most %s updates, through step %s",
            stop_after_steps,
            session_final_step,
        )
    else:
        session_final_step = int(args.total_steps)

    # Stage 5: enter the epoch/step training loop.
    start_epoch = (args.current_step - 1) // args.steps_per_epoch
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        header = f"Epoch: [{epoch + 1}/{args.epochs}]"
        step_iter = accumulation_windows(loader, args.steps_per_epoch, args.gradient_accumulation_steps)
        start_iteration = 0
        if epoch == start_epoch:
            start_iteration = args.current_step - 1 - epoch * args.steps_per_epoch
            for _ in range(start_iteration):
                next(step_iter)

        for batch_idx, batches in metric_logger.log_every(
            step_iter,
            args.print_freq,
            header=header,
            n_iterations=args.steps_per_epoch,
            start_iteration=start_iteration,
        ):
            step = epoch * args.steps_per_epoch + batch_idx + 1

            # Stage 6: run one optimizer step over the accumulation window.
            lr = adjust_learning_rate(optimizer, step, args)
            step_loss, step_metrics = train_step(
                args,
                train_model,
                batches,
                aux_loss_fn,
                endpoint_ot,
                step=step,
            )

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if args.grad_clip > 0.0 else get_grad_norm(model.parameters())
            )
            if torch.isfinite(grad_norm):
                optimizer.step()
                ema_model.step(model)
            else:
                logger.warning(f"[step {step}] NaN/Inf grad_norm; skipping optimizer and EMA update")
            optimizer.zero_grad(set_to_none=True)

            args.current_step = step + 1
            args.samples_seen += args.global_bsz

            reduced_metrics = reduce_scalar_metrics({
                "loss": step_loss,
                **step_metrics,
            })
            loss_value = reduced_metrics.pop("loss")
            loss_dict = reduced_metrics
            # values.tolist() above synchronizes the reduced CUDA tensor, so the
            # measured step includes compute, optimizer/EMA, and metric reduction
            # without an additional torch.cuda.synchronize().
            step_time = time.perf_counter() - step_start
            step_start = time.perf_counter()
            sps = args.batch_size / step_time if step_time > 0 else 0.0
            mem_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

            metric_logger.update(
                loss=loss_value,
                grad_norm=grad_norm,
                lr=lr,
                **{
                    "samples/s/device": sps,
                    "samples/s": sps * args.world_size,
                    "samples_seen(M)": args.samples_seen / 1e6,
                    "device_mem(GB)": mem_gb,
                },
                **loss_dict,
            )

            if wandb_logger and step % args.print_freq == 0:
                wandb_logger.update(
                    {
                        "train/loss": loss_value,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "train/samples_seen_M": args.samples_seen / 1e6,
                        **{f"train/{k}": v for k, v in loss_dict.items()},
                    },
                    step=step,
                )

            # Stage 7: save immediately if the preemption handler was triggered.
            if preempt_requested():
                logger.info(f"Preemption requested at step {args.current_step}; saving checkpoint")
                elapsed = time.time() - session_start + args.last_elapsed_time
                save_training_checkpoint(
                    args, step, model, optimizer, ema_model, elapsed, endpoint_ot
                )
                return 0

            if step >= session_final_step:
                logger.info(
                    "Reached bounded continuation endpoint at step %s; saving",
                    step,
                )
                # Returning from inside MetricLogger.log_every would close its
                # generator before the post-yield JSON write for this final
                # iteration. Persist the terminal row explicitly so one-step
                # calibrations and bounded pilots remain machine-readable.
                metric_logger.dump_in_output_file(
                    batch_idx,
                    step_time,
                    0.0,
                )
                elapsed = time.time() - session_start + args.last_elapsed_time
                save_training_checkpoint(
                    args, step, model, optimizer, ema_model, elapsed, endpoint_ot
                )
                metric_logger.synchronize_between_processes()
                logger.info(f"averaged stats: {metric_logger}")
                total = time.time() - session_start + args.last_elapsed_time
                logger.info(
                    "Training segment complete. Total time: "
                    f"{datetime.timedelta(seconds=int(total))}"
                )
                if wandb_logger:
                    wandb_logger.finish()
                if is_enabled():
                    torch.distributed.barrier()
                    torch.distributed.destroy_process_group()
                return 0

        # Stage 8: save checkpoints at configured epoch/final boundaries.
        should_save = (
            (epoch + 1) == args.epochs
            or (args.save_freq > 0 and (epoch + 1) % args.save_freq == 0)
            or (args.milestone_interval > 0 and (epoch + 1) % args.milestone_interval == 0)
        )
        if should_save:
            elapsed = time.time() - session_start + args.last_elapsed_time
            save_training_checkpoint(
                args,
                args.current_step - 1,
                model,
                optimizer,
                ema_model,
                elapsed,
                endpoint_ot,
            )

    # Stage 9: synchronize final metrics and shut down cleanly.
    metric_logger.synchronize_between_processes()
    logger.info(f"averaged stats: {metric_logger}")
    total = time.time() - session_start + args.last_elapsed_time
    logger.info(f"Training complete. Total time: {datetime.timedelta(seconds=int(total))}")
    if wandb_logger:
        wandb_logger.finish()
    if is_enabled():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


def get_args_parser():
    parser = argparse.ArgumentParser("OST ImageNet pretraining", add_help=True)
    parser.add_argument(
        "-c", "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config file loaded with OmegaConf (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--print_config",
        action="store_true",
        help="print the resolved default/config/override YAML and exit",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. model.model=OST_T train.micro_batch_size=64 model.rope_2d=true",
    )
    return parser


def _section_dict(cfg, section):
    if section not in cfg or cfg[section] is None:
        return {}
    section_cfg = cfg[section]
    return OmegaConf.to_container(section_cfg, resolve=True)


def _make_utils_args(cfg, config_path, overrides):
    values = {}
    for section in UTIL_ARG_SECTIONS:
        section_values = _section_dict(cfg, section)
        overlap = sorted(set(values).intersection(section_values))
        if overlap:
            names = ", ".join(overlap)
            raise ValueError(f"duplicate config keys across utility sections: {names}")
        values.update(section_values)
    values["config_path"] = config_path
    values["config"] = OmegaConf.to_container(cfg, resolve=True)
    values["overrides"] = list(overrides)
    return argparse.Namespace(**values)


def parse_config(argv=None):
    cli = get_args_parser().parse_args(argv)
    cfg = OmegaConf.load(cli.config)
    if cli.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli.overrides))
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    if cli.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        sys.exit(0)

    return cfg, _make_utils_args(cfg, cli.config, cli.overrides)


if __name__ == "__main__":
    sys.exit(train(*parse_config()))
