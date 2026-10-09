"""Reconstruction pretraining for Priming-B/L, launched directly with torchrun."""
import argparse
from contextlib import nullcontext
from functools import partial
import json
import logging
import os
from pathlib import Path
import time

from utils.paths import ROOT, configure_environment, resolve as resolve_path

resolve = partial(resolve_path, canonical=True)
LOG = logging.getLogger(__name__)


def load_config(path):
    from utils.config import load_yaml
    config = load_yaml(resolve(path))
    family = {"OST_B": "b", "OST_L": "l"}.get(config["model"]["model"])
    if family is None:
        raise ValueError("Pretraining supports OST_B and OST_L")
    train, optimizer, model = config["train"], config["optimizer"], config["model"]
    if (train["micro_batch_size"], train["gradient_accumulation_steps"], train["dtype"]) != (64, 2, "bf16"):
        raise ValueError("Production pretraining uses microbatch 64, accumulation 2, and BF16")
    if train["steps_per_epoch"] != 1251 or train["epochs"] != 160:
        raise ValueError("Production recipes use 1251 updates/epoch for 160 epochs")
    if optimizer["optimizer"] != "muon" or optimizer["lr_sched"] != "constant" or optimizer["warmup_epochs"] != 0:
        raise ValueError("Production reconstruction uses Muon with constant LR and no warmup")
    if model["label_drop_prob"] != 0 or model["t_dist"] != "uniform" or (model["t_min"], model["t_max"]) != (.8, .8):
        raise ValueError("Production reconstruction uses fixed t=0.8 and no label dropout")
    if family == "l" and not config["loss"]["aux"]["enabled"]:
        raise ValueError("There is no packaged L no-perceptual recipe")
    if train["ema_rates"] != [.9999] or train["ema_type"] != "const":
        raise ValueError("Production pretraining uses constant EMA 0.9999")
    if train["keep_n_ckpts"] < 1 or train["milestone_interval"] < 1:
        raise ValueError("Checkpoint retention settings must be positive")
    if config["data"]["num_workers"] < 0 or train["print_freq"] < 1:
        raise ValueError("Worker count must be nonnegative and print frequency positive")
    if optimizer["grad_clip"] != 0:
        raise ValueError("Production pretraining does not clip gradients")
    config["data"]["data_path"] = str(resolve(config["data"]["data_path"]))
    return config, family


def make_model(config, device="cuda"):
    import inspect
    from models import Priming
    kwargs = dict(config["model"])
    kwargs["backbone"] = kwargs.pop("model")
    kwargs["in_channels"] = kwargs.pop("token_channels")
    kwargs.update({k: v for k, v in config["loss"].items() if k != "aux"})
    kwargs.update(img_size=config["data"]["img_size"], num_classes=config["data"]["num_classes"])
    kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(Priming).parameters}
    return Priming(**kwargs).to(device)


def build_auxiliary(config, family, lpips_weights, device="cuda"):
    if not config["loss"]["aux"]["enabled"]:
        return None
    from losses.perceptual import PerceptualLoss
    kwargs = dict(config["loss"]["aux"])
    kwargs.pop("enabled")
    aux = PerceptualLoss(**kwargs, deterministic=family == "b",
                         lpips_weights=str(lpips_weights)).to(device).eval()

    def auxiliary(prediction, target, mask):
        prediction = (prediction * .5 + .5).clamp(0, 1)
        target = (target * .5 + .5).clamp(0, 1)
        return aux(prediction, target, mask)

    return auxiliary


class AuxiliaryStream:
    def __init__(self, fn, device="cuda"):
        self.fn, self.device, self.call_seed = fn, device, 1

    def __call__(self, prediction, target, mask):
        import torch
        devices = [torch.cuda.current_device()] if self.device == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if self.device == "cuda":
                torch.cuda.manual_seed(self.call_seed)
            else:
                torch.manual_seed(self.call_seed)
            return self.fn(prediction, target, mask)


def train_step(config, model, batches, auxiliary, step, *, family, rank, device="cuda"):
    """Preserve B/L RNG and accumulation arithmetic; CPU is for equivalence tests."""
    import torch
    from utils.random import mixed_seed
    model.train()
    accumulation = config["train"]["gradient_accumulation_steps"]
    total = torch.zeros((), device=device)
    metrics = {}
    for micro, batch in enumerate(batches):
        images, labels = (item.to(device, non_blocking=True) for item in batch[:2])
        if family == "b":
            seed = mixed_seed(config["train"]["seed"], 101, rank, step, micro)
            if device == "cuda":
                torch.cuda.manual_seed(seed)
            else:
                torch.manual_seed(seed)
            if auxiliary is not None:
                auxiliary.call_seed = mixed_seed(config["train"]["seed"], 202, rank, step, micro)
        sync = nullcontext() if micro + 1 == accumulation or not hasattr(model, "no_sync") else model.no_sync()
        with sync:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda", cache_enabled=False):
                loss, values = model(images, labels, aux_loss_fn=auxiliary)
                if family == "b":
                    (loss / accumulation).backward()
            if family == "l":
                (loss / accumulation).backward()
        total += loss.detach() / accumulation if family == "b" else loss.detach()
        for name, value in values.items():
            if family == "b":
                metrics[name] = metrics.get(name, torch.zeros((), device=device)) + value.detach() / accumulation
            else:
                metrics[name] = metrics.get(name, 0.) + float(value.detach().item())
    if family == "l":
        total = total / accumulation
        metrics = {name: value / accumulation for name, value in metrics.items()}
    return total, metrics


def train(config, family, args, output, hf_home, torch_home):
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from data.imagenet import build_dataset, ResumeSampler
    from utils.assets import pretraining_manifest
    from utils.distributed import initialize, broadcast_model, stopping
    from utils.ema import EMA
    from utils.optim import create_optimizer, grad_norm
    from utils.pretrain_checkpoint import save_pretraining, resume_pretraining
    from utils.random import mixed_seed, seed_all, snapshot_rng, restore_rng

    rank, world = initialize()
    try:
        torch.set_num_threads(4 if family == "b" else int(os.environ.get("OMP_NUM_THREADS", "8")))
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = family == "l"
        torch.backends.cudnn.deterministic = family == "b"
        if family == "b":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            torch.use_deterministic_algorithms(True)
        lpips = resolve(args.lpips_weights)
        manifest_message = [None]
        if rank == 0:
            try:
                if output.exists() and not args.resume:
                    raise FileExistsError(f"Output exists: {output}; use --resume or a new output")
                manifest_message[0] = dict(manifest=pretraining_manifest(config, root=ROOT, family=family,
                    hf_home=hf_home, torch_home=torch_home, lpips_weights=lpips))
                output.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                manifest_message[0] = dict(error=repr(exc))
        dist.broadcast_object_list(manifest_message, src=0)
        if "error" in manifest_message[0]:
            raise RuntimeError(manifest_message[0]["error"])
        manifest = manifest_message[0]["manifest"]
        dist.barrier()
        logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
                            format="%(asctime)s %(message)s")
        seed_all(config["train"]["seed"] + rank)
        dataset = build_dataset({**config["data"], "seed": config["train"]["seed"]}, family)
        if len(dataset) != 1281167:
            raise ValueError("Expected 1,281,167 ImageNet training images")
        sampler_class = ResumeSampler if family == "b" else DistributedSampler
        sampler = sampler_class(dataset, num_replicas=world, rank=rank, shuffle=True,
                                seed=config["train"]["seed"], drop_last=True)
        loader_generator = torch.Generator().manual_seed(mixed_seed(config["train"]["seed"], 303, rank)) if family == "b" else None
        loader = DataLoader(dataset, batch_size=config["train"]["micro_batch_size"], sampler=sampler,
                            num_workers=config["data"]["num_workers"], pin_memory=config["data"]["pin_mem"],
                            drop_last=True, persistent_workers=config["data"]["num_workers"] > 0,
                            generator=loader_generator)
        model = make_model(config)
        broadcast_model(model)
        ema = EMA(model, decay=.9999, batch_size=1024)
        optimizer = create_optimizer(model, config["optimizer"])
        saved_rng = snapshot_rng() if family == "b" else None
        raw_auxiliary = build_auxiliary(config, family, lpips)
        if saved_rng is not None:
            restore_rng(saved_rng)
        auxiliary = AuxiliaryStream(raw_auxiliary) if family == "b" and raw_auxiliary is not None else raw_auxiliary
        completed, elapsed = 0, 0.
        if args.resume:
            completed, elapsed = resume_pretraining(resolve(args.resume), model, optimizer, ema, config, manifest,
                family=family, loader_generator=loader_generator)
        if rank == 0:
            (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if family == "b" and not args.resume:
            save_pretraining(output, 0, model, optimizer, ema, config, manifest,
                             family=family, loader_generator=loader_generator)
        if completed == 0 and config["train"]["measure_flops"]:
            from torch.utils.flop_counter import FlopCounterMode
            saved_rng = snapshot_rng()
            dummy = [(torch.zeros(64, 3, 256, 256, device="cuda"),
                      torch.arange(64, device="cuda")) for _ in range(2)]
            with FlopCounterMode(display=False) as counter:
                train_step(config, model, dummy, auxiliary, 1, family=family, rank=rank)
            optimizer.zero_grad(set_to_none=True)
            restore_rng(saved_rng)
            flops = torch.tensor(counter.get_total_flops(), device="cuda", dtype=torch.int64)
            dist.all_reduce(flops)
            if rank == 0:
                (output / "training_flops.json").write_text(json.dumps(dict(
                    global_step_flops=flops.item(), planned_updates=200160,
                    scope="Supported forward/backward operations; excludes optimizer/EMA/communication"), indent=2) + "\n")
            del dummy
        if config["train"]["compile"]:
            model.net.compile(fullgraph=True)
            if raw_auxiliary is not None:
                compiled_auxiliary = torch.compile(raw_auxiliary, fullgraph=True)
                if family == "b":
                    auxiliary.fn = compiled_auxiliary
                else:
                    auxiliary = compiled_auxiliary
        training_model = torch.nn.parallel.DistributedDataParallel(model,
            device_ids=[int(os.environ["LOCAL_RANK"])], broadcast_buffers=False,
            gradient_as_bucket_view=True, find_unused_parameters=False)
        # B historically warms DDP buckets with two backwards and restores RNG.
        if family == "b":
            saved_rng = snapshot_rng()
            dummy = [(torch.zeros(64, 3, 256, 256, device="cuda"),
                      torch.arange(64, device="cuda"), torch.zeros(64, 2, dtype=torch.long)) for _ in range(2)]
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                train_step(config, training_model, dummy, auxiliary, max(completed, 1), family=family, rank=rank)
            optimizer.zero_grad(set_to_none=True)
            restore_rng(saved_rng)
            del dummy
        validation_rows = []
        if family == "b":
            from data.imagenet import BImages
            from utils.visualization import pretraining_panels
            if rank == 0:
                validation = BImages(Path(config["data"]["data_path"]) / "imagenet-latents-images/val", hflip=False)
                if len(validation) != 50000:
                    raise ValueError("Expected 50,000 validation images")
                for i in range(0, 50000, 6250):
                    image, label, _ = validation[(0, i)]
                    validation_rows.append((image, label))
                pretraining_panels(output, completed, model, ema, validation_rows)
            dist.barrier()
        train_cfg = config["train"]
        epoch_steps = train_cfg["steps_per_epoch"]
        target = min(epoch_steps * train_cfg["epochs"], args.smoke_steps or epoch_steps * train_cfg["epochs"])
        accumulation = train_cfg["gradient_accumulation_steps"]
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        while completed < target:
            epoch, offset = divmod(completed, epoch_steps)
            sampler.set_epoch(epoch)
            if family == "b":
                sampler.offset = offset * train_cfg["micro_batch_size"] * accumulation
            iterator = iter(loader)
            if family == "l":
                for _ in range(offset * accumulation):
                    next(iterator)
            while completed < min((epoch + 1) * epoch_steps, target):
                step = completed + 1
                loss, metrics = train_step(config, training_model, [next(iterator) for _ in range(accumulation)],
                    auxiliary, step, family=family, rank=rank)
                norm = grad_norm(model.parameters())
                if family == "b":
                    finite = torch.stack([loss, norm, *metrics.values()]).isfinite().all().to(torch.int32)
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise FloatingPointError(f"Nonfinite loss/gradient at update {step}")
                if torch.isfinite(norm):
                    if config["optimizer"]["grad_clip"] > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["optimizer"]["grad_clip"])
                    optimizer.step()
                    ema.step(model)
                else:
                    LOG.warning("Nonfinite gradient at step %d; skipping optimizer/EMA", step)
                optimizer.zero_grad(set_to_none=True)
                completed = step
                values = torch.tensor([loss.item(), norm.item(), *[float(v) for v in metrics.values()]], device="cuda")
                dist.all_reduce(values)
                values /= world
                if rank == 0:
                    row = dict(step=step, loss=values[0].item(), grad_norm=values[1].item(),
                               **{name: value.item() for name, value in zip(metrics, values[2:])})
                    with (output / "metrics.jsonl").open("a") as handle:
                        handle.write(json.dumps(row) + "\n")
                    if step <= 2 or step % train_cfg["print_freq"] == 0:
                        LOG.info("step=%d/%d loss=%.6f", step, target, row["loss"])
                stop = stopping("cuda")
                if stop or step == target or step % epoch_steps == 0:
                    save_pretraining(output, step, model, optimizer, ema, config, manifest, family=family,
                                     elapsed=elapsed + time.perf_counter() - start, loader_generator=loader_generator)
                if family == "b" and (step == target or step % (20 * epoch_steps) == 0):
                    if rank == 0:
                        pretraining_panels(output, step, model, ema, validation_rows)
                    dist.barrier()
                if stop:
                    return 75
        return 0
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="Fresh training output directory")
    parser.add_argument("--resume", help="Renewed pretraining checkpoint, or a historical L checkpoint")
    parser.add_argument("--smoke-steps", type=int, help="Stop a fresh run after this many updates")
    parser.add_argument("--data-path", help="Root containing imagenet-latents-images")
    parser.add_argument("--hf-home")
    parser.add_argument("--torch-home")
    parser.add_argument("--lpips-weights", default="assets/lpips/vgg.pth")
    parser.add_argument("--check-config", action="store_true", help="Validate the recipe/model schema without data or GPUs")
    args = parser.parse_args()
    if args.smoke_steps is not None and (args.smoke_steps < 1 or args.resume):
        parser.error("--smoke-steps must be positive and requires a fresh run")
    if not args.check_config and not args.output:
        parser.error("--output is required for training")
    config, family = load_config(args.config)
    if args.data_path:
        config["data"]["data_path"] = str(resolve(args.data_path))
    if args.check_config:
        import torch
        with torch.device("meta"):
            model = make_model(config, device="meta")
        print(json.dumps(dict(model=config["model"]["model"], depth=len(model.net.blocks),
                              parameters=sum(p.numel() for p in model.parameters()),
                              updates=config["train"]["epochs"] * config["train"]["steps_per_epoch"],
                              perceptual=config["loss"]["aux"]["enabled"]), indent=2))
        return 0
    output = resolve(args.output)
    hf_home, torch_home = configure_environment(output, phase="pretraining",
        hf_home=args.hf_home, torch_home=args.torch_home,
        threads=4 if family == "b" else 8, deterministic=family == "b")
    return train(config, family, args, output, hf_home, torch_home)


if __name__ == "__main__":
    raise SystemExit(main())
