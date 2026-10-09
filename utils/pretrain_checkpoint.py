"""Pretraining save/resume: all-rank B state and historical rank-zero L state."""
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .random import snapshot_rng, restore_rng
from .state import optimizer_layout, validate_optimizer_shard, exact_optimizer


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_cpu(item) for item in value)
    return value


def data_cursor(step, config):
    train = config["train"]
    return dict(epoch=step // train["steps_per_epoch"], offset=(step % train["steps_per_epoch"])
                * train["micro_batch_size"] * train["gradient_accumulation_steps"])


def save_pretraining(output, step, model, optimizer, ema, config, manifest,
                     *, family, elapsed=0., loader_generator=None):
    rank, world = dist.get_rank(), dist.get_world_size()
    global_batch = world * config["train"]["micro_batch_size"] * config["train"]["gradient_accumulation_steps"]
    rank_states = None
    if family == "b":
        shard = validate_optimizer_shard(optimizer, step, rank, world)
        local = dict(rank=rank, rng=snapshot_rng(), optimizer=to_cpu(optimizer.state_dict()),
                     loader_rng=loader_generator.get_state(), shard=shard,
                     next_data=data_cursor(step, config),
                     streams=dict(version="pixel_pair_stateless_v1", next_step=step + 1,
                                  seed=config["train"]["seed"], rank=rank))
        rank_states = [None] * world if rank == 0 else None
        dist.gather_object(local, rank_states, dst=0)
    error = [None]
    if rank == 0:
        try:
            directory = Path(output) / "checkpoints"
            directory.mkdir(parents=True, exist_ok=True)
            payload = dict(model=to_cpu(model.state_dict()), model_ema=to_cpu(ema.state_dict()),
                           optimizer=to_cpu(optimizer.state_dict()), step=step, current_step=step + 1,
                           samples_seen=step * global_batch, last_elapsed_time=elapsed,
                           config=config, manifest=manifest)
            if family == "b":
                payload.update(rank_states=rank_states, world_size=world,
                               optimizer_layout=optimizer_layout(model, optimizer),
                               contract=dict(version=2, stream="pixel_pair_stateless_v1",
                                             world_size=world, global_batch=global_batch,
                                             model=config["model"], data=config["data"], loss=config["loss"],
                                             optimizer=config["optimizer"], train=config["train"]))
            path = directory / f"step_{step:07d}.pth"
            temporary = path.with_suffix(".pth.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, path)
            link = directory / "latest.tmp"
            link.unlink(missing_ok=True)
            link.symlink_to(path.name)
            os.replace(link, directory / "latest.pth")
            milestone = config["train"]["steps_per_epoch"] * config["train"]["milestone_interval"]
            keep = config["train"]["keep_n_ckpts"]
            for old in sorted(directory.glob("step_*.pth"))[:-keep]:
                old_step = int(old.stem.split("_")[-1])
                if not milestone or old_step % milestone:
                    old.unlink()
        except Exception as exc:
            error[0] = repr(exc)
    dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])
    dist.barrier()


def resume_pretraining(path, model, optimizer, ema, config, manifest,
                       *, family, loader_generator=None):
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    rank, world = dist.get_rank(), dist.get_world_size()
    step = int(state["step"])
    global_batch = world * config["train"]["micro_batch_size"] * config["train"]["gradient_accumulation_steps"]
    if state.get("current_step") != step + 1 or state.get("samples_seen") != step * global_batch:
        raise ValueError("Checkpoint counters disagree")
    historical_l = family == "l" and "config" not in state and "manifest" not in state
    if not historical_l and (state.get("config") != config or state.get("manifest") != manifest):
        raise ValueError("Checkpoint recipe, source, data, or asset identity changed")
    if family == "b":
        if state.get("world_size") != world or len(state.get("rank_states", [])) != world:
            raise ValueError("Checkpoint does not contain this complete rank topology")
        if state["optimizer_layout"] != optimizer_layout(model, optimizer):
            raise ValueError("Muon ownership/parameter ordering differs")
        local = state["rank_states"][rank]
        if local["rank"] != rank or local["next_data"] != data_cursor(step, config):
            raise ValueError("Rank identity or data cursor differs")
        expected = dict(version="pixel_pair_stateless_v1", next_step=step + 1,
                        seed=config["train"]["seed"], rank=rank)
        if local["streams"] != expected:
            raise ValueError("Corruption/crop stream identity differs")
    model.load_state_dict(state["model"], strict=True)
    ema.load_state_dict(state["model_ema"])
    # L intentionally restores rank zero's optimizer on every rank, as before.
    optimizer.load_state_dict(local["optimizer"] if family == "b" else state["optimizer"])
    if family == "b":
        validate_optimizer_shard(optimizer, step, rank, world)
        if ema.step_count != step:
            raise ValueError("EMA counter differs")
        loader_generator.set_state(local["loader_rng"])
        restore_rng(local["rng"])
        exact_optimizer(model.state_dict(), state["model"])
        exact_optimizer(optimizer.state_dict(), local["optimizer"])
        exact_optimizer(ema.state_dict(), state["model_ema"])
        exact_optimizer(snapshot_rng(), local["rng"])
    return step, float(state["last_elapsed_time"])
