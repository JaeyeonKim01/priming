#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
from mmcv.transforms import Compose
from mmdet.utils import get_test_pipeline_cfg
from tqdm import tqdm

import evaluate_images as legacy


DEFAULT_MODEL = "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2"
DEFAULT_CONFIG = (
    Path(legacy.mmdet.__file__).resolve().parent / ".mim"
    / "configs"
    / "mask2former"
    / "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py"
)
# Source checkouts keep configs beside the package; wheels use mmdet/.mim.
if not DEFAULT_CONFIG.is_file():
    DEFAULT_CONFIG = (
        Path(legacy.mmdet.__file__).resolve().parents[1] / "configs" / "mask2former"
        / "mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py"
    )
DEFAULT_CLIP_CHECKPOINT = Path(
    os.environ.get("CLIP_CACHE_DIR", Path.home() / ".cache" / "clip")
) / "ViT-L-14.pt"
_create_model_and_transforms = legacy.open_clip.create_model_and_transforms


def patch_mmengine_device_id_for_uuid_visibility():
    """Keep MMEngine logging compatible with Slurm GPU/MIG UUID visibility."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return
    device_tokens = [token.strip() for token in visible_devices.split(",")]
    if all(token.isdigit() for token in device_tokens):
        return

    import mmengine.logging.logger as mmengine_logger

    mmengine_logger._get_device_id = lambda: int(os.environ.get("LOCAL_RANK", "0"))


def parse_launcher_args(argv):
    """Parse adapter-only arguments before using the legacy GenEval parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    launcher_args, legacy_argv = parser.parse_known_args(argv)
    if launcher_args.num_gpus < 1:
        parser.error("--num-gpus must be a positive integer")
    if launcher_args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be a positive integer")
    if "-h" in argv or "--help" in argv:
        print(
            "Evaluation options:\n"
            "  --num-gpus NUM_GPUS              evaluation workers (default: 1)\n"
            "  --eval-batch-size EVAL_BATCH_SIZE  images per detector batch (default: 4)\n"
        )
    return launcher_args, legacy_argv


def launch_workers(num_gpus, eval_batch_size, argv):
    """Relaunch this evaluator under torchrun."""
    visible_gpus = legacy.torch.cuda.device_count()
    if num_gpus > visible_gpus:
        raise SystemExit(
            f"requested {num_gpus} evaluation GPUs, but only {visible_gpus} are visible"
        )

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        os.fspath(Path(__file__).resolve()),
        "--eval-batch-size",
        str(eval_batch_size),
        *argv,
    ]
    print(f"[geneval] launching {num_gpus} evaluation workers", flush=True)
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def initialize_distributed(local_rank_arg):
    """Initialize independent data-parallel workers when launched by torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    local_rank = (
        local_rank_arg
        if local_rank_arg is not None
        else int(os.environ.get("LOCAL_RANK", rank))
    )
    legacy.torch.cuda.set_device(local_rank)
    # clip-benchmark passes this value to torch.autocast, which accepts the
    # device type ("cuda") but not an indexed device string ("cuda:1").
    # set_device above still keeps each worker's models on its assigned GPU.
    legacy.DEVICE = "cuda"
    legacy.torch.distributed.init_process_group(
        backend="gloo",
        timeout=timedelta(hours=12),
    )
    return rank, world_size, local_rank


def prompt_directories(imagedir):
    root = Path(imagedir)
    if not root.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {root}")
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: path.name,
    )


def build_image_tasks(folders):
    tasks = []
    for folderpath in folders:
        with (folderpath / "metadata.jsonl").open() as fp:
            metadata = json.load(fp)
        sample_dir = folderpath / "samples"
        imagepaths = sorted(
            (
                path
                for path in sample_dir.iterdir()
                if path.is_file() and path.suffix == ".png" and path.stem.isdigit()
            ),
            key=lambda path: int(path.stem),
        )
        tasks.extend((imagepath, metadata) for imagepath in imagepaths)
    return tasks


def inference_detector_batch(model, imagepaths, test_pipeline):
    """Run one real MMDetection batch instead of its sequential list helper."""
    prepared = [
        test_pipeline(dict(img_path=os.fspath(imagepath), img_id=0))
        for imagepath in imagepaths
    ]
    batch = {
        "inputs": [item["inputs"] for item in prepared],
        "data_samples": [item["data_samples"] for item in prepared],
    }
    try:
        with legacy.torch.inference_mode():
            results = model.test_step(batch)
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            raise RuntimeError(
                "Mask2Former evaluation ran out of memory; lower --eval-batch-size"
            ) from error
        raise
    if len(results) != len(imagepaths):
        raise RuntimeError(
            f"detector returned {len(results)} results for {len(imagepaths)} images"
        )
    return results


def convert_detection(result, metadata):
    """Convert MMDetection 3 output while copying only required masks to CPU."""
    instances = result.pred_instances
    label_tensor = instances.labels
    labels = label_tensor.detach().cpu().numpy()
    boxes = instances.bboxes.detach().cpu().numpy()
    scores = instances.scores.detach().cpu().numpy()

    color_classes = {
        requirement["class"]
        for requirement in metadata.get("include", [])
        if "color" in requirement
    }
    has_masks = "masks" in instances
    bbox_results = []
    segm_results = [] if color_classes and has_masks else None
    for class_index, classname in enumerate(legacy.classnames):
        selected = labels == class_index
        bbox_results.append(
            np.concatenate((boxes[selected], scores[selected, None]), axis=1).astype(
                np.float32, copy=False
            )
        )
        if segm_results is not None:
            if classname not in color_classes:
                segm_results.append(None)
            elif legacy.torch.is_tensor(instances.masks):
                class_masks = instances.masks[label_tensor == class_index]
                segm_results.append(list(class_masks.detach().cpu().numpy()))
            else:
                segm_results.append(list(np.asarray(instances.masks)[selected]))

    return bbox_results, segm_results


def evaluate_shard(args, rank, world_size, local_rank, eval_batch_size):
    """Evaluate a strided prompt shard so expensive task types stay balanced."""
    folders = prompt_directories(args.imagedir)
    local_folders = folders[rank::world_size]
    tasks = build_image_tasks(local_folders)
    test_pipeline = Compose(get_test_pipeline_cfg(legacy.object_detector.cfg.copy()))
    results = []
    progress = tqdm(
        total=len(tasks),
        desc=f"[geneval] GPU {local_rank} shard",
        unit="image",
        disable=rank != 0,
        dynamic_ncols=True,
    )
    for start in range(0, len(tasks), eval_batch_size):
        batch_tasks = tasks[start : start + eval_batch_size]
        detection_results = inference_detector_batch(
            legacy.object_detector,
            [imagepath for imagepath, _ in batch_tasks],
            test_pipeline,
        )
        converted_detections = [
            convert_detection(detection_result, metadata)
            for (_, metadata), detection_result in zip(
                batch_tasks, detection_results
            )
        ]
        del detection_results
        for (imagepath, metadata), (bbox, segm) in zip(
            batch_tasks, converted_detections
        ):
            results.append(
                legacy.evaluate_image_detections(
                    os.fspath(imagepath), metadata, bbox, segm
                )
            )
        progress.update(len(batch_tasks))
    progress.close()

    print(
        f"[geneval] rank={rank}/{world_size} GPU={local_rank} "
        f"prompts={len(local_folders)} images={len(results)} complete",
        flush=True,
    )
    return results


def write_results(results, outfile, rank, world_size):
    """Write one part per rank, then atomically merge them on rank zero."""
    output_path = Path(outfile)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if world_size == 1:
        temporary_output = output_path.with_name(f".{output_path.name}.tmp")
        legacy.pd.DataFrame(results).to_json(
            temporary_output, orient="records", lines=True
        )
        os.replace(temporary_output, output_path)
        return

    part_path = output_path.with_name(f".{output_path.name}.rank{rank:05d}.tmp")
    legacy.pd.DataFrame(results).to_json(part_path, orient="records", lines=True)
    legacy.torch.distributed.barrier()

    if rank == 0:
        temporary_output = output_path.with_name(f".{output_path.name}.merge.tmp")
        with temporary_output.open("wb") as output_file:
            for part_rank in range(world_size):
                current_part = output_path.with_name(
                    f".{output_path.name}.rank{part_rank:05d}.tmp"
                )
                with current_part.open("rb") as part_file:
                    shutil.copyfileobj(part_file, output_file)
        os.replace(temporary_output, output_path)
        for part_rank in range(world_size):
            output_path.with_name(
                f".{output_path.name}.rank{part_rank:05d}.tmp"
            ).unlink()
        print(f"[geneval] merged results: {output_path}", flush=True)

    legacy.torch.distributed.barrier()


def create_model_and_transforms_cached(model_name, *args, pretrained=None, **kwargs):
    """Reuse the cache while preserving OpenCLIP's OpenAI TorchScript loader.

    Replacing ``pretrained='openai'`` with a path selects torch.load instead;
    its weights_only default in PyTorch 2.6 rejects the OpenAI JIT archive.
    """
    kwargs.setdefault("cache_dir", os.fspath(DEFAULT_CLIP_CHECKPOINT.parent))
    return _create_model_and_transforms(model_name, *args, pretrained=pretrained, **kwargs)


def main() -> None:
    launcher_args, legacy_argv = parse_launcher_args(sys.argv[1:])
    patch_mmengine_device_id_for_uuid_visibility()
    launched_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if launched_world_size == 1 and launcher_args.num_gpus > 1:
        launch_workers(
            launcher_args.num_gpus,
            launcher_args.eval_batch_size,
            legacy_argv,
        )
        return

    sys.argv = [sys.argv[0], *legacy_argv]
    args = legacy.parse_args()
    config_was_explicit = any(
        arg == "--model-config" or arg.startswith("--model-config=")
        for arg in legacy_argv
    )
    if not config_was_explicit:
        args.model_config = os.fspath(DEFAULT_CONFIG)
    args.options.setdefault("model", DEFAULT_MODEL)

    rank, world_size, local_rank = initialize_distributed(launcher_args.local_rank)
    completed = False
    try:
        if rank == 0:
            print(
                f"[geneval] evaluating with {world_size} GPU(s), "
                f"detector_batch_size={launcher_args.eval_batch_size}",
                flush=True,
            )
        legacy.open_clip.create_model_and_transforms = create_model_and_transforms_cached
        legacy.args = args
        (
            legacy.object_detector,
            (legacy.clip_model, legacy.transform, legacy.tokenizer),
            legacy.classnames,
        ) = legacy.load_models(args)
        legacy.THRESHOLD = float(args.options.get("threshold", 0.3))
        legacy.COUNTING_THRESHOLD = float(args.options.get("counting_threshold", 0.9))
        legacy.MAX_OBJECTS = int(args.options.get("max_objects", 16))
        legacy.NMS_THRESHOLD = float(args.options.get("max_overlap", 1.0))
        legacy.POSITION_THRESHOLD = float(args.options.get("position_threshold", 0.1))

        results = evaluate_shard(
            args,
            rank,
            world_size,
            local_rank,
            launcher_args.eval_batch_size,
        )
        write_results(results, args.outfile, rank, world_size)
        completed = True
    finally:
        # Keep the group alive while Python's distributed exception hook
        # reports a worker failure. Process exit will clean it up on errors.
        if completed and legacy.torch.distributed.is_initialized():
            legacy.torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
