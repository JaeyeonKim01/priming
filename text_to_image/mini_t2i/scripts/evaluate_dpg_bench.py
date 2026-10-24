#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TextIO

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mini_t2i import settings
from mini_t2i.dpg_data import load_dpg_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPG-Bench evaluation with mPLUG VQA.")
    parser.add_argument("--image-root-path", required=True)
    parser.add_argument("--data", default=str(settings.DPG_BENCH_DATA))
    parser.add_argument("--csv", dest="data", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--pic-num", type=int, default=4, choices=[1, 4])
    parser.add_argument("--sample_nums", type=int, default=1065)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N matching images (useful for smoke tests).",
    )
    parser.add_argument("--res-path", default=None)
    parser.add_argument("--vqa-model", default="mplug", choices=["mplug"])
    parser.add_argument("--mplug-ckpt", default="damo/mplug_visual-question-answering_coco_large_en")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Maximum number of questions decoded together on each GPU.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Inference autocast dtype. FP32 preserves parity with the original evaluator.",
    )
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="Keep per-rank result files after rank 0 merges them.",
    )
    parser.add_argument(
        "--include-first-row",
        action="store_true",
        help="Do not skip the first CSV data row. The default mirrors the original DPG-Bench evaluator.",
    )
    return parser.parse_args()


def init_distributed() -> tuple[torch.device, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed DPG scoring requires CUDA")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(hours=6))
        return torch.device("cuda", local_rank), dist.get_rank(), dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        return torch.device("cuda", 0), 0, 1
    return torch.device("cpu"), 0, 1


class MPLUG(torch.nn.Module):
    def __init__(self, ckpt: str, device: torch.device, amp_dtype_name: str) -> None:
        super().__init__()
        from modelscope.models.multi_modal.mplug import CONFIG_NAME, MPlug, MPlugConfig
        from modelscope.utils.constant import Tasks
        from torchvision import transforms

        self.device = device
        self.amp_dtype_name = amp_dtype_name
        if self.device.type != "cuda" and amp_dtype_name != "fp32":
            print("[dpg] CUDA is unavailable; falling back to fp32 inference", flush=True)
            self.amp_dtype_name = "fp32"
        config = MPlugConfig.from_yaml_file(osp.join(ckpt, CONFIG_NAME))
        self.model = MPlug.from_pretrained(ckpt, task=Tasks.visual_question_answering)
        self.model.to(self.device).eval().requires_grad_(False)
        visual = getattr(self.model.visual_encoder, "visual", None)
        visual_transformer = getattr(visual, "transformer", None)
        if visual_transformer is not None and hasattr(visual_transformer, "use_grad_ckp"):
            visual_transformer.use_grad_ckp = False
        self.tokenizer = self.model.tokenizer
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (config.image_res, config.image_res),
                    interpolation=Image.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def autocast_context(self):
        if self.device.type == "cuda" and self.amp_dtype_name == "bf16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def prepare_image(self, image: Image.Image) -> torch.Tensor:
        return self.image_transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def prepare_questions(self, questions: list[str]) -> SimpleNamespace:
        tokens = self.tokenizer(
            [question.lower() for question in questions],
            padding="max_length",
            truncation=True,
            max_length=25,
            return_tensors="pt",
        ).to(self.device)
        return SimpleNamespace(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)

    @torch.inference_mode()
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with self.autocast_context():
            image_tensor = image_tensor.to(dtype=next(self.model.parameters()).dtype)
            image_embeds = self.model.visual_encoder.visual(image_tensor, skip_last_layer=True)
            if self.model.large:
                image_embeds = self.model.dropout(
                    self.model.visn_layer_norm(self.model.visn_fc(image_embeds))
                )
        return image_embeds

    @torch.inference_mode()
    def vqa_batch(
        self,
        image_embeds: torch.Tensor,
        question_tokens: SimpleNamespace,
        start: int,
        end: int,
    ) -> list[str]:
        input_ids = question_tokens.input_ids[start:end]
        attention_mask = question_tokens.attention_mask[start:end]
        batch_size = input_ids.shape[0]
        if batch_size == 0:
            return []
        batch_image_embeds = image_embeds.expand(batch_size, -1, -1)
        image_atts = torch.ones(
            batch_image_embeds.size()[:-1], dtype=torch.long, device=self.device
        )
        with self.autocast_context():
            text_output = self.model.text_encoder(
                input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            fusion_output = self.model.fusion_encoder(
                encoder_embeds=text_output.last_hidden_state,
                attention_mask=attention_mask,
                encoder_hidden_states=batch_image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=False,
            )
            image_output, question_output = fusion_output
            question_states = torch.cat([image_output, question_output], dim=1)
            question_atts = torch.cat([image_atts, attention_mask], dim=1)
            topk_ids, _ = self.model.generation(question_states, question_atts)
        return [
            self.tokenizer.decode(predictions[0], skip_special_tokens=True)
            if predictions
            else ""
            for predictions in topk_ids
        ]


def prepare_dpg_data(data_source: str | Path, *, skip_first_csv_row: bool = True) -> dict[str, dict]:
    question_dict: dict[str, dict] = {}
    previous_id = None
    source_path = Path(str(data_source))
    skip_first = skip_first_csv_row and source_path.exists() and source_path.suffix.lower() == ".csv"
    for row_idx, line in enumerate(load_dpg_rows(data_source)):
        if skip_first and row_idx == 0:
            continue
        current_id = line["item_id"]
        qid = int(line["proposition_id"])
        dependencies = [int(d.strip()) for d in str(line["dependency"]).split(",")]
        if current_id != previous_id:
            question_dict[current_id] = {
                "qid2tuple": {},
                "qid2dependency": {},
                "qid2question": {},
            }
        question_dict[current_id]["qid2tuple"][qid] = line["tuple"]
        question_dict[current_id]["qid2dependency"][qid] = dependencies
        question_dict[current_id]["qid2question"][qid] = line["question_natural_language"]
        previous_id = current_id
    return question_dict


def crop_image(input_image: Image.Image, crop_tuple: tuple[int, int, int, int] | None = None) -> Image.Image:
    return input_image if crop_tuple is None else input_image.crop(crop_tuple)


def compute_dpg_one_sample(
    args: argparse.Namespace,
    question_dict: dict[str, dict],
    image_path: str,
    vqa_model: MPLUG,
    result_file: TextIO,
    detail_file: TextIO,
) -> tuple[float, dict[int, str], dict[int, float]]:
    generated_image = Image.open(image_path).convert("RGB")
    resolution = args.resolution
    crop_tuples = [
        (0, 0, resolution, resolution),
        (resolution, 0, resolution * 2, resolution),
        (0, resolution, resolution, resolution * 2),
        (resolution, resolution, resolution * 2, resolution * 2),
    ][: args.pic_num]
    key = osp.basename(image_path).rsplit(".", 1)[0]
    value = question_dict[key]
    qid2tuple = value["qid2tuple"]
    qid2question = value["qid2question"]
    qid2dependency = value["qid2dependency"]
    question_items = list(qid2question.items())
    questions = [question for _, question in question_items]
    question_tokens = vqa_model.prepare_questions(questions)
    scores = []
    qid2scores_orig: dict[int, float] = {}
    for crop_tuple in crop_tuples:
        cropped_image = crop_image(generated_image, crop_tuple)
        image_tensor = vqa_model.prepare_image(cropped_image)
        image_embeds = vqa_model.encode_image(image_tensor)
        answers: list[str] = []
        for start in range(0, len(questions), args.batch_size):
            end = min(start + args.batch_size, len(questions))
            answers.extend(vqa_model.vqa_batch(image_embeds, question_tokens, start, end))
        if len(answers) != len(question_items):
            raise RuntimeError(
                f"mPLUG returned {len(answers)} answers for {len(question_items)} questions"
            )
        qid2scores: dict[int, float] = {}
        detail_lines = []
        for (qid, question), answer in zip(question_items, answers):
            qid2scores[qid] = float(answer == "yes")
            detail_lines.append(f"{image_path}, {crop_tuple}, {question}, {answer}\n")
        detail_file.writelines(detail_lines)
        qid2scores_orig = qid2scores.copy()
        for qid, parent_ids in qid2dependency.items():
            if any(parent_id != 0 and qid2scores[parent_id] == 0 for parent_id in parent_ids):
                qid2scores[qid] = 0
        scores.append(sum(qid2scores.values()) / len(qid2scores))
    average_score = sum(scores) / len(scores)
    result_file.write(
        image_path + ", " + ", ".join(str(score) for score in scores) + ", " + str(average_score) + "\n"
    )
    return average_score, qid2tuple, qid2scores_orig


def detail_path(result_path: Path) -> Path:
    return result_path.with_name(f"{result_path.stem}_detail{result_path.suffix}")


def part_path(path: Path, rank: int, world_size: int) -> Path:
    return path.with_name(f"{path.stem}.rank{rank:05d}-of-{world_size:05d}{path.suffix}")


def stats_part_path(result_path: Path, rank: int, world_size: int) -> Path:
    return result_path.with_name(
        f"{result_path.stem}.rank{rank:05d}-of-{world_size:05d}.stats.json"
    )


def build_summary(
    image_root: Path,
    result_path: Path,
    local_scores: list[float],
    category2scores: dict[str, list[float]],
) -> tuple[str, float]:
    mean_score = float(np.mean(local_scores) * 100) if local_scores else float("nan")
    output = [f"Image path: {image_root}", f"Save results to: {result_path}"]
    output.append("L1 category scores:")
    l1_scores: dict[str, list[float]] = defaultdict(list)
    for category, values in category2scores.items():
        l1_scores[category.split("-")[0].strip()].extend(values)
    for category in sorted(l1_scores):
        output.append(f"\t{category}: {float(np.mean(l1_scores[category]) * 100)}")
    output.append("L2 category scores:")
    for category in sorted(category2scores):
        output.append(f"\t{category}: {float(np.mean(category2scores[category]) * 100)}")
    output.append(f"DPG-Bench score: {mean_score}")
    return "\n".join(output), mean_score


def merge_rank_outputs(
    image_root: Path,
    result_path: Path,
    world_size: int,
    keep_parts: bool,
) -> tuple[str, float]:
    merged_scores: list[float] = []
    merged_categories: dict[str, list[float]] = defaultdict(list)
    result_lines: list[str] = []
    result_parts = [part_path(result_path, rank, world_size) for rank in range(world_size)]
    detail_parts = [part_path(detail_path(result_path), rank, world_size) for rank in range(world_size)]
    stats_parts = [stats_part_path(result_path, rank, world_size) for rank in range(world_size)]
    for result_part, stats_part in zip(result_parts, stats_parts):
        result_lines.extend(result_part.read_text(encoding="utf-8").splitlines())
        stats = json.loads(stats_part.read_text(encoding="utf-8"))
        merged_scores.extend(float(score) for score in stats["scores"])
        for category, values in stats["category2scores"].items():
            merged_categories[category].extend(float(value) for value in values)
    result_lines.sort(key=lambda line: line.split(", ", 1)[0])
    summary, mean_score = build_summary(image_root, result_path, merged_scores, merged_categories)
    with result_path.open("w", encoding="utf-8") as output_file:
        for line in result_lines:
            output_file.write(line + "\n")
        output_file.write(summary + "\n")
    with detail_path(result_path).open("w", encoding="utf-8") as output_file:
        for path in detail_parts:
            with path.open("r", encoding="utf-8") as input_file:
                shutil.copyfileobj(input_file, output_file)
    simple_path = result_path.with_name(f"{result_path.stem}_simple.txt")
    simple_path.write_text(str(mean_score), encoding="utf-8")
    if not keep_parts:
        for path in result_parts + detail_parts + stats_parts:
            path.unlink()
    return summary, mean_score


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be positive; got {args.batch_size}")
    if args.limit is not None and args.limit < 1:
        raise ValueError(f"--limit must be positive; got {args.limit}")
    image_root = Path(args.image_root_path)
    if args.res_path is None:
        args.res_path = str(image_root / f"dpg_sample{args.sample_nums}_results.txt")
    result_path = Path(args.res_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    device, rank, world_size = init_distributed()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    started_at = time.monotonic()
    try:
        question_dict = prepare_dpg_data(args.data, skip_first_csv_row=not args.include_first_row)
        image_paths = sorted(
            path for path in image_root.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        image_paths = [path for path in image_paths if path.stem in question_dict]
        if args.limit is not None:
            image_paths = image_paths[: args.limit]
        local_image_paths = image_paths[rank::world_size]
        print(
            f"[dpg] rank={rank}/{world_size} device={device} images={len(local_image_paths)}/{len(image_paths)} "
            f"batch_size={args.batch_size} amp_dtype={args.amp_dtype}",
            flush=True,
        )
        vqa_model = MPLUG(ckpt=args.mplug_ckpt, device=device, amp_dtype_name=args.amp_dtype)
        scoring_started_at = time.monotonic()
        print(
            f"[dpg] rank={rank} model_load_elapsed={scoring_started_at - started_at:.1f}s",
            flush=True,
        )

        local_scores: list[float] = []
        category2scores: dict[str, list[float]] = defaultdict(list)
        local_result_path = part_path(result_path, rank, world_size)
        local_detail_path = part_path(detail_path(result_path), rank, world_size)
        with local_result_path.open("w", encoding="utf-8") as result_file, local_detail_path.open(
            "w", encoding="utf-8"
        ) as detail_file:
            for n, path in enumerate(local_image_paths, start=1):
                try:
                    score, qid2tuple, qid2scores = compute_dpg_one_sample(
                        args,
                        question_dict,
                        str(path),
                        vqa_model,
                        result_file,
                        detail_file,
                    )
                except Exception as error:
                    print(f"[dpg] rank={rank} failed {path}: {error}", flush=True)
                    continue
                local_scores.append(score)
                for qid, tuple_text in qid2tuple.items():
                    category = tuple_text.split("(")[0].strip()
                    category2scores[category].append(qid2scores[qid])
                if n % 10 == 0 or n == len(local_image_paths):
                    score_elapsed = time.monotonic() - scoring_started_at
                    print(
                        f"[dpg] rank={rank} evaluated {n}/{len(local_image_paths)} "
                        f"score_elapsed={score_elapsed:.1f}s",
                        flush=True,
                    )
        stats_part_path(result_path, rank, world_size).write_text(
            json.dumps({"scores": local_scores, "category2scores": category2scores}) + "\n",
            encoding="utf-8",
        )

        if dist.is_initialized():
            dist.barrier(device_ids=[device.index])
        if rank == 0:
            summary, _ = merge_rank_outputs(
                image_root, result_path, world_size, args.keep_parts
            )
            print(summary, flush=True)
            finished_at = time.monotonic()
            print(
                f"[dpg] scoring complete score_elapsed={finished_at - scoring_started_at:.1f}s "
                f"total_elapsed={finished_at - started_at:.1f}s",
                flush=True,
            )
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index])
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
