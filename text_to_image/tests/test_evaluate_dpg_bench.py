from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from mini_t2i.scripts.evaluate_dpg_bench import (
    compute_dpg_one_sample,
    detail_path,
    merge_rank_outputs,
    part_path,
    stats_part_path,
)


class FakeMPLUG:
    def __init__(self) -> None:
        self.prepare_question_calls = 0
        self.prepare_image_calls = 0
        self.encode_image_calls = 0
        self.batch_ranges: list[tuple[int, int]] = []

    def prepare_questions(self, questions: list[str]) -> list[str]:
        self.prepare_question_calls += 1
        return questions

    def prepare_image(self, image: Image.Image) -> Image.Image:
        self.prepare_image_calls += 1
        return image

    def encode_image(self, image: Image.Image) -> Image.Image:
        self.encode_image_calls += 1
        return image

    def vqa_batch(
        self,
        image: Image.Image,
        questions: list[str],
        start: int,
        end: int,
    ) -> list[str]:
        self.batch_ranges.append((start, end))
        return ["yes"] * (end - start)


class DpgBatchingTest(unittest.TestCase):
    def test_questions_are_tokenized_once_and_batched_for_each_crop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "item.png"
            Image.new("RGB", (1024, 1024), color="white").save(image_path)
            args = argparse.Namespace(resolution=512, pic_num=4, batch_size=2)
            question_dict = {
                "item": {
                    "qid2tuple": {1: "entity(a)", 2: "entity(b)", 3: "global(c)"},
                    "qid2dependency": {1: [0], 2: [1], 3: [0]},
                    "qid2question": {1: "one?", 2: "two?", 3: "three?"},
                }
            }
            model = FakeMPLUG()
            result_file = io.StringIO()
            detail_file = io.StringIO()

            score, _, qid2scores = compute_dpg_one_sample(
                args,
                question_dict,
                str(image_path),
                model,
                result_file,
                detail_file,
            )

            self.assertEqual(score, 1.0)
            self.assertEqual(qid2scores, {1: 1.0, 2: 1.0, 3: 1.0})
            self.assertEqual(model.prepare_question_calls, 1)
            self.assertEqual(model.prepare_image_calls, 4)
            self.assertEqual(model.encode_image_calls, 4)
            self.assertEqual(model.batch_ranges, [(0, 2), (2, 3)] * 4)
            self.assertEqual(len(detail_file.getvalue().splitlines()), 12)
            self.assertEqual(len(result_file.getvalue().splitlines()), 1)

    def test_rank_outputs_are_merged_and_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "dpg_results.txt"
            entries = [
                (0, "b.png, 0.5", [0.5], {"entity-a": [1.0]}),
                (1, "a.png, 1.0", [1.0], {"entity-a": [0.0]}),
            ]
            for rank, result_line, scores, categories in entries:
                part_path(result_path, rank, 2).write_text(result_line + "\n", encoding="utf-8")
                part_path(detail_path(result_path), rank, 2).write_text(
                    f"rank {rank}\n", encoding="utf-8"
                )
                stats_part_path(result_path, rank, 2).write_text(
                    json.dumps({"scores": scores, "category2scores": categories}),
                    encoding="utf-8",
                )

            _, score = merge_rank_outputs(root, result_path, 2, keep_parts=False)

            self.assertEqual(score, 75.0)
            lines = result_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[:2], ["a.png, 1.0", "b.png, 0.5"])
            self.assertEqual(
                detail_path(result_path).read_text(encoding="utf-8"),
                "rank 0\nrank 1\n",
            )
            self.assertEqual(
                result_path.with_name("dpg_results_simple.txt").read_text(encoding="utf-8"),
                "75.0",
            )
            for rank in range(2):
                self.assertFalse(part_path(result_path, rank, 2).exists())
                self.assertFalse(part_path(detail_path(result_path), rank, 2).exists())
                self.assertFalse(stats_part_path(result_path, rank, 2).exists())


if __name__ == "__main__":
    unittest.main()
