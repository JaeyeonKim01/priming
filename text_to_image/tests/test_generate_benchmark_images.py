from __future__ import annotations

import types
import unittest

import torch
from torch import nn

from mini_t2i.scripts.generate_benchmark_images import (
    LegacyTimestepAdapter,
    PromptItem,
    extract_ost_online_state_dict,
    generate_for_prompt_batch,
)
from mini_t2i.one_step import (
    iterative_x0_sample,
    make_xt,
    refinement_time_schedule,
    validate_refinement_time_schedule,
)


class FakeTokenizer:
    def __call__(self, prompts, **kwargs):
        batch = len(prompts)
        return {
            "input_ids": torch.arange(batch * 3).reshape(batch, 3),
            "attention_mask": torch.ones(batch, 3, dtype=torch.long),
        }


class FakeTextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, input_ids, attention_mask):
        self.batch_sizes.append(input_ids.shape[0])
        embeddings = input_ids[:, :, None].float().expand(-1, -1, 4)
        return types.SimpleNamespace(last_hidden_state=embeddings)


class FakeDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, image, timestep, context, mask):
        self.batch_sizes.append(image.shape[0])
        return torch.zeros_like(image)


class PromptBatchGenerationTest(unittest.TestCase):
    def test_text_is_encoded_once_per_distinct_prompt(self) -> None:
        cfg = types.SimpleNamespace(
            prompt_length=3,
            image_size=4,
            noise_scale=1.0,
            model_variant="diffusion",
            n_T=1,
            cfg_scale=1.0,
        )
        items = [
            PromptItem(0, "a", "first"),
            PromptItem(1, "b", "second"),
        ]
        text_encoder = FakeTextEncoder()
        denoiser = FakeDenoiser()

        generated = generate_for_prompt_batch(
            denoiser,
            text_encoder,
            FakeTokenizer(),
            cfg,
            items,
            torch.device("cpu"),
            torch.float32,
            samples_per_prompt=4,
            sample_batch_size=4,
            seed=0,
            fuse_cfg=False,
        )

        self.assertEqual(text_encoder.batch_sizes, [2])
        self.assertEqual(denoiser.batch_sizes, [8])
        self.assertEqual([len(images) for images in generated], [4, 4])

    def test_iterative_x0_schedule_and_forward_process(self) -> None:
        class IdentityRecorder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = []

            def forward(self, state, time, text, mask):
                self.calls.append((state.clone(), time.clone()))
                return state

        model = IdentityRecorder()
        initial = torch.full((2, 3, 2, 2), 2.0)
        noises = (torch.full_like(initial, 1.0), torch.full_like(initial, 2.0),
                  torch.full_like(initial, 3.0))
        text = torch.zeros(2, 3, 4)
        mask = torch.ones(2, 3)
        output = iterative_x0_sample(
            model, initial, text, mask, steps=4, standard_noises=noises,
            noise_scale=2.0,
        )
        expected = initial
        schedule = refinement_time_schedule(4)
        for time_value, noise in zip(schedule[1:], noises):
            expected = make_xt(
                expected[:, None], noise[:, None],
                torch.full((2, 1), time_value), noise_scale=2.0,
            )[:, 0]
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(
            torch.stack([call[1] for call in model.calls]),
            torch.tensor(schedule)[:, None].expand(-1, 2),
        )
        self.assertEqual(refinement_time_schedule(1), (1.0,))
        self.assertEqual(refinement_time_schedule(2), (1.0, 0.5))
        self.assertEqual(refinement_time_schedule(3), (1.0, 0.75, 0.5))
        with self.assertRaisesRegex(ValueError, "1, 2, 3, or 4"):
            refinement_time_schedule(5)

    def test_custom_two_step_refinement_schedule(self) -> None:
        class IdentityRecorder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.times = []

            def forward(self, state, time, text, mask):
                self.times.append(time.clone())
                return state

        model = IdentityRecorder()
        initial = torch.full((2, 3, 2, 2), 2.0)
        noise = torch.full_like(initial, 1.0)
        output = iterative_x0_sample(
            model,
            initial,
            torch.zeros(2, 3, 4),
            torch.ones(2, 3),
            steps=2,
            time_schedule=(1.0, 0.25),
            standard_noises=(noise,),
            noise_scale=2.0,
        )
        expected = make_xt(
            initial[:, None], noise[:, None], torch.full((2, 1), 0.25), noise_scale=2.0,
        )[:, 0]
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(
            torch.stack(model.times), torch.tensor([[1.0, 1.0], [0.25, 0.25]]),
        )
        self.assertEqual(validate_refinement_time_schedule([1, 0.25], steps=2), (1.0, 0.25))
        with self.assertRaisesRegex(ValueError, "start at t=1.0"):
            validate_refinement_time_schedule([0.75, 0.25], steps=2)
        with self.assertRaisesRegex(ValueError, "strictly decreasing"):
            validate_refinement_time_schedule([1.0, 1.0], steps=2)

    def test_prompt_batch_uses_distmatch_refinement(self) -> None:
        cfg = types.SimpleNamespace(
            prompt_length=3, image_size=4, noise_scale=2.0,
            model_variant="diffusion", n_T=2, cfg_scale=1.0,
            mean_flow_t_min=0.2, mean_flow_t_max=0.8,
        )
        denoiser = FakeDenoiser()
        generated = generate_for_prompt_batch(
            denoiser, FakeTextEncoder(), FakeTokenizer(), cfg,
            [PromptItem(0, "a", "first"), PromptItem(1, "b", "second")],
            torch.device("cpu"), torch.float32, samples_per_prompt=4,
            sample_batch_size=4, seed=0, fuse_cfg=False,
            distmatch_refinement_steps=2,
            distmatch_refinement_times=(1.0, 0.25),
        )
        self.assertEqual(denoiser.batch_sizes, [8, 8])
        self.assertEqual([len(images) for images in generated], [4, 4])

    def test_ost_online_state_and_legacy_timestep_adaptation(self) -> None:
        state = {"net.weight": torch.ones(1), "net.bias": torch.zeros(1)}
        extracted = extract_ost_online_state_dict({"model": state})
        self.assertEqual(set(extracted), {"weight", "bias"})

        class RecordingTimestep(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.timestep = None

            def forward(self, images, timesteps, text_embeddings, attention_mask):
                self.timestep = timesteps
                return images

        model = RecordingTimestep()
        adapter = LegacyTimestepAdapter(model)
        images = torch.zeros(2, 3, 4, 4)
        adapter(images, torch.tensor([1.0, 0.25]), None, None)
        torch.testing.assert_close(model.timestep, torch.tensor([0.0, 0.75]))


if __name__ == "__main__":
    unittest.main()
