from __future__ import annotations

import unittest
from unittest import mock

import torch
from torch import nn

from mini_t2i.diffusion import euler_sample, training_loss, sample_discrete


class RecordingModel(nn.Module):
    def __init__(self, prediction: torch.Tensor | None = None) -> None:
        super().__init__()
        self.prediction = prediction
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, x, t, context, mask):
        self.calls.append((x.detach().clone(), t.detach().clone()))
        if self.prediction is None:
            return torch.zeros_like(x)
        return self.prediction.to(device=x.device, dtype=x.dtype).expand_as(x)


class DiffusionTimeConventionTest(unittest.TestCase):
    def test_unweighted_loss_and_gradient_are_independent_of_timestep(self) -> None:
        class ConstantModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.value = nn.Parameter(torch.tensor(0.0))

            def forward(self, x, t, context, mask):
                return self.value.expand_as(x)

        images = torch.ones(2, 3, 4, 4)
        for t in (0.5, 0.65, 0.8, 1.0):
            for weighting in ("none", "inverse_t_squared"):
                model = ConstantModel()
                loss, _, _ = training_loss(
                    model, images, torch.zeros(2, 1, 4), torch.ones(2, 1),
                    label_drop_rate=0.0, t_sample_schedule="uniform",
                    t_uniform_min=t, t_uniform_max=t, loss_weighting=weighting,
                )
                expected = 1.0 if weighting == "none" else t ** -2
                torch.testing.assert_close(loss, torch.tensor(expected))
                loss.backward()
                torch.testing.assert_close(model.value.grad, torch.tensor(-2 * expected))

    def test_discrete_mixture_uses_exact_endpoints_with_equal_probability(self) -> None:
        torch.manual_seed(42)
        times = sample_discrete(10000, [0.5, 1.0], torch.device("cpu"))
        self.assertEqual(set(times.tolist()), {0.5, 1.0})
        self.assertLess(abs(float((times == 0.5).float().mean()) - 0.5), 0.02)

    def test_discrete_training_interpolates_per_example(self) -> None:
        model = RecordingModel()
        images = torch.ones(2, 3, 4, 4)
        with mock.patch("torch.randint", return_value=torch.tensor([0, 1])), mock.patch(
            "torch.randn_like", side_effect=torch.zeros_like
        ):
            loss, _, _ = training_loss(
                model, images, torch.zeros(2, 1, 4), torch.ones(2, 1),
                label_drop_rate=0.0, t_sample_schedule="discrete",
                t_discrete_values=[0.5, 1.0],
            )
        x, t = model.calls[0]
        torch.testing.assert_close(t, torch.tensor([0.5, 1.0]))
        torch.testing.assert_close(x[0], images[0] * 0.5)
        torch.testing.assert_close(x[1], torch.zeros_like(images[1]))
        self.assertTrue(torch.isfinite(loss))

    def test_training_uses_data_at_zero_and_noise_at_one(self) -> None:
        images = torch.ones(2, 3, 4, 4)
        context = torch.zeros(2, 1, 4)
        mask = torch.ones(2, 1, dtype=torch.long)

        for timestep, expected_input in ((0.0, images), (1.0, torch.zeros_like(images))):
            model = RecordingModel(prediction=images[:1])
            with mock.patch("torch.randn_like", side_effect=torch.zeros_like):
                loss, _, _ = training_loss(
                    model,
                    images,
                    context,
                    mask,
                    label_drop_rate=0.0,
                    noise_scale=1.0,
                    t_sample_schedule="uniform",
                    t_uniform_min=timestep,
                    t_uniform_max=timestep,
                )

            model_input, model_t = model.calls[0]
            torch.testing.assert_close(model_input, expected_input)
            torch.testing.assert_close(model_t, torch.full((2,), timestep))
            torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_euler_sampling_runs_from_noise_time_to_data_time(self) -> None:
        model = RecordingModel()
        context = torch.zeros(2, 1, 4)
        mask = torch.ones(2, 1, dtype=torch.long)

        output = euler_sample(
            model,
            context,
            mask,
            image_size=4,
            steps=4,
            cfg_scale=1.0,
            noise_scale=1.0,
        )

        self.assertEqual(output.shape, (2, 3, 4, 4))
        torch.testing.assert_close(output, torch.zeros_like(output), atol=1e-6, rtol=0.0)
        expected_times = (1.0, 0.75, 0.5, 0.25)
        self.assertEqual(len(model.calls), len(expected_times))
        for (_, timestep), expected in zip(model.calls, expected_times):
            torch.testing.assert_close(timestep, torch.full((2,), expected))

    def test_fused_cfg_uses_one_doubled_model_call_per_step(self) -> None:
        model = RecordingModel()
        context = torch.zeros(2, 1, 4)
        mask = torch.ones(2, 1, dtype=torch.long)
        initial_noise = torch.ones(2, 3, 4, 4)

        output = euler_sample(
            model,
            context,
            mask,
            image_size=4,
            steps=3,
            cfg_scale=5.0,
            initial_noise=initial_noise,
            fuse_cfg=True,
        )

        self.assertEqual(output.shape, initial_noise.shape)
        self.assertEqual(len(model.calls), 3)
        for model_input, timestep in model.calls:
            self.assertEqual(model_input.shape[0], 4)
            self.assertEqual(timestep.shape[0], 4)

    def test_initial_noise_shape_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "initial_noise has shape"):
            euler_sample(
                RecordingModel(),
                torch.zeros(2, 1, 4),
                torch.ones(2, 1, dtype=torch.long),
                image_size=4,
                initial_noise=torch.zeros(1, 3, 4, 4),
            )


if __name__ == "__main__":
    unittest.main()
