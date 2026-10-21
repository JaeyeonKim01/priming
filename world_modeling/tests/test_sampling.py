import inspect
import copy
import sys
import unittest
from pathlib import Path

import torch


RT1_ROOT = Path(__file__).resolve().parents[1]


from world_modeling.losses.reconstruction import (  # noqa: E402
    interpolate,
    ode_step,
    x_prediction_to_velocity,
    x_prediction_velocity_loss,
)
from world_modeling.models.denoiser import Denoiser  # noqa: E402
from world_modeling.models.rt1.unet_cfg import InnerModelCFG  # noqa: E402


class _ConstantXModel(torch.nn.Module):
    def forward(self, state, history, actions):
        return torch.full_like(state, 0.25)


def _small_test_denoiser():
    denoiser = Denoiser.__new__(Denoiser)
    torch.nn.Module.__init__(denoiser)
    denoiser.model = _ConstantXModel()
    denoiser.ema_model = copy.deepcopy(denoiser.model)
    denoiser.cfg_conditioned = False
    denoiser.num_history_steps = 2
    denoiser.noise_scale = 1.0
    denoiser.t_eps = 0.05
    denoiser.sampling_steps = 1
    denoiser.solver = "euler"
    return denoiser


class FlowMatchingTests(unittest.TestCase):
    def test_path_endpoints_follow_fixed_convention(self):
        clean = torch.randn(3, 2, 4, 4)
        noise = torch.randn_like(clean)
        self.assertTrue(torch.equal(interpolate(clean, noise, torch.zeros(3)), clean))
        self.assertTrue(torch.equal(interpolate(clean, noise, torch.ones(3)), noise))

    def test_velocity_target_is_noise_minus_data(self):
        clean = torch.randn(4, 2, 3, 3)
        noise = torch.randn_like(clean)
        t = torch.tensor([0.1, 0.3, 0.7, 1.0])
        state = interpolate(clean, noise, t)
        velocity = x_prediction_to_velocity(state, clean, t, t_eps=0.05)
        torch.testing.assert_close(velocity, noise - clean)

    def test_perfect_x_prediction_has_zero_velocity_loss(self):
        clean = torch.randn(2, 3, 4, 4)
        noise = torch.randn_like(clean)
        t = torch.tensor([0.25, 0.9])
        state = interpolate(clean, noise, t)
        loss, per_example = x_prediction_velocity_loss(
            clean, state, clean, t, t_eps=0.05
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(per_example, torch.zeros_like(per_example)))

    def test_one_euler_step_from_noise_returns_x_prediction(self):
        state = torch.randn(2, 3, 4, 4)
        x_prediction = torch.randn_like(state)
        result = ode_step(
            state,
            x_prediction,
            torch.ones(2),
            torch.zeros(2),
            t_eps=0.05,
        )
        torch.testing.assert_close(result, x_prediction)

    def test_existing_unet_has_no_timestep_argument(self):
        parameters = inspect.signature(InnerModelCFG.forward).parameters
        self.assertEqual(
            list(parameters), ["self", "noise", "obs", "act", "cfg_scale"]
        )
        self.assertNotIn("t", parameters)
        self.assertNotIn("timestep", parameters)

    def test_model_sampler_supports_one_and_multistep_generation(self):
        denoiser = _small_test_denoiser()
        history = torch.randn(2, 2, 3, 4, 4)
        actions = torch.randn(2, 2, 10)
        noise = torch.randn(2, 3, 4, 4)
        one_step = denoiser.sample(
            history, actions, use_ema=True, noise=noise,
            sampling_steps=1, solver="euler",
        )
        torch.testing.assert_close(one_step, torch.full_like(one_step, 0.25))

        multistep = denoiser.sample(
            history, actions, use_ema=True, noise=noise,
            sampling_steps=4, solver="heun",
        )
        self.assertEqual(multistep.shape, noise.shape)
        self.assertTrue(torch.isfinite(multistep).all())

    def test_autoregressive_sampler_can_return_latents_without_loading_vae(self):
        denoiser = _small_test_denoiser()
        observations = torch.randn(1, 5, 3, 4, 4)
        actions = torch.randn(1, 5, 10)
        result = denoiser.sample_autoregressive(
            observations,
            actions,
            use_ema=True,
            gen_chunk=2,
            return_latents=True,
        )
        self.assertEqual(result.shape, observations.shape)
        torch.testing.assert_close(result[:, :2], observations[:, :2])
        torch.testing.assert_close(
            result[:, 2:], torch.full_like(result[:, 2:], 0.25)
        )

    def test_teacher_forced_sampler_can_return_latents_without_loading_vae(self):
        denoiser = _small_test_denoiser()
        observations = torch.randn(1, 5, 3, 4, 4)
        actions = torch.randn(1, 5, 10)
        result = denoiser.sample_not_autoregressive(
            observations,
            actions,
            use_ema=True,
            return_latents=True,
        )
        self.assertEqual(result.shape, observations.shape)
        torch.testing.assert_close(result[:, :2], observations[:, :2])
        torch.testing.assert_close(
            result[:, 2:], torch.full_like(result[:, 2:], 0.25)
        )


if __name__ == "__main__":
    unittest.main()
