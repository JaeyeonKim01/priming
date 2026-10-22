# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class PrimingSchedulerOutput(BaseOutput):
    r"""
    Output of [`PrimingScheduler.step`].

    Args:
        prev_sample (`torch.Tensor`):
            The model prediction. Priming returns the clean image in one evaluation.
    """

    prev_sample: torch.Tensor


class PrimingScheduler(SchedulerMixin, ConfigMixin):
    r"""
    One-step scheduler for Priming.

    The transformer predicts the clean sample directly. `set_timesteps` accepts a single evaluation,
    and `step` returns that prediction. `scale_model_input` leaves the noise unchanged.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            Stored for config compatibility with other Diffusers schedulers. Priming does not index it.
        prediction_type (`str`, defaults to `"sample"`):
            The model output is the clean sample.
    """

    order = 1
    _compatibles: list[str] = []

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        prediction_type: str = "sample",
    ):
        if num_train_timesteps < 1:
            raise ValueError(f"num_train_timesteps must be >= 1, got {num_train_timesteps}.")
        if prediction_type != "sample":
            raise ValueError(f"PrimingScheduler prediction_type must be 'sample', got {prediction_type!r}.")
        self.timesteps: torch.Tensor | None = None
        self.num_inference_steps: int | None = None
        self._step_index: int | None = None

    @property
    def init_noise_sigma(self) -> float:
        r"""
        Scale of the standard normal used to start sampling.

        Returns:
            `float`: `1.0`. The pipeline multiplies this by the checkpoint `noise_scale`.
        """
        return 1.0

    def scale_model_input(self, sample: torch.Tensor, timestep: float | torch.Tensor) -> torch.Tensor:
        r"""
        Return the sample unchanged.

        Args:
            sample (`torch.Tensor`):
                Current sample `(batch, channels, height, width)`.
            timestep (`float` or `torch.Tensor`):
                Unused. Priming has no timestep input.

        Returns:
            `torch.Tensor`: `sample`.
        """
        del timestep
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device | None = None,
    ) -> None:
        r"""
        Install the single evaluation step.

        Args:
            num_inference_steps (`int`):
                Must be 1. Priming generates the image in one forward pass.
            device (`str` or `torch.device`, *optional*):
                Device of `self.timesteps`.
        """
        if num_inference_steps != 1:
            raise ValueError(
                "Priming generates in one forward pass. "
                f"num_inference_steps must be 1, got {num_inference_steps}."
            )
        self.num_inference_steps = 1
        self.timesteps = torch.tensor([0.0], device=device, dtype=torch.float32)
        self._step_index = 0

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None = None,
        return_dict: bool = True,
    ) -> PrimingSchedulerOutput | tuple[torch.Tensor]:
        r"""
        Replace the noisy sample with the predicted clean image.

        Args:
            model_output (`torch.Tensor`):
                Clean-image prediction `(batch, channels, height, width)`.
            timestep (`float` or `torch.Tensor`):
                The single scheduled timestep. Accepted for the Diffusers step signature.
            sample (`torch.Tensor`):
                Current noisy sample. Same shape as `model_output`.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Accepted so pipelines can pass it through. Priming does not draw noise in `step`.
            return_dict (`bool`, defaults to `True`):
                Return [`PrimingSchedulerOutput`] when `True`, otherwise a one-tuple.

        Returns:
            [`PrimingSchedulerOutput`] or `tuple`:
                `prev_sample` is `model_output`.
        """
        del timestep, sample, generator
        if self.timesteps is None:
            raise ValueError("Call set_timesteps before step.")
        if self._step_index is None or self._step_index > 0:
            raise ValueError("PrimingScheduler has already consumed its single step. Call set_timesteps again.")
        self._step_index = 1
        if not return_dict:
            return (model_output,)
        return PrimingSchedulerOutput(prev_sample=model_output)
