# Copyright 2026 The HuggingFace Team. All rights reserved.
# Modified for the Priming repository; see NOTICE.md for provenance and changes.
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

import inspect
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any, Callable

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import deprecate, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from integrations.huggingface_diffusers import PrimingPipeline
        >>> pipe = PrimingPipeline.from_pretrained("outputs/diffusers/priming_b_160").to("cuda")
        >>> image = pipe(class_labels=207).images[0]
        ```
"""

class PrimingPipeline(DiffusionPipeline):
    r"""
    Class-conditional one-step pipeline for Priming.

    Priming maps unit Gaussian RGB noise to a 256×256 image in one forward pass. There is no VAE.
    Released post-training checkpoints have no CFG tokens, so sampling uses `guidance_scale=1`.

    The pipeline inherits from [`DiffusionPipeline`]. Check the superclass documentation for the
    generic methods (download, save, device placement, CPU offload).

    Args:
        transformer ([`ModelMixin`]):
            [`PrimingTransformer2DModel`] that predicts the clean RGB image.
        scheduler ([`PrimingScheduler`]):
            One-step scheduler. Loaded from `scheduler/`.
        id2label (`dict[int, str]`, *optional*):
            ImageNet class id to English label mapping. Values may contain comma-separated synonyms.
            Loaded from `model_index.json` when omitted.

        Sampling defaults live on `transformer.config`: `default_num_inference_steps`, `guidance_scale`,
        `guidance_interval_min`, `guidance_interval_max`, and `noise_scale`.
    """

    model_cpu_offload_seq = "transformer"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer: ModelMixin,
        scheduler: SchedulerMixin,
        id2label: dict[int, str] | dict[str, str] | None = None,
    ):
        super().__init__()
        if scheduler is None:
            raise ValueError("Pass a PrimingScheduler instance; scheduler cannot be None.")
        self._reject_karras_scheduler(scheduler)
        self.register_modules(transformer=transformer, scheduler=scheduler)
        normalized_labels = self._normalize_id2label(id2label)
        self.register_to_config(id2label={str(key): value for key, value in normalized_labels.items()})
        self._id2label = normalized_labels
        self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = bool(self._id2label)

    def _sampling_default(self, name: str, fallback: float) -> float:
        r"""
        Read a sampling default from the transformer config.

        Args:
            name (`str`):
                Key on `transformer.config`, such as `guidance_scale` or `noise_scale`.
            fallback (`float`):
                Value used when the transformer config does not define `name`.

        Returns:
            `float`: The configured value, or `fallback`.
        """
        value = getattr(self.transformer.config, name, None)
        if value is None:
            return fallback
        return value

    @staticmethod
    def _reject_karras_scheduler(scheduler: SchedulerMixin) -> None:
        r"""
        Reject multi-step Karras schedulers.

        Priming has no timestep channel, so a Karras noise schedule cannot be integrated.

        Args:
            scheduler ([`SchedulerMixin`]):
                Scheduler attached to the pipeline.
        """
        scheduler_name = scheduler.__class__.__name__
        if scheduler_name in KarrasDiffusionSchedulers.__members__:
            raise TypeError(
                "Priming is a one-step generator and does not integrate a Karras noise schedule. "
                f"Received {scheduler_name}. Use PrimingScheduler."
            )

    def _ensure_labels_loaded(self) -> None:
        r"""
        Fill `id2label` from `model_index.json` when the constructor did not receive it.
        """
        if self._labels_loaded_from_model_index:
            return
        loaded = self._read_id2label_from_model_index(getattr(self.config, "_name_or_path", None))
        if loaded:
            self._id2label = loaded
            self.labels = self._build_label2id(self._id2label)
        self._labels_loaded_from_model_index = True

    @staticmethod
    def _normalize_id2label(id2label: dict[int, str] | dict[str, str] | None) -> dict[int, str]:
        r"""
        Cast label-map keys to integers.

        Args:
            id2label (`dict`, *optional*):
                Class id to English label. Keys may be strings, which is how JSON stores them.

        Returns:
            `dict[int, str]`: Integer-keyed map. Empty when `id2label` is missing.
        """
        if not id2label:
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _read_id2label_from_model_index(variant_path: str | None) -> dict[int, str]:
        r"""
        Read `id2label` from a local variant directory.

        Args:
            variant_path (`str`, *optional*):
                Directory that contains `model_index.json`.

        Returns:
            `dict[int, str]`: Loaded map, or an empty dict when the file or key is absent.
        """
        if not variant_path:
            return {}
        model_index_path = Path(variant_path).resolve() / "model_index.json"
        if not model_index_path.exists():
            return {}
        raw = json.loads(model_index_path.read_text(encoding="utf-8"))
        id2label = raw.get("id2label")
        if not isinstance(id2label, dict):
            return {}
        return {int(key): value for key, value in id2label.items()}

    @staticmethod
    def _build_label2id(id2label: dict[int, str]) -> dict[str, int]:
        r"""
        Invert `id2label`, splitting comma-separated synonyms.

        Args:
            id2label (`dict[int, str]`):
                Class id to one or more English names.

        Returns:
            `dict[str, int]`: Synonym to class id, sorted by synonym.
        """
        label2id: dict[str, int] = {}
        for class_id, value in id2label.items():
            for synonym in value.split(","):
                synonym = synonym.strip()
                if synonym:
                    label2id[synonym] = int(class_id)
        return dict(sorted(label2id.items()))

    @property
    def id2label(self) -> dict[int, str]:
        r"""
        ImageNet id to English label map.

        Returns:
            `dict[int, str]`: The map loaded from the constructor or `model_index.json`.
        """
        self._ensure_labels_loaded()
        return self._id2label

    def get_label_ids(self, label: str | list[str]) -> list[int]:
        r"""
        Map ImageNet English labels to class ids.

        Args:
            label (`str` or `list[str]`):
                One label or a list of labels. Each string must match a synonym in `id2label`
                after stripping whitespace. Synonyms are the comma-separated pieces of each value.

        Returns:
            `list[int]`: Class ids in the same order as `label`.
        """
        self._ensure_labels_loaded()
        label2id = self.labels
        if not label2id:
            raise ValueError("No English labels loaded. Ensure `id2label` exists in model_index.json.")
        if isinstance(label, str):
            label = [label]
        missing = [item for item in label if item not in label2id]
        if missing:
            preview = ", ".join(list(label2id.keys())[:8])
            raise ValueError(f"Unknown English label(s): {missing}. Example valid labels: {preview}, ...")
        return [label2id[item] for item in label]

    def _normalize_class_labels(self, class_labels: int | str | list[int | str]) -> list[int]:
        r"""
        Turn a class-label argument into a list of integer ids.

        Args:
            class_labels (`int`, `str`, or `list`):
                One id, one English label, or a list of either. A list must not mix types.

        Returns:
            `list[int]`: One id per requested image.
        """
        if isinstance(class_labels, int):
            return [class_labels]
        if isinstance(class_labels, str):
            return self.get_label_ids(class_labels)
        if len(class_labels) == 0:
            raise ValueError("`class_labels` must contain at least one class id or label.")
        if isinstance(class_labels[0], str):
            if not all(isinstance(item, str) for item in class_labels):
                raise TypeError("`class_labels` must be all strings or all integers.")
            return self.get_label_ids(class_labels)
        if not all(isinstance(item, int) for item in class_labels):
            raise TypeError("`class_labels` must be all strings or all integers.")
        return list(class_labels)

    def prepare_extra_step_kwargs(
        self,
        generator: torch.Generator | list[torch.Generator] | None,
        eta: float,
    ) -> dict[str, Any]:
        r"""
        Extra arguments forwarded to `scheduler.step`, depending on what that method accepts.

        Args:
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Generator passed through when the scheduler step samples noise.
            eta (`float`):
                DDIM eta in `[0, 1]`. Ignored by schedulers whose `step` has no `eta` argument.

        Returns:
            `dict`: Keyword arguments for `scheduler.step`.
        """
        extra_step_kwargs: dict[str, Any] = {}
        step_params = set(inspect.signature(self.scheduler.step).parameters.keys())
        if "eta" in step_params:
            extra_step_kwargs["eta"] = eta
        if "generator" in step_params:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        class_labels: int | str | list[int | str],
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        guidance_interval_min: float,
        guidance_interval_max: float,
        noise_scale: float,
        output_type: str,
        callback_steps: int | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> None:
        r"""
        Reject arguments that the generator cannot run.

        Args:
            class_labels (`int`, `str`, or `list`):
                Class ids or English labels. A list must be non-empty.
            height (`int`):
                Output height in pixels. Must be positive and divisible by the patch size.
            width (`int`):
                Output width in pixels. Must be positive and divisible by the patch size.
            num_inference_steps (`int`):
                Must be 1. Priming evaluates the network once.
            guidance_scale (`float`):
                CFG scale. Must be finite and positive.
            guidance_interval_min (`float`):
                Lower endpoint of the guidance interval, in `[0, 1]`.
            guidance_interval_max (`float`):
                Upper endpoint of the guidance interval. Must be greater than the minimum and at most 1.
            noise_scale (`float`):
                Initial Gaussian scale. Must be positive.
            output_type (`str`):
                One of `"pil"`, `"np"`, `"pt"`, `"latent"`.
            callback_steps (`int`, *optional*):
                Deprecated callback period. When set, must be a positive integer.
            callback_on_step_end_tensor_inputs (`list[str]`, *optional*):
                Tensor names the step callback may read. Each name must be in `_callback_tensor_inputs`.
        """
        if isinstance(class_labels, list) and len(class_labels) == 0:
            raise ValueError("`class_labels` must contain at least one class id or label.")
        patch_size = int(self.transformer.config.patch_size)
        if height <= 0 or width <= 0:
            raise ValueError(f"`height` and `width` must be positive, got {(height, width)}.")
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by patch_size={patch_size}. Got {(height, width)}."
            )
        if (height, width) != (self.transformer.config.sample_size,) * 2:
            raise ValueError("Only the checkpoint native resolution is supported.")
        if num_inference_steps != 1:
            raise ValueError(
                "Priming generates in one forward pass. "
                f"`num_inference_steps` must be 1, got {num_inference_steps}."
            )
        if not torch.isfinite(torch.tensor(float(guidance_scale))) or float(guidance_scale) <= 0:
            raise ValueError(f"`guidance_scale` must be finite and positive, got {guidance_scale}.")
        if not 0.0 <= guidance_interval_min < guidance_interval_max <= 1.0:
            raise ValueError(
                "Guidance interval must satisfy 0 <= min < max <= 1, got "
                f"[{guidance_interval_min}, {guidance_interval_max}]."
            )
        if not torch.isfinite(torch.tensor(float(noise_scale))) or noise_scale <= 0:
            raise ValueError(f"`noise_scale` must be positive, got {noise_scale}.")
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError(f"Unsupported `output_type` {output_type!r}. Choose from 'pil', 'np', 'pt', 'latent'.")
        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps < 1):
            raise ValueError(f"`callback_steps` must be a positive integer, got {callback_steps}.")
        if callback_on_step_end_tensor_inputs is not None:
            unknown = [name for name in callback_on_step_end_tensor_inputs if name not in self._callback_tensor_inputs]
            if unknown:
                raise ValueError(
                    f"`callback_on_step_end_tensor_inputs` contains {unknown}, which are not in "
                    f"{self._callback_tensor_inputs}."
                )

    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None,
        noise_scale: float,
    ) -> torch.Tensor:
        r"""
        Sample the initial RGB noise, or validate a tensor the caller already sampled.

        Fresh noise is multiplied by `noise_scale`. A provided `latents` tensor is not rescaled.

        Args:
            batch_size (`int`):
                Number of images.
            num_channels (`int`):
                Channel count. 3 for RGB.
            height (`int`):
                Image height in pixels.
            width (`int`):
                Image width in pixels.
            dtype (`torch.dtype`):
                Dtype of freshly sampled noise.
            device (`torch.device`):
                Device of the returned tensor.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                Per-call RNG. A list must have length `batch_size`.
            latents (`torch.Tensor`, *optional*):
                Starting noise of shape `(batch_size, num_channels, height, width)`.
            noise_scale (`float`):
                Scale applied only when `latents` is `None`.

        Returns:
            `torch.Tensor`: Starting noise of shape `(batch_size, num_channels, height, width)`.
        """
        shape = (batch_size, num_channels, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You passed a list of {len(generator)} generators for a batch of {batch_size}. "
                "The two lengths must match."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = latents * float(noise_scale)
        else:
            latents = latents.to(device=device)
            if tuple(latents.shape) != shape:
                raise ValueError(f"`latents` shape {tuple(latents.shape)} does not match the expected {shape}.")
        return latents

    def _prepare_scheduler(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        r"""
        Install the single Priming evaluation on `self.scheduler`.

        Args:
            num_inference_steps (`int`):
                Must be 1.
            device (`torch.device`):
                Device of the scheduler tensors.

        Returns:
            `torch.Tensor`: The one timestep at which the generator is evaluated.
        """
        self._reject_karras_scheduler(self.scheduler)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        return self.scheduler.timesteps

    def _predict_sample(
        self,
        latents: torch.Tensor,
        class_labels: torch.Tensor,
        guidance_scale: float,
        guidance_interval_min: float,
        guidance_interval_max: float,
    ) -> torch.Tensor:
        r"""
        Run the one-step generator and return the predicted image in float32.

        Args:
            latents (`torch.Tensor`):
                Noise `(batch, channels, height, width)`.
            class_labels (`torch.Tensor`):
                Class ids `(batch,)`.
            guidance_scale (`float`):
                CFG scale. Applied only when the checkpoint has CFG tokens.
            guidance_interval_min (`float`):
                Interval start. Applied only when the checkpoint has interval tokens.
            guidance_interval_max (`float`):
                Interval end. Applied only when the checkpoint has interval tokens.

        Returns:
            `torch.Tensor`: Predicted image `(batch, channels, height, width)` in float32.
        """
        model_input = latents
        parameter_dtype = self.transformer.dtype
        device_type = model_input.device.type
        use_amp = device_type in {"cuda", "xpu"} and parameter_dtype in (torch.float16, torch.bfloat16)
        if not use_amp:
            model_input = model_input.to(dtype=parameter_dtype)
        cfg_tokens = int(getattr(self.transformer.config, "num_cfg_tokens", 0) or 0)
        if cfg_tokens == 0 and guidance_scale != 1.0:
            raise ValueError("This checkpoint has no CFG tokens; guidance_scale must be 1.")
        # Keep an explicit caller autocast context when weights remain FP32,
        # matching scripts/sample.py. Casting weights is a different mode.
        amp = torch.autocast(device_type, dtype=parameter_dtype) if use_amp else nullcontext()
        with amp:
            prediction = self.transformer(
                model_input,
                class_labels=class_labels,
                guidance_scale=guidance_scale,
                guidance_interval_min=guidance_interval_min,
                guidance_interval_max=guidance_interval_max,
            ).sample
        return prediction.float()

    @property
    def guidance_scale(self) -> float:
        r"""
        Classifier-free guidance scale of the call that is currently running.

        Returns:
            `float`: The scale set by the active `__call__`.
        """
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self) -> bool:
        r"""
        Whether this checkpoint can condition on a guidance scale above 1.

        Returns:
            `bool`: True when `guidance_scale > 1` and the transformer has CFG tokens.
        """
        cfg_tokens = int(getattr(self.transformer.config, "num_cfg_tokens", 0) or 0)
        return self._guidance_scale > 1.0 and cfg_tokens > 0

    @property
    def num_timesteps(self) -> int:
        r"""
        Number of scheduler timesteps in the active call.

        Returns:
            `int`: Length of the timestep schedule. Priming uses one timestep.
        """
        return self._num_timesteps

    @property
    def interrupt(self) -> bool:
        r"""
        Whether the active denoising loop should skip remaining steps.

        Returns:
            `bool`: True after the caller sets `pipeline._interrupt = True`.
        """
        return self._interrupt

    def _images_from_latents(self, latents: torch.Tensor, output_type: str) -> torch.Tensor | list[Image.Image]:
        r"""
        Convert pixel-space samples to the requested output type.

        Quantization matches the official sampler: `uint8(round(clamp((x + 1) / 2, 0, 1) * 255))`.

        Args:
            latents (`torch.Tensor`):
                Samples of shape `(batch, channels, height, width)` in the model range around `[-1, 1]`.
            output_type (`str`):
                `"latent"` returns `latents` unchanged. `"pt"` is float RGB in `[0, 1]` after that
                quantization. `"np"` is `uint8` HWC arrays. `"pil"` is a list of `PIL.Image.Image`.

        Returns:
            Images in the requested type.
        """
        if output_type == "latent":
            return latents
        images = ((latents.float() + 1) / 2).clamp(0, 1)
        images = (images * 255).round().to(torch.uint8)
        if output_type == "pt":
            return images.float() / 255.0
        arrays = images.permute(0, 2, 3, 1).cpu().numpy()
        if output_type == "np":
            return arrays
        return [Image.fromarray(image) for image in arrays]

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        class_labels: int | str | list[int | str],
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        guidance_interval_min: float | None = None,
        guidance_interval_max: float | None = None,
        noise_scale: float | None = None,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int, dict], dict] | PipelineCallback | MultiPipelineCallbacks | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        **kwargs,
    ) -> ImagePipelineOutput | tuple:
        r"""
        Generate images from ImageNet class labels.

        Args:
            class_labels (`int`, `str`, or `list`):
                One ImageNet class id, one English label, or a list of either. English labels are
                resolved with [`get_label_ids`]. The list length is the batch size.
            height (`int`, *optional*):
                Image height in pixels. Must equal `transformer.config.sample_size` (256).
            width (`int`, *optional*):
                Image width in pixels. Defaults to `transformer.config.sample_size` (256).
            num_inference_steps (`int`, *optional*):
                Must be 1. Defaults to `transformer.config.default_num_inference_steps`.
            guidance_scale (`float`, *optional*):
                CFG scale. Defaults to `transformer.config.guidance_scale` (1). Released checkpoints have
                no CFG tokens, so the forward uses the class label only.
            guidance_interval_min (`float`, *optional*):
                Interval start. Defaults to `transformer.config.guidance_interval_min`. Used when the
                checkpoint has interval tokens.
            guidance_interval_max (`float`, *optional*):
                Interval end. Defaults to `transformer.config.guidance_interval_max`.
            noise_scale (`float`, *optional*):
                Initial noise scale. Defaults to `transformer.config.noise_scale` (1). Ignored when
                `latents` is passed.
            eta (`float`, defaults to 0.0):
                DDIM eta. Ignored by [`PrimingScheduler`].
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                RNG for the initial noise.
            latents (`torch.Tensor`, *optional*):
                Initial noise `(batch, 3, height, width)`. Not multiplied by `noise_scale`.
            output_type (`str`, defaults to `"pil"`):
                `"pil"`, `"np"`, `"pt"` (float RGB in `[0, 1]`), or `"latent"` (pixels in model range).
            return_dict (`bool`, defaults to `True`):
                Return [`ImagePipelineOutput`] when `True`, otherwise a one-tuple of images.
            callback_on_step_end (`Callable` or `PipelineCallback`, *optional*):
                Called as `callback_on_step_end(pipeline, step, timestep, callback_kwargs)` after the
                scheduler step. Return a dict to replace tensors listed in
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`list[str]`, *optional*):
                Tensor names passed to the step callback. Defaults to `["latents"]`.

        Examples:

        Returns:
            [`ImagePipelineOutput`] or `tuple`:
                When `return_dict` is `True`, [`ImagePipelineOutput`] with the images. Otherwise a tuple
                whose first element is the images.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        if kwargs:
            raise TypeError(f"Unexpected arguments: {sorted(kwargs)}.")
        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = ["latents"]

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        image_size = int(self.transformer.config.sample_size)
        height = image_size if height is None else int(height)
        width = image_size if width is None else int(width)
        if num_inference_steps is None:
            num_inference_steps = int(self._sampling_default("default_num_inference_steps", 1))
        if guidance_scale is None:
            guidance_scale = float(self._sampling_default("guidance_scale", 1.0))
        if guidance_interval_min is None:
            guidance_interval_min = float(self._sampling_default("guidance_interval_min", 0.1))
        if guidance_interval_max is None:
            guidance_interval_max = float(self._sampling_default("guidance_interval_max", 1.0))
        if noise_scale is None:
            noise_scale = float(self._sampling_default("noise_scale", 1.0))

        # 1. Check inputs.
        self.check_inputs(
            class_labels,
            height,
            width,
            num_inference_steps,
            guidance_scale,
            guidance_interval_min,
            guidance_interval_max,
            noise_scale,
            output_type,
            callback_steps,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = float(guidance_scale)
        self._interrupt = False

        # 2. Define call parameters.
        class_label_ids = self._normalize_class_labels(class_labels)
        batch_size = len(class_label_ids)
        device = self._execution_device
        null_class = int(self.transformer.config.num_classes)
        class_labels_t = torch.tensor(class_label_ids, device=device, dtype=torch.long)
        if torch.any(class_labels_t < 0) or torch.any(class_labels_t > null_class):
            raise ValueError(f"Class ids must lie in [0, {null_class}], got {class_label_ids}.")

        # 3. Encode input condition (class ids are the condition).

        # 4. Prepare timesteps.
        scheduled_timesteps = self._prepare_scheduler(num_inference_steps, device)

        # 5. Prepare latent variables (pixel-space noise; there is no VAE).
        latents = self.prepare_latents(
            batch_size,
            int(self.transformer.config.in_channels),
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
            noise_scale,
        ).float()

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(scheduled_timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(scheduled_timesteps)

        # 7. Denoising loop.
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, timestep in enumerate(scheduled_timesteps):
                if self.interrupt:
                    continue

                model_output = self._predict_sample(
                    latents,
                    class_labels_t,
                    self.guidance_scale,
                    guidance_interval_min,
                    guidance_interval_max,
                )
                latents = self.scheduler.step(
                    model_output,
                    timestep,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {"latents": latents} if "latents" in callback_on_step_end_tensor_inputs else {}
                    callback_outputs = callback_on_step_end(self, i, timestep, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                if i == len(scheduled_timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % (callback_steps or 1) == 0:
                        callback(i, timestep, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        images = self._images_from_latents(latents, output_type)
        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)
        return ImagePipelineOutput(images=images)
