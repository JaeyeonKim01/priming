"""TensorFlow-compatible Inception-v3 features for ADM/FID evaluation."""

from __future__ import annotations

from collections.abc import Mapping
import os

import torch
from torch import Tensor, nn
import torch.nn.functional as F


INCEPTION_WEIGHTS_URL = (
    "https://github.com/toshas/torch-fidelity/releases/download/"
    "v0.2.0/weights-inception-2015-12-05-6726825d.pth"
)


def tensorflow_resize(
    images: Tensor, size: tuple[int, int] = (299, 299)
) -> Tensor:
    """Bilinear resize with the TensorFlow 1.x FID coordinate convention."""
    output_h, output_w = size
    input_h, input_w = images.shape[-2:]
    scale_y, scale_x = input_h / output_h, input_w / output_w

    grid_y = torch.arange(
        output_h, dtype=images.dtype, device=images.device
    ) * scale_y
    y_low = grid_y.long()
    y_high = (y_low + 1).clamp_max(input_h - 1)
    dy = grid_y - y_low.to(grid_y.dtype)

    grid_x = torch.arange(
        output_w, dtype=images.dtype, device=images.device
    ) * scale_x
    x_low = grid_x.long()
    x_high = (x_low + 1).clamp_max(input_w - 1)
    dx = grid_x - x_low.to(grid_x.dtype)

    in_00 = images[:, :, y_low, :][:, :, :, x_low]
    in_01 = images[:, :, y_low, :][:, :, :, x_high]
    in_10 = images[:, :, y_high, :][:, :, :, x_low]
    in_11 = images[:, :, y_high, :][:, :, :, x_high]
    in_0 = in_00 + (in_01 - in_00) * dx.view(1, 1, 1, output_w)
    in_1 = in_10 + (in_11 - in_10) * dx.view(1, 1, 1, output_w)
    return in_0 + (in_1 - in_0) * dy.view(1, 1, output_h, 1)


class _BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, inputs: Tensor) -> Tensor:
        return F.relu(self.bn(self.conv(inputs)), inplace=True)


class _InceptionA(nn.Module):
    def __init__(self, in_channels: int, pool_features: int) -> None:
        super().__init__()
        self.branch1x1 = _BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = _BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = _BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = _BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = _BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = _BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = _BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.cat(
            [
                self.branch1x1(inputs),
                self.branch5x5_2(self.branch5x5_1(inputs)),
                self.branch3x3dbl_3(
                    self.branch3x3dbl_2(self.branch3x3dbl_1(inputs))
                ),
                self.branch_pool(
                    F.avg_pool2d(inputs, 3, 1, 1, count_include_pad=False)
                ),
            ],
            dim=1,
        )


class _InceptionB(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch3x3 = _BasicConv2d(
            in_channels, 384, kernel_size=3, stride=2
        )
        self.branch3x3dbl_1 = _BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = _BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = _BasicConv2d(
            96, 96, kernel_size=3, stride=2
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.cat(
            [
                self.branch3x3(inputs),
                self.branch3x3dbl_3(
                    self.branch3x3dbl_2(self.branch3x3dbl_1(inputs))
                ),
                F.max_pool2d(inputs, 3, 2),
            ],
            dim=1,
        )


class _InceptionC(nn.Module):
    def __init__(self, in_channels: int, channels_7x7: int) -> None:
        super().__init__()
        self.branch1x1 = _BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7_1 = _BasicConv2d(
            in_channels, channels_7x7, kernel_size=1
        )
        self.branch7x7_2 = _BasicConv2d(
            channels_7x7,
            channels_7x7,
            kernel_size=(1, 7),
            padding=(0, 3),
        )
        self.branch7x7_3 = _BasicConv2d(
            channels_7x7, 192, kernel_size=(7, 1), padding=(3, 0)
        )
        self.branch7x7dbl_1 = _BasicConv2d(
            in_channels, channels_7x7, kernel_size=1
        )
        self.branch7x7dbl_2 = _BasicConv2d(
            channels_7x7,
            channels_7x7,
            kernel_size=(7, 1),
            padding=(3, 0),
        )
        self.branch7x7dbl_3 = _BasicConv2d(
            channels_7x7,
            channels_7x7,
            kernel_size=(1, 7),
            padding=(0, 3),
        )
        self.branch7x7dbl_4 = _BasicConv2d(
            channels_7x7,
            channels_7x7,
            kernel_size=(7, 1),
            padding=(3, 0),
        )
        self.branch7x7dbl_5 = _BasicConv2d(
            channels_7x7, 192, kernel_size=(1, 7), padding=(0, 3)
        )
        self.branch_pool = _BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        branch7 = self.branch7x7_3(
            self.branch7x7_2(self.branch7x7_1(inputs))
        )
        branch7_double = self.branch7x7dbl_5(
            self.branch7x7dbl_4(
                self.branch7x7dbl_3(
                    self.branch7x7dbl_2(self.branch7x7dbl_1(inputs))
                )
            )
        )
        return torch.cat(
            [
                self.branch1x1(inputs),
                branch7,
                branch7_double,
                self.branch_pool(
                    F.avg_pool2d(inputs, 3, 1, 1, count_include_pad=False)
                ),
            ],
            dim=1,
        )


class _InceptionD(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch3x3_1 = _BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = _BasicConv2d(
            192, 320, kernel_size=3, stride=2
        )
        self.branch7x7x3_1 = _BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = _BasicConv2d(
            192, 192, kernel_size=(1, 7), padding=(0, 3)
        )
        self.branch7x7x3_3 = _BasicConv2d(
            192, 192, kernel_size=(7, 1), padding=(3, 0)
        )
        self.branch7x7x3_4 = _BasicConv2d(
            192, 192, kernel_size=3, stride=2
        )

    def forward(self, inputs: Tensor) -> Tensor:
        branch7 = self.branch7x7x3_4(
            self.branch7x7x3_3(
                self.branch7x7x3_2(self.branch7x7x3_1(inputs))
            )
        )
        return torch.cat(
            [
                self.branch3x3_2(self.branch3x3_1(inputs)),
                branch7,
                F.max_pool2d(inputs, 3, 2),
            ],
            dim=1,
        )


class _InceptionE(nn.Module):
    def __init__(self, in_channels: int, *, final: bool) -> None:
        super().__init__()
        self.final = final
        self.branch1x1 = _BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = _BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = _BasicConv2d(
            384, 384, kernel_size=(1, 3), padding=(0, 1)
        )
        self.branch3x3_2b = _BasicConv2d(
            384, 384, kernel_size=(3, 1), padding=(1, 0)
        )
        self.branch3x3dbl_1 = _BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = _BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = _BasicConv2d(
            384, 384, kernel_size=(1, 3), padding=(0, 1)
        )
        self.branch3x3dbl_3b = _BasicConv2d(
            384, 384, kernel_size=(3, 1), padding=(1, 0)
        )
        self.branch_pool = _BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        branch3 = self.branch3x3_1(inputs)
        branch3_double = self.branch3x3dbl_2(self.branch3x3dbl_1(inputs))
        pooled = (
            F.max_pool2d(inputs, 3, 1, 1)
            if self.final
            else F.avg_pool2d(inputs, 3, 1, 1, count_include_pad=False)
        )
        return torch.cat(
            [
                self.branch1x1(inputs),
                torch.cat(
                    [
                        self.branch3x3_2a(branch3),
                        self.branch3x3_2b(branch3),
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        self.branch3x3dbl_3a(branch3_double),
                        self.branch3x3dbl_3b(branch3_double),
                    ],
                    dim=1,
                ),
                self.branch_pool(pooled),
            ],
            dim=1,
        )


class _FIDInceptionV3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.Conv2d_1a_3x3 = _BasicConv2d(
            3, 32, kernel_size=3, stride=2
        )
        self.Conv2d_2a_3x3 = _BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = _BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.MaxPool_1 = nn.MaxPool2d(3, 2)
        self.Conv2d_3b_1x1 = _BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = _BasicConv2d(80, 192, kernel_size=3)
        self.MaxPool_2 = nn.MaxPool2d(3, 2)
        self.Mixed_5b = _InceptionA(192, 32)
        self.Mixed_5c = _InceptionA(256, 64)
        self.Mixed_5d = _InceptionA(288, 64)
        self.Mixed_6a = _InceptionB(288)
        self.Mixed_6b = _InceptionC(768, 128)
        self.Mixed_6c = _InceptionC(768, 160)
        self.Mixed_6d = _InceptionC(768, 160)
        self.Mixed_6e = _InceptionC(768, 192)
        self.Mixed_7a = _InceptionD(768)
        self.Mixed_7b = _InceptionE(1280, final=False)
        self.Mixed_7c = _InceptionE(2048, final=True)
        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, 1008)

    def forward(self, images: Tensor) -> Tensor:
        inputs = tensorflow_resize(images.float()) * 2.0 - 1.0
        inputs = self.Conv2d_1a_3x3(inputs)
        inputs = self.Conv2d_2a_3x3(inputs)
        inputs = self.Conv2d_2b_3x3(inputs)
        inputs = self.MaxPool_1(inputs)
        inputs = self.Conv2d_3b_1x1(inputs)
        inputs = self.Conv2d_4a_3x3(inputs)
        inputs = self.MaxPool_2(inputs)
        inputs = self.Mixed_5b(inputs)
        inputs = self.Mixed_5c(inputs)
        inputs = self.Mixed_5d(inputs)
        inputs = self.Mixed_6a(inputs)
        inputs = self.Mixed_6b(inputs)
        inputs = self.Mixed_6c(inputs)
        inputs = self.Mixed_6d(inputs)
        inputs = self.Mixed_6e(inputs)
        inputs = self.Mixed_7a(inputs)
        inputs = self.Mixed_7b(inputs)
        inputs = self.Mixed_7c(inputs)
        return torch.flatten(self.AvgPool(inputs), 1).float()


class FIDInceptionExtractor(nn.Module):
    """Return the 2,048-dimensional ADM/TF-FID pooled representation.

    Inputs are floating RGB images in ``[0,1]``. When ``weights_path`` is not
    supplied, the official torch-fidelity conversion is downloaded through
    PyTorch Hub and cached below ``TORCH_HOME``.
    """

    def __init__(self, weights_path: str | os.PathLike[str] | None = None) -> None:
        super().__init__()
        self.model = _FIDInceptionV3()
        if weights_path is None:
            state = torch.hub.load_state_dict_from_url(
                INCEPTION_WEIGHTS_URL,
                progress=True,
                map_location="cpu",
                check_hash=True,
            )
        else:
            state = torch.load(
                os.fspath(weights_path), map_location="cpu", weights_only=True
            )
        if isinstance(state, Mapping) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("FIDInceptionExtractor expects RGB [N,3,H,W]")
        return self.model(images)


__all__ = [
    "FIDInceptionExtractor",
    "INCEPTION_WEIGHTS_URL",
    "tensorflow_resize",
]
