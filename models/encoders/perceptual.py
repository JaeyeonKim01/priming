"""Frozen VGG/LPIPS and ConvNeXt encoders; weights must be prepared locally."""
from collections import namedtuple
from pathlib import Path
import hashlib
import logging

import torch
from torch import Tensor, nn
from torchvision import models

logger = logging.getLogger(__name__)

VggOutputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])

_VGG_SLICE_BOUNDS = [0, 4, 9, 16, 23, 30]


class VGG16(nn.Module):
    def __init__(self, requires_grad: bool = False, pretrained: bool = True):
        super().__init__()
        weight_file = Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"
        if not weight_file.is_file():
            raise FileNotFoundError(weight_file)
        vgg = models.vgg16(weights=None)
        vgg.load_state_dict(torch.load(weight_file, map_location="cpu", weights_only=True))
        feats = vgg.features
        self.slices = nn.ModuleList([
            nn.Sequential(*[feats[i] for i in range(s, e)])
            for s, e in zip(_VGG_SLICE_BOUNDS[:-1], _VGG_SLICE_BOUNDS[1:])
        ])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> VggOutputs:
        outputs = []
        for s in self.slices:
            x = s(x)
            outputs.append(x)
        return VggOutputs(*outputs)


def normalize_tensor(x: Tensor, eps: float = 1e-10) -> Tensor:
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x: Tensor, keepdim: bool = True) -> Tensor:
    return x.mean([2, 3], keepdim=keepdim)


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv."""

    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class LPIPS(nn.Module):
    def __init__(self, ckpt_pth="assets/lpips/vgg.pth", use_dropout=True):
        super().__init__()
        self.chns = [64, 128, 256, 512, 512]  # VGG16 feature channels
        self.net = VGG16(pretrained=True, requires_grad=False)
        self.lins = nn.ModuleList([NetLinLayer(c, use_dropout=use_dropout) for c in self.chns])
        self.load_from_pretrained(ckpt_pth=ckpt_pth)
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, ckpt_pth="assets/lpips/vgg.pth", name="vgg_lpips"):
        ckpt = Path(ckpt_pth)
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        if hashlib.md5(ckpt.read_bytes()).hexdigest() != "d507d7349b931f0638a25a48a722f98a":
            raise ValueError("LPIPS calibration differs from the released weights")
        state = torch.load(ckpt, map_location="cpu")
        remapped = {}
        for k, v in state.items():
            if not k.startswith("lin"):
                continue
            for i in range(5):
                k = k.replace(f"lin{i}.", f"lins.{i}.")
            remapped[k] = v
        self.load_state_dict(remapped, strict=False)
        logger.info(f"Loaded pretrained LPIPS loss from {ckpt}")

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # we assume input and target are imagenet-normalized.
        # see https://github.com/richzhang/PerceptualSimilarity/issues/86#issuecomment-1578100825
        # x (0-1), x = (x - imgnet_mean)/imgnet_val, then no need to use scaling layer.
        # in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(input), self.net(target)
        val = sum(
            spatial_average(lin.model((normalize_tensor(o0) - normalize_tensor(o1)) ** 2))
            for lin, o0, o1 in zip(self.lins, outs0, outs1)
        )
        return val.reshape(-1)


# =============================================================================
# ConvNext
# =============================================================================


class ConvNextFeatureExtractor(nn.Module):
    def __init__(self, name: str = "facebook/convnextv2-base-22k-224"):
        super().__init__()
        from transformers import ConvNextV2ForImageClassification

        model = ConvNextV2ForImageClassification.from_pretrained(name, local_files_only=True)
        self.model = model.convnextv2
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        # This is the tensor-only body of ConvNextV2Model.forward. Calling the
        # Hugging Face wrapper here adds runtime signature inspection that
        # cannot be captured by torch.compile(fullgraph=True).
        hidden_states = self.model.embeddings(x)
        for stage in self.model.encoder.stages:
            hidden_states = stage(hidden_states)
        return self.model.layernorm(hidden_states.mean(dim=(-2, -1)))
