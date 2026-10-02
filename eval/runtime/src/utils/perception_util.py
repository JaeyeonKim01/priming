import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import os
from collections import namedtuple
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models
from tqdm import tqdm

import logging
logger = logging.getLogger("FD_loss")

# =============================================================================
# inception v3
# =============================================================================


INCEPTION_URL = (
    "https://github.com/toshas/torch-fidelity/releases/download/"
    "v0.2.0/weights-inception-2015-12-05-6726825d.pth"
)

def resize_tf(x, size=(299, 299)):
    """bilinear resize matching tensorflow 1.x behavior.
    input: NCHW float tensor.
    output: NCHW float tensor."""
    oh, ow = size
    ih, iw = x.shape[2], x.shape[3]
    scale_y, scale_x = ih / oh, iw / ow

    gy = torch.arange(oh, dtype=x.dtype, device=x.device) * scale_y
    gy_lo = gy.long()
    gy_hi = (gy_lo + 1).clamp_max(ih - 1)
    dy = gy - gy_lo.float()

    gx = torch.arange(ow, dtype=x.dtype, device=x.device) * scale_x
    gx_lo = gx.long()
    gx_hi = (gx_lo + 1).clamp_max(iw - 1)
    dx = gx - gx_lo.float()

    in_00 = x[:, :, gy_lo, :][:, :, :, gx_lo]
    in_01 = x[:, :, gy_lo, :][:, :, :, gx_hi]
    in_10 = x[:, :, gy_hi, :][:, :, :, gx_lo]
    in_11 = x[:, :, gy_hi, :][:, :, :, gx_hi]

    in_0 = in_00 + (in_01 - in_00) * dx.view(1, 1, 1, ow)
    in_1 = in_10 + (in_11 - in_10) * dx.view(1, 1, 1, ow)
    out = in_0 + (in_1 - in_0) * dy.view(1, 1, oh, 1)
    return out


class BasicConv2d(nn.Module):
    def __init__(self, in_c, out_c, **kw):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, bias=False, **kw)
        self.bn = nn.BatchNorm2d(out_c, eps=0.001)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class InceptionA(nn.Module):
    def __init__(self, in_c, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_c, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_c, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_c, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(in_c, pool_features, kernel_size=1)

    def forward(self, x):
        return torch.cat([
            self.branch1x1(x),
            self.branch5x5_2(self.branch5x5_1(x)),
            self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x))),
            self.branch_pool(F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)),
        ], 1)


class InceptionB(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.branch3x3 = BasicConv2d(in_c, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = BasicConv2d(in_c, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        return torch.cat([
            self.branch3x3(x),
            self.branch3x3dbl_3(self.branch3x3dbl_2(self.branch3x3dbl_1(x))),
            F.max_pool2d(x, 3, 2),
        ], 1)


class InceptionC(nn.Module):
    def __init__(self, in_c, c7):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_c, 192, kernel_size=1)
        self.branch7x7_1 = BasicConv2d(in_c, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_1 = BasicConv2d(in_c, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch_pool = BasicConv2d(in_c, 192, kernel_size=1)

    def forward(self, x):
        b7 = self.branch7x7_3(self.branch7x7_2(self.branch7x7_1(x)))
        b7d = self.branch7x7dbl_5(self.branch7x7dbl_4(self.branch7x7dbl_3(
            self.branch7x7dbl_2(self.branch7x7dbl_1(x)))))
        return torch.cat([
            self.branch1x1(x), b7, b7d,
            self.branch_pool(F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)),
        ], 1)


class InceptionD(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.branch3x3_1 = BasicConv2d(in_c, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = BasicConv2d(in_c, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        b7 = self.branch7x7x3_4(self.branch7x7x3_3(
            self.branch7x7x3_2(self.branch7x7x3_1(x))))
        return torch.cat([
            self.branch3x3_2(self.branch3x3_1(x)), b7,
            F.max_pool2d(x, 3, 2),
        ], 1)


class InceptionE1(nn.Module):
    """first InceptionE block (uses avg_pool)"""
    def __init__(self, in_c):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_c, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(in_c, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = BasicConv2d(in_c, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = BasicConv2d(in_c, 192, kernel_size=1)

    def forward(self, x):
        b3 = self.branch3x3_1(x)
        b3d = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        return torch.cat([
            self.branch1x1(x),
            torch.cat([self.branch3x3_2a(b3), self.branch3x3_2b(b3)], 1),
            torch.cat([self.branch3x3dbl_3a(b3d), self.branch3x3dbl_3b(b3d)], 1),
            self.branch_pool(F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)),
        ], 1)


class InceptionE2(nn.Module):
    """second InceptionE block (uses max_pool -- matches tf bug)"""
    def __init__(self, in_c):
        super().__init__()
        self.branch1x1 = BasicConv2d(in_c, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(in_c, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = BasicConv2d(in_c, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = BasicConv2d(in_c, 192, kernel_size=1)

    def forward(self, x):
        b3 = self.branch3x3_1(x)
        b3d = self.branch3x3dbl_2(self.branch3x3dbl_1(x))
        return torch.cat([
            self.branch1x1(x),
            torch.cat([self.branch3x3_2a(b3), self.branch3x3_2b(b3)], 1),
            torch.cat([self.branch3x3dbl_3a(b3d), self.branch3x3dbl_3b(b3d)], 1),
            # tf uses max_pool here (bug in original tf inception)
            self.branch_pool(F.max_pool2d(x, 3, 1, 1)),
        ], 1)


class InceptionV3(nn.Module):
    """tf-compatible InceptionV3. returns (pool_2048, logits_unbiased)."""

    def __init__(self, normalize=True):
        super().__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.MaxPool_1 = nn.MaxPool2d(3, 2)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.MaxPool_2 = nn.MaxPool2d(3, 2)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, c7=128)
        self.Mixed_6c = InceptionC(768, c7=160)
        self.Mixed_6d = InceptionC(768, c7=160)
        self.Mixed_6e = InceptionC(768, c7=192)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE1(1280)
        self.Mixed_7c = InceptionE2(2048)
        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, 1008)
        self.normalize = normalize
        
    def forward(self, x):
        # normalize=True:  x in [0, 255] (uint8 range) -> [-1, 1]
        # normalize=False: x in [0, 1] -> [-1, 1]
        x = x.float()
        x = resize_tf(x, (299, 299))
        if self.normalize:
            x = (x - 128) / 128
        else:
            x = x * 2 - 1

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.MaxPool_1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.MaxPool_2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.AvgPool(x)
        pool = torch.flatten(x, 1).float()                  # (N, 2048)
        logits_unbiased = pool.mm(self.fc.weight.T).float()  # (N, 1008)
        return pool, logits_unbiased


def load_inception(device="cuda", normalize=True):
    model = InceptionV3(normalize=normalize)
    state = torch.hub.load_state_dict_from_url(INCEPTION_URL, progress=True)
    model.load_state_dict(state)
    model.to(device).eval().requires_grad_(False)
    return model



# =============================================================================
# VGG16 and LPIPS
# =============================================================================


URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}
CKPT_MAP = {"vgg_lpips": "vgg.pth"}
MD5_MAP = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}
LPIPS_CKPT_DIR = "checkpoints/lpips"


VggOutputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])

_VGG_SLICE_BOUNDS = [0, 4, 9, 16, 23, 30]


class VGG16(nn.Module):
    def __init__(self, requires_grad: bool = False, pretrained: bool = True):
        super().__init__()
        feats = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
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


def download(url: str, local_path: str, chunk_size: int = 1024) -> None:
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path: str) -> str:
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name: str, root: str, check: bool = False) -> str:
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        logger.info("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path


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
    def __init__(self, ckpt_pth=LPIPS_CKPT_DIR, use_dropout=True):
        super().__init__()
        self.chns = [64, 128, 256, 512, 512]  # VGG16 feature channels
        self.net = VGG16(pretrained=True, requires_grad=False)
        self.lins = nn.ModuleList([NetLinLayer(c, use_dropout=use_dropout) for c in self.chns])
        self.load_from_pretrained(ckpt_pth=ckpt_pth)
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, ckpt_pth=LPIPS_CKPT_DIR, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, ckpt_pth, check=True)
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

        model = ConvNextV2ForImageClassification.from_pretrained(name)
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
