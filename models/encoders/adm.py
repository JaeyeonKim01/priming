"""Native ADM Inception features and unbiased logits for IS and precision/recall."""
import torch
from torch import nn
import torch.nn.functional as F

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


def load_inception(weights_path, device="cuda"):
    model = InceptionV3(normalize=False)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval().requires_grad_(False)
