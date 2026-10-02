"""Unchanged bicubic crop forward with a deterministic input-gradient adjoint.

The native CUDA bicubic backward uses unordered atomic additions. Its native
CPU counterpart accumulates deterministically. Only the generated-image crop
needs gradients; the grid is random data and is never differentiated. The
input adjoint depends on the grid and upstream gradient, not input values.
"""
import torch
import torch.nn.functional as F

class _BicubicCrop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, grid):
        assert inputs.dtype == grid.dtype == torch.float32
        assert not grid.requires_grad
        ctx.save_for_backward(grid)
        ctx.input_shape = inputs.shape
        ctx.input_device = inputs.device
        return F.grid_sample(inputs, grid, mode='bicubic', padding_mode='border', align_corners=True)

    @staticmethod
    def backward(ctx, gradient):
        grid, = ctx.saved_tensors
        # output_mask disables the grid derivative, so input values are unused.
        input_gradient, _ = torch.ops.aten.grid_sampler_2d_backward.default(
            gradient.to(device='cpu', dtype=torch.float32),
            torch.zeros(ctx.input_shape, dtype=torch.float32, device='cpu'),
            grid.to(device='cpu', dtype=torch.float32),
            2, 1, True, [True, False],
        )
        return input_gradient.to(device=ctx.input_device), None


def bicubic_crop(inputs, grid):
    return _BicubicCrop.apply(inputs, grid)
