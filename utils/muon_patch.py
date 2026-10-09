"""
Compatibility patch for the Muon optimizer.

Muon releases differ in a few distributed-step details. This patch keeps all
ranks participating in each all_gather by replacing missing gradients with
zeros, reshapes Muon updates back to the parameter shape for convolutional
weights, and pads Muon parameter groups to the next multiple of world_size
without allocating an unnecessary full padding batch.

Usage:
    # At the top of your training script, BEFORE creating the optimizer:
    import utils.muon_patch  # noqa: F401  — applies the patch on import
"""

import torch
import torch.distributed as dist
import muon as _muon_module


# ---------- patched Muon.step ----------

@torch.no_grad()
def _patched_muon_step(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        params = group["params"]
        # fix: pad to next multiple of world_size (not remainder)
        params_pad = params + [torch.empty_like(params[-1])] * (
            (-len(params)) % dist.get_world_size()
        )
        for base_i in range(len(params_pad))[::dist.get_world_size()]:
            if base_i + dist.get_rank() < len(params):
                p = params[base_i + dist.get_rank()]
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = _muon_module.muon_update(
                    p.grad, state["momentum_buffer"], beta=group["momentum"]
                )
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])
            dist.all_gather(
                params_pad[base_i : base_i + dist.get_world_size()],
                params_pad[base_i + dist.get_rank()],
            )
    return loss


# ---------- patched MuonWithAuxAdam.step ----------

@torch.no_grad()
def _patched_muon_with_aux_adam_step(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        if group["use_muon"]:
            params = group["params"]
            # fix: pad to next multiple of world_size (not remainder)
            params_pad = params + [torch.empty_like(params[-1])] * (
                (-len(params)) % dist.get_world_size()
            )
            for base_i in range(len(params_pad))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = _muon_module.muon_update(
                        p.grad, state["momentum_buffer"], beta=group["momentum"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(
                    params_pad[base_i : base_i + dist.get_world_size()],
                    params_pad[base_i + dist.get_rank()],
                )
        else:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] = 0
                state["step"] += 1
                update = _muon_module.adam_update(
                    p.grad, state["exp_avg"], state["exp_avg_sq"],
                    state["step"], group["betas"], group["eps"],
                )
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])
    return loss


# ---------- apply patches ----------

_muon_module.Muon.step = _patched_muon_step
_muon_module.MuonWithAuxAdam.step = _patched_muon_with_aux_adam_step
