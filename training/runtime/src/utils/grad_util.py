import torch
from torch import inf


def get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    grads = [p.grad.detach() for p in parameters]
    if float(norm_type) == inf:
        return max(g.abs().max().to(device) for g in grads)
    return torch.norm(
        torch.stack([torch.norm(g, norm_type).to(device) for g in grads]),
        norm_type,
    )