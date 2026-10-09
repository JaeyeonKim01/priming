"""The production Muon/auxiliary-Adam grouping and post-training AdamW."""
import torch


def muon_groups(model):
    matrices, auxiliary = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        special = any(key in name for key in ("pos_embed", "token", "in_context", "embedding"))
        (matrices if parameter.ndim == 2 and not special else auxiliary).append(parameter)
    return matrices, auxiliary


def create_optimizer(model, config):
    if config["optimizer"] == "muon":
        from . import muon_patch  # Apply the production distributed update fixes.
        from muon import MuonWithAuxAdam
        matrices, auxiliary = muon_groups(model)
        groups = []
        if matrices:
            groups.append(dict(params=matrices, use_muon=True, lr=config["lr"],
                               momentum=config["muon_momentum"], weight_decay=config["muon_weight_decay"]))
        if auxiliary:
            groups.append(dict(params=auxiliary, use_muon=False, lr=config["lr"],
                               betas=(config["beta1"], config["beta2"]), eps=1e-8, weight_decay=0.))
        return MuonWithAuxAdam(groups)
    if config["optimizer"] != "adamw":
        raise ValueError("optimizer must be muon or adamw")
    groups = [[], []]
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            no_decay = parameter.ndim < 2 or any(key in name for key in
                ("ln", "bias", "embedding", "norm", "gamma", "embed", "token", "diffloss"))
            groups[0 if no_decay else 1].append(parameter)
    return torch.optim.AdamW([
        dict(params=groups[0], weight_decay=0.),
        dict(params=groups[1], weight_decay=config["weight_decay"]),
    ], lr=config["lr"], betas=(config["beta1"], config["beta2"]))


def grad_norm(parameters):
    gradients = [p.grad.detach() for p in parameters if p.grad is not None]
    if not gradients:
        return torch.tensor(0.)
    return torch.norm(torch.stack([torch.norm(g, 2).to(gradients[0].device) for g in gradients]), 2)
