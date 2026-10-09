"""Constant-decay parameter EMA in the released checkpoint format."""
from contextlib import contextmanager

import torch


class EMA:
    def __init__(self, model, decay=.9999, batch_size=1024):
        self.decay = decay
        self.label = str(decay)
        self.batch_size = batch_size
        self.step_count = 0
        self.shadow = {name: p.clone().detach() if p.requires_grad else p
                       for name, p in model.named_parameters()}

    @torch.no_grad()
    def step(self, model):
        self.step_count += 1
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.shadow[name].lerp_(parameter.data, 1. - self.decay)
            else:
                self.shadow[name].copy_(parameter.data)

    def state_dict(self):
        return dict(step_count=self.step_count, schedule="const_schedule",
                    batch_size=self.batch_size, schedules=[(self.label, self.decay)],
                    shadows={self.label: dict(self.shadow)})

    def to(self, device):
        self.shadow = {name: value.to(device) for name, value in self.shadow.items()}
        return self

    @torch.no_grad()
    def load_state_dict(self, state):
        # Post-training inherits the pretraining shadow under its new decay label.
        shadows = state["shadows"]
        if len(shadows) != 1:
            raise ValueError("Expected one EMA shadow")
        source = next(iter(shadows.values()))
        if set(source) != set(self.shadow):
            raise ValueError("EMA tensor names differ")
        for name, value in source.items():
            if value.shape != self.shadow[name].shape:
                raise ValueError(f"EMA tensor shape differs: {name}")
        for name, value in source.items():
            self.shadow[name].copy_(value.to(self.shadow[name].device))
        self.step_count = state["step_count"]

    @contextmanager
    def swap(self, model):
        original = {name: p.data.clone() for name, p in model.named_parameters()}
        try:
            for name, p in model.named_parameters():
                p.data.copy_(self.shadow[name])
            yield
        finally:
            for name, p in model.named_parameters():
                p.data.copy_(original[name])
