from __future__ import annotations

import unittest

import torch
from torch import nn

from mini_t2i.utils import unwrap_model, update_ema


class Wrapper(nn.Module):
    def __init__(self, module: nn.Module, attribute: str) -> None:
        super().__init__()
        setattr(self, attribute, module)


class ModelUnwrapTest(unittest.TestCase):
    def test_unwraps_distributed_and_compiled_wrappers(self) -> None:
        model = nn.Linear(2, 2)
        compiled = Wrapper(model, "_orig_mod")
        distributed = Wrapper(compiled, "module")

        self.assertIs(unwrap_model(distributed), model)
        self.assertEqual(set(unwrap_model(distributed).state_dict()), {"weight", "bias"})

    def test_ema_reads_unwrapped_parameters(self) -> None:
        model = nn.Linear(2, 2)
        ema_model = nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.fill_(2.0)
            model.bias.fill_(3.0)
            ema_model.weight.zero_()
            ema_model.bias.zero_()

        update_ema(ema_model, Wrapper(Wrapper(model, "_orig_mod"), "module"), decay=0.5)

        torch.testing.assert_close(ema_model.weight, torch.ones_like(ema_model.weight))
        torch.testing.assert_close(ema_model.bias, torch.full_like(ema_model.bias, 1.5))


if __name__ == "__main__":
    unittest.main()
