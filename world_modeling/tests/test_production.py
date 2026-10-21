import copy
import unittest

import torch

from world_modeling.scripts.production import validate_payload


class ProductionChecks(unittest.TestCase):
    def fixture(self):
        return dict(step=84608,
            training_contract=dict(stage='pretrain', world_size=8, batch_size=8, rt1={}),
            config={'train':dict(num_epochs=64, max_steps=None)},
            rng_states=[{'torch':torch.tensor([1])} for _ in range(8)],
            optimizer={'state':{0:{'step':torch.tensor(84608), 'exp_avg':torch.zeros(2)}}},
            model={'weight':torch.ones(2)})

    def test_accepts_complete_recipe_and_rejects_pilot_or_wrong_task(self):
        state = self.fixture()
        validate_payload(state, 'rt1', 'pretrain')
        for field, value in [('max_steps', 84608), ('num_epochs', 1)]:
            invalid = copy.deepcopy(state)
            invalid['config']['train'][field] = value
            with self.assertRaises(ValueError): validate_payload(invalid, 'rt1', 'pretrain')
        invalid = copy.deepcopy(state)
        invalid['training_contract'].pop('rt1')
        invalid['training_contract']['bridge'] = {}
        with self.assertRaises(ValueError): validate_payload(invalid, 'rt1', 'pretrain')

    def test_rejects_corrupt_or_partial_checkpoint(self):
        for change in ('nonfinite_model', 'nonfinite_optimizer', 'missing_rng', 'stale_optimizer', 'partial'):
            state = self.fixture()
            if change == 'nonfinite_model': state['model']['weight'][0] = float('nan')
            if change == 'nonfinite_optimizer': state['optimizer']['state'][0]['exp_avg'][0] = float('inf')
            if change == 'missing_rng': state['rng_states'].pop()
            if change == 'stale_optimizer': state['optimizer']['state'][0]['step'] = torch.tensor(1000)
            if change == 'partial': state['step'] = 1000
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_payload(state, 'rt1', 'pretrain')


if __name__ == '__main__': unittest.main()
