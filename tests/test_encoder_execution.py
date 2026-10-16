"""Execution settings preserve calibrated weights, RNG and checkpoint contracts."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

import posttrain
from models.encoders.load import FrozenEncoders, configure_full_execution
from losses.matching import Matching, input_vjp


class SmallEncoders(nn.Module):
    family_names = ('mae', 'convnext')
    microbatch = 2

    def __init__(self):
        super().__init__()
        self.mae = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1, bias=False), nn.GroupNorm(2, 4))
        self.convnext = nn.Conv2d(3, 4, 1)
        self.eval().requires_grad_(False)

    def encode_family(self, images, family):
        with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=family == 'mae'):
            return {family: getattr(self, family)(images).flatten(1)}


class PassthroughMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


class EncoderExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_conversion_only_changes_mae_conv_weights_and_preserves_rng(self):
        encoder = SmallEncoders()
        before = deepcopy(encoder.state_dict())
        rng = torch.get_rng_state().clone()
        configure_full_execution(encoder, mae_bf16_channels_last=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        conv = encoder.mae[0].weight
        self.assertEqual(conv.dtype, torch.bfloat16)
        self.assertTrue(conv.is_contiguous(memory_format=torch.channels_last))
        torch.testing.assert_close(conv, before['mae.0.weight'].bfloat16(), rtol=0, atol=0)
        for name, value in encoder.state_dict().items():
            if name != 'mae.0.weight':
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
        images = torch.randn(3, 3, 8, 8)
        rows = {family: torch.cat([encoder.encode_family(x, family)[family]
                for x in images.split(2)]) for family in encoder.family_names}
        gradient = input_vjp(encoder, images, {name: torch.ones_like(x) for name, x in rows.items()})
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)
        self.assertEqual(rows['convnext'].dtype, torch.float32)

    def test_dispatch_fallback_does_not_disable_later_compilation_or_replay(self):
        encoder = SmallEncoders()
        reference = deepcopy(encoder)
        compile_calls = []
        compiled_calls = []

        def compile_for_test(eager, **options):
            compile_calls.append(options)
            # Exercise real Dynamo graph capture on CPU, without GPU code generation.
            compiled = torch.compile(eager, backend='eager', fullgraph=True, dynamic=False)
            def call(*args):
                compiled_calls.append((args[0].shape[0], args[1], torch.is_grad_enabled()))
                return compiled(*args)
            return call

        real_compile = torch.compile
        def factory(eager, **options):
            with patch('torch.compile', real_compile):
                return compile_for_test(eager, **options)

        with patch('torch.compile', side_effect=factory):
            configure_full_execution(encoder, compile_encoders=True)
        self.assertEqual(compile_calls, [dict(fullgraph=True, dynamic=False,
                                             options={'layout_optimization': False})])
        images = torch.randn(3, 3, 8, 8)
        with PassthroughMode(), torch.no_grad():
            encoder.encode_family(images, 'mae')
        self.assertFalse(compiled_calls)
        cotangents = {}
        with torch.no_grad():
            for family in encoder.family_names:
                expected = reference.encode_family(images, family)[family]
                actual = encoder.encode_family(images, family)[family]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                cotangents[family] = torch.ones_like(actual)
        actual = input_vjp(encoder, images, cotangents)
        expected = input_vjp(reference, images, cotangents)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIn((1, 'mae', True), compiled_calls)  # Partial replay microbatch.
        self.assertIn((2, 'convnext', True), compiled_calls)

    def test_fast_recipes_only_change_execution(self):
        for family in ('b', 'l'):
            baseline = posttrain.load_config(f'configs/posttrain/{family}160.yaml')
            fast = posttrain.load_config(f'configs/posttrain/{family}160_fast.yaml')
            self.assertEqual(fast['matching'].pop('mae_bf16_channels_last'), True)
            self.assertEqual(fast['matching'].pop('compile_encoders'), True)
            self.assertEqual(fast['matching']['microbatch'], 32)
            fast['matching']['microbatch'] = 8
            self.assertEqual(fast, baseline)
        baseline = posttrain.load_config('configs/posttrain/b160.yaml')
        for field, value in (('microbatch', 16), ('microbatch', True),
                             ('compile_encoders', 'yes'), ('mae_bf16_channels_last', 1),
                             ('samples_per_class', 32), ('tolerance', 1e-4)):
            bad = deepcopy(baseline)
            bad['matching'][field] = value
            with patch('utils.config.load_yaml', return_value=bad), self.assertRaises(ValueError):
                posttrain.load_config('unused.yaml')
        bad = deepcopy(baseline)
        bad['matching'].update(stack='sim', compile_encoders=True)
        with patch('utils.config.load_yaml', return_value=bad), self.assertRaisesRegex(ValueError, 'SIM'):
            posttrain.load_config('unused.yaml')

    def test_checkpoint_rejects_changed_execution_metadata(self):
        def metadata(**execution):
            encoder = FrozenEncoders.__new__(FrozenEncoders)
            nn.Module.__init__(encoder)
            encoder.stack, encoder.manifest_digest = 'convnext_full_mae', 'original-calibration'
            encoder.execution = execution
            return encoder.checkpoint_metadata()
        baseline = metadata(microbatch=8, mae_bf16_channels_last=False, compile_encoders=False)
        fast = metadata(microbatch=32, mae_bf16_channels_last=True, compile_encoders=True)
        self.assertNotIn('encoder_execution', baseline)
        self.assertEqual(baseline['manifest_sha256'], fast['manifest_sha256'])
        def matcher(meta):
            return Matching(SmallEncoders(), bandwidths={'mae': 1., 'convnext': 1.},
                            weights={'mae': .5, 'convnext': .5}, distributed=False, checkpoint_metadata=meta)
        state = matcher(fast).running_state_dict()
        matcher(fast).load_running_state_dict(state)
        with self.assertRaises(ValueError):
            matcher(baseline).load_running_state_dict(state)
        for key, value in (('microbatch', 8), ('mae_bf16_channels_last', False), ('compile_encoders', False)):
            changed = deepcopy(fast)
            changed['encoder_execution'][key] = value
            with self.assertRaises(ValueError):
                matcher(changed).load_running_state_dict(state)


if __name__ == '__main__':
    unittest.main()
