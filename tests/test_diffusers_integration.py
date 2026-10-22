"""Optional integration: run with diffusers installed."""
import tempfile
import unittest
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import torch

if importlib.util.find_spec('diffusers') is None:
    raise unittest.SkipTest('Optional diffusers dependency is not installed')

from integrations.huggingface_diffusers import PrimingPipeline, PrimingScheduler, PrimingTransformer2DModel
from models.priming import OST


class DiffusersIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.native = OST(input_size=32, patch_size=16, hidden_size=32, depth=2,
                          num_heads=4, num_classes=10, num_class_tokens=2,
                          num_cfg_tokens=0, num_interval_tokens=0).eval()
        # Exercise every branch: released final/residual layers are nonzero.
        with torch.no_grad():
            for name, parameter in self.native.named_parameters():
                if name.endswith('_scale') or name.startswith('final_layer.linear'):
                    parameter.normal_(0, .1)
        self.model = PrimingTransformer2DModel(sample_size=32, hidden_size=32, depth=2,
                     num_attention_heads=4, num_classes=10, num_class_tokens=2).eval()
        self.model.load_state_dict(self.native.state_dict(), strict=True)
        self.pipe = PrimingPipeline(self.model, PrimingScheduler(), id2label={1:'one'})
        self.pipe.set_progress_bar_config(disable=True)
        self.noise = torch.randn(2,3,32,32)
        self.labels = torch.tensor([1,2])

    def test_native_parity_and_save_reload(self):
        with torch.no_grad():
            expected = self.native(self.noise, self.labels)
        actual = self.pipe([1,2], latents=self.noise, output_type='latent').images
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            self.pipe.save_pretrained(directory)
            loaded = PrimingPipeline.from_pretrained(directory, local_files_only=True)
            torch.testing.assert_close(loaded([1,2], latents=self.noise, output_type='latent').images,
                                       expected, rtol=0, atol=0)
            self.assertEqual(loaded.get_label_ids('one'), [1])

    def test_explicit_noise_callback_and_postprocess(self):
        raw = self.pipe([1,2], latents=self.noise, noise_scale=9, output_type='latent').images
        base = self.pipe([1,2], latents=self.noise, output_type='latent').images
        torch.testing.assert_close(raw, base, rtol=0, atol=0)
        def callback(pipe, step, time, tensors):
            return {'latents': torch.zeros_like(tensors['latents'])}
        out = self.pipe([1,2], latents=self.noise, callback_on_step_end=callback, output_type='np').images
        self.assertTrue((out == 128).all())

    def test_rejected_inputs(self):
        for kwargs in [dict(guidance_scale=2), dict(noise_scale=float('nan')),
                       dict(num_inference_steps=2), dict(height=48), dict(class_labels=[11])]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.pipe(**({'class_labels':[1,2]} | kwargs))

    def test_converter_uses_released_sampling_args(self):
        from integrations.huggingface_diffusers.convert import convert
        # cfg lives outside sampling_args in the release JSON; the sampler uses 1.
        sampling_args = dict(noise_scale=1, interval_min=.1, interval_max=1,
                             rope_2d=True, learned_pe=True)
        with tempfile.TemporaryDirectory() as directory, patch(
            'utils.checkpoint.load_released_model',
            return_value=(SimpleNamespace(net=self.native), sampling_args),
        ) as loader:
            pipeline, record = convert('priming_b_160', directory)
            loader.assert_called_once_with('priming_b_160', checkpoint=None, terminal=True)
            self.assertEqual(pipeline.transformer.config.guidance_scale, 1)
            self.assertEqual(record['weights'], 'ema')
            with self.assertRaises(ValueError):
                convert('priming_b_160', directory)

    def test_standalone_custom_pipeline_export(self):
        import json
        from pathlib import Path
        from diffusers import DiffusionPipeline
        from integrations.huggingface_diffusers.export_hub import export_hub
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source'
            self.pipe.save_pretrained(source)
            (source/'conversion.json').write_text(json.dumps({'test_fixture': True}))
            output = export_hub(source, Path(directory)/'standalone')
            loaded = DiffusionPipeline.from_pretrained(
                output, custom_pipeline=str(output/'pipeline.py'),
                trust_remote_code=True, local_files_only=True,
            )
            self.assertTrue(loaded.__class__.__module__.startswith('diffusers_modules.'))
            self.assertTrue(loaded.transformer.__class__.__module__.startswith('diffusers_modules.'))
            expected = self.pipe([1,2], latents=self.noise, output_type='latent').images
            actual = loaded([1,2], latents=self.noise, output_type='latent').images
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
