"""Direct-loop continuation, interrupted evaluation, and recipe validation."""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from data.imagenet import ClasswiseOTStream
from eval import BASELINES
from losses.matching import Matching
import posttrain
from utils.ema import EMA
from utils.evaluation import scheduled_evaluation
from utils.optim import create_optimizer
from utils.posttrain_checkpoint import resume_posttraining, snapshot_rng, restore_rng
from test_posttrain import TinyGenerator, Encoder, Features, References


def metric_rows(total=24):
    rows = []
    for label in ('online', '0.999'):
        fd = {name: (i+1)*.17 for i, name in enumerate(BASELINES)}
        fdr = {name: fd[name]/baseline for name, baseline in BASELINES.items()}
        rows.append(dict(ema_label=label, cfg=1., num_samples=total, fd=fd, fdr=fdr,
                         fdr6=sum(fdr.values())/6, **{'is': 1.3, 'precision': .7, 'recall': .4}))
    return rows


def assert_rng(test, first, second):
    test.assertEqual(first['py'], second['py'])
    np.testing.assert_equal(first['np'], second['np'])
    test.assertTrue(torch.equal(first['torch'], second['torch']))


class PosttrainRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self):
        return dict(train=dict(steps=6, save_every=2, print_freq=1),
            optimizer=dict(optimizer='adamw', lr=1e-5, beta1=.9, beta2=.95, weight_decay=0.,
                           grad_clip=1., warmup_steps=2),
            sampling=dict(cfg=1., t_min=.1, t_max=1., num_steps=1),
            evaluation=dict(every=2, num_images=24))

    def components(self):
        model = TinyGenerator()
        stream = ClasswiseOTStream(class_counts=[10]*4, global_classes_per_step=2,
            samples_per_class=3, real_references_per_class=4, rank=0, world_size=1, seed=7919, device='cpu')
        matcher = Matching(Encoder(), bandwidths=Features.sigmas, weights=Features.weights,
            global_classes=2, samples=3, references=4, iterations=30, distributed=False)
        ema = EMA(model, decay=.999)
        ema.step_count = 7
        return model, create_optimizer(model, self.config()['optimizer']), ema, matcher, References(), stream

    def test_full_loop_checkpoint_and_scheduled_draw_continuation(self):
        config = self.config()
        with TemporaryDirectory() as folder:
            root = Path(folder)
            direct, paused = root/'direct', root/'paused'
            direct.mkdir(); paused.mkdir()
            original = self.components()
            random.seed(91); np.random.seed(91); torch.manual_seed(91)
            initial = snapshot_rng()
            def consume_rng():
                random.random(); np.random.rand(3); torch.randn(4)
                return metric_rows()
            completed = posttrain.run_training(config, *original, direct, amp=False, evaluate=consume_rng)
            self.assertEqual(completed, 6)
            expected_rng = snapshot_rng()
            self.assertEqual(len(list((direct/'evaluation').glob('*.json'))), 3)
            first = self.components()
            restore_rng(initial)
            self.assertEqual(posttrain.run_training(config, *first, paused, amp=False, smoke_steps=2, evaluate=consume_rng), 2)
            checkpoint = paused/'checkpoints/step_000002.pth'
            resumed = list(self.components())
            model, _, _, matcher, _, stream = resumed
            resumed[1], resumed[2], completed = resume_posttraining(checkpoint, model, matcher, stream, config['optimizer'], config=config)
            calls = []
            def evaluated():
                calls.append(True)
                return consume_rng()
            self.assertEqual(posttrain.run_training(config, *resumed, paused, amp=False, completed=completed,
                             checkpoint=checkpoint, evaluate=evaluated), 6)
            self.assertEqual(len(calls), 2)  # The completed step-two draw is reused.
            torch.testing.assert_close(original[0].weight, resumed[0].weight, rtol=0, atol=0)
            torch.testing.assert_close(original[2].shadow['weight'], resumed[2].shadow['weight'], rtol=0, atol=0)
            self.assertEqual(resumed[2].step_count, 13)
            for key, value in original[1].state[original[0].weight].items():
                torch.testing.assert_close(value, resumed[1].state[resumed[0].weight][key], rtol=0, atol=0)
            assert_rng(self, expected_rng, snapshot_rng())
            calls.clear()
            final = paused/'checkpoints/step_000006.pth'
            self.assertEqual(posttrain.run_training(config, *resumed, paused, amp=False, completed=6,
                             checkpoint=final, evaluate=evaluated), 6)
            self.assertFalse(calls)

    def test_interrupted_draw_retries_before_next_update(self):
        config = self.config()
        with TemporaryDirectory() as folder:
            output = Path(folder)
            values = self.components()
            def fail():
                torch.randn(5); np.random.rand(); random.random()
                raise RuntimeError('interrupted draw')
            with self.assertRaisesRegex(RuntimeError, 'interrupted draw'):
                posttrain.run_training(config, *values, output, amp=False, evaluate=fail)
            checkpoint = output/'checkpoints/step_000002.pth'
            state = torch.load(checkpoint, weights_only=False)
            self.assertEqual(state['current_step'], 2)
            self.assertFalse((output/'evaluation/step_000002.json').exists())
            assert_rng(self, state['mmd_rng_state']['rank_states'][0], snapshot_rng())
            resumed = list(self.components())
            resumed[1], resumed[2], completed = resume_posttraining(checkpoint, resumed[0], resumed[3], resumed[5], config['optimizer'], config=config)
            observed = []
            def draw():
                observed.append(resumed[2].step_count)
                return metric_rows()
            posttrain.run_training(config, *resumed, output, amp=False, completed=completed, checkpoint=checkpoint,
                smoke_steps=1, evaluate=draw)
            self.assertEqual(observed, [9])  # The inherited count seven plus two completed updates.
            self.assertEqual(resumed[2].step_count, 10)
            self.assertTrue((output/'evaluation/step_000002.json').is_file())

    def test_checkpoint_preemption_between_milestones(self):
        config = self.config()
        with TemporaryDirectory() as folder:
            output = Path(folder)
            count = posttrain.run_training(config, *self.components(), output, amp=False,
                                          evaluate=metric_rows, stop=lambda: True)
            self.assertEqual(count, 1)
            latest = json.loads((output/'checkpoints/latest.json').read_text())
            self.assertEqual(latest, dict(checkpoint='step_000001.pth', completed=1))
            self.assertFalse((output/'evaluation').exists())

    def test_evaluation_record_is_bound_to_checkpoint_and_protocol(self):
        config = self.config()
        with TemporaryDirectory() as folder:
            output = Path(folder); checkpoint = output/'checkpoint.pth'
            checkpoint.write_bytes(b'first weights')
            calls = []
            def draw():
                calls.append(True)
                return metric_rows()
            scheduled_evaluation(output, checkpoint, 2, config, draw)
            self.assertEqual(scheduled_evaluation(output, checkpoint, 2, config, draw), metric_rows())
            self.assertEqual(len(calls), 1)
            checkpoint.write_bytes(b'changed weights')
            scheduled_evaluation(output, checkpoint, 2, config, draw)
            self.assertEqual(len(calls), 2)
            path = output/'evaluation/step_000002.json'
            bad = json.loads(path.read_text()); bad['rows'][1]['num_samples'] = 23
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(RuntimeError, 'sample count'):
                scheduled_evaluation(output, checkpoint, 2, config, draw)

    def test_all_five_recipes_preserve_the_original_settings(self):
        root = os.environ.get('PRIMING_REFERENCE_ROOT')
        if not root:
            self.skipTest('Set PRIMING_REFERENCE_ROOT for recipe comparisons')
        for name in ('b160', 'l160', 'b160_sim', 'b160_energy', 'b160_laplacian'):
            with self.subTest(recipe=name):
                config = posttrain.load_config(f'configs/posttrain/{name}.yaml')
                old = json.loads((Path(root)/'training/configs'/f'{name}.json').read_text())
                self.assertEqual(config['train']['steps'], old['train']['epochs']*old['train']['steps_per_epoch'])
                self.assertEqual(config['train']['global_batch_size'], old['train']['rollout_size'])
                for key in ('model', 'class_tokens', 'guidance_tokens', 'interval_tokens', 'rope_2d', 'learned_pe', 'norm_eps'):
                    self.assertEqual(config['model'][key], old['model'][key])
                for key in ('optimizer', 'lr', 'beta1', 'beta2', 'weight_decay', 'grad_clip'):
                    self.assertEqual(config['optimizer'][key], old['optimizer'][key])
                self.assertEqual(config['optimizer']['warmup_steps'], int(old['optimizer']['warmup_epochs']*old['train']['steps_per_epoch']))
                for new, original in (('kind', 'discrepancy_kind'), ('stack', 'ot_feature_stack'), ('energy_delta', 'energy_delta'),
                    ('coefficient', 'lambda_ot'), ('samples_per_class', 'ot_samples_per_class'),
                    ('real_references_per_class', 'ot_num_real_refs'), ('epsilon_mixture', 'ot_epsilon_mixture')):
                    self.assertEqual(config['matching'][new], old['ot'][original])

    def test_run_identity_does_not_require_initial_checkpoint_on_resume(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); output = root/'output'; source = root/'source.pth'
            source.write_bytes(b'custom pretrained checkpoint')
            packed = root/'data/imagenet-latents-images'; packed.mkdir(parents=True)
            for name in ('dataset_info.json', 'state.json', 'dataset.arrow'):
                (packed/name).write_text('original metadata')
            config = posttrain.load_config('configs/posttrain/b160.yaml')
            config['data']['data_path'] = str(root/'data')
            original = deepcopy(config)
            args = SimpleNamespace(resume=None, init_checkpoint=str(source))
            configured = posttrain.prepare_run(config, args, output)
            self.assertIn('_provenance', configured)
            source.unlink()
            args.resume = 'latest'; args.init_checkpoint = None
            restored = posttrain.prepare_run(deepcopy(original), args, output)
            self.assertEqual(configured, restored)
            (packed/'state.json').write_text('changed metadata')
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                posttrain.prepare_run(deepcopy(original), args, output)


if __name__ == '__main__':
    unittest.main()
