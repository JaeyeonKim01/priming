"""Numerical evaluation comparisons against the untouched production checkout."""
from collections.abc import Callable, Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import eval as evaluation
from metrics.frechet import covariance_sqrt, frechet_distance
from metrics.inception import compute_isc, compute_precision_recall
from metrics.moments import ExactMoments
from metrics.statistics import FeatureStats, load_feature_stats
from models.encoders import adm
from models.encoders.base import ExtractorSpec
from utils.collectives import DistributedContext
from utils.ema import EMA
from test_pretrain import reference_definitions


class PixelFeatures(nn.Module):
    def __init__(self, name, index=0):
        super().__init__()
        self.index = index
        self.spec = ExtractorSpec(family='synthetic', model=name, feature_dim=3)

    def forward(self, images):
        pixels = images.mean((2, 3))
        return pixels*(self.index+1) if self.index % 2 else pixels.square()*(self.index+1)


class EvaluationGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.7))
        self.noise_scale, self.in_channels, self.input_size = 1., 3, 2

    def generate(self, n_samples, labels, cfg=1., args=None, verbose=False):
        return torch.randn(n_samples, 3, 2, 2)*self.weight+labels[:, None, None, None]*.1


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get('PRIMING_REFERENCE_ROOT')
        if not root:
            raise unittest.SkipTest('Set PRIMING_REFERENCE_ROOT for old/new evaluation comparisons')
        cls.root = Path(root)/'training/runtime'
        cls.old = dict(torch=torch, nn=nn, F=F, np=np, dataclass=dataclass, field=field,
            Mapping=Mapping, Iterable=Iterable, Callable=Callable, Any=Any, os=os,
            json=__import__('json'), ExtractorSpec=ExtractorSpec, DistributedContext=DistributedContext,
            DataLoader=DataLoader, Dataset=Dataset, Feature=object, ImageGetter=object,
            REFERENCE_FORMAT='distmatch.frechet.reference', REFERENCE_FORMAT_VERSION=1, REFERENCE_PAYLOAD_VERSION=1)
        folder = cls.root/'packages/distmatch/src/distmatch/objectives/frechet'
        reference_definitions(folder/'loss.py', {'_checked_nonnegative', 'covariance_sqrt', 'frechet_distance'}, cls.old)
        reference_definitions(folder/'moments.py', {'ExactMoments'}, cls.old)
        reference_definitions(folder/'artifacts.py', {'FeatureStats', '_scalar_string', '_load_json', 'load_feature_stats'}, cls.old)
        reference_definitions(cls.root/'src/frechet_distance/metrics.py',
            {'compute_isc', '_knn_radius', '_fraction_in_manifold', 'compute_precision_recall'}, cls.old)
        reference_definitions(cls.root/'src/utils/perception_util.py',
            {'resize_tf', 'BasicConv2d', 'InceptionA', 'InceptionB', 'InceptionC', 'InceptionD', 'InceptionE1', 'InceptionE2', 'InceptionV3'}, cls.old)
        torch.set_num_threads(2)

    def test_moments_frechet_score_and_manifolds_match_exactly(self):
        torch.manual_seed(62)
        real, generated = torch.randn(31, 5), torch.randn(27, 5)
        moments = []
        for constructor in (self.old['ExactMoments'], ExactMoments):
            value = constructor.empty(5, device='cpu')
            for rows in generated.split(7):
                value.update(rows)
            moments.append(value.globalized(DistributedContext(False)))
        self.assertEqual(moments[0].count, moments[1].count)
        for attribute in ('sum', 'sum_outer'):
            torch.testing.assert_close(getattr(moments[0], attribute), getattr(moments[1], attribute), rtol=0, atol=0)
        ref = FeatureStats(real.double().mean(0), torch.cov(real.double().T))
        results = []
        for moment, sqrt, fd in zip(moments, (self.old['covariance_sqrt'], covariance_sqrt), (self.old['frechet_distance'], frechet_distance)):
            mean, cov = moment.mean_covariance()
            results.append(fd(mean, cov, ref.mean, ref.covariance, reference_sqrt=sqrt(ref.covariance)))
        torch.testing.assert_close(*results, rtol=0, atol=0)
        logits = torch.randn(27, 11)
        before = np.random.get_state()
        self.assertEqual(self.old['compute_isc'](logits), compute_isc(logits))
        np.testing.assert_array_equal(before[1], np.random.get_state()[1])
        self.assertEqual(self.old['compute_precision_recall'](real, generated, k=3, batch_size=7),
                         compute_precision_recall(real, generated, k=3, batch_size=7))

    def test_reference_statistics_feature_identity_and_shapes(self):
        spec = ExtractorSpec(family='test', model='test', feature_dim=3)
        with TemporaryDirectory() as folder:
            path = Path(folder)/'stats.npz'
            np.savez(path, mu=np.zeros(3), sigma=np.eye(3), count=42, feature_id=spec.feature_id, feature_dim=3)
            old = self.old['load_feature_stats'](path, expected_spec=spec)
            new = load_feature_stats(path, expected_spec=spec)
            for name in ('mean', 'covariance'):
                torch.testing.assert_close(getattr(old, name), getattr(new, name), rtol=0, atol=0)
            self.assertEqual(old.count, new.count)
            for loader in (self.old['load_feature_stats'], load_feature_stats):
                with self.assertRaisesRegex(ValueError, 'identity'):
                    loader(path, expected_spec=ExtractorSpec(family='changed', model='test', feature_dim=3))
                with self.assertRaisesRegex(ValueError, 'dimension'):
                    loader(path, expected_spec=ExtractorSpec(family='test', model='test', feature_dim=4))
                with self.assertRaises(KeyError):
                    loader(path, pool='avg')

    def test_native_inception_features_logits_and_constructor_rng(self):
        torch.manual_seed(13)
        old = self.old['InceptionV3'](normalize=False).eval().requires_grad_(False)
        with TemporaryDirectory() as folder:
            path = Path(folder)/'inception.pth'
            torch.save(old.state_dict(), path)
            initial = torch.get_rng_state()
            expected = self.old['InceptionV3'](normalize=False)
            rng_after_constructor = torch.get_rng_state()
            del expected
            torch.set_rng_state(initial)
            new = adm.load_inception(path, device='cpu')
            self.assertTrue(torch.equal(rng_after_constructor, torch.get_rng_state()))
            for dtype in (torch.float32, torch.bfloat16):
                images = torch.linspace(0., 1., 3*32*32).reshape(1, 3, 32, 32).to(dtype)
                with torch.inference_mode():
                    a, b = old(images), new(images)
                for first, second in zip(a, b):
                    self.assertEqual(second.dtype, torch.float32)
                    torch.testing.assert_close(first, second, rtol=0, atol=0)
            self.assertFalse(new.training)
            self.assertTrue(all(not p.requires_grad for p in new.parameters()))

    def test_complete_draw_matches_original_online_and_ema_evaluation(self):
        names = tuple(evaluation.BASELINES)
        extractors = {name: PixelFeatures(name, i) for i, name in enumerate(names)}
        references = {name: FeatureStats(torch.zeros(3), torch.eye(3)) for name in names}
        real = torch.linspace(0, 1, 45).reshape(15, 3)
        inception = lambda images: (images.mean((2, 3)), images.mean((2, 3))*3)
        old_ns = dict(self.old)
        class Resolver:
            def __init__(self, device): pass
            def resolve_all(self, features):
                return [SimpleNamespace(name=name, spec=extractors[name].spec, extractor=extractors[name]) for name in names]
        old_ns.update(FeatureResolver=Resolver,
            match_feature_stats=lambda resolved, stats: stats,
            validate_images=lambda *a, **kw: None,
            extract_features=lambda resolved, images: {item.name: item.extractor(images) for item in resolved})
        folder = self.root/'packages/distmatch/src/distmatch/objectives/frechet'
        reference_definitions(folder/'evaluation.py', {'FrechetEvaluation', 'FrechetEvaluator'}, old_ns)
        old_evaluator = old_ns['FrechetEvaluator']
        old_ns.update(argparse=SimpleNamespace(Namespace=object), FrozenJointSIMIntervention=object,
            FDR6_SPECS=(), FDR6_VALIDATION_FD=evaluation.BASELINES, SIM3_EVAL_NAMES=(),
            _features=lambda specs: [], _cached_stats=lambda *a, **kw: references,
            FrechetEvaluator=lambda **kw: old_evaluator(**kw),
            _prepare_inception_evaluation=lambda args, **kw: (inception, real),
            get_world_size=lambda: 1, get_global_rank=lambda: 0, is_enabled=lambda: False,
            get_start_end_indices=evaluation.partition,
            _prepare_eval_classes=lambda args, total, start, end: np.arange(start, end)%4,
            _append_eval_csv=lambda *a: None, logger=logging.getLogger('test'))
        sampling_ns = dict(torch=torch, argparse=SimpleNamespace(Namespace=object), nullcontext=nullcontext)
        reference_definitions(self.root/'src/utils/sampling_util.py', {'generate_images'}, sampling_ns, cpu=True)
        old_ns['generate_images'] = sampling_ns['generate_images']
        def accumulate(model, images, features, logits):
            x, y = model(images)
            features.append(x.float().cpu()); logits.append(y.cpu())
        def native(args, features, logits, reference, **kw):
            score, _ = self.old['compute_isc'](torch.cat(logits))
            p, r = self.old['compute_precision_recall'](reference, torch.cat(features), k=args.prc_k, batch_size=args.prc_batch_size)
            return dict(is_=score, precision=p, recall=r, num_samples=sum(map(len, features))) | {'is': score}
        old_ns.update(_accumulate_inception_batch=accumulate, run_inception_evaluation=native)
        reference_definitions(self.root/'scripts/main_mmd_and_eval.py', {'run_fdr6_evaluation'}, old_ns)
        args = SimpleNamespace(cfg_list=[1.], cfg=1., eval_ema_labels=['online', '0.999'],
            num_images=24, eval_bsz=5, interval_min=.1, interval_max=1., enable_amp=False,
            amp_dtype=torch.bfloat16, num_sampling_steps=1, current_step=1250, prc_k=3, prc_batch_size=7)
        model = EvaluationGenerator(); ema = EMA(model, decay=.999)
        ema.shadow['weight'].fill_(.3)
        class OldEMA:
            def swap(self, model, label):
                return nullcontext() if label == 'online' else ema.swap(model)
        torch.manual_seed(52); initial = torch.get_rng_state()
        old_rows = old_ns['run_fdr6_evaluation'](args, model, OldEMA(), None, wandb_logger=None)
        expected_rng = torch.get_rng_state()
        torch.set_rng_state(initial)
        evaluator = evaluation.FrechetEvaluator(extractors, references, device='cpu', distributed=False)
        config = dict(num_images=24, batch_size=5, prc_k=3, prc_batch_size=7)
        rows = evaluation.evaluate_samples(model, ema, evaluator, inception, real, config,
            dict(cfg=1., t_min=.1, t_max=1., num_steps=1), num_classes=4)
        for old, new in zip(old_rows, rows):
            self.assertEqual(new, {key: old[key] for key in new})
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(float(model.weight), float(torch.tensor(.7)))

    def test_uneven_partition_matches_reference(self):
        namespace = {}
        reference_definitions(self.root/'src/utils/eval_util.py', {'get_start_end_indices'}, namespace)
        for total, world in ((50000, 8), (23, 8), (3, 8)):
            shards = [evaluation.partition(total, world, rank) for rank in range(world)]
            self.assertEqual(shards, [namespace['get_start_end_indices'](total, world, rank) for rank in range(world)])
            self.assertEqual([i for start, end in shards for i in range(start, end)], list(range(total)))

    def test_distributed_native_metric_rounding_matches_scalar_broadcasts(self):
        torch.manual_seed(86)
        features, logits, real = torch.randn(24, 3), torch.randn(24, 7), torch.randn(15, 3)
        score, _ = self.old['compute_isc'](logits)
        precision, recall = self.old['compute_precision_recall'](real, features, k=3, batch_size=7)
        namespace = dict(torch=torch, is_enabled=lambda: True,
            dist=SimpleNamespace(broadcast=lambda tensor, src: None))
        reference_definitions(self.root/'src/utils/distributed_util.py', {'broadcast_scalar'}, namespace)
        expected = {name: namespace['broadcast_scalar'](value, device='cpu')
                    for name, value in zip(('is', 'precision', 'recall'), (score, precision, recall))}
        context = DistributedContext(False)
        context.enabled = True  # Exercise the initialized-group rounding, without a GPU process group.
        actual = evaluation.inception_metrics([features], [logits], real, context,
                                              expected=24, device='cpu', k=3, batch_size=7)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
