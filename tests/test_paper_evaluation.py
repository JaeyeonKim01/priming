"""Official FID and standalone sampling protocol against the read-only old repo."""
import ast
from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _ntuple

import eval as evaluation
from metrics.fid import feature_statistics, fid_distance
from metrics.statistics import FeatureStats
from models.encoders.fid import FIDInception, resize_uint8_features
from test_evaluation import PixelFeatures
from test_pretrain import reference_definitions


class PaperEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get('PRIMING_REFERENCE_ROOT')
        if not root:
            raise unittest.SkipTest('Set PRIMING_REFERENCE_ROOT for paper evaluation comparisons')
        cls.root = Path(root)
        cls.vendor = cls.root/'eval/vendor/torch-fidelity/torch_fidelity'
        def vassert(condition, message):
            if not condition:
                raise ValueError(message)
        cls.old = dict(torch=torch, nn=nn, F=F, np=np, _ntuple=_ntuple,
            math=__import__('math'), vassert=vassert, text_to_dtype=lambda selected, default: getattr(torch, selected or default),
            KEY_METRIC_FID='frechet_inception_distance', vprint=lambda *args: None)
        reference_definitions(cls.vendor/'feature_extractor_base.py', {'FeatureExtractorBase'}, cls.old)
        reference_definitions(cls.vendor/'interpolate_compat_tensorflow.py', {'interpolate_bilinear_2d_like_tensorflow1x'}, cls.old)
        names = {'FeatureExtractorInceptionV3', 'BasicConv2d', 'InceptionA', 'InceptionB', 'InceptionC', 'InceptionD', 'InceptionE_1', 'InceptionE_2'}
        reference_definitions(cls.vendor/'feature_extractor_inceptionv3.py', names, cls.old)
        reference_definitions(cls.vendor/'metric_fid.py', {'fid_features_to_statistics', 'fid_statistics_to_metric'}, cls.old)
        torch.set_num_threads(2)

    def test_official_resize_and_full_inception_match_exactly(self):
        image = torch.linspace(0., 255., 3*37*53).reshape(1, 3, 37, 53)
        expected = self.old['interpolate_bilinear_2d_like_tensorflow1x'](image, size=(299, 299), align_corners=False)
        torch.testing.assert_close(expected, resize_uint8_features(image), rtol=0, atol=0)
        # Use a real complete network, replacing only the external pretrained weights.
        from models.encoders.adm import InceptionV3
        torch.manual_seed(11)
        weights = InceptionV3().state_dict()
        with TemporaryDirectory() as folder:
            path = Path(folder)/'inception.pth'
            torch.save(weights, path)
            torch.manual_seed(72)
            initial = torch.get_rng_state()
            old = self.old['FeatureExtractorInceptionV3']('inception-v3-compat', ['2048'], feature_extractor_weights_path=str(path))
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(initial)
            new = FIDInception(path)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            self.assertEqual({key: value.shape for key, value in old.state_dict().items()},
                             {key: value.shape for key, value in new.state_dict().items()})
            image = image.round().to(torch.uint8)
            with torch.inference_mode():
                expected, actual = old(image)[0], new(image)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(all(not p.requires_grad for p in new.parameters()))

    def test_official_numpy_statistics_and_reference_dtype_arithmetic(self):
        torch.manual_seed(68)
        features = torch.randn(23, 5)
        first = self.old['fid_features_to_statistics'](features)
        second = feature_statistics(features)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        for dtype in (np.float32, np.float64):
            reference = dict(mu=np.linspace(0., .4, 5).astype(dtype), sigma=np.eye(5, dtype=dtype)*.7)
            generated = {key: first[key].astype(reference[key].dtype) for key in ('mu', 'sigma')}
            expected = self.old['fid_statistics_to_metric'](generated, reference, verbose=False)['frechet_inception_distance']
            self.assertEqual(expected, fid_distance(second, reference))

    def test_interleaved_balanced_batches_include_the_original_padded_tail(self):
        for total, batch, world in ((1000, 32, 4), (50000, 32, 4), (3000, 64, 8)):
            ids = []
            for rank in range(world):
                actual = list(evaluation.paper_batches(total, batch, world, rank, device='cpu'))
                starts = list(range(rank*batch, total, world*batch))
                self.assertEqual([row[0] for row in actual], starts)
                for start, valid, labels in actual:
                    expected = (torch.arange(start, start+batch)//(total//1000)).clamp(max=999).long()
                    torch.testing.assert_close(expected, labels, rtol=0, atol=0)
                    self.assertEqual(len(labels), batch)
                    ids.extend(range(start, start+valid))
            self.assertEqual(sorted(ids), list(range(total)))

    def test_actual_old_sampling_loop_matches_bytes_features_and_rng(self):
        names = [row[0] for row in evaluation.PAPER_SPECS]
        def make_evaluator():
            extractors = {name: PixelFeatures(name, i) for i, name in enumerate(names)}
            references = {name: FeatureStats(torch.zeros(3), torch.eye(3)) for name in names}
            return evaluation.FrechetEvaluator(extractors, references, device='cpu', distributed=False)
        def sample(labels, seed, start):
            images = torch.randn(len(labels), 3, 1, 1)*.4+labels[:, None, None, None]*.0003
            return ((images+1)/2).clamp(0, 1).expand(-1, -1, 256, 256)
        def official(images):
            return images.float().mean((2, 3))*.01
        source = ast.parse((self.root/'eval/evaluate.py').read_text())
        loop = next(node for node in ast.walk(source) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == 'start')
        class CPU(ast.NodeTransformer):
            def visit_Constant(self, node):
                return ast.copy_location(ast.Constant('cpu'), node) if node.value == 'cuda' else node
        tree = CPU().visit(ast.Module(body=[loop], type_ignores=[]))
        ast.fix_missing_locations(tree)
        with TemporaryDirectory() as folder:
            old_output, new_output = Path(folder)/'old', Path(folder)/'new'
            old_output.mkdir(); new_output.mkdir()
            old_evaluator = make_evaluator()
            namespace = dict(torch=torch, np=np, time=time, rank=0, world=1, seed=3,
                tick=time.monotonic(), opt=SimpleNamespace(batch=32, num_images=1000, model='test'),
                sample=sample, official=lambda images: (official(images),), evaluator=old_evaluator,
                ids=[], parts=[], ihash=hashlib.sha256(), out=old_output)
            torch.manual_seed(3*1000003); np.random.seed(3*1000003)
            with torch.inference_mode():
                exec(compile(tree, 'old sampling loop', 'exec'), namespace)
            expected_rng = torch.get_rng_state()
            fd = evaluation.evaluate_paper_seed(sample, make_evaluator(), official, new_output,
                                                total=1000, batch=32, seed=3)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            np.testing.assert_array_equal(np.load(new_output/'features_rank0.npy'), torch.cat(namespace['parts']).numpy())
            np.testing.assert_array_equal(np.load(new_output/'ids_rank0.npy'), namespace['ids'])
            np.testing.assert_array_equal(np.load(new_output/'first_bytes_rank0.npy'), np.load(old_output/'first_bytes_rank0.npy'))
            import json
            self.assertEqual(json.loads((new_output/'rank0.json').read_text())['image_bytes_sha256'], namespace['ihash'].hexdigest())
            self.assertEqual(fd, old_evaluator.compute()[0])
            reference = dict(mu=np.zeros(3, dtype=np.float32), sigma=np.eye(3, dtype=np.float64))
            result = evaluation.finalize_paper_seed(new_output, total=1000, world=1, fd=fd, reference=reference)
            stat = self.old['fid_features_to_statistics'](torch.cat(namespace['parts']))
            stat = {key: stat[key].astype(reference[key].dtype) for key in ('mu', 'sigma')}
            expected_fid = self.old['fid_statistics_to_metric'](stat, reference, False)['frechet_inception_distance']
            self.assertEqual(result['fid'], expected_fid)
            ratios = {name: value/evaluation.BASELINES[name] for name, value in fd.items()}
            self.assertEqual(result['fdr5'], sum(ratios.values())/5)


if __name__ == '__main__':
    unittest.main()
