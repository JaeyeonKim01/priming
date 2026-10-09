"""Frozen feature extraction and asset checks against the read-only old repo."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from models.encoders.full import encoders as full
from models.encoders.full.vendor.convnext import ConvNextV2
from models.encoders import sim, timm_features, inception
from models.encoders.load import FrozenEncoders, full_source_hash
from losses.matching import input_vjp
from utils import posttrain_assets as assets
from data.imagenet import RealImages
from test_pretrain import reference_definitions


def reference_package(name, directory, modules):
    package = ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        for module_name in modules:
            spec = importlib.util.spec_from_file_location(f'{name}.{module_name}', directory/f'{module_name}.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return {module_name: sys.modules[f'{name}.{module_name}'] for module_name in modules}


def tiny_convnext(module):
    with torch.device('meta'):
        model = module.ConvNextV2(depths=(1,1,1,1), dims=(4,8,16,32))
    del model.head
    return model


def materialize(model):
    model.to_empty(device='cpu')
    with torch.no_grad():
        for value in model.parameters():
            value.copy_(torch.randn_like(value)*.1)
    return model


class FakeTimmModel(nn.Module):
    num_features = 3

    def __init__(self, name):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3))
        self.num_prefix_tokens = 1 if 'mae' in name else 0
        self.attn_pool = None if self.num_prefix_tokens else nn.Identity()
        self.pretrained_cfg = {'hf_hub_id': f'timm/{name}', 'mean':(.4,.5,.6),
                               'std':(.2,.3,.4), 'input_size':(3,224,224)}

    def forward_features(self, images):
        tokens = images.mean((2,3)) * self.weight
        return torch.stack([tokens, tokens.square(), tokens*2], dim=1)

    def pool(self, tokens):
        return tokens.mean(1)*.7


class FakeInception(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3))
        self.spec = SimpleNamespace(feature_id='synthetic-inception',feature_dim=3)

    def forward(self, images):
        return images.mean((2,3))*self.weight


class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get('PRIMING_REFERENCE_ROOT')
        if not root:
            raise unittest.SkipTest('Set PRIMING_REFERENCE_ROOT for encoder comparisons')
        cls.root = Path(root)/'training/runtime'
        cls.old_full = reference_package('_full_encoder_reference',
            cls.root/'src/ot_representations_full', ('encoders',))['encoders']
        cls.old_extractors = reference_package('_sim_extractor_reference',
            cls.root/'packages/distmatch/src/distmatch/extractors', ('base','model_families','inception'))
        cls.old_conv = sys.modules['_full_encoder_reference.vendor.convnext']
        torch.set_num_threads(2)

    def test_calibration_source_hash_and_meta_architecture(self):
        directory = self.root/'src/ot_representations_full'
        paths = [directory/'encoders.py', *sorted((directory/'vendor').glob('*.py'))]
        expected = hashlib.sha256(b''.join(path.read_bytes() for path in paths)).hexdigest()
        self.assertEqual(full_source_hash(),expected)
        for build_old,build_new in ((self.old_full.make_convnext,full.make_convnext),
                (lambda:self.old_full.make_mae({'base_channels':640}),lambda:full.make_mae({'base_channels':640}))):
            old,new=build_old(),build_new()
            self.assertEqual({k:(v.shape,v.dtype) for k,v in old.state_dict().items()},
                             {k:(v.shape,v.dtype) for k,v in new.state_dict().items()})
            self.assertTrue(all(value.device.type=='meta' for value in new.parameters()))
            self.assertFalse(hasattr(new,'head') or hasattr(new,'decoder') or hasattr(new,'fc'))

    def test_full_features_precision_and_gradient_replay(self):
        config={'base_channels':4,'layers':[2,2,2,2]}
        with TemporaryDirectory() as folder:
            cache=Path(folder)
            (cache/'converted_manifest.json').write_text(json.dumps({'mae_config':config}))
            torch.manual_seed(43)
            torch.save(materialize(full.make_mae(config)).state_dict(),cache/'mae_encoder.pt')
            torch.save(materialize(tiny_convnext(sys.modules[ConvNextV2.__module__])).state_dict(),cache/'convnext_encoder.pt')
            for mixed in (False,True):
                with self.subTest(mixed_precision=mixed), \
                     patch.object(self.old_full,'make_convnext',lambda:tiny_convnext(self.old_conv)), \
                     patch.object(full,'make_convnext',lambda:tiny_convnext(sys.modules[ConvNextV2.__module__])):
                    old=self.old_full.PixelDriftingEncoders(cache,device='cpu',microbatch=2,
                        mixed_precision=mixed,checkpoint_inputs=False)
                    new=full.PixelDriftingEncoders(cache,device='cpu',microbatch=2,
                        mixed_precision=mixed,checkpoint_inputs=False)
                    new.family_names=('mae','convnext')
                    images=torch.linspace(-.8,1.9,3*3*16*16).reshape(3,3,16,16)
                    rng=torch.get_rng_state()
                    with torch.no_grad(), torch.autocast('cpu',dtype=torch.bfloat16):
                        a,b=old(images),new(images)
                    self.assertEqual(tuple(a),tuple(b))
                    for key in a:
                        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
                        if key.startswith('convnext__'): self.assertEqual(b[key].dtype,torch.float32)
                    self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                    gradients={name:torch.randn_like(value) for name,value in a.items()}
                    namespace=dict(torch=torch)
                    reference_definitions(self.root/'src/ot_representations_full/provider.py',{'RepresentationProvider'},namespace)
                    provider=namespace['RepresentationProvider'].__new__(namespace['RepresentationProvider'])
                    provider.encoders=old
                    old_gradient=provider.input_vjp(images,gradients)
                    new_gradient=input_vjp(new,images,gradients)
                    torch.testing.assert_close(old_gradient,new_gradient,rtol=0,atol=0)
                    self.assertGreater(float(new_gradient.abs().sum()),0.)
                    self.assertTrue(all(p.grad is None and not p.requires_grad for p in new.parameters()))

    def test_full_loader_preserves_calibration_and_checkpoint_identity(self):
        class FakeFull(nn.Module):
            def __init__(self,cache,*,device,microbatch,mixed_precision,checkpoint_inputs):
                super().__init__()
                self.microbatch=microbatch
                self.mixed_precision=mixed_precision
                self.checkpoint_inputs=checkpoint_inputs
            def forward(self,images):
                return {name:images.flatten(1) for name in names}
        names=[f'mae__{i}' for i in range(84)]+[f'convnext__{i}' for i in range(13)]
        with TemporaryDirectory() as folder:
            cache=Path(folder)
            calibration=dict(encoder_source_sha256=full_source_hash(),
                bandwidths={name:1. for name in names},weights={name:1./97 for name in names},
                sorted_row_indices=[1,2],dataset_indices=[7,8])
            for filename,value in [('downloads.json',{}),('converted_manifest.json',{}),('calibration.json',calibration)]:
                (cache/filename).write_text(json.dumps(value))
            expected=hashlib.sha256(b''.join((cache/name).read_bytes() for name in
                                           ('downloads.json','converted_manifest.json','calibration.json'))).hexdigest()
            with patch('models.encoders.load.verify_references'),patch.object(full,'PixelDriftingEncoders',FakeFull):
                encoder=FrozenEncoders(cache,device='cpu',microbatch=8)
                self.assertEqual(encoder.manifest_digest,expected)
                self.assertEqual(encoder.bandwidths,calibration['bandwidths'])
                self.assertEqual(encoder.weights,calibration['weights'])
                self.assertFalse(encoder.training)
                self.assertFalse(encoder.encoders.checkpoint_inputs)
                self.assertTrue(encoder.encoders.mixed_precision)
                calibration['encoder_source_sha256']='changed'
                (cache/'calibration.json').write_text(json.dumps(calibration))
                with self.assertRaisesRegex(ValueError,'calibration source'):
                    FrozenEncoders(cache,device='cpu')

    def test_sim_loader_rejects_changed_feature_identity(self):
        class WrongSIM(nn.Module):
            family_names=('siglip','mae','inception')
            def __init__(self,**kwargs):
                super().__init__()
                self.specs={name:SimpleNamespace(feature_id='changed',feature_dim=1) for name in self.family_names}
        with patch('models.encoders.load.verify_references'),patch.object(sim,'SIMEncoders',WrongSIM):
            with self.assertRaisesRegex(ValueError,'feature identity mismatch'):
                FrozenEncoders('/unused',stack='sim',device='cpu')

    def test_timm_local_loading_and_historical_feature_identity(self):
        timm_module=ModuleType('timm');data_module=ModuleType('timm.data')
        calls=[]
        def create_model(name,**kwargs):
            calls.append(kwargs)
            return FakeTimmModel(name)
        timm_module.create_model=create_model
        data_module.resolve_data_config=lambda cfg:cfg
        with patch.dict(sys.modules,{'timm':timm_module,'timm.data':data_module}):
            for family,name in sim.MODELS.items():
                torch.manual_seed(11)
                old=self.old_extractors['model_families'].ModelFamilyExtractor(family,name,pool='cls',target_size=224)
                torch.manual_seed(11)
                new=timm_features.ModelFamilyExtractor(family,name,pool='cls',target_size=224,weights_path='/local/pinned.safetensors')
                self.assertEqual(old.spec.to_dict(),new.spec.to_dict())
                self.assertEqual(calls[-1]['pretrained_cfg_overlay'],{'file':'/local/pinned.safetensors'})
                x=torch.linspace(-1.,2.,2*3*8*8).reshape(2,3,8,8).requires_grad_()
                a,b=old(x),new(x)
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                grad_a,=torch.autograd.grad(a.sum(),x);grad_b,=torch.autograd.grad(b.sum(),x)
                torch.testing.assert_close(grad_a,grad_b,rtol=0,atol=0)

    def test_sim_microbatch_precision_and_constructor_rng(self):
        def constructor(family,name,**kwargs):
            torch.randn(5)  # A pretrained constructor consumes RNG before loading.
            model=FakeTimmModel(name)
            model.spec=SimpleNamespace(feature_id=f'synthetic-{family}',feature_dim=3)
            model.forward=lambda images:model.forward_features(images)[:,0]
            return model
        declarations=[SimpleNamespace(name=name) for name in ('siglip','mae','inception')]
        class Resolver:
            def __init__(self,device): pass
            def resolve_all(self,features):
                result=[]
                for feature in features:
                    module=FakeInception() if feature.name=='inception' else constructor(feature.name,sim.MODELS[feature.name])
                    result.append(SimpleNamespace(name=feature.name,extractor=module,spec=module.spec))
                return result
        namespace=dict(torch=torch,nn=nn,FEATURES=declarations,FeatureResolver=Resolver)
        reference_definitions(self.root/'src/ot_representations_sim/provider.py',{'SIMEncoders'},namespace)
        paths={name:'/local/pinned' for name in ('siglip','mae','inception')}
        for mixed in (False,True):
            with self.subTest(mixed_precision=mixed),patch.object(sim,'sim_weights',return_value=paths), \
                 patch.object(sim,'ModelFamilyExtractor',side_effect=constructor),patch.object(sim,'InceptionExtractor',FakeInception):
                torch.manual_seed(47);before=torch.get_rng_state()
                old=namespace['SIMEncoders']('cpu',microbatch=2,mixed_precision=mixed)
                self.assertTrue(torch.equal(before,torch.get_rng_state()))
                new=sim.SIMEncoders('cpu',microbatch=2,mixed_precision=mixed)
                self.assertTrue(torch.equal(before,torch.get_rng_state()))
                pixels=torch.linspace(-.8,1.8,3*3*8*8).reshape(3,3,8,8).requires_grad_()
                a,b=old(pixels),new(pixels)
                self.assertEqual(tuple(a),tuple(b))
                for name in a: torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
                grad_a,=torch.autograd.grad(sum(v.sum() for v in a.values()),pixels)
                grad_b,=torch.autograd.grad(sum(v.sum() for v in b.values()),pixels)
                torch.testing.assert_close(grad_a,grad_b,rtol=0,atol=0)
                provider_namespace=dict(torch=torch)
                reference_definitions(self.root/'src/ot_representations/provider.py',{'RepresentationProvider'},provider_namespace)
                reference=provider_namespace['RepresentationProvider'].__new__(provider_namespace['RepresentationProvider']);reference.encoders=old
                cotangents={name:torch.randn_like(value) for name,value in a.items()}
                torch.testing.assert_close(reference.input_vjp(pixels,cotangents),
                                           input_vjp(new,pixels,cotangents),rtol=0,atol=0)

    def test_inception_offline_loading_outputs_identity_and_input_gradient(self):
        old_module=self.old_extractors['inception']
        with TemporaryDirectory() as folder:
            torch.manual_seed(19)
            weights=old_module._InceptionV3().state_dict()
            path=Path(folder)/'weights.pth';torch.save(weights,path)
            torch.manual_seed(31)
            with patch.object(torch.hub,'load_state_dict_from_url',return_value=weights):
                old=old_module.InceptionExtractor(model='inception',metadata={}).eval().requires_grad_(False)
            rng_old=torch.get_rng_state()
            torch.manual_seed(31)
            with patch.object(torch.hub,'load_state_dict_from_url',side_effect=AssertionError('Network access')):
                new=inception.InceptionExtractor(model='inception',metadata={},weights_path=path).eval().requires_grad_(False)
            self.assertTrue(torch.equal(rng_old,torch.get_rng_state()))
            self.assertEqual(old.spec.feature_id,new.spec.feature_id)
            calibration=json.loads((Path(__file__).parents[1]/'models/encoders/sim_calibration.json').read_text())
            self.assertEqual(new.spec.feature_id,calibration['inception']['feature_id'])
            pixels=torch.linspace(-.7,1.7,3*24*24).reshape(1,3,24,24).requires_grad_()
            for mixed in (False,True):
                with torch.autocast('cpu',dtype=torch.bfloat16,enabled=mixed):
                    a,b=old(pixels),new(pixels)
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                grad_a,=torch.autograd.grad(a.sum(),pixels)
                grad_b,=torch.autograd.grad(b.sum(),pixels)
                torch.testing.assert_close(grad_a,grad_b,rtol=0,atol=0)

    def test_reference_fetch_preserves_sorted_rows_and_pixels(self):
        # Build the lookup directly to keep this test independent of the 259-GB dataset.
        old_namespace=dict(torch=torch,np=np,pil_to_tensor=__import__('torchvision.transforms.functional',fromlist=['pil_to_tensor']).pil_to_tensor)
        reference_definitions(self.root/'src/ot_representations_full/provider.py',{'RealImages'},old_namespace)
        from concurrent.futures import ThreadPoolExecutor
        image=Image.fromarray(np.arange(256*256*3,dtype=np.uint8).reshape(256,256,3))
        dataset=[{'image':image,'label':value} for value in (2,0,1)]
        providers=[]
        for cls in (old_namespace['RealImages'],RealImages):
            reference=cls.__new__(cls)
            reference.dataset=dataset;reference.order=np.array([1,2,0]);reference.original_labels=np.array([2,0,1])
            reference.pool=ThreadPoolExecutor(max_workers=2);providers.append(reference)
        try:
            values=[provider.fetch(torch.tensor([2,0]),torch.tensor([2,0])) for provider in providers]
            torch.testing.assert_close(*values,rtol=0,atol=0)
            self.assertEqual(values[0].shape,(2,3,256,256))
            providers[1].validate_calibration(dict(sorted_row_indices=[0,2],dataset_indices=[1,0]))
            with self.assertRaisesRegex(ValueError,'dataset order'):
                providers[1].validate_calibration(dict(sorted_row_indices=[0,2],dataset_indices=[0,1]))
            with self.assertRaises(ValueError): providers[1].fetch(torch.tensor([0]),torch.tensor([2]))
        finally:
            for provider in providers: provider.pool.shutdown(wait=True)


class AssetTests(unittest.TestCase):
    def test_checksum_editions_missing_files_and_corruption(self):
        with TemporaryDirectory() as folder:
            path=Path(folder)/'weights'
            original=b'original';published=b'published'
            row=dict(bytes=len(original),sha256=hashlib.sha256(original).hexdigest(),
                     published_bytes=len(published),published_sha256=hashlib.sha256(published).hexdigest())
            for content in (original,published):
                path.write_bytes(content);self.assertEqual(assets.verify_file(path,row),path)
            path.write_bytes(b'corrupted')
            with self.assertRaises(ValueError): assets.verify_file(path,row)
            with self.assertRaises(FileNotFoundError): assets.verify_file(Path(folder)/'missing',row)

    def test_sim_paths_use_pinned_revisions_and_explicit_cache_roots(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);(root/'old/releases').mkdir(parents=True)
            records=[]
            for family,repo_id in [('siglip','timm/vit_so400m_patch16_siglip_256.v2_webli'),('mae','timm/vit_large_patch16_224.mae')]:
                path=root/'hf/hub'/('models--'+repo_id.replace('/','--'))/'snapshots/pinned/model.safetensors'
                path.parent.mkdir(parents=True);path.write_bytes(family.encode())
                records.append(dict(path=str(path),repo_id=repo_id,revision='pinned',filename='model.safetensors',
                                    bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            path=root/'torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth'
            path.parent.mkdir(parents=True);path.write_bytes(b'inception')
            records.append(dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            (root/'old/releases/encoders.json').write_text(json.dumps({'files':records}))
            with patch.object(assets,'ROOT',root):
                paths=assets.sim_weights(hf_home=root/'hf',torch_home=root/'torch')
                self.assertEqual(set(paths),{'siglip','mae','inception'})
                self.assertIn('/snapshots/pinned/',str(paths['mae']))
                paths['mae'].write_bytes(b'changed')
                with self.assertRaises(ValueError): assets.sim_weights(hf_home=root/'hf',torch_home=root/'torch')


if __name__=='__main__': unittest.main()
