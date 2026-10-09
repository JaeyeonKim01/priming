"""Post-training numerical comparisons with the untouched reference checkout."""
import ast
from copy import deepcopy
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from tempfile import TemporaryDirectory
import numpy as np
import random
from unittest.mock import patch
import utils.collectives as collectives
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn

from data.imagenet import ClasswiseOTStream
from losses.discrepancy import FeatureDiscrepancy
from losses.matching import Matching
from utils.collectives import DistributedContext
from utils.ema import EMA
from utils.posttrain_checkpoint import initialize_posttraining, save_posttraining, resume_posttraining, snapshot_rng
import posttrain
from test_pretrain import reference_definitions


class TinyGenerator(nn.Module):
    in_channels = 1
    input_size = 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.7))

    def sample_images_with_grad(self, noise, labels, sampling_args):
        return noise * self.weight + labels[:, None, None, None] * .01


class Features:
    sigmas = {"squared": 1.3, "linear": .7}
    weights = {"squared": .25, "linear": .75}
    representation_swap = True

    def running_state_dict(self):
        return {"format": "synthetic-frozen-features", "version": 1}

    def load_running_state_dict(self, state):
        if state != self.running_state_dict():
            raise ValueError("Feature contract differs")

    def extract_features(self, images, **kwargs):
        rows = images.flatten(1)
        return {"squared": rows.square(), "linear": rows * 1.7}

    def target_features(self, batch):
        pixels = batch.target_indices.float()[:, None, None, None].expand(-1, 1, 2, 2) / 20
        return self.extract_features(pixels)

    def input_vjp(self, images, cotangents):
        pixels = images.detach().requires_grad_(True)
        values = self.extract_features(pixels)
        return torch.autograd.grad(tuple(values.values()), pixels,
                                   grad_outputs=tuple(cotangents[k] for k in values))[0]


class Encoder(nn.Module):
    family_names = ('all',)
    microbatch = 6

    def forward(self, images):
        return Features().extract_features(images)

    def encode_family(self, images, family):
        return self(images)


class References:
    def fetch(self, indices, labels):
        return indices.float()[:,None,None,None].expand(-1,1,2,2)/20


class PosttrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("PRIMING_REFERENCE_ROOT"):
            raise unittest.SkipTest("Set PRIMING_REFERENCE_ROOT for old/new comparisons")
        cls.root = Path(os.environ["PRIMING_REFERENCE_ROOT"])
        cls.package = cls.root/'training/runtime/packages/distmatch/src/distmatch'
        ns = dict(torch=torch, math=math, Mapping=Mapping, dataclass=dataclass,
                  DistributedContext=DistributedContext,
                  _INTEGER_DTYPES=frozenset({torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64}))
        reference_definitions(cls.package/'objectives/sinkhorn_ot.py',
                              {'SinkhornOTResult','BalancedSinkhornOT','balanced_sinkhorn_plan'}, ns)
        reference_definitions(cls.package/'objectives/loss_ablation.py',
                              {'LossAblationOT','featurewise_sinkhorn_plan','_laplacian_cost'}, ns)
        ns['_CDIST_PAIRWISE_BYTE_BUDGET'] = 1 << 30
        cls.old_loss = ns['LossAblationOT']
        stream_ns = dict(torch=torch, dataclass=dataclass, Mapping=Mapping, Sequence=Sequence, Any=Any)
        reference_definitions(cls.root/'training/runtime/src/utils/ot_util.py',
                              {'ClasswiseOTBatch','ClasswiseOTStream'}, stream_ns)
        cls.old_stream = stream_ns['ClasswiseOTStream']
        torch.set_num_threads(2)

    def stream(self, old=False):
        return (self.old_stream if old else ClasswiseOTStream)(class_counts=[10]*4,
            global_classes_per_step=2, samples_per_class=3, real_references_per_class=4,
            rank=0, world_size=1, seed=7919, device='cpu')

    def options(self, kind):
        return dict(bandwidths=Features.sigmas, weights=Features.weights,
                    discrepancy_kind=kind, energy_delta=.04549544751644135,
                    global_classes_per_step=2, samples_per_class=3,
                    real_references_per_class=4, epsilon=.001,
                    epsilon_mixture={"0.001": .75, "0.01": .25},
                    num_iterations=40, tolerance=1e-6, distributed=False)

    def test_discrepancies_values_plans_metrics_and_gradients(self):
        for kind in ('featurewise_ot','featurewise_energy','featurewise_laplacian_mmd'):
            with self.subTest(kind=kind):
                torch.manual_seed(3)
                features = {k: torch.randn(6,5,requires_grad=True) for k in Features.sigmas}
                targets = {k: torch.randn(8,5,requires_grad=True) for k in Features.sigmas}
                kwargs = dict(generated_labels=torch.tensor([0]*3+[2]*3),
                              target_labels=torch.tensor([0]*4+[2]*4))
                results = [loss(**self.options(kind)).score_features(features,targets,**kwargs)
                           for loss in (self.old_loss,FeatureDiscrepancy)]
                for attribute in ('loss','cost_matrix','transport_plan'):
                    torch.testing.assert_close(getattr(results[0],attribute),getattr(results[1],attribute),rtol=0,atol=0)
                self.assertEqual(set(results[0].metrics),set(results[1].metrics))
                for name,value in results[0].metrics.items():
                    torch.testing.assert_close(value,results[1].metrics[name],rtol=0,atol=0)
                gradients = [torch.autograd.grad(r.loss,tuple(features.values()),retain_graph=True) for r in results]
                for old,new in zip(*gradients): torch.testing.assert_close(old,new,rtol=0,atol=0)
                self.assertTrue(all(v.grad is None for v in targets.values()))

    def test_sampler_private_rng_and_resume(self):
        old,new=self.stream(True),self.stream()
        torch.manual_seed(13); before=torch.get_rng_state()
        for _ in range(4):
            a,b=old.next_batch(),new.next_batch()
            for name in ('generated_labels','target_indices','target_labels','class_ids','global_class_ids'):
                torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        torch.randn(3,generator=new.negative_noise_generator)
        saved=new.state_dict(); resumed=self.stream();resumed.load_state_dict(saved)
        torch.testing.assert_close(new.next_batch().target_indices,resumed.next_batch().target_indices,rtol=0,atol=0)
        torch.testing.assert_close(torch.randn(3,generator=new.negative_noise_generator),
                                   torch.randn(3,generator=resumed.negative_noise_generator),rtol=0,atol=0)

    def test_full_update_against_old_gradient_replay(self):
        # Execute the original production branch itself; stub only frozen encoders.
        ns = dict(torch=torch, argparse=SimpleNamespace(Namespace=object),Mapping=Mapping,
                  Sequence=Sequence,RolloutChunk=tuple,MMD=object,FullClassAttraction=object,
                  BalancedSinkhornOT=object,ClasswiseOTStream=object,ClasswiseOTBatch=object,
                  FrozenJointSIMIntervention=object,sync_gradients=lambda *a,**kw:None)
        def extract(args,model,tokenizer,sampling,provider,chunk,**kwargs):
            noise,labels=chunk
            return provider.extract_features(model.sample_images_with_grad(noise,labels,sampling_args=sampling)*.5+.5)
        ns['extract_chunk_features']=extract
        reference_definitions(self.root/'training/runtime/scripts/main_mmd_and_eval.py',{'backward_ot_rollout'},ns)
        args=SimpleNamespace(ot_log_grad_norms=False,batch_size=6,noise_scale=1.,
                             ot_baseline_weight=0.,lambda_ot=5.,amp_dtype=torch.bfloat16,enable_amp=False)
        for kind in ('featurewise_ot','featurewise_energy','featurewise_laplacian_mmd'):
            with self.subTest(kind=kind):
                old,new=TinyGenerator(),TinyGenerator();provider=Features()
                streams=[self.stream(True),self.stream()]
                objectives=[]
                for loss in (self.old_loss,FeatureDiscrepancy):
                    options=self.options(kind);cross=loss(**options)
                    options['real_references_per_class']=3
                    objectives.append((cross,loss(**options)))
                torch.manual_seed(41);noise=torch.randn(6,1,2,2);initial=torch.get_rng_state()
                batch=streams[0].next_batch()
                old_result=ns['backward_ot_rollout'](args,old,None,{},provider,None,*objectives[0],
                            streams[0],[(noise,batch.generated_labels)],batch,replay=False)
                old_rng=torch.get_rng_state();torch.set_rng_state(initial)
                batch=streams[1].next_batch()
                matcher=Matching(Encoder(),bandwidths=Features.sigmas,weights=Features.weights,kind=kind,
                                 global_classes=2,samples=3,references=4,iterations=40,distributed=False)
                new_result=posttrain.backward_rollout(new,matcher,References(),streams[1],batch,noise,{},amp=False)
                torch.testing.assert_close(old_result[0],new_result[0],rtol=0,atol=0)
                torch.testing.assert_close(old.weight.grad,new.weight.grad,rtol=0,atol=0)
                self.assertEqual(set(old_result[2]),set(new_result[1]))
                for name,value in old_result[2].items():
                    torch.testing.assert_close(value,new_result[1][name],rtol=0,atol=0)
                self.assertTrue(torch.equal(old_rng,torch.get_rng_state()))
                self.assertTrue(torch.equal(streams[0].negative_noise_generator.get_state(),
                                            streams[1].negative_noise_generator.get_state()))

    def test_matcher_sample_interface_detaches_references_and_freezes_encoders(self):
        class StatefulEncoder(Encoder):
            def __init__(self):
                super().__init__()
                self.norm=nn.BatchNorm2d(1)
                self.dropout=nn.Dropout(.7)
            def forward(self,images):
                return super().forward(self.dropout(self.norm(images)))
        encoder=StatefulEncoder()
        matcher=Matching(encoder,bandwidths=Features.sigmas,weights=Features.weights,
                         kind='featurewise_energy',global_classes=2,samples=3,references=4,
                         distributed=False)
        matcher.train()
        self.assertTrue(matcher.training)
        self.assertFalse(encoder.training)
        before={name:value.clone() for name,value in encoder.named_buffers()}
        x=torch.linspace(.1,.9,24).reshape(2,3,1,2,2).requires_grad_()
        x_prime=torch.linspace(-.2,.5,24).reshape(2,3,1,2,2).requires_grad_()
        y=torch.linspace(.6,1.2,32).reshape(2,4,1,2,2).requires_grad_()
        loss,metrics=matcher(x=x,x_prime=x_prime,y=y)
        (3*loss).backward()
        self.assertIsNone(x_prime.grad);self.assertIsNone(y.grad)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in encoder.parameters()))
        self.assertTrue(all(torch.equal(before[name],value) for name,value in encoder.named_buffers()))
        self.assertTrue(all(not value.requires_grad for value in metrics.values()))
        # Energy has no detached transport plan, so finite differences check the
        # sample-gradient replay itself, including the caller's loss multiplier.
        plus=x.detach().clone();minus=plus.clone()
        plus.flatten()[5]+=.001;minus.flatten()[5]-=.001
        with torch.no_grad():
            a,_=matcher(plus,x_prime,y);b,_=matcher(minus,x_prime,y)
        finite_difference=3*(a-b)/.002
        torch.testing.assert_close(x.grad.flatten()[5],finite_difference,rtol=.01,atol=.001)
        self.assertFalse(a.requires_grad)
        with self.assertRaisesRegex(ValueError,'Real samples'):
            matcher(x,x_prime,y[:,:3])

    def test_scalar_local_backward_and_final_parameter_sum(self):
        def double(tensor, **kwargs):
            tensor.mul_(2)
        with patch.object(collectives.dist,'is_initialized',return_value=True), \
             patch.object(collectives.dist,'get_world_size',return_value=2), \
             patch.object(collectives.dist,'get_rank',return_value=0), \
             patch.object(collectives.dist,'all_reduce',side_effect=double):
            value=torch.tensor(3.,requires_grad=True)
            result=DistributedContext(True).differentiable_sum(value)
            self.assertEqual(float(result),6.)
            result.backward()
            self.assertEqual(float(value.grad),1.)
            parameter=nn.Parameter(torch.tensor(0.));parameter.grad=torch.tensor(3.)
            collectives.sync_gradients([parameter],reduction='sum')
            self.assertEqual(float(parameter.grad),6.)

    def test_zero_based_warmup_clipping_and_ema(self):
        model=TinyGenerator();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.95),weight_decay=0.)
        ema=EMA(model,decay=.999);original=model.weight.detach().clone()
        model.weight.grad=torch.tensor(2.)
        norm,lr=posttrain.optimizer_step(model,optimizer,ema,0)
        self.assertEqual(lr,0);self.assertEqual(float(norm),2.)
        torch.testing.assert_close(model.weight,original,rtol=0,atol=0)
        self.assertEqual(ema.step_count,1)
        for step,expected in ((1,1e-5/125),(125,1e-5),(126,1e-5)):
            model.weight.grad=torch.tensor(.5)
            _,lr=posttrain.optimizer_step(model,optimizer,ema,step)
            self.assertEqual(lr,expected)
        model.weight.grad=torch.tensor(float('nan'));count=ema.step_count
        with self.assertRaises(FloatingPointError): posttrain.optimizer_step(model,optimizer,ema,127)
        self.assertEqual(ema.step_count,count)


    def test_initialization_and_exact_update_recovery(self):
        options = dict(optimizer='adamw',lr=1e-5,beta1=.9,beta2=.95,weight_decay=0.)
        with TemporaryDirectory() as folder:
            source = Path(folder)/'pretrained.pth'
            model = TinyGenerator()
            pretrained_ema = EMA(model,decay=.9999)
            pretrained_ema.shadow['weight'].fill_(.3)
            pretrained_ema.step_count=7
            source_contract=dict(model=dict(model='OST_B',patch_size=16,class_tokens=8,guidance_tokens=0,
                                  interval_tokens=0,label_drop_prob=0.,rope_2d=True,learned_pe=True),
                                  data={'num_classes':1000},loss={'aux':{'enabled':True}})
            torch.save(dict(model=model.state_dict(),model_ema=pretrained_ema.state_dict(),step=7,current_step=8,
                            contract=source_contract),source)
            optimizer,ema=initialize_posttraining(model,source,options,config=source_contract)
            self.assertFalse(optimizer.state)
            self.assertEqual(ema.step_count,7)
            self.assertEqual(ema.label,'0.999')
            self.assertEqual(float(ema.shadow['weight']),float(torch.tensor(.3)))
            stream=self.stream();provider=Features()
            matcher=Matching(Encoder(),bandwidths=Features.sigmas,weights=Features.weights,
                             global_classes=2,samples=3,references=4,iterations=40,distributed=False)
            config={'kind':'featurewise_ot','seed':1}
            torch.manual_seed(29);np.random.seed(29);random.seed(29)
            for step in range(3):
                posttrain.train_step(model,matcher,References(),stream,{},amp=False)
                posttrain.optimizer_step(model,optimizer,ema,step)
            saved=Path(folder)/'resume.pth'
            save_posttraining(saved,model,optimizer,ema,matcher,stream,3,config=config)
            expected_loss,expected_metrics=posttrain.train_step(model,matcher,References(),stream,{},amp=False)
            posttrain.optimizer_step(model,optimizer,ema,3)
            expected_rng=snapshot_rng()
            resumed=TinyGenerator();new_stream=self.stream()
            opt,new_ema,completed=resume_posttraining(saved,resumed,matcher,new_stream,options,config=config)
            self.assertEqual(completed,3)
            actual_loss,actual_metrics=posttrain.train_step(resumed,matcher,References(),new_stream,{},amp=False)
            posttrain.optimizer_step(resumed,opt,new_ema,3)
            torch.testing.assert_close(expected_loss,actual_loss,rtol=0,atol=0)
            torch.testing.assert_close(model.weight,resumed.weight,rtol=0,atol=0)
            torch.testing.assert_close(ema.shadow['weight'],new_ema.shadow['weight'],rtol=0,atol=0)
            for key,value in optimizer.state[model.weight].items():
                torch.testing.assert_close(value,opt.state[resumed.weight][key],rtol=0,atol=0)
            self.assertTrue(torch.equal(expected_rng['torch'],snapshot_rng()['torch']))
            for key,value in expected_metrics.items():
                torch.testing.assert_close(value,actual_metrics[key],rtol=0,atol=0)
            bad=deepcopy(config);bad['seed']=2
            with self.assertRaisesRegex(ValueError,'recipe'):
                resume_posttraining(saved,TinyGenerator(),matcher,self.stream(),options,config=bad)
            payload=torch.load(saved,weights_only=False)
            del payload['sinkhorn_ot_stream_state']['rank_states'][0]['negative_noise_generator_state']
            torch.save(payload,saved)
            with self.assertRaisesRegex(ValueError,'private'):
                resume_posttraining(saved,TinyGenerator(),matcher,self.stream(),options,config=config)


if __name__ == '__main__': unittest.main()
