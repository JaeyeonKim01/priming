import argparse
import copy
import os
from pathlib import Path
import tempfile
import unittest
import torch
from omegaconf import OmegaConf
from world_modeling.losses.reconstruction import interpolate, x_prediction_velocity_loss, ode_step
from world_modeling.losses.matching import load_distmatch
from world_modeling.models.generator import load_initial_flow_weights
from world_modeling.utils.replay import backward_with_latent_replay
from world_modeling.utils.config import load_config, parser
from world_modeling.utils.training import step_generator

ROOT = Path(__file__).resolve().parents[2]

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.scale=torch.nn.Parameter(torch.tensor(.7)); self.bias=torch.nn.Parameter(torch.tensor(-.1))
    def forward(self,z,h,a): return self.scale*z+self.bias+.05*h[:,:z.shape[1]]+.01*a.mean((1,2))[:,None,None,None]

class Branches(torch.nn.Module):
    def forward(self,x): return {'spatial':x,'pooled':x.mean((2,3))}

class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(17)

    def test_weighted_reconstruction_and_gradient(self):
        clean=torch.randn(4,2,3,3); noise=torch.randn_like(clean)
        t=torch.tensor([.001,.05,.4,.99]); pred=torch.randn_like(clean,requires_grad=True)
        loss,_=x_prediction_velocity_loss(clean,interpolate(clean,noise,t),pred,t)
        expected=((pred-clean).square().flatten(1).mean(1)/t.clamp_min(.05).square()).mean()
        torch.testing.assert_close(loss,expected)
        torch.testing.assert_close(torch.autograd.grad(loss,pred,retain_graph=True)[0],torch.autograd.grad(expected,pred)[0])
        torch.testing.assert_close(ode_step(noise,pred,torch.ones(4),torch.zeros(4)),pred)

    def test_source_noise_stream(self):
        generator=step_generator('cpu',7,33,201,3)
        expected=torch.Generator().manual_seed((7+1_000_003*33+10_007*201+3)%(2**63-1))
        torch.testing.assert_close(torch.randn(12,generator=generator),torch.randn(12,generator=expected),rtol=0,atol=0)

    def test_configs_and_override_interpolation(self):
        for stage in ('pretrain','posttrain','eval'):
            for ds in ('rt1','bridge'):
                args=parser(stage).parse_args(['--config',str(ROOT/f'world_modeling/configs/{stage}/{ds}.yaml'),
                    '--storage-root','/example/assets','--output','/example/output','--check-config'])
                cfg=load_config(args,stage)
                self.assertEqual(cfg.assets.sd3,'/example/assets/models/sd3-medium-diffusers')
                self.assertTrue(cfg[ds].dataset_dir.startswith('/example/assets/'))
                self.assertEqual(cfg.output_dir,'/example/output')

    def test_ema_stage_initialization(self):
        class Pair(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.model=torch.nn.Linear(2,2); self.ema_model=copy.deepcopy(self.model)
        source=Pair()
        with torch.no_grad():
            for p in source.model.parameters():p.fill_(-3)
            for p in source.ema_model.parameters():p.fill_(7)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.pth';torch.save(dict(objective='flow_matching_x_prediction',step=9,model=source.state_dict()),path)
            model=Pair();metadata=load_initial_flow_weights(model,path)
            self.assertEqual(metadata['weights'],'ema')
            import hashlib
            self.assertEqual(metadata['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            for p in model.model.parameters():torch.testing.assert_close(p,torch.full_like(p,7))
            for p,q in zip(model.model.parameters(),model.ema_model.parameters()):torch.testing.assert_close(p,q)

    @unittest.skipUnless(os.environ.get('DISTMATCH_SOURCE'),'Set DISTMATCH_SOURCE for matcher/replay integration')
    def test_latent_replay_equals_full_population_gradient(self):
        dm=load_distmatch(os.environ['DISTMATCH_SOURCE'])
        matcher=dm.DistMatch([dm.FeatureBranches(Branches(),{'spatial':'pair_rms','pooled':'pair_rms'})],dm.Sinkhorn(),mode='conditional')
        z=torch.randn(4,16,2,3,3);zp=torch.randn_like(z);h=torch.randn(4,16,8,3,3);a=torch.randn(4,16,4,10)
        y=torch.randn(4,1,2,3,3,requires_grad=True)
        direct=Toy();replayed=copy.deepcopy(direct)
        x=direct(z.flatten(0,1),h.flatten(0,1),a.flatten(0,1)).reshape_as(z)
        xp=direct(zp.flatten(0,1),h.flatten(0,1),a.flatten(0,1)).reshape_as(z)
        loss,_=matcher(x,xp,y);loss.backward()
        value,_=backward_with_latent_replay(replayed,matcher,z,zp,h,a,y,microbatch_size=8,verify_replay=True)
        torch.testing.assert_close(value,loss.detach())
        self.assertIsNone(y.grad)
        for p,q in zip(direct.parameters(),replayed.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=1e-6)


class RecoveryTests(unittest.TestCase):
    def test_checkpoint_restores_model_optimizer_rng_and_rejects_changed_contract(self):
        from world_modeling.utils import checkpoints
        torch.manual_seed(81)
        cfg=OmegaConf.create({'stage':'pretrain'})
        model=Toy();optimizer=torch.optim.AdamW(model.parameters(),lr=.03)
        (model.scale+model.bias).square().backward();optimizer.step();optimizer.zero_grad()
        expected=copy.deepcopy(model.state_dict())
        signature={'stage':'pretrain','world_size':1}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ckpt-step1.pth'
            checkpoints.save(path,model,optimizer,step=1,epoch=0,batch_in_epoch=1,cfg=cfg,signature=signature)
            expected_random=torch.randn(4)
            with torch.no_grad():model.scale.fill_(44)
            state=checkpoints.resume(path,model,optimizer,signature,0)
            checkpoints.restore_rng(state['rng'])
            torch.testing.assert_close(torch.randn(4),expected_random)
            for key,value in expected.items():torch.testing.assert_close(model.state_dict()[key],value)
            self.assertEqual(state['step'],1)
            self.assertEqual(optimizer.state[model.scale]['step'].item(),1)
            self.assertTrue((Path(tmp)/'ckpt-latest.pth').is_symlink())
            with self.assertRaisesRegex(ValueError,'contract differs'):
                checkpoints.resume(path,model,optimizer,{'stage':'posttrain'},0)
            with self.assertRaises(FileExistsError):
                checkpoints.save(path,model,optimizer,step=1,epoch=0,batch_in_epoch=1,cfg=cfg,signature=signature)

    def test_evaluation_reuse_rejects_changed_recipe(self):
        from world_modeling.utils.eval_identity import evaluation_identity, validate_output_identity
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);checkpoint=path/'checkpoint.pth';checkpoint.write_bytes(b'fixture')
            cfg=OmegaConf.create({'checkpoint':str(checkpoint),'train':{'seed':1},'evaluation':{'max_videos_per_shard':2}})
            identity=evaluation_identity(cfg,2,'ema')
            validate_output_identity(path/'eval',identity)
            validate_output_identity(path/'eval',identity,aggregate_only=True)
            for changed in ({**identity,'weights':'online'},{**identity,'seed':3},{**identity,'world_size':8}):
                with self.assertRaisesRegex(ValueError,'different checkpoint or recipe'):
                    validate_output_identity(path/'eval',changed)

if __name__ == '__main__':
    unittest.main()
