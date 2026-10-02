"""Adapters around pinned, unmodified upstream baseline sampling code."""
import contextlib, hashlib, subprocess, sys
from pathlib import Path
from types import SimpleNamespace, ModuleType
import torch
HERE=Path(__file__).resolve().parent
ASSETS=Path('/n/netscratch/sham_lab/Lab/jaykim/three_seed_eval_20260925/assets/checkpoints/base')
EXPECTED={
 'pmf_l':('pMF-L_256.pth','930eea77d2874031740af7c56b24e440520e820f8c298ed0f16a728ec512f1d3'),
 'pmf_h':('pMF-H_256.pth','0822f66af5be5d0a2d7e127779ade472e0a0a04c6b6db02d42e7af29a1377e99'),
 'imf_xl':('iMF-XL.pth','8a0b02d2fbd640e3a6d045d088946ef300314460a7e5d43bcad15d086095da67'),
 'jit_h':('JiT-H.pth','f89a34c373fce0d5189fad4c8802119028837be94634e535794308ccbcfe2039'),
}

def load_baseline(key):
 filename,expected=EXPECTED[key]; ckpt=ASSETS/filename
 repo='pMF' if key.startswith('pmf') else 'iMF' if key=='imf_xl' else 'JiT'
 vendor=HERE/'vendor'/repo; sys.path.insert(0,str(vendor))
 # Upstream repos use namespace directories; prevent W-Flow's regular packages
 # with the same names from winning Python's import search.
 for package in ('models','utils') if repo!='JiT' else ('util',):
  for name in list(sys.modules):
   if name==package or name.startswith(package+'.'): del sys.modules[name]
  module=ModuleType(package); module.__path__=[str(vendor/package)]
  sys.modules[package]=module
 state=torch.load(ckpt,map_location='cpu',mmap=True,weights_only=False)
 print('CHECKPOINT',key,'top_keys',list(state)[:6],flush=True)
 metadata=dict(checkpoint=str(ckpt),expected_checkpoint_sha256=expected,checkpoint_release='jjiaweiyang/FD-Loss@88ce2bb4061b0dd58839a1827235ffcffa51eed2',official_sampler_source=str(vendor),adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 if repo!='JiT':
  metadata['official_sampler_commit']=subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()
  from utils.torch_util import BatchGenerator
  if repo=='pMF':
   from pmf import pixelMeanFlow
   net=pixelMeanFlow(f'pmfDiT_{key[-1].upper()}_16',img_size=256)
   cfg=7.; low=.2; high=.7 if key=='pmf_l' else .6
   decode=lambda z:z
  else:
   from imf import iMeanFlow
   from utils.vae_util import VAEWrapper
   net=iMeanFlow('imfDiT_XL_2')
   vae=VAEWrapper(decode_batch_size=32)
   # PyTorch 2.6 + diffusers 0.38 compilation of the accelerate wrapper
   # raises NameError(torch). Keep the exact upstream decoder in eager FP32.
   vae.compiled_decode=vae.vae.decode
   metadata['vae_execution']='official VAEWrapper normalization + eager FP32 decode (compiler compatibility)'
   metadata['vae_revision']='31f26fdeee1355a5c34592e401dd41e45d25a493'
   decode=vae.decode
   cfg=8.; low=.42; high=.62
  # These are byte-identical to the original authors' HF release (LFS hashes).
  net.load_state_dict(state,strict=True,assign=True)
  net.cuda().eval().requires_grad_(False)
  def sample(labels,seed,start):
   # Disjoint sample RNG seeds, unlike XOR with 1/2/3 (which permutes the same seeds).
   rng=BatchGenerator(device=labels.device,seeds=torch.arange(start,start+len(labels))+seed*1000003)
   z=net.generate(n_sample=len(labels),rng=rng,num_steps=1,omega=cfg,t_min=low,t_max=high,labels=labels)
   return ((decode(z).float()+1)/2).clamp(0,1)
  metadata.update(cfg=cfg,interval=[low,high],sampling_steps=1,ema='released_inference',generation_precision='upstream FP32/TF32',noise_scale=getattr(net,'noise_scale',1.),rng='upstream CPU BatchGenerator; per-image seed = 1000003 * replicate + image_index')
 else:
  from denoiser import Denoiser
  args=SimpleNamespace(model='JiT-H/16',img_size=256,class_num=1000,attn_dropout=0.,proj_dropout=.2,label_drop_prob=.1,P_mean=-.8,P_std=.8,t_eps=.05,noise_scale=1.,ema_decay1=.9999,ema_decay2=.9996,sampling_method='heun',num_sampling_steps=50,cfg=2.2,interval_min=.1,interval_max=1.)
  net=Denoiser(args)
  assert 'model_ema1' in state, 'Official JiT evaluation requires first EMA weights'
  net.load_state_dict(state['model_ema1'],strict=True,assign=True)
  net.cuda().eval().requires_grad_(False)
  def sample(labels,seed,start):
   with torch.autocast('cuda',dtype=torch.bfloat16): z=net.generate(labels)
   return ((z.float()+1)/2).clamp(0,1)
  metadata.update(cfg=2.2,interval=[.1,1.],sampling_steps=50,solver='Heun; last Euler',ema='model_ema1',generation_precision='upstream BF16 autocast',noise_scale=1.,official_sampler_commit='cbc743a2ada5e9762697da2c83f8c4f8379e8c17')
 return sample,contextlib.nullcontext(),metadata
