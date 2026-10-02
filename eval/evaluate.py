"""Three sampling seeds, paired official FID and frozen legacy FDr5."""
import argparse, contextlib, csv, hashlib, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from priming_paths import asset, configure_caches, resolve
configure_caches()
HERE=Path(__file__).resolve().parent
RUNTIME=HERE/'runtime'
CACHE=asset('PRIMING_COMPARISON_CACHE','assets/baselines/comparison')
VENDOR=HERE/'vendor/torch-fidelity'
for p in (HERE/'vendor/W-Flow', RUNTIME, RUNTIME/'packages/distmatch/src', VENDOR): sys.path.insert(0,str(p))
os.environ['DRIFT_COMPILE']='0'
import numpy as np
import torch
import torch.distributed as dist

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()

def load_generator(key):
 if key.startswith('priming'):
  from src.models import OSTDenoiser_models
  from src.utils.sampling_util import generate_images
  item=json.loads((ROOT/'checkpoints/manifest.json').read_text())[key]
  ckpt=ROOT/item['checkpoint']
  argsfile=ROOT/item['config']
  config=json.loads(argsfile.read_text())
  assert ckpt.stat().st_size==item['checkpoint_bytes']
  if dist.get_rank()==0: assert digest(ckpt)==item['checkpoint_sha256']
  dist.barrier()
  args=SimpleNamespace(**config['sampling_args'])
  args.enable_amp,args.amp_dtype=True,torch.bfloat16
  assert args.num_sampling_steps==1 and config['completed_updates']==25000
  net=OSTDenoiser_models[args.model](
   img_size=args.img_size,patch_size=args.patch_size,in_channels=args.token_channels,
   tokenizer_patch_size=1,num_classes=args.num_classes,label_drop_prob=args.label_drop_prob,
   corruption='denoising',t_dist='lognorm',t_min=0.,t_max=1.,
   P_mean=args.P_mean,P_std=args.P_std,cfg_beta=args.cfg_beta,
   cfg_omega_max=args.cfg_omega_max,class_tokens=args.class_tokens,
   guidance_tokens=args.guidance_tokens,interval_tokens=args.interval_tokens,
   rope_2d=args.rope_2d,learned_pe=args.learned_pe,noise_scale=args.noise_scale,
   norm_eps=args.norm_eps,norm_p=args.norm_p,perceptual_threshold=args.perceptual_threshold)
  state=torch.load(ckpt,map_location='cpu',mmap=True,weights_only=False)
  assert state['step']==24999 and state['current_step']==25000
  assert state['model_ema']['schedules']==[('0.999',0.999)]
  shadow=state['model_ema']['shadows']['0.999']
  assert set(shadow)==set(state['model'])
  net.load_state_dict(shadow,strict=True,assign=True)
  net.cuda().eval().requires_grad_(False)
  def sample(labels,seed,start): return generate_images(args,net,labels=labels,cfg=1.)
  return sample,contextlib.nullcontext(),dict(checkpoint=str(ckpt),checkpoint_sha256=item['checkpoint_sha256'],cfg=1.,ema='0.999',args_source=str(argsfile),args_sha256=digest(argsfile))
 if key in ('wflow_xl','drifting_b','drifting_l'):
  import yaml
  from models.generator import DitGen
  if key=='wflow_xl':
   ckpt=CACHE/'wflow/checkpoints/latent_sota_XL_ot/state_00180000.pt'
   config=HERE/'vendor/W-Flow/configs/gen/latent_sota_XL_ot_8node.yaml'
   config=yaml.safe_load(config.read_text())['model']; config={**config,'num_classes':1000}
   cfg=1.09
  else:
   ckpt=CACHE/f'drifting/models/gen/jax/pixel_{key[-1].upper()}_sota/converted_ema.pt'
   cfg=1.
  state=torch.load(ckpt,map_location='cpu',mmap=True,weights_only=False)
  if key!='wflow_xl': config=state['metadata']['model_config']
  net=DitGen(**config); net.load_state_dict(state['ema_model'],strict=True,assign=True)
  net.cuda().eval().requires_grad_(False)
  vae=None
  if key=='wflow_xl':
   from diffusers import AutoencoderKL
   vae=AutoencoderKL.from_pretrained(CACHE/'vae',local_files_only=True).cuda().eval().requires_grad_(False)
  def sample(labels,seed,start):
   rng=torch.Generator(device='cuda').manual_seed(seed ^ start)
   z=net(c=labels,cfg_scale=cfg,deterministic=True,train=False,rng=rng)['samples'].permute(0,3,1,2).contiguous().float()
   if vae is not None: z=vae.decode(z/.18215).sample
   return ((z+1)/2).clamp(0,1)
  return sample,contextlib.nullcontext(),dict(checkpoint=str(ckpt),cfg=cfg,ema='released_ema',sampler='official DitGen; upstream batch-index RNG; PNG writes omitted')
 from baseline_loaders import load_baseline
 return load_baseline(key)

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--num-images',type=int,default=50000); p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3]); p.add_argument('--batch',type=int,default=32); p.add_argument('--output-dir',type=Path,required=True)
 opt=p.parse_args(); opt.output_dir=resolve(opt.output_dir); rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE'])
 assert opt.num_images%1000==0
 torch.cuda.set_device(int(os.environ['LOCAL_RANK'])); dist.init_process_group('nccl'); torch.set_num_threads(4)
 from scripts import main_mmd_and_eval as mmd
 from distmatch.objectives.frechet.evaluation import FrechetEvaluator
 from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
 from torch_fidelity.metric_fid import fid_features_to_statistics,fid_statistics_to_metric
 mmd.STATS_DIR=asset('PRIMING_FDR5_STATS','assets/fid_stats')
 specs=tuple(s for s in mmd.FDR6_SPECS if s[0]!='convnext')
 started=time.monotonic()
 evaluator=FrechetEvaluator(features=mmd._features(specs),stats=mmd._cached_stats(specs,device=torch.device('cuda')),stats_dtype=torch.float64,distributed=True,device='cuda')
 weights=asset('PRIMING_INCEPTION_WEIGHTS','assets/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth')
 official=FeatureExtractorInceptionV3('inception-v3-compat',['2048'],feature_extractor_weights_path=str(weights)).cuda().eval().requires_grad_(False)
 sample,swap,metadata=load_generator(opt.model)
 print('READY',opt.model,rank,'load_seconds',time.monotonic()-started,flush=True)
 refpath=asset('PRIMING_FID_REFERENCE','assets/fid_stats/jit_in256_stats.npz')
 assert digest(refpath)=='412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c'
 with swap:
  for seed in opt.seeds:
   out=opt.output_dir.resolve()/opt.model/f'n{opt.num_images}_seed{seed}'
   if out.exists() and any(out.iterdir()): raise RuntimeError(f'Refusing to overwrite {out}')
   out.mkdir(parents=True,exist_ok=True)
   if (out/'results.json').exists(): raise RuntimeError(f'Refusing to overwrite {out}')
   torch.manual_seed(seed*1000003+rank); np.random.seed(seed*1000003+rank)
   evaluator.reset(); parts=[]; ids=[]; ihash=hashlib.sha256(); tick=time.monotonic()
   # Same balanced class ordering and padded batches as upstream W-Flow.
   for start in range(rank*opt.batch,opt.num_images,world*opt.batch):
    valid=min(opt.batch,opt.num_images-start)
    indices=torch.arange(start,start+opt.batch,device='cuda')
    labels=(indices//(opt.num_images//1000)).clamp(max=999).long()
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    images=sample(labels,seed,start)[:valid].float().clamp(0,1)
    assert images.shape==(valid,3,256,256) and torch.isfinite(images).all()
    bytes_=images.mul(255).round().to(torch.uint8)
    with torch.autocast('cuda',dtype=torch.bfloat16): evaluator.update(images)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    with torch.autocast('cuda',enabled=False): parts.append(official(bytes_)[0].cpu())
    arr=bytes_.cpu().numpy(); ihash.update(arr.tobytes()); ids.extend(range(start,start+valid))
    if len(ids)<=opt.batch:
     np.save(out/f'first_bytes_rank{rank}.npy',arr)
    if rank==0 and (len(ids)%1024<opt.batch): print('PROGRESS',opt.model,seed,len(ids)*world,'seconds',round(time.monotonic()-tick,1),flush=True)
   duration=time.monotonic()-tick
   np.save(out/f'features_rank{rank}.npy',torch.cat(parts).numpy()); np.save(out/f'ids_rank{rank}.npy',np.array(ids))
   (out/f'rank{rank}.json').write_text(json.dumps(dict(num_images=len(ids),image_bytes_sha256=ihash.hexdigest(),sampling_and_features_seconds=duration)))
   result=evaluator.compute(); assert result.num_samples==opt.num_images
   dist.barrier()
   if rank==0:
    allids=np.concatenate([np.load(out/f'ids_rank{r}.npy') for r in range(world)])
    assert np.array_equal(np.sort(allids),np.arange(opt.num_images))
    feat=np.concatenate([np.load(out/f'features_rank{r}.npy') for r in range(world)])[np.argsort(allids)]
    stat=fid_features_to_statistics(torch.from_numpy(feat)); ref=dict(np.load(refpath))
    np.savez(out/'official_generated_stats.npz',**stat)
    stat={k:stat[k].astype(ref[k].dtype) for k in ('mu','sigma')}
    fid=fid_statistics_to_metric(stat,{k:ref[k] for k in ('mu','sigma')},verbose=True)['frechet_inception_distance']
    ratios={k:v/mmd.FDR6_VALIDATION_FD[k] for k,v in result.fd.items()}
    report=dict(model=opt.model,seed=seed,num_images=opt.num_images,rng_rank_seed=seed*1000003+rank,world_size=world,batch=opt.batch,fid=fid,fdr5=sum(ratios.values())/5,fd_legacy=result.fd,fdr_components=ratios,**metadata,sampling_and_features_seconds=duration,seconds_since_load=time.monotonic()-started,official_reference=str(refpath),official_reference_sha256=digest(refpath),script_sha256=digest(__file__),protocol='FID: LTH14 FP32 rounded uint8, JiT reference. FDr5: frozen BF16 float images, FP64 moments, excludes ConvNeXt; legacy ADM FID component retained.',gpu=torch.cuda.get_device_name())
    assert np.isfinite(fid) and np.isfinite(report['fdr5'])
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('COMPLETE',json.dumps(report),flush=True)
   dist.barrier()
 dist.destroy_process_group()
if __name__=='__main__': main()
