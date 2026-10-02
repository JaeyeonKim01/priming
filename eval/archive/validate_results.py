"""Independent artifact checks; no regeneration or metric retuning."""
import hashlib,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
OUTPUT=Path('/n/netscratch/sham_lab/Lab/jaykim/three_seed_eval_20260925')
DENOM={'FID(ADM)':1.68,'dinov2_cls':14.19,'mae_cls':.04,'siglip_cls':.60,'clip_cls':5.60}
CFG={'wflow_xl':1.09,'drifting_b':1.,'drifting_l':1.,'pmf_l':7.,'pmf_h':7.,'imf_xl':8.,'jit_h':2.2,'priming_b_80':1.,'priming_b_160':1.,'priming_l_80':1.,'priming_l_160':1.}
code_hashes={hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['evaluate.py','evaluate_v2.py']}
by_model={}; checked=[]
for p in sorted((HERE/'reports').glob('*_n50000_seed*.json')):
 j=json.loads(p.read_text()); name=j['model']; seed=j['seed']; world=j['world_size']
 assert name in CFG and seed in [1,2,3] and j['num_images']==50000
 assert j['cfg']==CFG[name] and j['script_sha256'] in code_hashes
 assert j['official_reference_sha256']=='412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c'
 assert set(j['fd_legacy'])==set(DENOM)
 assert math.isclose(j['fdr5'],sum(j['fd_legacy'][k]/v for k,v in DENOM.items())/5,abs_tol=1e-10)
 assert math.isfinite(j['fid']) and j['fid']>0
 out=OUTPUT/name/f'n50000_seed{seed}'
 assert json.loads((out/'results.json').read_text())==j
 ids=np.concatenate([np.load(out/f'ids_rank{r}.npy') for r in range(world)])
 assert np.array_equal(np.sort(ids),np.arange(50000))
 assert np.array_equal(np.bincount(ids//50),np.full(1000,50))
 ranks=[json.loads((out/f'rank{r}.json').read_text()) for r in range(world)]
 assert sum(r['num_images'] for r in ranks)==50000
 hashes=tuple(r['image_bytes_sha256'] for r in ranks)
 stats=dict(np.load(out/'official_generated_stats.npz'))
 assert stats['mu'].shape==(2048,) and stats['sigma'].shape==(2048,2048)
 assert all(np.isfinite(x).all() for x in stats.values())
 peers=by_model.setdefault(name,[])
 for peer in peers:
  assert peer['hashes']!=hashes,'Repeated image stream across seeds'
  assert peer['checkpoint']==j['checkpoint']
 peers.append(dict(seed=seed,hashes=hashes,checkpoint=j['checkpoint']))
 checked.append(dict(model=name,seed=seed,sample_count=50000,world_size=world))
report=dict(checked_replicates=len(checked),expected_replicates=33,complete=len(checked)==33,models_complete=sum(len(rows)==3 for rows in by_model.values()),checks=['exact sample-ID coverage','50 images per class','per-rank counts','distinct generated streams across seeds','fixed checkpoints and CFG','common official reference hash','evaluation source hashes','raw FDr5 aggregation','finite FID and generated moments','scratch/report parity'],replicates=checked)
(HERE/'reports/validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='replicates'},indent=2))
