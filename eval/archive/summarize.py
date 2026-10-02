"""Validate completed 50k replicates and write mean +/- sample SD."""
import csv,json,statistics,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent
MODELS=['wflow_xl','drifting_b','drifting_l','pmf_l','pmf_h','jit_h','imf_xl','priming_b_80','priming_b_160','priming_l_80','priming_l_160']
rows=[]; raw=[]
for model in MODELS:
 results=[]
 for seed in (1,2,3):
  p=HERE/'reports'/f'{model}_n50000_seed{seed}.json'
  if not p.exists(): continue
  j=json.loads(p.read_text()); assert j['num_images']==50000 and j['seed']==seed and j['model']==model
  assert j['official_reference_sha256']=='412046720c0d496dc6d72a30eac3e22b9ef6ddac3501cce7f996674ad227ba4c'
  assert len(j['fd_legacy'])==5 and 'convnext' not in j['fd_legacy']
  results.append(j); raw.append(dict(model=model,seed=seed,fid=j['fid'],fdr5=j['fdr5']))
 row=dict(model=model,completed=len(results))
 for metric in ['fid','fdr5']:
  values=[j[metric] for j in results]
  row[metric+'_mean']=statistics.mean(values) if len(values)==3 else ''
  row[metric+'_sd']=statistics.stdev(values) if len(values)==3 else ''
 rows.append(row)
for name,data,fields in [('summary.csv',rows,list(rows[0])),('per_seed.csv',raw,['model','seed','fid','fdr5'])]:
 with (HERE/'reports'/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)
text=['# Three-seed ImageNet evaluation','',f'Updated {datetime.datetime.now(datetime.timezone.utc).isoformat()}. Complete: {len(raw)}/33 replicates.','',
'50,000 balanced images per sampling seed (1, 2, 3), one fixed released/terminal EMA checkpoint per model. SD is sample SD across sampling seeds, not training seeds. No pooled-sample FID.','',
'FID uses pinned LTH14 torch-fidelity, FP32 features without TF32, rounded uint8 images, and the official JiT ImageNet-256 reference. FDr5 retains the separate frozen local definition (BF16 feature extraction on float images, FP64 moments, ADM Inception component, excludes ConvNeXt). Official FID is not substituted into the FDr5 composite.','',
'| Model | Completed seeds | FID mean ± SD | FDr5 mean ± SD |','|---|---:|---:|---:|']
for row in rows:
 values=[f"{row[m+'_mean']:.4f} ± {row[m+'_sd']:.4f}" if row['completed']==3 else 'pending' for m in ['fid','fdr5']]
 text.append(f"| {row['model']} | {row['completed']}/3 | {values[0]} | {values[1]} |")
(HERE/'reports/RESULTS.md').write_text('\n'.join(text)+'\n')
print('\n'.join(text))
