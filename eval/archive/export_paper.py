"""Export only after all 33 independent 50k replicates pass validation."""
import csv,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
summary=list(csv.DictReader((HERE/'reports/summary.csv').open()))
assert len(summary)==11 and all(int(r['completed'])==3 for r in summary), 'All 33 benchmark replicates must complete first'
old={r['model']:r for r in csv.DictReader((ROOT/'final_paper/data/headline_all_models.csv').open())}
names={'wflow_xl':'WF-XL/2','drifting_b':'Drifting-B/16','drifting_l':'Drifting-L/16','pmf_l':'pMF-L/16','pmf_h':'pMF-H/16','jit_h':'JiT-H/16','imf_xl':'iMF-XL/2',**{f'priming_{s}_{e}':f'Priming-{s.upper()}-{e}' for s in ('b','l') for e in ('80','160')}}
rows=[]
for r in summary:
 key=r['model']; name=names[key]
 if key=='wflow_xl':
  source=ROOT/'final_paper/data/wflow_metrics.csv'; meta=next(x for x in csv.DictReader(source.open()) if x['model']=='W-Flow-XL/2')
 elif key=='imf_xl':
  source=ROOT/'final_paper/data/imf_xl_training_flops.json'; meta={'eflops':json.loads(source.read_text())['total_eflops'],'space':'latent','compute_source':str(source)}
 else:
  meta=old[name.replace('Priming','OST')]; source=ROOT/'final_paper/data/headline_all_models.csv'
 report=json.loads((HERE/'reports'/f'{key}_n50000_seed1.json').read_text())
 rows.append(dict(model=name,space=meta['space'],fid_mean=float(r['fid_mean']),fid_sd=float(r['fid_sd']),fdr5_mean=float(r['fdr5_mean']),fdr5_sd=float(r['fdr5_sd']),sampling_seeds='1;2;3',num_images_per_seed=50000,nfe=198 if key=='jit_h' else 1,eflops=float(meta['eflops']),cfg=report['cfg'],checkpoint=report['checkpoint'],fid_protocol='LTH14 FP32 uint8 / JiT ImageNet256 reference',fdr5_protocol='frozen BF16 float / ADM Inception / excludes ConvNeXt',metric_source=str(HERE/'reports'),compute_source=meta.get('compute_source',str(source))))
path=ROOT/'final_paper/data/imagenet_three_seed_20260925.csv'
with path.open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
lines=['% Three sampling seeds, 50k per seed, mean +/- sample SD. Fixed trained checkpoints.', '% FID: official FP32 uint8/JiT reference. FDr5: separately frozen legacy definition.', '% Training EFLOPs copied from existing audited compute sources; not remeasured.', r'\begin{tabular}{lrrr}', r'\toprule', r'Model & FID $\downarrow$ & FDr5 $\downarrow$ & Training EFLOPs \\', r'\midrule']
for r in rows:
 lines.append(f"{r['model']} & ${r['fid_mean']:.3f} \\pm {r['fid_sd']:.3f}$ & ${r['fdr5_mean']:.3f} \\pm {r['fdr5_sd']:.3f}$ & {r['eflops']:.2f} \\\\")
lines += [r'\bottomrule',r'\end{tabular}']
(HERE/'reports/paper_table.tex').write_text('\n'.join(lines)+'\n')
print('Exported',path)
