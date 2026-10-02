"""Keep original-paper FIDs distinct from release and old-dashboard numbers."""
import csv,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
# Original-paper values and original dashboard provenance were audited in
# final_paper/METRIC_AUDIT_20260925.md. Release values from pinned READMEs.
REFS={
 'wflow_xl':(1.3476836112471915,1.29,None,'https://github.com/hanjq17/W-Flow'),
 'drifting_b':(1.5469637818006277,1.76,1.73,'https://github.com/lambertae/drifting'),
 'drifting_l':(1.426036859378371,1.61,1.62,'https://github.com/lambertae/drifting'),
 'pmf_l':(2.72,2.52,2.51,'https://github.com/Lyy-iiis/pMF/tree/990e81a84249dbd68a128accef67eb95621d10b1'),
 'pmf_h':(2.29,2.22,2.15,'https://github.com/Lyy-iiis/pMF/tree/990e81a84249dbd68a128accef67eb95621d10b1'),
 'imf_xl':(1.82,1.72,1.72,'https://github.com/Lyy-iiis/imeanflow/tree/04687983e821b3ad01f54f03dc44194a33c20c54'),
 'jit_h':(1.97,1.86,None,'https://arxiv.org/html/2511.13720v1'),
}
rows=[]
for row in csv.DictReader((HERE/'reports/summary.csv').open()):
 if row['model'] not in REFS: continue
 old,paper,release,source=REFS[row['model']]
 mean=float(row['fid_mean']) if row['fid_mean'] else None
 rows.append(dict(model=row['model'],completed=int(row['completed']),old_dashboard_fid=old,original_paper_fid=paper,official_release_fid=release,new_fid_mean=mean,new_fid_sd=float(row['fid_sd']) if mean is not None else None,delta_vs_paper=mean-paper if mean is not None else None,source=source))
(HERE/'reports/published_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
text=['# Original-paper comparison','','Original paper, official release reproduction, historical dashboard, and our new common-protocol results are separate measurements. Differences include sampling draws, framework/hardware, and evaluator precision/reference. Do not interpret the differences below as pure seed effects.','','| Model | Old dashboard | Original paper | Official release | New mean ± SD | New − paper |','|---|---:|---:|---:|---:|---:|']
for row in rows:
 mean=row['new_fid_mean']; result=f"{mean:.4f} ± {row['new_fid_sd']:.4f}" if mean is not None else 'pending'
 delta=f"{row['delta_vs_paper']:+.4f}" if mean is not None else '—'
 release=f"{row['official_release_fid']:.2f}" if row['official_release_fid'] is not None else '—'
 text.append(f"| [{row['model']}]({row['source']}) | {row['old_dashboard_fid']:.4f} | {row['original_paper_fid']:.2f} | {release} | {result} | {delta} |")
(HERE/'reports/PUBLISHED_COMPARISON.md').write_text('\n'.join(text)+'\n')
