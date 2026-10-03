# ImageNet evaluation dashboard

Training compute vs. generation quality for Priming Models and comparison
baselines on ImageNet-256. Lower FID and FDr⁵ indicate better generation quality.

## Paper figures

Figures from the ICLR manuscript. Click either preview to open its PDF.

<table>
  <tr>
    <th align="center">Training compute vs. FID</th>
    <th align="center">Training compute vs. FDr⁵</th>
  </tr>
  <tr>
    <td width="50%" align="center"><a href="figures/compute_vs_fid.pdf"><img src="figures/compute_vs_fid.png" width="420" alt="Training compute vs. FID on ImageNet-256"></a></td>
    <td width="50%" align="center"><a href="figures/compute_vs_fdr5.pdf"><img src="figures/compute_vs_fdr5.png" width="420" alt="Training compute vs. FDr⁵ on ImageNet-256"></a></td>
  </tr>
</table>

## Reproduced results

- [Evaluation guide and commands](../eval/README.md)
- [Full reproduction report](../eval/full_reproduction_20260930.md)
- [Full-precision results (CSV)](../eval/full_reproduction_20260930.csv)
- [Per-seed results (CSV)](../eval/full_reproduction_20260930_per_seed.csv)
- [Figure sources and checksums](figures/manifest.json)

The [HTML dashboard](index.html) includes reproduced trade-off plots, model
comparisons, and per-seed scores. Open it locally in a browser. To rebuild it
from the checked-in evaluation results, run this command from the repository root:

```bash
python dashboard/build.py
```

[Back to the main README](../README.md)
