# Fresh FID / FDr⁵ reproduction — 20260930

All 11 checkpoints × three independent sampling seeds × 50,000 class-balanced ImageNet-256 images completed and passed artifact validation (33 runs; 1,650,000 images). FID and FDr⁵ were computed as separate metrics under the final paper protocol.

The fresh and paper means match at the manuscript's two-decimal precision. Δ is fresh minus the September 25–26 paper batch.

| Model | Fresh FID mean ± SD | Δ FID | Fresh FDr⁵ mean ± SD | Δ FDr⁵ | Matching rank streams |
|---|---:|---:|---:|---:|---:|
| WF-XL/2 | 1.3279 ± 0.0136 | +0.000346 | 11.0703 ± 0.0466 | +0.000297 | 0/12 |
| Drifting-B/16 | 1.7387 ± 0.0173 | +0.000000 | 15.3034 ± 0.0399 | -0.000006 | 12/12 |
| Drifting-L/16 | 1.6149 ± 0.0205 | +0.000000 | 12.5892 ± 0.0705 | +0.000002 | 12/12 |
| pMF-L/16 | 2.5544 ± 0.0152 | +0.000000 | 10.8473 ± 0.0182 | -0.000002 | 12/12 |
| pMF-H/16 | 2.1501 ± 0.0182 | -0.000000 | 8.1446 ± 0.0058 | -0.000000 | 12/12 |
| JiT-H/16 | 1.8797 ± 0.0177 | -0.000291 | 8.6928 ± 0.0526 | +0.000113 | 9/24 |
| iMF-XL/2 | 1.7397 ± 0.0150 | -0.000014 | 9.7455 ± 0.0382 | +0.000101 | 6/12 |
| Priming-B-80 | 1.4683 ± 0.0128 | -0.000000 | 12.7089 ± 0.0188 | -0.000001 | 12/12 |
| Priming-B-160 | 1.3841 ± 0.0130 | -0.000000 | 11.2286 ± 0.0157 | +0.000003 | 12/12 |
| Priming-L-80 | 1.2695 ± 0.0128 | +0.000000 | 8.9430 ± 0.0134 | +0.000003 | 12/12 |
| Priming-L-160 | 1.1871 ± 0.0147 | +0.000000 | 7.1048 ± 0.0196 | +0.000007 | 12/12 |

Byte-stream counts compare each GPU rank and seed with the retained paper batch; differing streams can still yield close aggregate metrics. FDr⁵ uses the five frozen normalized feature distances and excludes ConvNeXt. The three seeds are sampling replicates of one checkpoint per model, not training replicates.

Full-precision results: [CSV](full_reproduction_20260930.csv), [per-seed CSV](full_reproduction_20260930_per_seed.csv), and [validation/comparison JSON](full_reproduction_20260930.json). Raw image and rank artifacts are not distributed.
