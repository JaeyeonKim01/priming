# Three-seed ImageNet evaluation

Updated 2026-09-26T02:38:06.202978+00:00. Complete: 33/33 replicates.

50,000 balanced images per sampling seed (1, 2, 3), one fixed released/terminal EMA checkpoint per model. SD is sample SD across sampling seeds, not training seeds. No pooled-sample FID.

FID uses pinned LTH14 torch-fidelity, FP32 features without TF32, rounded uint8 images, and the official JiT ImageNet-256 reference. FDr5 retains the separate frozen local definition (BF16 feature extraction on float images, FP64 moments, ADM Inception component, excludes ConvNeXt). Official FID is not substituted into the FDr5 composite.

| Model | Completed seeds | FID mean ± SD | FDr5 mean ± SD |
|---|---:|---:|---:|
| wflow_xl | 3/3 | 1.3275 ± 0.0137 | 11.0700 ± 0.0462 |
| drifting_b | 3/3 | 1.7387 ± 0.0173 | 15.3034 ± 0.0399 |
| drifting_l | 3/3 | 1.6149 ± 0.0205 | 12.5892 ± 0.0705 |
| pmf_l | 3/3 | 2.5544 ± 0.0152 | 10.8473 ± 0.0182 |
| pmf_h | 3/3 | 2.1501 ± 0.0182 | 8.1446 ± 0.0058 |
| jit_h | 3/3 | 1.8800 ± 0.0177 | 8.6927 ± 0.0528 |
| imf_xl | 3/3 | 1.7397 ± 0.0150 | 9.7454 ± 0.0382 |
| priming_b_80 | 3/3 | 1.4683 ± 0.0128 | 12.7089 ± 0.0188 |
| priming_b_160 | 3/3 | 1.3841 ± 0.0130 | 11.2285 ± 0.0157 |
| priming_l_80 | 3/3 | 1.2695 ± 0.0128 | 8.9430 ± 0.0134 |
| priming_l_160 | 3/3 | 1.1871 ± 0.0147 | 7.1048 ± 0.0196 |
