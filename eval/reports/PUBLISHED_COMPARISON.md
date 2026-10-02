# Original-paper comparison

Original paper, official release reproduction, historical dashboard, and our new common-protocol results are separate measurements. Differences include sampling draws, framework/hardware, and evaluator precision/reference. Do not interpret the differences below as pure seed effects.

| Model | Old dashboard | Original paper | Official release | New mean ± SD | New − paper |
|---|---:|---:|---:|---:|---:|
| [wflow_xl](https://github.com/hanjq17/W-Flow) | 1.3477 | 1.29 | — | 1.3275 ± 0.0137 | +0.0375 |
| [drifting_b](https://github.com/lambertae/drifting) | 1.5470 | 1.76 | 1.73 | 1.7387 ± 0.0173 | -0.0213 |
| [drifting_l](https://github.com/lambertae/drifting) | 1.4260 | 1.61 | 1.62 | 1.6149 ± 0.0205 | +0.0049 |
| [pmf_l](https://github.com/Lyy-iiis/pMF/tree/990e81a84249dbd68a128accef67eb95621d10b1) | 2.7200 | 2.52 | 2.51 | 2.5544 ± 0.0152 | +0.0344 |
| [pmf_h](https://github.com/Lyy-iiis/pMF/tree/990e81a84249dbd68a128accef67eb95621d10b1) | 2.2900 | 2.22 | 2.15 | 2.1501 ± 0.0182 | -0.0699 |
| [jit_h](https://arxiv.org/html/2511.13720v1) | 1.9700 | 1.86 | — | 1.8800 ± 0.0177 | +0.0200 |
| [imf_xl](https://github.com/Lyy-iiis/imeanflow/tree/04687983e821b3ad01f54f03dc44194a33c20c54) | 1.8200 | 1.72 | 1.72 | 1.7397 ± 0.0150 | +0.0197 |
