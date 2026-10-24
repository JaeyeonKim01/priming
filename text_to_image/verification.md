# Port verification, 2026-10-24

The port was checked with Python 3.11.16, PyTorch 2.6.0+cu124,
Transformers 4.40.2, and external DistMatch 0.6.0. The source files came from
the `~/minit2i-torch` working tree, including its uncommitted one-step code.
`source_manifest.json` records hashes for 73 source files; 15 have only the
documented path/dependency adaptations.

## Checks completed

| Check | Result |
| --- | --- |
| Source reconstruction, Mean Flow, and benchmark generation CPU tests | 15 passed |
| Selected one-step CPU tests: forward process, one-call gradient, tiny model backward, checkpoint resume, config, and MMD parameterization | 7 passed |
| `train-distmatch --dry-run` with the local B/16 checkpoint and 120K data | Passed, same JSON result as the original checkout |
| One-step evaluation command generation | Passed with `--dry-run`; uses one step and CFG scale 1 |
| Sample and post-training CLI help | Passed |
| Full copied test suite collection | 63 tests collected without import errors |
| Read-only dataset check | CC12M: 6,933,021 samples / 6,771 chunks; fine-tuning: 120,000 / 118 |
| B/16 training data loaders | Pretraining and fine-tuning each yielded `[2,3,512,512]` uint8 images and `[2,256]` int64 token IDs |
| Data-free CPU smoke using Priming's environment | One-step MM-JiT forward and backward passed |

Both dry runs found 120,000 caption/example IDs, 83,951 distinct caption
strings, and dataset fingerprint
`b1a1a6e1a529475258acb3c3f5740c2cff7423ecefd09b4ff4d239d5a5755774`.
Both selected the same B/16 checkpoint, SIM branches, marginal Laplacian MMD
settings, and replay configuration.

The login environment has no CUDA device. Model sampling, GPU training,
full-size memory use, and benchmark scores were not checked here. The source
report's four-GPU smoke test is historical evidence for its own checkout, not
a GPU validation of this port.

The historical T2I overlay has Transformers 4.40.2 and Hugging Face Hub
0.36.0. In this workspace, importing the installed Diffusers 0.40.0 pipeline
with that overlay fails because Diffusers expects `get_cached_repo_tree` from
Hub. Multi-step Hub inference therefore needs a separately checked dependency
set. The training imports and CPU tests passed in the overlay.

The dataset check compared contiguous chunk filenames, completion markers,
prior full verification records, dimensions, tokenizer, and current file sizes.
It did not rescan every image. The CC12M full chunk scan finished on October 19;
the 120K scan finished on August 27. The staged raw fine-tuning WebDataset has
links into another storage area that this sandbox cannot read. The materialized
tensor chunks were readable by both training loaders, so this does not block
training from `local_folder` data.
