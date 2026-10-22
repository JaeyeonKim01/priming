# Optional Diffusers inference

ImageNet B/L inference adapted from [BiliSakura's community implementation](https://huggingface.co/BiliSakura/Priming-diffusers), pinned to revision
`a0c1743f49eb0f972feec280356ccaab90298b20`. This is a repository integration,
not an upstream Diffusers pipeline. Training and world modeling do not depend on it.

From the repository root, in an environment with the repository's PyTorch dependencies:

```bash
python -m pip install -r integrations/huggingface_diffusers/requirements.txt
python -m integrations.huggingface_diffusers.convert \
  --model priming_b_160 --output outputs/diffusers/priming_b_160
```

Download the [released checkpoint](../../docs/checkpoints.md) first. The converter
verifies its release SHA-256, selects terminal post-training EMA 0.999, and saves
Safetensors without changing tensor dtype. All four B/L 80/160 releases are
accepted. `--checkpoint` can point to another location for the same verified file.
The output directory must be empty. `conversion.json` records provenance.

```python
import torch
from integrations.huggingface_diffusers import PrimingPipeline

pipe = PrimingPipeline.from_pretrained(
    'outputs/diffusers/priming_b_160', local_files_only=True,
).to('cuda')
# Keep FP32 weights and use BF16 autocast, as in scripts/sample.py.
with torch.autocast('cuda', dtype=torch.bfloat16):
    image = pipe(
        class_labels=207,
        generator=torch.Generator(device='cuda').manual_seed(1),
    ).images[0]
image.save('sample.png')
```

Run local repository imports from the repository root. For standalone loading,
use the official [B-160](https://huggingface.co/jaykim1102/priming/tree/main/diffusers/priming_b_160)
and [L-160](https://huggingface.co/jaykim1102/priming/tree/main/diffusers/priming_l_160)
exports. They include custom Python components and do not require this repository:

```python
from pathlib import Path
from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline

variant = "priming_b_160"  # or priming_l_160
snapshot = snapshot_download(
    "jaykim1102/priming", revision="b5928fce72a3f7aa29ffcf14c121df57c00ffa36",
    allow_patterns=[f"diffusers/{variant}/*"],
)
folder = Path(snapshot) / "diffusers" / variant
pipe = DiffusionPipeline.from_pretrained(
    str(folder), custom_pipeline=str(folder / "pipeline.py"),
    local_files_only=True,
).to("cuda")
# Sample with the FP32-weights/BF16-autocast example above.
```

This executes the included custom Python code. Inspect its source and pin the
Hub revision for reproducible use. To package your own local conversion:

```bash
python -m integrations.huggingface_diffusers.export_hub \
  --source outputs/diffusers/priming_b_160 \
  --output outputs/hub/priming_b_160
```

Sampling supports the native 256×256 resolution, one forward pass, and
`guidance_scale=1`. Use ImageNet IDs 0–999 (1000 is the null class); English names
require an explicit `id2label` mapping. `latents=` accepts an explicit NCHW noise
tensor and bypasses noise sampling/scaling. `output_type='latent'` returns raw
predictions; `pil`/`np`/`pt` apply the original rounded uint8 image conversion
(`np`: NHWC uint8, `pt`: NCHW float in [0,1]).

Loading with `torch_dtype=torch.bfloat16` casts the weights and is a distinct
numerical mode from FP32 weights under BF16 autocast. Use fixed noise and class
IDs for comparisons; a seed alone does not establish equivalence.

Validation:

```bash
python -m unittest discover -s tests -p test_diffusers_integration.py -v
# On one CUDA GPU; local community weights named Priming-B/L.safetensors:
python -m integrations.huggingface_diffusers.validate \
  --community-weights /path/to/pinned/community/weights \
  --output /path/to/new/validation-directory
```

The full check compares community tensors against both 80/160 EMA releases,
then checks B/L FP32, BF16 autocast, BF16 weight casting, and saved/reloaded
outputs using identical noise and labels. This is conversion/inference validation,
not a new 50k-sample FID evaluation. See [attribution](NOTICE.md).

## Verified conversion

On one H100 with PyTorch 2.6.0 and Diffusers 0.37.1, the pinned community B/L
weights exactly matched the released **B-160/L-160 post-training EMA** (218/426
tensors). Fixed-input comparisons used three classes and one noise seed:

| Comparison against native code | B-160 max error | L-160 max error |
|---|---:|---:|
| FP32 | 0 | 0 |
| FP32 weights, BF16 autocast | 0 | 0 |
| BF16 weights, BF16 autocast | 0 | 0 |
| FP32 save/reload | 0 | 0 |

These rows compare the same precision mode on both sides. Casting weights to
BF16 instead of retaining FP32 weights changed raw outputs (mean absolute
change: B 0.01369, L 0.02451). Five CPU regression tests also passed, including standalone custom-component loading.
[Machine-readable results](validation.json) include hashes and exact errors.
GPU parity was checked for the 160-epoch releases; 80-epoch conversion uses the
same architecture and verified loader but was not separately GPU-tested.

The [published release check](hub-release.json) verifies downloaded artifact hashes
and exact pre-upload versus downloaded H100 outputs without importing Priming.
