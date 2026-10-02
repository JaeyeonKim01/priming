# distmatch

`distmatch` is a repository-independent package for image-distribution matching
during post-training. It provides EMA Fréchet and exact MMD losses, reference
construction, multiple feature spaces, distributed execution, gradient replay,
and exact Fréchet evaluation. Exact MMD can optionally retain detached,
class-specific generated-feature queues.

The package boundary is floating RGB `NCHW`, nominally in `[0, 1]`. That range
is a convention rather than a hard validation bound: finite generator
overshoot is passed through without clamping. Generation, latent or tokenizer
decoding, `[-1, 1] -> [0, 1]` conversion, optimizers, parameter EMA, logging,
and checkpoint files stay in the host repository.

## Installation

From this source tree:

```bash
pip install -e packages/distmatch
```

SigLIP, MAE, DINO/DINOv2, CLIP, and ConvNeXt model identifiers use the optional
model loader dependency:

```bash
pip install -e 'packages/distmatch[models]'
```

## Reference statistics

```python
from distmatch import Feature, feature_stats

features = [
    Feature(model="vit_so400m_patch16_siglip_256.v2_webli",
            name="siglip", pool="cls", target_size=224),
    Feature(model="inception", name="inception"),
]
stats = feature_stats(
    dataset_or_loader, features=features, batch_size=64, device="cuda",
    stats_dir="reference_stats",
)
```

`stats_dir` is cache-first. Existing, compatible `<feature-name>.npz` files are
loaded without consuming `dataset_or_loader`; omit the source entirely when all
statistics are expected to exist. Only missing features are generated.

Bare `.npz` files containing `mu` and `sigma` are accepted as stats values.
Artifacts written by `distmatch` additionally preserve feature identity and
provenance metadata.

## EMA Fréchet training

```python
from distmatch import EMAFrechet, gradcache_backward

fd = EMAFrechet(
    features=features,
    stats=stats,
    weights={"siglip": 1.0, "inception": 1.0},
    beta=0.999, normalize="value", normalize_eps=0.01,
    device="cuda", distributed="auto",
)

# Optional exact warm start; only moments are retained.
fd.initialize(generate_initial_images, num_samples=50_000, batch_size=256)

# Optional: compile each feature extractor without capturing EMA state mutation.
fd.compile(fullgraph=True)

images = generate_images_with_grad()  # [B, 3, H, W], nominally in [0, 1]
features = fd.extract_features(images)
loss, raw_fd = fd.score_features(features)
loss.backward()
fd.update_state()

checkpoint["distmatch"] = fd.running_state_dict()
```

`fd(images)` is shorthand for the explicit extraction and scoring calls above.
`raw_fd` is a detached dictionary keyed by feature name. `update_state()`
explicitly installs the staged EMA moments after backward.

For a full-rollout loss that does not fit in one autograd graph, cache detached
features and replay one generator chunk at a time:

```python
chunks = make_replayable_chunks(grad_accum)

def encode(chunk):
    images = generate_chunk_with_grad(chunk)
    return fd.extract_features(images)

loss, raw_fd = gradcache_backward(chunks, encode, fd.score_features)
fd.update_state()
```

`gradcache_backward()` performs backward internally and returns a detached loss.
It calls `score_features()` once for the complete logical rollout, so EMA state
still advances once when `update_state()` is called. `encode(chunk)` must
deterministically reproduce the same feature rows on both passes. Distributed
parameter-gradient synchronization remains host-owned, as in the direct path.

## Exact MMD training

Exact MMD compares the generated and real feature rows supplied for the current
step using exact, tiled kernel reductions:

```python
import torch
from distmatch import MMD, gradcache_backward

mmd = MMD(
    features=features,
    bandwidths={"siglip": 4.2, "inception": 18.0},
    weights={"siglip": 1.0, "inception": 1.0},
    normalize="none",
    kernel="rbf",
    kernel_chunk_size=1024,
    attraction_mode="within_class",
    attraction_balance="class",
    repulsion_mode="within_class",
    repulsion_queue_size=128,        # capacity per class
    repulsion_queue_num_classes=1000,
    repulsion_queue_weight=1.0,
    repulsion_queue_kernel="laplacian_l2",
    repulsion_queue_bandwidths={"siglip": 3.1, "inception": 12.4},
    device="cuda",
    distributed="auto",
)

generated_features = mmd.extract_features(generate_images_with_grad())
with torch.no_grad():
    real_features = mmd.extract_features(real_images)

loss, raw_mmd = mmd.score_features(
    generated_features,
    real_features,
    generated_labels=generated_labels,
    attraction_features=full_reference_features,
    attraction_labels=full_reference_labels,
)
loss.backward()
```

`mmd(generated_images, real_images)` is the image-level shorthand and extracts
the real branch without a gradient graph. For the RBF kernel, each `bandwidths`
value is sigma in `exp(-||x-y||^2 / (2 sigma^2))`. `bandwidths="median"`
computes a detached real-only median sigma per feature from the current global
real rows. Kernel dispatch supports RBF and L2-Laplacian. The latter is
`exp(-||x-y||_2 / ell)` and uses a finite zero subgradient at exact equality.

Queue weight, kernel, and bandwidths are independent from the attraction
kernel. `repulsion_queue_bandwidths="attraction"` reuses attraction bandwidths
only when the kernels match; a different queue kernel requires a complete
name-keyed mapping. Diagnostics report both `<name>_queue_kernel_mean` and
`<name>_queue_weighted_repulsion`.

With a positive queue size, current generated rows are attracted to supplied
real rows from the same class and repelled only from detached historical
generated rows of that same class. Current-current generated repulsion is not
added. Queue capacity is independent per class, varying class sets are valid,
and the queue advances after scoring so a rollout never repels itself. Save and
restore `running_state_dict()` to preserve the exact queue on resume.

For gradient replay, extract the step's real features once and bind them into
the scoring callback:

```python
with torch.no_grad():
    real_features = mmd.extract_features(real_images)

def encode(chunk):
    return mmd.extract_features(generate_chunk_with_grad(chunk))

loss, raw_mmd = gradcache_backward(
    chunks,
    encode,
    lambda generated: mmd.score_features(generated, real_features),
)
```

Each distributed rank supplies its local generated and real rows; both sets are
gathered globally, while only generated rows remain differentiable. Kernel
tiles bound working memory. With class-conditional reference banks, the cross
term scales as `sum_c |G_c| |R_c|` instead of a dense `G R` product.

Repository-local real-feature bank construction and exact nearest-neighbor
selection, including ImageNet class-conditioned search, are documented in
[`tools/README.md`](tools/README.md); these host utilities are intentionally
outside the installable `distmatch` API.

## Custom features

```python
custom = Feature(
    model=encoder, name="encoder", feature_dim=768,
    feature_id="encoder:weights-v2:preprocess-v1",
)
```

The callable receives validated floating RGB and returns `[B, D]`. Values are
nominally `[0, 1]`, but finite overshoot is preserved. It owns its
feature-specific resize, normalization, and pooling. An `nn.Module` is moved to
the configured device, put in evaluation mode, and frozen while input gradients
remain enabled.

## Exact evaluation

```python
from distmatch import FrechetEvaluator

evaluator = FrechetEvaluator(features=eval_features, stats=eval_stats, device="cuda",
                             distributed="auto")
evaluator.reset()
for images in generated_batches:
    evaluator.update(images)
result = evaluator.compute()
print(result.fd, result.num_samples)
```

In distributed runs, the host initializes the process group and supplies a
non-overlapping local image stream on each rank. If sampling bypasses a DDP
forward, `distmatch.distributed.sync_gradients(..., reduction="mean")` provides
the explicit parameter-gradient synchronization used by the direct workflow.

## OST integration material

The legacy FD interface demo and FD parity tests live in
[`integrations/ost`](integrations/ost/). They are development and migration
assets, not dependencies of the repository-independent runtime package. The
active OST MMD integration is owned and documented by the host repository.
