# Gradient Replay Memory Analysis

## Scope

This note records the unexpectedly high GPU-memory use observed when OST-L
uses DistMatch feature-gradient replay for a batch-coupled Fréchet objective.
It separates memory that is inherent to replay from memory caused by the
current implementation or runtime. The large fixed overhead is not yet fully
attributed; claims about `torch.compile` and allocator behavior below are
hypotheses to verify with stage-level profiling.

## Observed runs

All three runs use OST-L, BF16 autocast, a per-GPU forward microbatch of 128,
and the same SigLIP, MAE, and Inception training feature spaces.

| Run | Per-GPU microbatch | Replay chunks | Per-GPU logical rollout | Global logical rollout | Compile | Peak reserved |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `OST_L_aux_unif_t08_pretrain_e320_fd_e20_gbs256_then_fd_e50_lr1e6_gbs1024_40678795` | 128 | 1 | 128 | 1,024 | yes | 90.49 GiB initially; 94.09 GiB steady |
| `OST_L_pretrain_e320_FD_noEMA_e50_lr1e5_cosine_warm5_modelEMA099_gbs5120_gc5_s12500_42924285` | 128 | 5 | 640 | 5,120 | yes | 128.50 GiB |
| `OST_L_from_42924285_FD_noEMA_e50_lr1e6_const_modelEMA099_gbs51200_gc50_s1250_43128476` | 128 | 50 | 6,400 | 51,200 | yes | 137.27 GiB |

The microbatch is the same, but the logical batch resident or reproducible
during one optimizer update is not. In particular, "batch size per GPU = 128"
only bounds the size of one generator/extractor graph; it does not imply that
the complete update has the same memory footprint.

These runs are not identical in every training setting. For example, the
original direct run uses FD EMA and a 50k initialization queue, whereas the
replay runs use `fd_ema_beta=0` and no queue. Those differences should be kept
in mind, but they do not explain the rollout-proportional memory measurements
below. The old queue would, if anything, tend to make the direct run larger.

## What the logged memory means

The OST driver logs:

```python
torch.cuda.max_memory_reserved()
```

This is the maximum amount reserved by PyTorch's CUDA allocator since peak
statistics were reset. It is not peak live tensor allocation. Freed blocks,
fragmentation, compiled/CUDA-graph pools, and other reusable allocator state
can remain counted. `reset_peak_memory_stats()` resets the statistic but does
not release memory that was already reserved.

Consequently, the logged number is operationally important for OOM headroom,
but it cannot by itself identify which live tensors caused the high-water
mark. Both `max_memory_allocated()` and `max_memory_reserved()` must be recorded
at replay stages to obtain that breakdown.

## Replay execution and its natural memory

The current replay algorithm has three phases:

1. Run every chunk under `torch.no_grad()` and retain detached feature rows.
2. Score the full logical rollout and compute `d(loss) / d(features)`.
3. Recompute one chunk at a time with gradients enabled and inject the cached
   feature-gradient slice through a surrogate dot product.

The intended peak live state is therefore:

- one microbatch's generator and extractor graph;
- compact feature rows for the logical rollout;
- compact feature-gradient rows for the logical rollout;
- small FD moment/covariance state; and
- parameter gradients accumulated from replayed chunks.

This natural cache is much smaller than tens of GiB. Gradient replay should
have some overhead over a direct 128-sample update, but a 34--45 GiB increase
is not theoretically required by the method.

## Confirmed rollout-proportional implementation overhead

### GPU-resident rollout inputs

The OST integration currently constructs every noise and label chunk up front
on the GPU. For 256x256 RGB FP32 noise, one 128-sample chunk occupies:

```text
128 * 3 * 256 * 256 * 4 bytes = 0.09375 GiB
```

Therefore noise alone occupies:

- 0.46875 GiB for 5 chunks;
- 4.6875 GiB for 50 chunks.

Labels are negligible. Only one noise chunk needs to be live at a time. The
rest can be represented by deterministic RNG seeds and CPU labels, then
regenerated for both the cache and replay passes. At 50 chunks, this would save
about 4.59 GiB relative to retaining all noise tensors while preserving one
active chunk.

### Detached CLS views retain complete token tensors

SigLIP and MAE return the CLS row with an operation equivalent to:

```python
output = tokens[:, 0]
```

The logical output has shape `[B, D]`, but it is a view into the full
`[B, 197, D]` token tensor. The current cache operation:

```python
rows.detach()
```

breaks autograd history but does not create new storage. Retaining the detached
CLS view therefore retains the full token tensor.

For the 6,400 local samples in the 50-chunk run, assuming BF16 ViT outputs and
FP32 Inception outputs, the backing storage is approximately:

| Feature | Retained current storage | Compact `[N, D]` storage |
| --- | ---: | ---: |
| SigLIP, `D=1152`, 197 tokens | 2.71 GiB | 14.1 MiB |
| MAE, `D=1024`, 197 tokens | 2.40 GiB | 12.5 MiB |
| Inception, `D=2048` | 50 MiB | 50 MiB |
| Total | about 5.16 GiB | about 76.6 MiB |

The cache boundary should instead retain an independent compact tensor:

```python
cached.setdefault(name, []).append(
    rows.detach().clone(memory_format=torch.contiguous_format)
)
```

`clone()` is deliberately used rather than relying only on `contiguous()`.
For some shapes, especially a batch of one, a view may be reported contiguous
while still sharing a larger backing allocation. Cloning always establishes
independent compact storage.

This does not change values, dtype, the loss, or replay gradients. The cache
pass is already intentionally detached, and the graph is reconstructed in the
replay pass. The copied logical features total only tens of MiB per GPU, so the
copy cost should be negligible relative to the extractor forwards.

### The gc5-to-gc50 measurement validates this accounting

Moving from 5 to 50 chunks increased logged peak reserved memory by:

```text
137.2656 - 128.4961 = 8.7695 GiB
```

The expected increase from the current retained tensors is approximately:

| Source | Expected increase |
| --- | ---: |
| FP32 noise | 4.2188 GiB |
| SigLIP full-token backing | 2.4348 GiB |
| MAE full-token backing | 2.1643 GiB |
| Inception output | 0.0439 GiB |
| Total | 8.8618 GiB |

The prediction is within about 0.1 GiB of the observed increase. Allocator
rounding and reuse can reasonably account for the small remainder. This is
strong evidence that GPU-resident inputs and non-compact feature views explain
the rollout-proportional part of the problem.

Applying both compact feature caching and seed-based input regeneration should
reduce the 50-chunk run by roughly 9--10 GiB after a fresh process start. The
exact change in `max_memory_reserved()` may differ from the live-tensor saving
because allocator behavior is part of that metric.

## The remaining fixed replay gap

The direct run's steady peak is about 94.09 GiB, while the 5-chunk replay run
uses about 128.50 GiB: a gap of roughly 34.4 GiB. At only five chunks, the
explicit additional input and full-token cache storage is less than about
1 GiB. It cannot explain this fixed gap.

The leading mechanisms to investigate are:

1. **Separate compiled grad-mode specializations.** The first replay pass calls
   the compiled generator and extractors under `torch.no_grad()`, while replay
   calls them with gradients enabled. Grad mode is normally guarded by
   `torch.compile`, so these calls may produce separate compiled variants and,
   depending on the backend, separate CUDA-graph or workspace pools.
2. **Allocator high-water mark and fragmentation.** A no-grad execution phase
   followed by a differently shaped/lived grad-enabled phase can reserve blocks
   that are not efficiently reused. `max_memory_reserved()` continues to count
   those blocks even after the first pass has released its live tensors.
3. **Gradient/forward overlap during later replay chunks.** After the first
   replay backward, parameter `.grad` buffers remain resident while the next
   chunk's complete forward graph is built. This overlap is natural, but the
   405M-parameter OST-L gradient buffers are not large enough by themselves to
   explain the whole 34 GiB gap.
4. **Other implementation or runtime lifetimes.** Compiled backward buffers,
   feature-output views that survive assignment boundaries, and objective
   workspaces should be measured rather than guessed.

The first two items are plausible explanations, not yet demonstrated causes.
The current logs contain only one aggregate reserved-memory high-water mark,
so assigning the full gap to `torch.compile` would be premature.

## Minimal fixes and recommended order

1. **Compact cached feature rows.** Clone detached rows at the cache boundary.
   This is local to `grad_caching.py`, simple, and estimator-independent.
2. **Store chunk descriptors instead of GPU inputs.** Retain a noise seed and
   CPU labels per chunk, and deterministically regenerate the GPU tensors in
   `encode_fn`. The cache and replay passes must reproduce identical inputs.
3. **Profile the fixed gap before changing compilation.** Compile behavior is
   performance-sensitive, and a speculative change could trade memory for a
   large slowdown without addressing the actual cause.
4. **Choose a compile/allocator mitigation from evidence.** Possibilities to
   benchmark include an eager first pass with compiled replay, avoiding CUDA
   graphs for these modules, using one grad-mode specialization for both
   passes, or an allocator configuration that reduces fragmentation. None
   should become the default until its memory and throughput are measured.

Calling `torch.cuda.empty_cache()` between replay phases or on every update is
not a preferred default solution. It may lower reserved memory in some cases,
but can introduce synchronization and allocation overhead, and it will not
necessarily release private compiled/CUDA-graph pools.

## Verification plan

Run a small stage-instrumented matrix using the same OST-L checkpoint,
microbatch 128, feature extractors, and loss configuration:

| Execution | Compile off | Compile on |
| --- | --- | --- |
| Direct, 1 chunk | measure | measure |
| Replay, 5 chunks | measure | measure |

Optionally repeat 50-chunk replay after the two confirmed storage fixes.

At minimum, synchronize CUDA and record all four values at each stage:

```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

Record them after:

1. setup and peak-stat reset;
2. construction of chunk inputs or descriptors;
3. every no-grad cache encode;
4. concatenating cached rows;
5. the full-rollout loss backward into feature leaves;
6. every replay encode and replay backward;
7. gradient clipping, optimizer step, and zeroing gradients.

The profile should answer:

- whether the 34 GiB gap exists in live allocation or only reservation;
- whether it appears during the no-grad compiled specialization;
- whether it appears only after the grad-enabled specialization;
- how much later replay forwards overlap with accumulated gradients; and
- whether disabling compile removes the fixed gap or merely moves the peak.

Functional verification must accompany memory measurements:

- cached and replayed feature rows match for fixed seeds;
- direct and replay loss values match for the same complete rollout;
- direct and replay parameter gradients retain the established parity;
- compact cloning and seed regeneration do not change objective state updates;
- throughput remains effectively unchanged for the two confirmed storage
  fixes.

## Current conclusion

Some replay overhead is natural, but the present magnitude is not. The
rollout-proportional increase is well explained and has two simple fixes. The
large fixed direct-to-replay gap is most likely a runtime/execution-lifetime
effect and should be treated as diagnosable and potentially reducible, not as
an unavoidable property of gradient replay.
