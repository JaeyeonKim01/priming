# Minimal Gradient Caching Plan

## Goal

Add full-rollout gradient caching to `distmatch` with the same simple three-pass
pattern as `~/woobin/RDM/rdm/train/grad_caching.py`, while supporting
`EMAFrechet` without changing its EMA equations or state lifecycle.

This is needed because Fréchet loss depends on moments of the entire rollout.
Computing and backpropagating a separate loss for each microbatch is not
equivalent to backpropagating the loss of the concatenated rollout.

## Public interface

Add one standalone helper:

```python
gradcache_backward(chunks, encode_fn, loss_fn, *, scale=1.0)
```

- `chunks` contains replayable per-microbatch inputs.
- `encode_fn(chunk)` returns `{feature_name: Tensor[B, D]}` and must be
  deterministic for a given chunk.
- `loss_fn(features)` returns `(scalar_loss, logs)`.
- `scale` scales only the backward gradient. It does not change the reported
  loss or the EMA update.
- The function accumulates parameter gradients but does not synchronize them,
  update EMA state, or perform an optimizer step.

Add `EMAFrechet.extract_features(images)` so an EMA Fréchet objective can be
wired to the helper without accessing internal extractor objects.

Example:

```python
def encode(chunk):
    images = generate_chunk(chunk)
    return fd.extract_features(images)

loss, raw_fd = gradcache_backward(
    chunks,
    encode,
    fd.score_features,
)

sync_gradients(model.parameters(), reduction="mean")
fd.update_state()
```

For `grad_accum == 1`, the host should retain the existing direct forward and
backward path to avoid the extra forward pass.

## Algorithm

The implementation should closely follow the RDM helper.

### 1. Feature pass

Run `encode_fn` over every local chunk under `torch.no_grad()`. Cache detached
feature tensors and concatenate them by feature name into autograd leaves.

Only feature tensors are retained; generator and extractor graphs are freed
after every chunk.

### 2. Full-rollout loss pass

Call `loss_fn` once with the concatenated feature leaves:

```python
loss, logs = loss_fn(leaves)
(loss * scale).backward()
feature_grads = {name: leaf.grad.detach() for name, leaf in leaves.items()}
```

Unlike RDM, the helper must not all-gather raw features. `EMAFrechet` already
reduces differentiable global sufficient statistics inside `score_features()`.

### 3. Replay pass

Run `encode_fn` again for one chunk at a time. Slice that chunk's cached feature
gradient and backpropagate through a surrogate:

```python
surrogate = sum(
    (features[name] * gradient_slice[name].detach()).sum()
    for name in features
)
surrogate.backward()
```

The resulting parameter gradients accumulate across chunks while retaining an
autograd graph for only one microbatch at a time.

## EMA Fréchet behavior

`EMAFrechet.score_features()` is called exactly once with all locally cached
features. Its existing distributed reductions construct moments for the global
rollout and stage one EMA transition.

After replay and parameter-gradient synchronization succeed, the host calls
`fd.update_state()` exactly once. Therefore:

- EMA decay is applied once per logical rollout.
- Cold state initializes from the complete rollout.
- `samples_seen` increases by the global rollout size.
- `update_count` increases by one.
- No cached features or gradients enter the running state or checkpoint.

## Gradient scaling and distributed behavior

Use `scale=1.0` to match a naive backward of one full-rollout EMA Fréchet
loss. If a host deliberately follows RDM's learning-rate convention, it may
pass `scale=1.0 / grad_accum` explicitly.

The cache helper owns no distributed parameter synchronization. Callers that
bypass a DDP forward continue to use:

```python
sync_gradients(model.parameters(), reduction="mean")
```

This preserves the existing `distmatch` distributed convention.

## File changes

Create:

- `src/distmatch/grad_caching.py`: the small RDM-style
  `gradcache_backward()` helper.

Modify:

- `src/distmatch/objectives/frechet/objective.py`: add
  `EMAFrechet.extract_features()` and make `__call__()` reuse it.
- `src/distmatch/__init__.py`: export `gradcache_backward`.
- `tests/test_distmatch.py`: add single-process cached-versus-direct parity.
- `tests/test_distributed.py`: extend distributed parity coverage to cached
  replay.
- `README.md`: document the short training example and deterministic replay
  requirement.
- `integrations/ost/distmatch_interface_example.py`: use the cache helper when
  `grad_accum > 1` instead of rejecting the configuration.

## Tests

### Single-process parity

Compare a naive concatenated rollout with a cached rollout using identical
chunks. Assert matching:

- reported loss and raw Fréchet values;
- generator parameter gradients;
- staged and committed EMA mean and raw-second moment;
- `samples_seen` and `update_count`.

The test should cover both cold and initialized EMA state. One multi-feature
case is sufficient to exercise feature-name slicing.

### Distributed parity

Extend the existing two-rank test to divide each rank's local samples into
multiple replay chunks. Compare cached and direct paths before and after the
existing mean parameter-gradient synchronization, and assert identical EMA
state on every rank.

### Basic validation

Check only the errors needed to prevent silent misuse:

- an empty chunk list;
- inconsistent per-feature chunk sizes;
- a replayed feature slice with the wrong shape.

## Deferred work

Keep the initial implementation deliberately small. Do not add:

- automatic RNG capture or replay verification;
- CPU feature offloading;
- a class-based or protocol-based cache framework;
- automatic gradient synchronization, EMA updates, or optimizer steps;
- AMP/GradScaler-specific callbacks;
- rollback APIs or new checkpoint state;
- raw-feature distributed gathering.

Replay determinism remains the caller's responsibility. Chunk descriptors
should contain all sampled noise, labels, conditioning, or explicit seeds
needed for `encode_fn` to reproduce the same features on both passes.

## Completion criteria

The feature is complete when cached replay matches a naive full-rollout
backward and produces the same committed EMA state in both single-process and
two-rank tests, while the generator graph memory is bounded by one microbatch.
