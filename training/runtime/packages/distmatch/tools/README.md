# Host tools

`tools` contains repository-local runtime helpers that are intentionally not
part of the installable `distmatch` package under `src/`.

## Build real feature banks

Use the objective's extraction callback so real and generated rows have exactly
the same preprocessing and feature identity:

```python
from tools import build_real_feature_banks

real_feature_banks = build_real_feature_banks(
    real_dataset_or_loader,
    extract_fn=mmd.extract_features,
    batch_size=64,
    device=mmd.device,
    bank_device="cpu",
    class_conditional=True,  # retain ImageNet target IDs
)

for name, bank in real_feature_banks.items():
    bank.save(f"real_feature_banks/{name}.pt")
```

Building extracts every configured feature in one pass over the source. Use
`max_samples=N` to build a deterministic prefix rather than the complete
source. A caller-provided shuffled loader preserves its own order. With
`class_conditional=True`, conventional `(images, labels)` batches (including
`ImageFolder` loaders) are handled automatically. Mapping batches may use
`label`, `labels`, `class_id`, `class_ids`, `target`, or `targets`; pass
`label_getter=` for another layout. Labels are stored in every saved bank.

Banks can be loaded without repeating feature extraction and optionally moved
to a GPU when they fit:

```python
from tools import RealFeatureBank

bank = RealFeatureBank.load("real_feature_banks/siglip.pt", device="cpu")
bank = bank.to("cuda")  # optional: faster repeated search
```

## Select nearest real features

`RealFeatureBank` precomputes reference squared norms once and performs exact
blocked matrix-multiplication search on its resident device. It never allocates
the complete query-by-dataset distance matrix.

```python
selected_real = {}
selected_indices = {}
for name, generated_rows in generated_features.items():
    selection = real_feature_banks[name].nearest(
        generated_rows,
        class_ids=generated_class_ids,
        query_chunk_size=1024,
        reference_chunk_size=16384,
    )
    selected_real[name] = selection.features
    selected_indices[name] = selection.indices

loss, raw_mmd = mmd.score_features(generated_features, selected_real)
```

`generated_class_ids` is a one-dimensional integer tensor with one ImageNet
class ID per generated row. Each query searches only real rows having that
same ID. The result indices are offsets into the full bank (in bank-generation
order), not per-class offsets. Every requested class must have at least one row
in the bank.

Search is detached and non-differentiable. Its selected rows, indices, and
distances are returned on the generated-feature device. Different feature
spaces select independently, so their chosen indices need not refer to the
same real image. Use fixed MMD bandwidths estimated from the full real banks
when the selected target changes each step.

Class labels are pre-indexed when a bank is created or loaded. On ImageNet this
turns each exact search from a scan of the whole dataset into a scan of just
the matching class, while retaining blocked distance computation and exact
nearest-neighbor results. Omitting both bank labels and `class_ids` preserves
the original global-search behavior.

For a one-off search over existing tensors,
`nearest_real_features(query, rows, class_ids=query_labels,
real_class_ids=row_labels)` is also available. Reuse `RealFeatureBank` during
training so bank validation, class indexing, and squared norms are not
recomputed every step.
