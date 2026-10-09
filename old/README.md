# Setup and release metadata

These files were moved here while simplifying the public repository:

- `environment/`: pinned runtime dependencies and optional encoder conversion dependencies.
- `releases/`: dataset and encoder revisions, filenames, and checksums.
- `checkpoints/`: released generator manifests and inference configurations.

Setup, asset, and released-checkpoint tools still read this metadata. The
directories contain no generator weights. Published checkpoint paths remain
relative to the repository root; training outputs stay under the chosen output
directory. `tests/` remains at the root for comparisons with the original checkout.
