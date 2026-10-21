# Source provenance

The source archive supplied for this migration is
`/n/netscratch/sitanc_lab/Lab/jaykim/t2i_and_world.tar.gz`.
The native RT-1 and Bridge U-Net files were copied from its
`driftworld/{rt1_driftworld,bridge_driftworld}/unet/` directories. Their GPC
attribution comments remain in place. VAE/DINO wrappers, reconstruction and
replay, episode readers, sampling, and streaming metrics were adapted from
those two projects, with imports changed to the `world_modeling` package.

[The source manifest](results/source-manifest.json) records original file hashes
and migrated destinations. Shared stage entrypoints, configuration handling,
checkpoint recovery, explicit variant selection, and the DistMatch loader are
migration code.

DistMatch is an external dependency loaded from the user-provided source path,
not vendored here. Its fingerprint is recorded in post-training checkpoints.
The migration assessment documents the recovered 0.6.0 versus historical 0.4.0
version gap. Model weights, data, DINOv3, pytorch-fid, and the I3D detector remain
external assets with their own provenance and license terms.
