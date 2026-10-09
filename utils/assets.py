"""Freeze setup identities without copying code or large assets into run bundles."""
import hashlib
from pathlib import Path

from .files import sha256


def pretraining_manifest(config, *, root, family, hf_home, torch_home, lpips_weights):
    root = Path(root)
    source_files = [root / "pretrain.py"]
    for directory in ("models", "losses", "utils", "data"):
        source_files.extend(sorted((root / directory).rglob("*.py")))
    sources = {str(p.relative_to(root)): sha256(p) for p in source_files}
    packed = Path(config["data"]["data_path"]) / "imagenet-latents-images"
    metadata, shards = {}, {}
    for directory in [packed] + ([packed / "val"] if family == "b" else []):
        for name in ("dataset_info.json", "state.json"):
            path = directory / name
            metadata[str(path.relative_to(packed))] = sha256(path)
        files = sorted(directory.glob("*.arrow"))
        if not files:
            raise FileNotFoundError(f"No packed ImageNet Arrow shards in {directory}")
        for path in files:
            stat = path.stat()
            shards[str(path.relative_to(packed))] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    assets = {}
    if config["loss"]["aux"]["enabled"]:
        lpips = Path(lpips_weights)
        if hashlib.md5(lpips.read_bytes()).hexdigest() != "d507d7349b931f0638a25a48a722f98a":
            raise ValueError("LPIPS calibration differs from the released weights")
        vgg = Path(torch_home) / "hub/checkpoints/vgg16-397923af.pth"
        assets.update(lpips=dict(path=str(lpips), sha256=sha256(lpips)),
                      vgg16=dict(path=str(vgg), sha256=sha256(vgg)))
        hub = Path(hf_home) / "hub/models--facebook--convnextv2-base-22k-224"
        ref = hub / "refs/main"
        revision = ref.read_text().strip()
        snapshot = hub / "snapshots" / revision
        files = sorted(p for p in snapshot.rglob("*") if p.is_file())
        if not (snapshot / "config.json").is_file() or not any(p.suffix in (".bin", ".safetensors") for p in files):
            raise FileNotFoundError(f"Incomplete ConvNeXt cache: {snapshot}")
        assets["convnext"] = dict(revision=revision, files={str(p): sha256(p) for p in [ref, *files]})
    return dict(version=2, runtime=sources, data_metadata_sha256=metadata,
                arrow_files=shards, assets=assets,
                data_identity_scope="Metadata hashes and shard names/sizes/mtime; image bytes are not fully hashed.")
