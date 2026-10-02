"""Link the four release checkpoint names to the production training files.

The links use the repository's fdloss_work_dirs mount, so the large model
files remain on scratch rather than being copied into OST_public.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def main() -> None:
    manifest_path = HERE / "checkpoints/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest) == 4
    for key, item in sorted(manifest.items()):
        source = Path(item["source_checkpoint"])
        if not source.is_file():
            raise FileNotFoundError(source)
        run = source.parents[1].name
        repo_source = ROOT / "fdloss_work_dirs" / "OST" / run / "checkpoints" / "step_0024999.pth"
        if repo_source.resolve() != source.resolve():
            raise RuntimeError(f"Repository checkpoint link does not match {source}")
        link = HERE / "checkpoints" / f"{key}.pth"
        target = os.path.relpath(repo_source, link.parent)
        if link.is_symlink():
            if link.readlink() != Path(target):
                raise RuntimeError(f"Unexpected existing link: {link}")
        elif link.exists():
            raise FileExistsError(link)
        else:
            link.symlink_to(target)
        if link.resolve() != source.resolve():
            raise RuntimeError(f"Broken production link: {link}")
        if item["checkpoint"].endswith(".safetensors"):
            item["previous_ema_export"] = {
                "sha256": item["checkpoint_sha256"],
                "bytes": item["checkpoint_bytes"],
                "tensor_count": item["ema_tensor_count"],
                "verification": item["verified"],
            }
        item.update(
            checkpoint=str(link.relative_to(HERE)),
            checkpoint_bytes=source.stat().st_size,
            checkpoint_sha256=item["source_checkpoint_sha256"],
            storage="symlink_to_production_full_checkpoint",
            link_target=target,
            verified="Link resolves to the SHA256-identified production checkpoint; evaluator loads EMA 0.999 directly",
        )
        print(f"{key}: {link} -> {target}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
