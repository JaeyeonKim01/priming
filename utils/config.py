"""Load a YAML recipe with one optional shared base beside it."""
from pathlib import Path


def load_yaml(path):
    from omegaconf import OmegaConf
    path = Path(path)
    config = OmegaConf.load(path)
    base = config.pop('_base_', None)
    if base is not None:
        config = OmegaConf.merge(OmegaConf.load(path.parent/base), config)
    return OmegaConf.to_container(config, resolve=True)
