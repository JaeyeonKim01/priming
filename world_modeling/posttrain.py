"""Conditional distributional post-training for RT-1 and Bridge."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from omegaconf import OmegaConf
from world_modeling.utils.config import parser, load_config, configure_assets


def main():
    args = parser('posttrain').parse_args()
    cfg = load_config(args, 'posttrain')
    if args.check_config:
        print(OmegaConf.to_yaml(cfg)); return
    configure_assets(cfg)
    from world_modeling.utils.training import train
    raise SystemExit(train(cfg))


if __name__ == '__main__': main()
