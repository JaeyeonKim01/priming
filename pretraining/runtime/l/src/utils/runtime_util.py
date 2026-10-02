import torch


def _warmup(fn, n=3):
    """run fn n times, then sync."""
    for _ in range(n):
        fn()
    torch.cuda.synchronize()


def normalize_param_name(name: str) -> str:
    """
    normalize parameter name by stripping wrapper prefixes.
    
    handles prefixes added by:
    - torch.compile(): '_orig_mod.'
    - DistributedDataParallel: 'module.'
    - combinations like '_orig_mod.module.' or 'module._orig_mod.'
    
    this ensures consistent parameter naming regardless of wrapper order or nesting.
    useful for EMA, checkpoint loading, and parameter matching across different training setups.
    """
    prefixes = ('_orig_mod.', 'module.')
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
    return name
