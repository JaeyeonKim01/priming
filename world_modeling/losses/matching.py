"""Load a pinned external DistMatch source without colliding with older installs."""
import hashlib
import importlib.util
from pathlib import Path
import sys


def load_distmatch(source):
    root = Path(source).expanduser().resolve()
    if (root / 'src/distmatch').is_dir():
        root = root / 'src/distmatch'
    elif (root / 'distmatch').is_dir():
        root = root / 'distmatch'
    if not (root / '__init__.py').is_file():
        raise FileNotFoundError(f'DistMatch source missing: {root}')
    name = '_priming_world_distmatch'
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).parent != root:
            raise RuntimeError('Cannot switch DistMatch sources in the same process')
        return module
    spec = importlib.util.spec_from_file_location(name, root / '__init__.py',
                                                 submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        for attr in ('DistMatch', 'FeatureBranches', 'Sinkhorn'):
            getattr(module, attr)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def source_identity(source):
    module = load_distmatch(source)
    root = Path(module.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes() + b'\0')
    return dict(version=module.__version__, source=str(root), sha256=digest.hexdigest())


def make_matcher(cfg, device):
    from .features import LatentDinoFeatureBranches
    package = load_distmatch(cfg.distmatch.source)
    extractor = LatentDinoFeatureBranches(cfg).to(device)
    scales = {name: str(cfg.feature_branches.scale) for name in extractor.branch_names}
    matcher = package.DistMatch(
        [package.FeatureBranches(extractor, branches=scales)],
        package.Sinkhorn(epsilon=float(cfg.distmatch.epsilon), max_iter=int(cfg.distmatch.max_iter),
                         tolerance=float(cfg.distmatch.tolerance), strict=bool(cfg.distmatch.strict)),
        mode='conditional', distributed='local', weights=dict(cfg.feature_branches.weights),
        max_pairwise_elements=int(cfg.distmatch.max_pairwise_elements),
    ).to(device).eval()
    # The historical world-model recipe uses rank-local scaling and matching.
    if getattr(matcher, 'distributed', 'local') != 'local':
        raise ValueError('World-model reproduction requires local matching')
    return matcher
