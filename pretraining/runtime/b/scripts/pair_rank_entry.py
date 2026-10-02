"""Set private compiler caches before importing Torch in each rank."""
import os
from pathlib import Path
import runpy

root = Path(os.environ['PAIR_CACHE_ROOT']) / ('rank_' + os.environ['RANK'])
for name, folder in [('TORCHINDUCTOR_CACHE_DIR', 'inductor'), ('TRITON_CACHE_DIR', 'triton'), ('CUDA_CACHE_PATH', 'cuda')]:
    path = root / folder; path.mkdir(parents=True, exist_ok=True); os.environ[name] = str(path)
runpy.run_path(str(Path(__file__).with_name('train_pixel_pair.py')), run_name='__main__')
