"""Repository paths, offline assets, and private training compiler caches."""
import hashlib
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def resolve(value, root=ROOT, *, canonical=False):
    path = Path(os.path.expandvars(str(value))).expanduser()
    path = path if path.is_absolute() else Path(root)/path
    return path.resolve() if canonical else path


def configure_caches(*, hf_home=None, torch_home=None, canonical=False):
    homes = []
    for name, selected, default in (('HF_HOME', hf_home, 'assets/huggingface'), ('TORCH_HOME', torch_home, 'assets/torch')):
        home = resolve(selected or os.environ.get(name, default), canonical=canonical)
        homes.append(home)
        os.environ[name] = str(home)
    os.environ.update(HF_HUB_CACHE=str(Path(os.environ['HF_HOME'])/'hub'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return tuple(homes)


def configure_environment(output, *, phase, hf_home=None, torch_home=None, threads=8, deterministic=False):
    """Configure before importing Torch; keep compiler caches private to each run/job/rank."""
    os.environ.setdefault('OMP_NUM_THREADS', str(threads))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '4')
    if deterministic:
        os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', NCCL_ALGO='Ring', NCCL_PROTO='Simple')
    homes = configure_caches(hf_home=hf_home, torch_home=torch_home, canonical=True)
    identity = hashlib.sha256(str(output).encode()).hexdigest()[:16]
    cache = Path(os.environ.get('PRIMING_LOCAL_TMP', '/tmp'))/f'priming-{os.getuid()}'/phase/identity
    cache = cache/os.environ.get('SLURM_JOB_ID', str(os.getppid()))/f"rank_{os.environ.get('RANK', '0')}"
    for variable, name in (('TMPDIR', 'tmp'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                           ('TRITON_CACHE_DIR', 'triton'), ('CUDA_CACHE_PATH', 'cuda')):
        folder = cache/name
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(folder)
    tempfile.tempdir = os.environ['TMPDIR']
    return homes
