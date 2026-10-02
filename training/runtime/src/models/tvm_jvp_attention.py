"""Checked adapter for TVM's JVP-only Triton attention implementation."""

import importlib
import sys
from pathlib import Path

import torch


_TVM_ROOT = (Path(__file__).resolve().parents[2] / "external" / "tvm").resolve()
if not _TVM_ROOT.is_dir():
    raise ImportError(f"TVM source tree not found at {_TVM_ROOT}")
if str(_TVM_ROOT) not in sys.path:
    sys.path.append(str(_TVM_ROOT))

_jvp_functional = importlib.import_module("jvp_utils.functional")
if _TVM_ROOT not in Path(_jvp_functional.__file__).resolve().parents:
    raise ImportError(
        "Imported jvp_utils from an unexpected location: "
        f"{_jvp_functional.__file__}"
    )
_sdpa_jvp = _jvp_functional.sdpa_jvp


@torch._dynamo.disable
def tvm_jvp_attention(q, k, v):
    """Run TVM SDPA after validating its narrow JVP-only contract."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "TVM JVP attention expects q, k, and v with shape [B, H, S, D]"
        )
    if q.device != k.device or q.device != v.device or not q.is_cuda:
        raise ValueError("TVM JVP attention requires q, k, and v on the same CUDA device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("TVM JVP attention requires matching q, k, and v dtypes")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"TVM JVP attention supports fp16/bf16, got {q.dtype}"
        )

    batch, heads, _, head_dim = q.shape
    key_length = k.shape[-2]
    expected_kv_shape = (batch, heads, key_length, head_dim)
    if k.shape != expected_kv_shape or v.shape != expected_kv_shape:
        raise ValueError(
            "TVM JVP attention requires matching batch/head/head-dim shapes; "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if not 16 <= head_dim <= 128:
        raise ValueError(
            f"TVM JVP attention requires 16 <= head_dim <= 128, got {head_dim}"
        )

    _, q_tangent = torch.autograd.forward_ad.unpack_dual(q)
    if q_tangent is None:
        raise RuntimeError(
            "TVM sdpa_jvp must only be called inside torch.func.jvp; "
            "its output is invalid outside forward AD"
        )

    return _sdpa_jvp(q, k, v)


__all__ = ["tvm_jvp_attention"]
