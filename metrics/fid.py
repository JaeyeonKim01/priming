"""Official FID statistics and the pinned LTH14 NumPy eigenspectrum formula.

Adapted from torch-fidelity / TTUR (Apache 2.0); see the encoder license.
"""
import numpy as np
import torch


def feature_statistics(features):
    if not torch.is_tensor(features) or features.ndim != 2:
        raise ValueError('Official FID needs feature rows [N, D]')
    rows = features.numpy()
    return dict(mu=np.mean(rows, axis=0), sigma=np.cov(rows, rowvar=False))


def fid_distance(generated, reference):
    # Preserve the reference dtypes, including the historical FP32 mean term.
    mu1, sigma1 = (generated[key].astype(reference[key].dtype) for key in ('mu', 'sigma'))
    mu2, sigma2 = reference['mu'], reference['sigma']
    if mu1.ndim != 1 or mu1.shape != mu2.shape or sigma1.shape != sigma2.shape:
        raise ValueError('Generated and reference FID shapes differ')
    difference = mu1-mu2
    trace_sqrt = np.sum(np.sqrt(np.linalg.eigvals(sigma1.dot(sigma2)).astype('complex128')).real)
    value = float(difference.dot(difference)+np.trace(sigma1)+np.trace(sigma2)-2*trace_sqrt)
    if not np.isfinite(value):
        raise FloatingPointError('Nonfinite official FID')
    return value
