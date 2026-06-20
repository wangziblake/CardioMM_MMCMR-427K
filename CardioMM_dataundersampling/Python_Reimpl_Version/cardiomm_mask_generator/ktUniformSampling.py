"""Python translation of ktUniformSampling.m."""

import numpy as np
from scipy.linalg import toeplitz

from ._utils import matlab_round_positive
from .ktdup import ktdup
from .zpad import zpad


def ktUniformSampling(nx: int, ny: int, nt: int, ncalib: int, R: int) -> np.ndarray:
    ptmp = np.zeros(ny)
    ptmp[matlab_round_positive(np.arange(1, ny + 1, R)) - 1] = 1
    ttmp = np.zeros(nt)
    ttmp[matlab_round_positive(np.arange(1, nt + 1, R)) - 1] = 1
    top = toeplitz(ptmp, ttmp)
    linear_indices = np.flatnonzero(top.reshape(-1, order="F")) + 1
    ph = np.remainder(linear_indices - 1, ny) - np.floor(ny / 2).astype(int)
    ti = np.floor_divide(linear_indices - 1, ny) - np.floor(nt) / 2
    ph, ti = ktdup(ph, ti, ny, nt)

    sample = np.zeros((ny, nt), order="F")
    indices = matlab_round_positive(ny * (ti + np.floor(nt / 2)) + (ph + np.floor(ny / 2) + 1))
    indices[indices <= 0] = 1
    sample.reshape(-1, order="F")[indices - 1] = 1
    acs = zpad(np.ones((nx, ncalib, nt)), (nx, ny, nt))
    ktus = np.repeat(sample[np.newaxis, :, :], nx, axis=0)
    result = (ktus + acs > 0).astype(np.float64)
    return result.squeeze(axis=-1) if nt == 1 else result


kt_uniform_sampling = ktUniformSampling
