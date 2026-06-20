"""Python translation of UniformSampling.m."""

import numpy as np

from .zpad import zpad


def UniformSampling(nx: int, ny: int, ncalib: int, R: int) -> np.ndarray:
    mask = zpad(np.ones((nx, ncalib), dtype=np.float64), (nx, ny))
    mask[:, ::R] = 1
    return mask


uniform_sampling = UniformSampling
