"""Python translation of ktGaussianSampling.m."""

import numpy as np

from ._utils import matlab_round_positive
from .ktdup import ktdup
from .randp import randp
from .zpad import zpad


def ktGaussianSampling(
    nx: int, ny: int, nt: int, ncalib: int, R: int, alpha: float, seed: int
) -> np.ndarray:
    p1 = np.arange(-np.floor(ny / 2), np.ceil(ny / 2), dtype=np.float64)
    readout_lines = int(matlab_round_positive(ny / R))
    ti = np.zeros(readout_lines * nt, dtype=np.int64)
    ph = np.zeros(readout_lines * nt, dtype=np.int64)
    sigma = ny / 5
    probability = 0.1 + alpha / (1 - alpha + 1e-10) * np.exp(-(p1**2) / (sigma**2))
    temporary_seeds = matlab_round_positive(1e6 * np.random.RandomState(int(seed)).rand(nt))

    index = 0
    for frame_offset, frame in enumerate(range(-int(np.floor(nt / 2)), int(np.ceil(nt / 2)))):
        chosen = randp(probability, int(temporary_seeds[frame_offset]), readout_lines, 1).reshape(-1, order="F")
        chosen = chosen - int(np.floor(ny / 2)) - 1
        ti[index : index + readout_lines] = frame
        ph[index : index + readout_lines] = chosen
        index += readout_lines

    ph, ti = ktdup(ph, ti, ny, nt)
    sample = np.zeros((ny, nt), order="F")
    indices = matlab_round_positive(ny * (ti + np.floor(nt / 2)) + (ph + np.floor(ny / 2) + 1))
    sample.reshape(-1, order="F")[indices - 1] = 1
    acs = zpad(np.ones((nx, ncalib, nt)), (nx, ny, nt))
    ktus = np.repeat(sample[np.newaxis, :, :], nx, axis=0)
    result = (ktus + acs > 0).astype(np.float64)
    return result.squeeze(axis=-1) if nt == 1 else result


kt_gaussian_sampling = ktGaussianSampling
