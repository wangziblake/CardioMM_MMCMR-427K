"""Python translation of ktMaskGenerator.m."""

from __future__ import annotations

import numpy as np

from .UniformSampling import UniformSampling
from .ktGaussianSampling import ktGaussianSampling
from .ktRadialSampling import ktRadialSampling
from .ktUniformSampling import ktUniformSampling


def ktMaskGenerator(
    nx: int,
    ny: int,
    nt: int,
    ncalib: int,
    R: int,
    pattern: str,
    *,
    gaussian_seed: int | None = None,
) -> np.ndarray:
    if pattern == "Uniform":
        mask_2d = UniformSampling(nx, ny, ncalib, R)
        mask = np.repeat(mask_2d[:, :, np.newaxis], nt, axis=2)
        return mask.squeeze(axis=-1) if nt == 1 else mask
    if pattern == "ktUniform":
        return ktUniformSampling(nx, ny, nt, ncalib, R)
    if pattern == "ktGaussian":
        seed = int(np.random.randint(1, 1001)) if gaussian_seed is None else gaussian_seed
        return ktGaussianSampling(nx, ny, nt, ncalib, R, 0.2, seed)
    if pattern == "ktRadial":
        return ktRadialSampling(nx, ny, nt, ncalib, R * 0.6, 137.5, True)
    raise ValueError("No selected undersampling pattern. Please choose the proper one.")


kt_mask_generator = ktMaskGenerator
