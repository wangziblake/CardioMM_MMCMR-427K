"""Python translation of randp.m."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._utils import target_shape


def randp(probabilities: np.ndarray, seed: int, *shape: int | Sequence[int]) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1, order="F")
    if np.any(probabilities < 0):
        raise ValueError("All probabilities should be 0 or larger.")
    output_shape = target_shape(*shape)
    total = probabilities.sum()
    if probabilities.size == 0 or total == 0:
        return np.zeros(output_shape, dtype=np.int64)
    # Generate a flat sequence then reshape in Fortran (column-major) order to
    # match MATLAB's rand(M,N), which fills arrays column by column.
    n = int(np.prod(output_shape))
    flat = np.random.RandomState(int(seed)).rand(n)
    random_values = flat.reshape(output_shape, order="F")
    edges = np.concatenate(([0.0], np.cumsum(probabilities))) / total
    return np.searchsorted(edges, random_values, side="right").astype(np.int64)
