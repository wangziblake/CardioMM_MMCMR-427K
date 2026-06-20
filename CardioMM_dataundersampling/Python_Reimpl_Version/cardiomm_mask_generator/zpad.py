"""Python translation of zpad.m."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._utils import center_slices, target_shape


def zpad(x: np.ndarray, *sizes: int | Sequence[int]) -> np.ndarray:
    target = target_shape(*sizes)
    source = np.asarray(x)
    source_shape = source.shape + (1,) * (len(target) - source.ndim)
    if len(source_shape) != len(target) or any(a > b for a, b in zip(source_shape, target)):
        raise ValueError(f"cannot pad shape {source.shape} to {target}")
    source = source.reshape(source_shape)
    if source_shape == target:
        return source.copy()
    result = np.zeros(target, dtype=source.dtype)
    result[center_slices(target, source_shape)] = source
    return result
