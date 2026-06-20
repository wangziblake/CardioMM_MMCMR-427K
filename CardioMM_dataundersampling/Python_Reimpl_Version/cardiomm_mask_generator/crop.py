"""Python translation of crop.m."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._utils import center_slices, target_shape


def crop(x: np.ndarray, *sizes: int | Sequence[int]) -> np.ndarray:
    target = target_shape(*sizes)
    source = np.asarray(x)
    if len(target) < source.ndim:
        target += (1,) * (source.ndim - len(target))
    if len(target) != source.ndim or any(a > b for a, b in zip(target, source.shape)):
        raise ValueError(f"cannot crop shape {source.shape} to {target}")
    if source.shape == target:
        return source.copy()
    return source[center_slices(source.shape, target)].copy()
