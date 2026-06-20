"""Shared MATLAB-compatibility helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def target_shape(*sizes: int | Sequence[int]) -> tuple[int, ...]:
    if len(sizes) == 1 and not np.isscalar(sizes[0]):
        return tuple(int(value) for value in sizes[0])
    return tuple(int(value) for value in sizes)


def center_slices(container: Sequence[int], content: Sequence[int]) -> tuple[slice, ...]:
    starts = [int(np.floor(big / 2) + np.ceil(-small / 2)) for big, small in zip(container, content)]
    return tuple(slice(start, start + small) for start, small in zip(starts, content))


def matlab_round_positive(value: np.ndarray | float) -> np.ndarray:
    return np.floor(np.asarray(value) + 0.5).astype(np.int64)
