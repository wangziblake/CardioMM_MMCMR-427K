"""Python translation of ktdup.m."""

from __future__ import annotations

import numpy as np


def _ind2xy(indices: np.ndarray, x_size: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices)
    x = indices - np.floor_divide(indices - 1, x_size) * x_size
    y = np.ceil(indices / x_size)
    return x, y


def ktdup(ph: np.ndarray, ti: np.ndarray, ny: int, nt: int) -> tuple[np.ndarray, np.ndarray]:
    ph = np.asarray(ph).reshape(-1, order="F") + int(np.ceil((ny + 1) / 2))
    ti = np.asarray(ti).reshape(-1, order="F") + int(np.ceil((nt + 1) / 2))
    points = (ti - 1) * ny + ph

    duplicate_indices: list[int] = []
    unique_points, counts = np.unique(points, return_counts=True)
    for repeated in unique_points[counts != 1]:
        occurrences = np.flatnonzero(points == repeated)
        duplicate_indices.extend(occurrences[1:].tolist())

    empty = np.setdiff1d(np.arange(1, ny * nt + 1, dtype=np.int64), points)
    for duplicate_index in duplicate_indices:
        x0, y0 = _ind2xy(points[duplicate_index], ny)
        x, y = _ind2xy(empty, ny)
        distances = np.where(y == y0, np.square(x - x0), np.inf)
        chosen_position = int(np.argmin(distances))
        points[duplicate_index] = empty[chosen_position]
        empty = np.delete(empty, chosen_position)

    ph_out, ti_out = _ind2xy(points, ny)
    return ph_out - int(np.ceil((ny + 1) / 2)), ti_out - int(np.ceil((nt + 1) / 2))
