"""Small MATLAB v7.3 I/O helpers for generated mask files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def _matlab_v73_header() -> bytes:
    text = (
        "MATLAB 7.3 MAT-file, Platform: GLNXA64, "
        "Created by CardioMM Python mask generator, HDF5 schema 1.00 ."
    )
    header = text.encode("ascii")[:116].ljust(116, b" ")
    # MATLAB v7.3 files use a 512-byte HDF5 user block. The first 128 bytes
    # contain the MAT-file banner, version, and endian indicator.
    return header + b"\x00" * 8 + b"\x00\x02" + b"IM"


def save_mat_v73(path: str | Path, variables: dict[str, np.ndarray]) -> None:
    """Save simple numeric arrays as MATLAB v7.3/HDF5 .mat files.

    This is intentionally small and only supports the numeric arrays needed by
    this package. MATLAB stores arrays in reversed dimension order in HDF5, so
    arrays are transposed on write to preserve their MATLAB-visible shape.
    """
    path = Path(path)
    with h5py.File(path, "w", userblock_size=512) as handle:
        for name, value in variables.items():
            array = np.asarray(value)
            stored = np.transpose(array, tuple(range(array.ndim - 1, -1, -1)))
            dataset = handle.create_dataset(name, data=stored, compression="gzip")
            dataset.attrs["MATLAB_class"] = np.bytes_("double")

    with path.open("r+b") as handle:
        handle.write(_matlab_v73_header())


def load_mat_v73_array(path: str | Path, name: str) -> np.ndarray:
    """Load one numeric array saved in MATLAB v7.3/HDF5 layout."""
    with h5py.File(path, "r") as handle:
        array = np.array(handle[name])
    return np.transpose(array, tuple(range(array.ndim - 1, -1, -1)))
