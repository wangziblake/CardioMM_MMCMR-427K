"""Python translation of ChallengeMaskGen_TestAnalysisSet_Fast.m."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .ktMaskGenerator import ktMaskGenerator
from .mat_io import save_mat_v73


def _kspace_shape(path: Path) -> tuple[int, ...]:
    try:
        content = loadmat(path, variable_names=["kspace_full", "kspace"])
        for name in ("kspace_full", "kspace"):
            if name in content:
                return content[name].shape
    except NotImplementedError:
        try:
            import h5py
        except (ImportError, ValueError) as exc:
            raise RuntimeError("Reading MATLAB v7.3 files requires a working h5py installation.") from exc
        with h5py.File(path) as content:
            for name in ("kspace_full", "kspace"):
                if name in content:
                    return tuple(reversed(content[name].shape))
    raise ValueError(f"No k-space data in {path}")


def ChallengeMaskGen_TestAnalysisSet_Fast(
    data_path: str | Path,
    mainSavePath_mask: str | Path,
    mainSavePath_kus: str | Path,
    file_nameCT: str,
    file_nameVD: str,
    file_nameID: str,
    dataName: str,
    setName: str,
) -> int:
    del mainSavePath_kus
    shape = _kspace_shape(Path(data_path))
    nx, ny = shape[:2]
    nt = shape[4] if len(shape) == 5 else 1
    if setName not in {"AnalysisSet", "TestSet", "TestAnalSet"}:
        return 0

    save_dir = Path(mainSavePath_mask) / file_nameCT / file_nameVD / file_nameID
    save_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    for pattern in ("Uniform", "ktGaussian", "ktRadial"):
        for R in (8, 16, 24):
            mask = ktMaskGenerator(nx, ny, nt, 20, R, pattern)
            save_mat_v73(save_dir / f"{dataName}_mask_{pattern}{R}.mat", {"mask": np.asarray(mask)})
            generated += 1
    return generated


challenge_mask_gen_test_analysis_set_fast = ChallengeMaskGen_TestAnalysisSet_Fast
