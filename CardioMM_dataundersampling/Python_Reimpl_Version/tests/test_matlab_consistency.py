from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from cardiomm_mask_generator import (
    crop,
    kt_gaussian_sampling,
    kt_radial_sampling,
    kt_uniform_sampling,
    ktMaskGenerator,
    randp,
    uniform_sampling,
    zpad,
)
from cardiomm_mask_generator.ktdup import ktdup


REFERENCE = Path(__file__).with_name("matlab_reference.mat")


def test_against_matlab_reference():
    reference = loadmat(REFERENCE)
    x2 = np.arange(1, 25).reshape((4, 6), order="F")
    x3 = np.arange(1, 25).reshape((2, 3, 4), order="F")

    actual = {
        # --- original cases ---
        "zpad_even": zpad(x2, (8, 10)),
        "zpad_odd": zpad(x2, (7, 9)),
        "crop_even": crop(zpad(x2, (8, 10)), (4, 6)),
        "crop_odd": crop(zpad(x2, (7, 9)), (4, 6)),
        "uniform": uniform_sampling(32, 31, 8, 4),
        "kt_uniform": kt_uniform_sampling(32, 31, 7, 8, 4),
        "kt_gaussian": kt_gaussian_sampling(32, 31, 7, 8, 4, 0.2, 10),
        "kt_radial": kt_radial_sampling(32, 31, 3, 8, 4.8, 137.5, True),

        # --- 3D zpad / crop ---
        "zpad_3d": zpad(x3, (4, 6, 8)),
        "crop_3d": crop(zpad(x3, (4, 6, 8)), (2, 3, 4)),

        # --- ktRadialSampling with cropcorner=False ---
        "kt_radial_nocrop": kt_radial_sampling(32, 31, 3, 8, 4.8, 137.5, False),

        # --- ktMaskGenerator end-to-end (all four patterns) ---
        # Uniform: dispatch + repmat logic
        "ktmask_uniform": ktMaskGenerator(32, 31, 7, 8, 8, "Uniform"),
        # ktUniform: dispatch logic
        "ktmask_ktuniform": ktMaskGenerator(32, 31, 7, 8, 8, "ktUniform"),
        # ktGaussian: dispatch + alpha=0.2 default; gaussian_seed fixes RNG for reproducibility
        "ktmask_ktgaussian": ktMaskGenerator(32, 31, 7, 8, 8, "ktGaussian", gaussian_seed=42),
        # ktRadial: dispatch applies R*0.6 before calling ktRadialSampling
        "ktmask_ktradial": ktMaskGenerator(32, 31, 3, 8, 8, "ktRadial"),

        # --- nt=1 (4D kspace case, all four patterns) ---
        "uniform_nt1":     ktMaskGenerator(32, 31, 1, 8, 8, "Uniform"),
        "kt_uniform_nt1":  kt_uniform_sampling(32, 31, 1, 8, 4),
        "kt_gaussian_nt1": kt_gaussian_sampling(32, 31, 1, 8, 8, 0.2, 7),
        "kt_radial_nt1":   kt_radial_sampling(32, 31, 1, 8, 4.8, 137.5, True),
    }

    for name, value in actual.items():
        np.testing.assert_array_equal(value, reference[name], err_msg=name)

    # nt=1 cases must return a 2-D array (Python squeezes the trailing dim).
    # numpy broadcasting makes assert_array_equal pass even with (nx,ny) vs
    # (nx,ny,1), so we check shape explicitly here.
    for name in ("uniform_nt1", "kt_uniform_nt1", "kt_gaussian_nt1", "kt_radial_nt1"):
        assert actual[name].ndim == 2, (
            f"{name}: expected 2-D array for nt=1, got shape {actual[name].shape}"
        )


def test_against_matlab_reference_extended():
    """Direct MATLAB comparisons for randp, ktdup, and UniformSampling(R=8).

    Requires regenerating matlab_reference.mat by running:
        generate_matlab_reference('<toolbox_path>', '<output_path>')
    in MATLAB after updating generate_matlab_reference.m.
    """
    reference = loadmat(REFERENCE)

    ph_in = np.array([0, 0, 1, 2], dtype=np.int64)
    ti_in = np.array([0, 0, 0, 1], dtype=np.int64)
    ph_out, ti_out = ktdup(ph_in, ti_in, ny=6, nt=4)

    extended = {
        "randp_weighted": randp(np.array([1.0, 2.0, 1.0]), 42, 20, 1),
        "randp_equal":    randp(np.array([1.0, 1.0, 1.0, 1.0]), 7, 5, 3),
        "ktdup_ph":       ph_out,
        "ktdup_ti":       ti_out,
        "uniform_r8":     uniform_sampling(32, 31, 8, 8),
    }

    missing = [k for k in extended if k not in reference]
    if missing:
        pytest.skip(
            f"Keys {missing} not in matlab_reference.mat — "
            "re-run generate_matlab_reference.m in MATLAB to regenerate."
        )

    for name, value in extended.items():
        ref = reference[name]
        # MATLAB saves 1-D vectors as (N,1) column vectors; squeeze before
        # comparing so shape (4,) matches (4,1) when values are identical.
        if value.ndim == 1:
            ref = ref.squeeze()
        np.testing.assert_array_equal(value, ref, err_msg=name)


if __name__ == "__main__":
    test_against_matlab_reference()
    print("All MATLAB/Python consistency checks passed.")
