"""Property-based tests for the mask generator — no MATLAB reference required."""

from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiomm_mask_generator import ktMaskGenerator, ktdup, randp, zpad, crop
from cardiomm_mask_generator._utils import matlab_round_positive
from cardiomm_mask_generator.ChallengeMaskGen_TestAnalysisSet_Fast import (
    ChallengeMaskGen_TestAnalysisSet_Fast,
)
from cardiomm_mask_generator.mat_io import load_mat_v73_array

PATTERNS = ["Uniform", "ktUniform", "ktGaussian", "ktRadial"]
R_VALUES = [8, 16, 24]


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("nt", [1, 7])
def test_mask_shape(pattern, nt):
    nx, ny, ncalib = 32, 31, 8
    m = ktMaskGenerator(nx, ny, nt, ncalib, 8, pattern, gaussian_seed=1)
    expected = (nx, ny) if nt == 1 else (nx, ny, nt)
    assert m.shape == expected, f"{pattern} nt={nt}: expected {expected}, got {m.shape}"


# ---------------------------------------------------------------------------
# Binary values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("R", R_VALUES)
def test_mask_is_binary(pattern, R):
    m = ktMaskGenerator(32, 31, 7, 8, R, pattern, gaussian_seed=1)
    assert set(np.unique(m)).issubset({0.0, 1.0}), f"{pattern} R={R}: non-binary values"


# ---------------------------------------------------------------------------
# ACS (autocalibration signal) region is always fully sampled
#
# ktGaussian / ktUniform / Uniform: ACS = full-height strip of ncalib center columns
# ktRadial:                          ACS = ncalib × ncalib center square
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pattern", PATTERNS)
def test_acs_coverage(pattern):
    nx, ny, nt, ncalib = 32, 31, 7, 8
    m = ktMaskGenerator(nx, ny, nt, ncalib, 8, pattern, gaussian_seed=1)
    if pattern == "ktRadial":
        acs = zpad(np.ones((ncalib, ncalib, nt)), (nx, ny, nt))
    else:
        acs = zpad(np.ones((nx, ncalib, nt)), (nx, ny, nt))
    assert np.all(m[acs > 0] == 1), f"{pattern}: ACS region not fully sampled"


# ---------------------------------------------------------------------------
# Acceleration factor is in a sensible range
# Actual AF < R because ACS adds extra lines; ktRadial further diverges due to
# R*0.6 scaling and radial geometry.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pattern,R,af_max", [
    ("Uniform",    8,  8),
    ("ktUniform",  8,  8),
    ("ktGaussian", 8,  8),
    ("ktRadial",   8, 12),   # wider tolerance: radial geometry + R*0.6 scaling
])
def test_acceleration_factor_upper_bound(pattern, R, af_max):
    m = ktMaskGenerator(256, 255, 12, 20, R, pattern, gaussian_seed=1)
    af = m.size / m.sum()
    assert af <= af_max, f"{pattern} R={R}: AF={af:.2f} exceeds upper bound {af_max}"


# ---------------------------------------------------------------------------
# randp: probability-weighted random integer sampler
# ---------------------------------------------------------------------------

def test_randp_deterministic():
    """Same seed always gives the same result."""
    p = np.array([1.0, 2.0, 1.0])
    r1 = randp(p, 7, 500)
    r2 = randp(p, 7, 500)
    np.testing.assert_array_equal(r1, r2)


def test_randp_different_seeds_differ():
    p = np.array([1.0, 1.0, 1.0])
    r1 = randp(p, 0, 200)
    r2 = randp(p, 1, 200)
    assert not np.array_equal(r1, r2)


def test_randp_concentrated_probability():
    """All mass on bin 2 → every output must be 2."""
    p = np.array([0.0, 1.0, 0.0])
    r = randp(p, 42, 100)
    assert np.all(r == 2), f"Expected all 2s, got unique values: {np.unique(r)}"


def test_randp_output_range():
    """Outputs are in [1, len(p)]."""
    p = np.array([1.0, 1.0, 1.0, 1.0])
    r = randp(p, 0, 1000)
    assert r.min() >= 1
    assert r.max() <= len(p)


def test_randp_approximate_distribution():
    """With uniform probabilities, each bin appears roughly equally."""
    p = np.array([1.0, 1.0, 1.0])
    r = randp(p, 0, 30000)
    counts = np.bincount(r.reshape(-1), minlength=len(p) + 1)[1:]
    expected = r.size / len(p)
    assert np.all(np.abs(counts - expected) < 0.05 * r.size), (
        f"Distribution too uneven: {counts}"
    )


# ---------------------------------------------------------------------------
# ktdup: duplicate-point remover on the k-t grid
# ---------------------------------------------------------------------------

def test_ktdup_removes_all_duplicates():
    """After ktdup, every (ph, ti) pair is unique."""
    ph = np.array([0, 0, 1, 2])
    ti = np.array([0, 0, 0, 1])
    ph_out, ti_out = ktdup(ph, ti, ny=6, nt=4)
    pairs = list(zip(ph_out.tolist(), ti_out.tolist()))
    assert len(pairs) == len(set(pairs)), f"Duplicates remain: {pairs}"


def test_ktdup_preserves_count():
    """Output has the same number of points as input."""
    ph = np.array([0, 0, 1])
    ti = np.array([0, 0, 1])
    ph_out, ti_out = ktdup(ph, ti, ny=5, nt=3)
    assert len(ph_out) == len(ph)
    assert len(ti_out) == len(ti)


def test_ktdup_no_op_when_unique():
    """When there are no duplicates, positions are unchanged."""
    ph = np.array([0, 1, 2])
    ti = np.array([0, 0, 0])
    ph_out, ti_out = ktdup(ph, ti, ny=5, nt=3)
    np.testing.assert_array_equal(ph_out, ph)
    np.testing.assert_array_equal(ti_out, ti)


# ---------------------------------------------------------------------------
# zpad / crop: edge cases and error handling
# ---------------------------------------------------------------------------

def test_zpad_noop():
    """zpad returns a copy when source already matches target size."""
    x = np.arange(24, dtype=float).reshape(4, 6)
    out = zpad(x, (4, 6))
    np.testing.assert_array_equal(out, x)
    assert out is not x  # must be a copy, not the same object

def test_crop_noop():
    """crop returns a copy when source already matches target size."""
    x = np.arange(24, dtype=float).reshape(4, 6)
    out = crop(x, 4, 6)
    np.testing.assert_array_equal(out, x)
    assert out is not x

def test_zpad_4d():
    """zpad works correctly on 4-D arrays."""
    x = np.ones((2, 3, 4, 5))
    out = zpad(x, (4, 6, 8, 10))
    assert out.shape == (4, 6, 8, 10)
    assert out.sum() == x.sum()  # no values lost

def test_zpad_raises_when_target_smaller():
    with pytest.raises(ValueError):
        zpad(np.ones((5, 5)), (3, 3))

def test_crop_raises_when_target_larger():
    with pytest.raises(ValueError):
        crop(np.ones((3, 3)), 5, 5)


# ---------------------------------------------------------------------------
# ktMaskGenerator / ktRadialSampling: error handling
# ---------------------------------------------------------------------------

def test_ktmask_generator_invalid_pattern():
    with pytest.raises(ValueError, match="No selected undersampling pattern"):
        ktMaskGenerator(10, 10, 3, 4, 8, "InvalidPattern")

def test_ktradial_r_too_large():
    from cardiomm_mask_generator.ktRadialSampling import ktRadialSampling
    with pytest.raises(ValueError, match="R is too large"):
        ktRadialSampling(32, 31, 3, 8, 200.0, 137.5, True)


# ---------------------------------------------------------------------------
# ChallengeMaskGen_TestAnalysisSet_Fast: integration test with synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_mat(tmp_path):
    """Write a small synthetic 5-D k-space .mat file and return its path."""
    kspace = np.ones((32, 31, 4, 3, 7), dtype=np.complex64)
    path = tmp_path / "cine_sax.mat"
    savemat(str(path), {"kspace_full": kspace})
    return path, tmp_path


def test_challenge_mask_gen_returns_correct_count(synthetic_mat):
    mat_path, tmp_path = synthetic_mat
    count = ChallengeMaskGen_TestAnalysisSet_Fast(
        mat_path, tmp_path / "masks", tmp_path / "kus",
        "Center001", "Siemens", "P001", "cine_sax", "TestSet",
    )
    assert count == 9  # 3 patterns × 3 R values


def test_challenge_mask_gen_creates_all_files(synthetic_mat):
    mat_path, tmp_path = synthetic_mat
    ChallengeMaskGen_TestAnalysisSet_Fast(
        mat_path, tmp_path / "masks", tmp_path / "kus",
        "Center001", "Siemens", "P001", "cine_sax", "TestSet",
    )
    save_dir = tmp_path / "masks" / "Center001" / "Siemens" / "P001"
    expected = {
        f"cine_sax_mask_{pat}{R}.mat"
        for pat in ("Uniform", "ktGaussian", "ktRadial")
        for R in (8, 16, 24)
    }
    actual = {f.name for f in save_dir.glob("*.mat")}
    assert actual == expected


def test_challenge_mask_gen_mask_shape_and_values(synthetic_mat):
    mat_path, tmp_path = synthetic_mat
    ChallengeMaskGen_TestAnalysisSet_Fast(
        mat_path, tmp_path / "masks", tmp_path / "kus",
        "Center001", "Siemens", "P001", "cine_sax", "TestSet",
    )
    save_dir = tmp_path / "masks" / "Center001" / "Siemens" / "P001"
    for mat_file in save_dir.glob("*.mat"):
        mask = load_mat_v73_array(mat_file, "mask")
        assert set(np.unique(mask)).issubset({0.0, 1.0}), f"{mat_file.name}: not binary"
        # 5-D kspace → nt=7, expect 3-D mask (nx, ny, nt)
        assert mask.ndim == 3, f"{mat_file.name}: expected 3-D mask, got {mask.ndim}-D"
        assert mask.shape == (32, 31, 7), f"{mat_file.name}: unexpected shape {mask.shape}"


def test_challenge_mask_gen_skips_wrong_set(synthetic_mat):
    """Function must produce no output for sets other than TestSet/AnalysisSet/TestAnalSet."""
    mat_path, tmp_path = synthetic_mat
    count = ChallengeMaskGen_TestAnalysisSet_Fast(
        mat_path, tmp_path / "masks", tmp_path / "kus",
        "Center001", "Siemens", "P001", "cine_sax", "TrainingSet",
    )
    assert count == 0


def test_challenge_mask_gen_4d_kspace(tmp_path):
    """4-D k-space (no temporal dim) → nt=1, masks should be 2-D."""
    kspace = np.ones((32, 31, 4, 3), dtype=np.complex64)
    mat_path = tmp_path / "cine_sax.mat"
    savemat(str(mat_path), {"kspace_full": kspace})

    ChallengeMaskGen_TestAnalysisSet_Fast(
        mat_path, tmp_path / "masks", tmp_path / "kus",
        "Center001", "Siemens", "P001", "cine_sax", "TestSet",
    )
    save_dir = tmp_path / "masks" / "Center001" / "Siemens" / "P001"
    for mat_file in save_dir.glob("*.mat"):
        mask = load_mat_v73_array(mat_file, "mask")
        assert mask.ndim == 2, f"{mat_file.name}: expected 2-D mask for nt=1, got {mask.ndim}-D"


# ---------------------------------------------------------------------------
# matlab_round_positive: MATLAB-compatible rounding (0.5 rounds up, not
# to even like Python's built-in round)
# ---------------------------------------------------------------------------

def test_matlab_round_half_up():
    """0.5 must round to 1, not 0 (Python banker's rounding gives 0)."""
    assert matlab_round_positive(0.5) == 1
    assert matlab_round_positive(1.5) == 2
    assert matlab_round_positive(2.5) == 3   # Python round(2.5) == 2

def test_matlab_round_differs_from_python_builtin():
    """Confirm the difference from Python's round() for the .5 case."""
    assert round(2.5) == 2                   # Python banker's rounding
    assert matlab_round_positive(2.5) == 3   # MATLAB-style

def test_matlab_round_array():
    result = matlab_round_positive(np.array([0.4, 0.5, 1.5, 2.5, -0.5]))
    np.testing.assert_array_equal(result, [0, 1, 2, 3, 0])

def test_matlab_round_integers_unchanged():
    """Integers should pass through without change."""
    result = matlab_round_positive(np.array([1, 2, 3, 10]))
    np.testing.assert_array_equal(result, [1, 2, 3, 10])

def test_matlab_round_negative_half():
    """-0.5 rounds to 0 (floor(-0.5 + 0.5) = floor(0) = 0)."""
    assert matlab_round_positive(-0.5) == 0
    assert matlab_round_positive(-1.5) == -1


# ---------------------------------------------------------------------------
# ktdup: algorithmic constraints
# ---------------------------------------------------------------------------

def test_ktdup_stays_in_same_frame():
    """A displaced duplicate must land in the same time frame, not another.

    MATLAB nearestvac uses dis2(dis2>eps)=inf, so cross-frame moves are
    forbidden. Python mirrors this with np.where(y == y0, ..., np.inf).
    """
    # two points at (ph=0, ti=0); vacant slots available in both ti=0 and ti=1
    ph = np.array([0, 0])
    ti = np.array([0, 0])
    ph_out, ti_out = ktdup(ph, ti, ny=5, nt=3)
    # one point stays at ti=0, the displaced duplicate must also stay at ti=0
    assert np.all(ti_out == 0), (
        f"Displaced duplicate moved to a different time frame: ti_out={ti_out}"
    )

def test_ktdup_unique_after_multiple_duplicates():
    """Multiple simultaneous duplicates are all resolved."""
    ph = np.array([0, 0, 1, 1, 2])
    ti = np.array([0, 0, 0, 0, 0])
    ph_out, ti_out = ktdup(ph, ti, ny=8, nt=4)
    pairs = list(zip(ph_out.tolist(), ti_out.tolist()))
    assert len(pairs) == len(set(pairs)), f"Duplicates remain: {pairs}"
    assert len(ph_out) == len(ph)


# ---------------------------------------------------------------------------
# randp: edge cases
# ---------------------------------------------------------------------------

def test_randp_zero_sum_probability():
    """All-zero probability vector should return all zeros (not crash)."""
    p = np.array([0.0, 0.0, 0.0])
    r = randp(p, 42, 10)
    assert np.all(r == 0), f"Expected all zeros, got {np.unique(r)}"

def test_randp_single_element():
    """Single-element probability → always returns 1."""
    p = np.array([1.0])
    r = randp(p, 0, 20)
    assert np.all(r == 1)

def test_randp_shape_2d():
    """randp respects 2D output shape request."""
    p = np.array([1.0, 1.0, 1.0])
    r = randp(p, 5, 4, 6)
    assert r.shape == (4, 6)
    assert r.min() >= 1
    assert r.max() <= len(p)


# ---------------------------------------------------------------------------
# nt=1 squeeze contract: ktMaskGenerator must return 2-D when nt=1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pattern", PATTERNS)
def test_mask_nt1_is_2d(pattern):
    """All patterns must squeeze (nx, ny, 1) → (nx, ny) for nt=1."""
    m = ktMaskGenerator(32, 31, 1, 8, 8, pattern, gaussian_seed=1)
    assert m.ndim == 2, f"{pattern} nt=1: expected 2D, got shape {m.shape}"
    assert m.shape == (32, 31), f"{pattern} nt=1: expected (32,31), got {m.shape}"
