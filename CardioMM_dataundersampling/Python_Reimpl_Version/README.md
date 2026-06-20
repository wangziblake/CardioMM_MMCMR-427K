# CardioMM data undersampling: Python implementation

This folder provides a Python implementation of the mask-generation pipeline in
`CardioMM_dataundersampling`. It is intended to mirror the MATLAB scripts used
for testing-set and analysis-set undersampling in the CardioMM / MMCMR-427K
workflow.

The implementation supports the same mask families used by the MATLAB code:

- `Uniform`
- `ktUniform`
- `ktGaussian`
- `ktRadial`

Generated mask files are saved as MATLAB v7.3/HDF5 `.mat` files, matching the
MATLAB implementation's `save(..., '-v7.3')` output format. MATLAB can load the
Python-generated files directly with `load(...)`.

For use inside this repository, load generated Cartesian masks through
`CardioMM_reconstruction.utils.load_maskdata`. That loader restores the
MATLAB-visible dimensions for 2D/3D masks while preserving the existing 4D
radial trajectory-mask layout used by the reconstruction scripts. Direct HDF5
reads of MATLAB v7.3 files expose the reversed on-disk dimension order.

## Installation

Install the Python dependencies from this directory:

```bash
pip install -r requirements.txt
```

Reading and writing MATLAB v7.3/HDF5 files requires `h5py`, which is included in
`requirements.txt`.

## Generate one mask

```python
from cardiomm_mask_generator import kt_mask_generator

mask = kt_mask_generator(256, 255, 12, 20, 8, "Uniform")
```

For a reproducible Gaussian mask, pass `gaussian_seed`, for example:

```python
mask = kt_mask_generator(256, 255, 12, 20, 8, "ktGaussian", gaussian_seed=10)
```

## Batch generation on MMCMR-427K

Run the batch script from this `python` directory:

```bash
python MaskGeneration_TestAnalysisSet_Fast.py /mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil --set-name TestSet
```

The script follows the same directory convention as the MATLAB version. For
example, generated masks are written under:

```text
/mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil/Cine/TestSet/Mask_TaskAll/...
```

For testing, or when you do not want to write into the original data tree, call
`ChallengeMaskGen_TestAnalysisSet_Fast` directly and pass a temporary output
directory as `mainSavePath_mask`.

## MATLAB/Python consistency

The MATLAB reference used by the unit tests can be regenerated with:

```bash
matlab -batch "addpath('tests'); generate_matlab_reference('../Toolbox_Mask_Generator','tests/matlab_reference.mat')"
```

Run the consistency and property tests with:

```bash
python -m pytest -q tests/test_matlab_consistency.py tests/test_mask_properties.py
```

The tests compare center padding/cropping, random sampling helpers, and all mask
families element by element against MATLAB references.

On a 1000-file random sample from
`/mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil`, the sampled files covered
273 unique `(nx, ny, nt)` shapes. MATLAB and Python matched exactly for
`Uniform`, fixed-seed `ktGaussian`, and `ktRadial` after excluding the known odd
45-degree spokes from both sides.

## Known ktRadial difference

`ktRadial` draws pseudo-radial spokes by rotating a horizontal line with
nearest-neighbor interpolation. MATLAB `imrotate(..., 'nearest', 'crop')` and
Python `skimage.rotate(order=0)` use slightly different tie-breaking for exact
odd multiples of 45 degrees (`45/135/225/315`, including equivalent angles such
as `405`).

This can leave a small number of samples different between MATLAB and Python.
For example, in a `446 x 212` crop at 45 degrees, MATLAB had 123 additional
samples and Python had no extra samples. The Python implementation intentionally
keeps the deterministic `skimage.rotate` behavior because a robust,
shape-independent emulation of MATLAB's tie handling is not available.

## Citation

If you use this implementation or the MMCMR-427K database, please cite the
CardioMM / MMCMR-427K papers listed in the repository root README.
