"""Python translation of ktRadialSampling.m."""

import numpy as np
from skimage.transform import rotate

from ._utils import matlab_round_positive
from .crop import crop
from .zpad import zpad


def ktRadialSampling(
    nx: int, ny: int, nt: int, ncalib: int, R: float, angle4next: float, cropcorner: bool
) -> np.ndarray:
    beams = int(np.floor((1 / R) * 180))
    if beams <= 0:
        raise ValueError("R is too large: radial mask would contain no beams")
    side = max(nx, ny) if cropcorner else int(np.ceil(np.sqrt(2) * max(nx, ny)))
    auxiliary = np.zeros((side, side))
    auxiliary[int(matlab_round_positive(side / 2)) - 1, :] = 1
    angle_step = 180 / beams

    ktus = np.zeros((nx, ny, nt))
    for frame in range(nt):
        start = angle4next * frame
        angles = np.arange(start, 180 + start + angle_step * 1e-10, angle_step)
        image = np.zeros((nx, ny))
        for angle in angles:
            # Known difference from MATLAB: skimage.rotate(order=0) and
            # MATLAB imrotate(..., 'nearest', 'crop') break nearest-neighbor
            # ties differently for exact odd multiples of 45 degrees. We keep
            # skimage's deterministic behavior here because a robust,
            # shape-independent MATLAB tie emulation is not available.
            rotated = rotate(
                auxiliary, angle, resize=False, order=0, mode="constant", cval=0, preserve_range=True
            )
            image += crop(rotated, nx, ny)
        ktus[:, :, frame] = image
    acs = zpad(np.ones((ncalib, ncalib, nt)), (nx, ny, nt))
    result = (ktus + acs > 0).astype(np.float64)
    return result.squeeze(axis=-1) if nt == 1 else result


kt_radial_sampling = ktRadialSampling
