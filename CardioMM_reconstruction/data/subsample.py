"""
More undersampling with 1D-Random, 1D-Uniform, 2D-Radial with fixed ACS - pytorch
Created on 2025/12/09
@author: Zi Wang
Email: Zi Wang (zi.wang@imperial.ac.uk)
If you want to use this code, please cite our relevant papers in the GitHub page.
"""

import contextlib
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

from fastmri.data.subsample import MagicMaskFunc, MagicMaskFractionFunc


@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]]):
    """A context manager for temporarily adjusting the random seed."""
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    When called, ``MaskFunc`` uses internal functions create mask by 1)
    creating a mask for the k-space center, 2) create a mask outside of the
    k-space center, and 3) combining them into a total mask. The internals are
    handled by ``sample_mask``, which calls ``calculate_center_mask`` for (1)
    and ``calculate_acceleration_mask`` for (2). The combination is executed
    in the ``MaskFunc`` ``__call__`` function.

    If you would like to implement a new mask, simply subclass ``MaskFunc``
    and overwrite the ``sample_mask`` logic. See examples in ``RandomMaskFunc``
    and ``EquispacedMaskFunc``.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        allow_any_combination: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            allow_any_combination: Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``.
            seed: Seed for starting the internal random number generator of the
                ``MaskFunc``.
        """
        if len(center_fractions) != len(accelerations) and not allow_any_combination:
            raise ValueError(
                "Number of center fractions should match number of accelerations "
                "if allow_any_combination is False."
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int, int, str]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_mask, accel_mask, num_low_frequencies, acceleration, mask_type = self.sample_mask(shape, offset)

        # combine masks together
        return torch.max(center_mask, accel_mask), num_low_frequencies, acceleration, mask_type

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, str]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_cols = shape[-2]
        num_rows = shape[-3]
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, num_rows, acceleration, offset, num_low_frequencies
            ),
            shape,
        )
        mask_type = "unknown"
        return center_mask, acceleration_mask, num_low_frequencies, acceleration, mask_type

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape."""
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols

        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        num_rows: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking (for equispaced masks).
            num_low_frequencies: Integer count of low-frequency lines sampled.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:
        """
        Build center mask based on number of low frequencies.

        Args:
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample.

        Returns:
            A mask for hte low spatial frequencies of k-space.
        """
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad: pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs

        return mask

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        if self.allow_any_combination:
            return self.rng.choice(self.center_fractions), self.rng.choice(
                self.accelerations
            )
        else:
            choice = self.rng.randint(len(self.center_fractions))
            return self.center_fractions[choice], self.accelerations[choice]


class RandomMaskFunc(MaskFunc):
    """
    Creates a random sub-sampling mask of a given shape. FastMRI multi-coil knee dataset uses this mask type.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the ``RandomMaskFunc`` object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        num_rows: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        prob = (num_cols / acceleration - num_low_frequencies) / (
            num_cols - num_low_frequencies
        )

        return self.rng.uniform(size=num_cols) < prob


class EquiSpacedMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced k-space lines.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        num_rows: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask


class EquispacedMaskFractionFunc(MaskFunc):
    """
    Equispaced mask with approximate acceleration matching. FastMRI multi-coil brain dataset uses this mask type.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        num_rows: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Number of low frequencies. Used to adjust mask
                to exactly match the target acceleration.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_frequencies - num_cols)) / (
            num_low_frequencies * acceleration - num_cols
        )
        if offset is None:
            offset = self.rng.randint(0, high=round(adjusted_accel))

        mask = np.zeros(num_cols)
        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = 1.0

        return mask


class FixedLowEquiSpacedMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced k-space lines and a fixed number of low-frequency lines. CMRxRecon dataset uses this mask type.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def sample_mask(self, shape, offset):

        num_cols = shape[-2]
        num_rows = shape[-3]
        # changes below
        num_low_frequencies, acceleration = self.choose_acceleration()
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, num_rows, acceleration, 0, num_low_frequencies
            ),
            shape,
        )
        mask_type = "uniform"
        # print(mask_type, acceleration)
        return center_mask, acceleration_mask, num_low_frequencies, acceleration, mask_type

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        num_rows: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask


class FixedLowRandomMaskFunc(MaskFunc):
    """
    Sample data with random k-space lines and a fixed number of low-frequency lines. CMRxRecon2024 dataset uses this mask type.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
    1. N_low_freqs = (N * center_fraction) columns in the center
       corresponding to low-frequencies.
    2. The other columns are selected uniformly at random with a
    probability equal to: prob = (N / acceleration - N_low_freqs) /
    (N - N_low_freqs). This ensures that the expected number of columns
    selected is equal to (N / acceleration).

    ``acceleration`` will be greater than the true acceleration rate.
    """

    def sample_mask(self, shape, offset):

        num_cols = shape[-2]
        num_rows = shape[-3]
        # changes below
        num_low_frequencies, acceleration = self.choose_acceleration()
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, num_rows, acceleration, 0, num_low_frequencies
            ),
            shape,
        )
        mask_type = "random"
        # print(mask_type, acceleration)
        return center_mask, acceleration_mask, num_low_frequencies, acceleration, mask_type

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        num_rows: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        prob = (num_cols / acceleration) / (
            num_cols
        )

        return self.rng.uniform(size=num_cols) < prob


class FixedLowRandomMaskFunc_EqualAF(MaskFunc):  # Not used in CMRxRecon
    """
    Sample data with random k-space lines and a fixed number of low-frequency lines. CMRxRecon2024 dataset uses this mask type.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
    1. N_low_freqs = (N * center_fraction) columns in the center
       corresponding to low-frequencies.
    2. The other columns are selected uniformly at random with a
    probability equal to: prob = (N / acceleration - N_low_freqs) /
    (N - N_low_freqs). This ensures that the expected number of columns
    selected is equal to (N / acceleration).

    ``acceleration`` will be equal to the true acceleration rate.
    """

    def sample_mask(self, shape, offset):

        num_cols = shape[-2]
        num_rows = shape[-3]
        # changes below
        num_low_frequencies, acceleration = self.choose_acceleration()
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, num_rows, acceleration, 0, num_low_frequencies
            ),
            shape,
        )
        mask_type = "random"
        # print(mask_type, acceleration)
        return center_mask, acceleration_mask, num_low_frequencies, acceleration, mask_type

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        num_rows: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        prob = (num_cols / acceleration - num_low_frequencies) / (
            num_cols - num_low_frequencies
        )

        return self.rng.uniform(size=num_cols) < prob


class FixedLowRadialMaskFunc(MaskFunc):
    """
    Sample data with pseudo 2D radial k-space trajectories and a fixed number of 2D low-frequency region. CMRxRecon2024 dataset uses this mask type.

    ``acceleration`` will be greater than the true acceleration rate.
    """

    def sample_mask(self, shape, offset):

        num_cols = shape[-2]
        num_rows = shape[-3]
        # changes below
        num_low_frequencies, acceleration = self.choose_acceleration()
        center_mask = self.reshape_mask(
            self.calculate_2Dcenter_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, num_rows, acceleration, 0, num_low_frequencies
            ),
            shape,
        )
        mask_type = "radial"
        # print(mask_type, acceleration)
        return center_mask, acceleration_mask, num_low_frequencies, acceleration, mask_type

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape."""
        num_cols = shape[-2]
        num_rows = shape[-3]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask_shape[-3] = num_rows

        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_2Dcenter_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:
        """
        Build 2D center mask based on number of low frequencies.

        Args:
            shape: Shape of k-space to mask. [..., FE, PE, 2]
            num_low_freqs: Number of low-frequency lines to sample.

        Returns:
            A mask for hte low spatial frequencies of k-space.
        """
        num_cols = shape[-2]
        num_rows = shape[-3]
        mask = np.zeros((num_rows, num_cols), dtype=np.float32)
        pad_cols = (num_cols - num_low_freqs + 1) // 2
        pad_rows = (num_rows - num_low_freqs + 1) // 2
        mask[pad_rows: pad_rows + num_low_freqs, pad_cols: pad_cols + num_low_freqs] = 1

        assert mask.sum() == num_low_freqs * num_low_freqs

        return mask

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        num_rows: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:

        from scipy.ndimage import rotate

        R = acceleration * 0.6
        rate = 1 / R
        cropcorner = True  # True or False
        angle4next = 137.5
        beams = int(np.floor(rate * 180))  # beams is the number of angles

        if cropcorner:
            a = max(num_rows, num_cols)
        else:
            a = int(np.ceil(np.sqrt(2) * max(num_rows, num_cols)))

        aux = np.zeros((a, a))
        aux[a // 2, :] = 1
        angle = 180 / beams

        import random
        i = random.randint(1, 30)
        angles = np.arange(0 + angle4next * (i - 1), 180 + angle4next * (i - 1), angle)
        image = np.zeros((num_rows, num_cols), dtype=np.float32)

        for ang in angles:
            temp = self.crop(rotate(aux, ang, reshape=False, order=0), num_rows, num_cols)
            image = image + temp

        mask = (image > 0).astype(np.float32)
        return mask

    def crop(self, image, nx, ny):
        """ Crop the image to the desired size (nx, ny) """
        start_x = (image.shape[0] - nx) // 2
        start_y = (image.shape[1] - ny) // 2
        return image[start_x:start_x + nx, start_y:start_y + ny]


class FixedLowEquiSpaced_RandomMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced/random k-space lines and a fixed number of low-frequency region. CMRxRecon2024 dataset uses this mask type.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def sample_mask(self, shape, offset):

        num_cols = shape[-2]
        num_rows = shape[-3]
        # changes below
        num_low_frequencies, acceleration = self.choose_acceleration()

        import random
        pattern = random.choice(['uniform', 'random'])

        if pattern == 'uniform':
            center_mask = self.reshape_mask(
                self.calculate_center_mask(shape, num_low_frequencies), shape
            )
            acceleration_mask = self.reshape_mask(
                self.calculate_acceleration_mask_uniform(
                    num_cols, num_rows, acceleration, 0, num_low_frequencies
                ),
                shape,
            )
            mask_type = "uniform"
            # print(mask_type, acceleration)

        elif pattern == 'random':
            center_mask = self.reshape_mask(
                self.calculate_center_mask(shape, num_low_frequencies), shape
            )
            acceleration_mask = self.reshape_mask(
                self.calculate_acceleration_mask_random(
                    num_cols, num_rows, acceleration, 0, num_low_frequencies
                ),
                shape,
            )
            mask_type = "random"
            # print(mask_type, acceleration)

        return center_mask, acceleration_mask, num_low_frequencies, acceleration, mask_type

    def calculate_acceleration_mask_uniform(
        self,
        num_cols: int,
        num_rows: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask

    def calculate_acceleration_mask_random(
        self,
        num_cols: int,
        num_rows: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        prob = (num_cols / acceleration) / (
            num_cols
        )

        return self.rng.uniform(size=num_cols) < prob


class FixedLowEquiSpaced_Random_RadialMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced/random/2D radial k-space lines and a fixed number of low-frequency region. CMRxRecon2024 dataset uses this mask type.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def sample_mask(self, shape, offset):

        num_cols = shape[-2]
        num_rows = shape[-3]
        # changes below
        num_low_frequencies, acceleration = self.choose_acceleration()

        import random
        pattern = random.choice(['uniform', 'random', 'radial'])

        if pattern == 'uniform':
            center_mask = self.reshape_mask(
                self.calculate_center_mask(shape, num_low_frequencies), shape
            )
            acceleration_mask = self.reshape_mask(
                self.calculate_acceleration_mask_uniform(
                    num_cols, num_rows, acceleration, 0, num_low_frequencies
                ),
                shape,
            )
            mask_type = "uniform"
            # print(mask_type, acceleration)

        elif pattern == 'random':
            center_mask = self.reshape_mask(
                self.calculate_center_mask(shape, num_low_frequencies), shape
            )
            acceleration_mask = self.reshape_mask(
                self.calculate_acceleration_mask_random(
                    num_cols, num_rows, acceleration, 0, num_low_frequencies
                ),
                shape,
            )
            mask_type = "random"
            # print(mask_type, acceleration)

        elif pattern == 'radial':
            center_mask = self.reshape_mask_radial(
                self.calculate_2Dcenter_mask(shape, num_low_frequencies), shape
            )
            acceleration_mask = self.reshape_mask_radial(
                self.calculate_acceleration_mask_radial(
                    num_cols, num_rows, acceleration, 0, num_low_frequencies
                ),
                shape,
            )
            mask_type = "radial"
            # print(mask_type, acceleration)

        return center_mask, acceleration_mask, num_low_frequencies, acceleration, mask_type

    def calculate_acceleration_mask_uniform(
            self,
            num_cols: int,
            num_rows: int,
            acceleration: int,
            offset: Optional[int],
            num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask

    def calculate_acceleration_mask_random(
            self,
            num_cols: int,
            num_rows: int,
            acceleration: int,
            offset: Optional[int],
            num_low_frequencies: int,
    ) -> np.ndarray:
        prob = (num_cols / acceleration) / (
                num_cols
        )

        return self.rng.uniform(size=num_cols) < prob

    def reshape_mask_radial(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape."""
        num_cols = shape[-2]
        num_rows = shape[-3]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask_shape[-3] = num_rows

        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_2Dcenter_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:
        """
        Build 2D center mask based on number of low frequencies.

        Args:
            shape: Shape of k-space to mask. [..., FE, PE, 2]
            num_low_freqs: Number of low-frequency lines to sample.

        Returns:
            A mask for hte low spatial frequencies of k-space.
        """
        num_cols = shape[-2]
        num_rows = shape[-3]
        mask = np.zeros((num_rows, num_cols), dtype=np.float32)
        pad_cols = (num_cols - num_low_freqs + 1) // 2
        pad_rows = (num_rows - num_low_freqs + 1) // 2
        mask[pad_rows: pad_rows + num_low_freqs, pad_cols: pad_cols + num_low_freqs] = 1

        assert mask.sum() == num_low_freqs * num_low_freqs

        return mask

    def calculate_acceleration_mask_radial(
        self,
        num_cols: int,
        num_rows: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:

        from scipy.ndimage import rotate

        R = acceleration * 0.6
        rate = 1 / R
        cropcorner = True  # True or False
        angle4next = 137.5
        beams = int(np.floor(rate * 180))  # beams is the number of angles

        if cropcorner:
            a = max(num_rows, num_cols)
        else:
            a = int(np.ceil(np.sqrt(2) * max(num_rows, num_cols)))

        aux = np.zeros((a, a))
        aux[a // 2, :] = 1
        angle = 180 / beams

        import random
        i = random.randint(1, 30)
        angles = np.arange(0 + angle4next * (i - 1), 180 + angle4next * (i - 1), angle)
        image = np.zeros((num_rows, num_cols), dtype=np.float32)

        for ang in angles:
            temp = self.crop(rotate(aux, ang, reshape=False, order=0), num_rows, num_cols)
            image = image + temp

        mask = (image > 0).astype(np.float32)
        return mask

    def crop(self, image, nx, ny):
        """ Crop the image to the desired size (nx, ny) """
        start_x = (image.shape[0] - nx) // 2
        start_y = (image.shape[1] - ny) // 2
        return image[start_x:start_x + nx, start_y:start_y + ny]


def create_mask_for_mask_type(
    mask_type_str: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
    num_low_frequencies: Optional[int] = None,
) -> MaskFunc:
    """
    Creates a mask of the specified type.

    Args:
        mask_type_str
        center_fractions: What fraction of the center of k-space to include.
        accelerations: What accelerations to apply.
        num_low_frequencies

    Returns:
        A mask func for the target mask type.
    """
    if mask_type_str == "random":  # FastMRI multi-coil knee dataset
        return RandomMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced":
        return EquiSpacedMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced_fraction": # FastMRI multi-coil brain dataset
        return EquispacedMaskFractionFunc(center_fractions, accelerations)
    elif mask_type_str == "magic":
        return MagicMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "magic_fraction":
        return MagicMaskFractionFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced_fixed":  # CMRxRecon dataset
        return FixedLowEquiSpacedMaskFunc(num_low_frequencies, accelerations, allow_any_combination=True)
    elif mask_type_str == "random_fixed":  # CMRxRecon2024 dataset
        return FixedLowRandomMaskFunc(num_low_frequencies, accelerations, allow_any_combination=True)
    elif mask_type_str == "radial_fixed":  # CMRxRecon2024 dataset
        return FixedLowRadialMaskFunc(num_low_frequencies, accelerations, allow_any_combination=True)
    elif mask_type_str == "random_equispaced_fixed":  # CMRxRecon2024 dataset
        return FixedLowEquiSpaced_RandomMaskFunc(num_low_frequencies, accelerations, allow_any_combination=True)
    elif mask_type_str == "random_equispaced_radial_fixed":  # CMRxRecon2024 dataset
        return FixedLowEquiSpaced_Random_RadialMaskFunc(num_low_frequencies, accelerations, allow_any_combination=True)
    else:
        raise ValueError(f"{mask_type_str} not supported")
