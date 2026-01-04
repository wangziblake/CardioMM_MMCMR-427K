"""
Data transform functions for the CardioMM model - pytorch - two lmhead for metadata and undersampling
- move llm to pl_module
- no medical condition injection
- modified metadata text information
Created on 2025/12/09
@author: Zi Wang
Email: Zi Wang (zi.wang@imperial.ac.uk)
If you want to use this code, please cite our relevant papers in the GitHub page.

Some codes are modified based on https://arxiv.org/abs/2309.13839
"""

from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import fastmri

from data.subsample import MaskFunc
from fastmri import fft2c, ifft2c, rss_complex, complex_abs
import re
from transformers import AutoTokenizer


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, str]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data.
            mask: The generated mask.
            num_low_frequencies: The number of low-resolution frequency samples
                in the mask.
    """
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies, acceleration, mask_type = mask_func(shape, offset, seed)
    if padding is not None:
        mask[..., : padding[0], :] = 0
        mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies, acceleration, mask_type


class CardioMMSample(NamedTuple):
    """
    A sample of masked k-space for CardioMM reconstruction.

    Args:
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
    """

    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]
    metadata: Dict[str, torch.Tensor]
    usdata: Dict[str, torch.Tensor]
    mask_type: str


class CardioMMDataTransform:
    """
    Data Transformer for training CardioMM models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, llm_model_name: Optional[str] = "../bge-micro-v2", use_seed: bool = True):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.llm_model_name = llm_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """

        if target is not None:
            target_torch = to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))  # so in validation, the same fname (volume) will have the same acc
        acq_start = 0
        acq_end = attrs["padding_right"]
        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies, acceleration, mask_type = apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )

            # TODO: add more metadata from dataset attributes in the future !!!!!!
            # Metadata: Default value should be unknown
            # -- Lifespan
            # -- Reconstruct
            # -- Center
            # -- Field strength, Vendor
            # -- Modality, View
            # -- Healthy, Patients with diseases
            metadata_lifespan = attrs['lifespan']  # Adult, Pediatric, Fetal
            metadata_task = "cardiac MRI reconstruction"
            metadata_center = attrs['center']
            metadata_modality, metadata_view = attrs['modality'], attrs['view']
            metadata_field, metadata_vendor, metadata_scanner = attrs['field'], attrs['vendor'], attrs['scanner']
            # metadata_medcon = attrs['medcon']  # [healthy, HCM, ..., or unknown]
            ori_metadata = f"{metadata_lifespan} {metadata_task}." \
                           f"Vendor: {metadata_field}, {metadata_vendor}, {metadata_scanner}. " \
                           f"Modality: {metadata_modality}, {metadata_view}. " \
                           # f"Disease: {metadata_medcon}."
            metadata_tuple = re.sub(r'\s+', ' ', ori_metadata.strip()).replace("\n", "")
            metadata_list = [metadata_tuple]  # tuple -> list as input

            # USdata: Default value should be unknown
            # -- Undersampling pattern, AF
            metadata_af = str(acceleration) + "x"  # "4x", "8x", "10x", "12x", "16x"
            metadata_us = mask_type  # "random", "uniform", "radial"
            ori_usdata = f"Undersampling: {metadata_af} {metadata_us}."
            usdata_tuple = re.sub(r'\s+', ' ', ori_usdata.strip()).replace("\n", "")
            usdata_list = [usdata_tuple]  # tuple -> list as input

            metadata = self.tokenizer(metadata_list, padding=True, max_length=512, truncation=True, return_tensors="pt")  # text including metadata for tensor input
            usdata = self.tokenizer(usdata_list, padding=True, max_length=512, truncation=True, return_tensors="pt")  # text including usdata for tensor input

            sample = CardioMMSample(
                masked_kspace=masked_kspace.to(torch.float32),
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=num_low_frequencies,
                target=target_torch.to(torch.float32),
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
                metadata=metadata,
                usdata=usdata,
                mask_type=mask_type,  # "uniform", "random", "radial"
            )

        else:
            masked_kspace = kspace_torch
            shape = np.array(kspace_torch.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask_torch = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask_torch = mask_torch.reshape(*mask_shape)
            mask_torch[:, :, :acq_start] = 0
            mask_torch[:, :, acq_end:] = 0

            # TODO: add metadata from dataset attributes in the future !!!!!!
            print("There is no any metadata.")

            sample = CardioMMSample(
                masked_kspace=masked_kspace.to(torch.float32),
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=0,
                target=target_torch.to(torch.float32),
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
                metadata={},
                usdata={},
                mask_type="random",
            )

        return sample
