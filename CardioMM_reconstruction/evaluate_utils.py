"""
Created on 2025/08/25
@author: Zi Wang
Email: Zi Wang (zi.wang@imperial.ac.uk)
If you want to use this code, please cite our relevant papers in the GitHub page.
"""
import os
import numpy as np
import csv
import pandas as pd
import random

############### criteria function
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from os.path import join


def crop_firstnx(image, nx):
    """
    Center-crop an image array along the first/x dimension.

    Args:
        image: Image array whose first dimension is the x/readout dimension.
        nx: Target size for the first dimension.

    Returns:
        Cropped array with shape ``[nx, ...]``.
    """
    start_x = (image.shape[0] - nx) // 2
    return image[start_x:start_x + nx, ...]


def image_halfcropnx(sosimage):
    """
    Remove 2x readout oversampling by half-cropping the first dimension.

    Args:
        sosimage: Sum-of-squares image array shaped ``[nx, ny, nz, nt]``,
            ``[nx, ny, nz]``, or ``[nx, ny]``.

    Returns:
        Center-cropped image with first dimension ``nx / 2`` and all remaining
        dimensions unchanged.
    """
    # The x/readout dimension is doubled by oversampling, so keep its center half.
    nx_halfcrop = sosimage.shape[0] // 2
    sosimage_crop = crop_firstnx(sosimage, nx_halfcrop)
    return sosimage_crop


def image_centercrop(sosimage):
    """
    Center-crop a 2D image to a square display region.

    Args:
        sosimage: 2D image array shaped ``[nx, ny]``.

    Returns:
        Square image with side length ``min(nx / 2, ny)``.
    """

    nx, ny = sosimage.shape
    crop_size = int(min(nx / 2, ny))

    x_start = (nx - crop_size) // 2
    y_start = (ny - crop_size) // 2

    sosimage_crop = sosimage[x_start:x_start + crop_size, y_start:y_start + crop_size]

    return sosimage_crop


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Compute mean squared error between ground truth and prediction arrays.

    Args:
        gt: Ground-truth image array.
        pred: Predicted image array with the same shape as ``gt``.

    Returns:
        Scalar MSE value.
    """
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Compute normalized mean squared error using ground-truth energy.

    Args:
        gt: Ground-truth image array.
        pred: Predicted image array with the same shape as ``gt``.

    Returns:
        Scalar NMSE value as a NumPy array scalar.
    """
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> np.ndarray:
    """
    Compute peak signal-to-noise ratio (PSNR).

    Args:
        gt: Ground-truth image array.
        pred: Predicted image array with the same shape as ``gt``.
        maxval: Data range used by PSNR. If omitted, ``gt.max()`` is used.

    Returns:
        Scalar PSNR value.
    """
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """
    Compute structural similarity index metric (SSIM) for two 2D images.

    Args:
        gt: Ground-truth 2D image array.
        pred: Predicted 2D image array with the same shape as ``gt``.
        maxval: Data range used by SSIM. If omitted, ``gt.max()`` is used.

    Returns:
        Scalar SSIM value.
    """
    if maxval is None:
        maxval = gt.max()
    return structural_similarity(gt, pred, data_range=maxval)


def normalize_percentile_clip(array, percentile=99.5):
    """
    Normalize an array by a percentile value and clip to ``[0, 1]``.

    Args:
        array: Input image or metric array.
        percentile: Percentile used as the intensity scale.

    Returns:
        Percentile-normalized array with values below 0 or above 1 clipped.
    """
    norm_array = array / np.percentile(array, percentile)
    # Values above the percentile scale are truncated for display stability.
    norm_clip_array = np.clip(norm_array, 0, 1)
    return norm_clip_array


def normalize_percentile(array, percentile=99.5):
    """
    Normalize an array by a percentile value without clipping.

    Args:
        array: Input image or metric array.
        percentile: Percentile used as the intensity scale.

    Returns:
        Percentile-normalized array.
    """
    norm_array = array / np.percentile(array, percentile)
    return norm_array


def normalize_maxval(array):
    """
    Normalize an array by its maximum value.

    Args:
        array: Input image or metric array.

    Returns:
        Array scaled by ``array.max()``.
    """
    norm_array = array / array.max()
    return norm_array


def normalize_std(array):
    """
    Normalize an array using its mean and standard deviation.

    Args:
        array: Input image or metric array.

    Returns:
        Zero-mean, unit-variance array. A small epsilon is added to the standard
        deviation to avoid division by zero.
    """
    mean = np.mean(array)
    std = np.std(array) + 1e-8
    return (array - mean) / std


def cal_objcriteria(pred_recon, gt_recon, norm_scheme):
    """
    Calculate PSNR, SSIM, and NMSE for 2D, 3D, or 4D reconstructions.

    Args:
        pred_recon: Predicted reconstruction with shape compatible with
            ``gt_recon``. Singleton dimensions may be present for 3D inputs.
        gt_recon: Ground-truth reconstruction shaped ``[nx, ny, nz, nt]``,
            ``[nx, ny, nz]``, or ``[nx, ny]``.
        norm_scheme: Normalization mode applied before metrics. Supported
            values are ``"percentile"``, ``"maxval"``, and ``"std"``.

    Returns:
        Tuple ``(psnr_array, ssim_array, nmse_array)``. For 4D inputs the arrays
        are shaped ``[nz, nt]``; for 3D inputs they are ``[1, nz]``; for 2D
        inputs each metric is a scalar.
    """
    # 4D data is evaluated independently for each slice/time pair.
    if gt_recon.ndim == 4:   # gt_recon: [nx,ny,nz,nt]
        psnr_array = np.zeros((gt_recon.shape[-2], gt_recon.shape[-1]))
        ssim_array = np.zeros((gt_recon.shape[-2], gt_recon.shape[-1]))
        nmse_array = np.zeros((gt_recon.shape[-2], gt_recon.shape[-1]))

        for i in range(gt_recon.shape[-2]):
            for j in range(gt_recon.shape[-1]):
                pred, gt = pred_recon[:, :, i, j], gt_recon[:, :, i, j]
                # Normalize each 2D frame before metrics for comparable scaling.
                if norm_scheme == 'percentile':
                    pred_normalized = normalize_percentile(pred, 99.5)
                    gt_normalized = normalize_percentile(gt, 99.5)
                elif norm_scheme == 'maxval':
                    pred_normalized = normalize_maxval(pred)
                    gt_normalized = normalize_maxval(gt)
                elif norm_scheme == 'std':
                    pred_normalized = normalize_std(pred)
                    gt_normalized = normalize_std(gt)

                psnr_array[i, j] = psnr(gt_normalized, pred_normalized)
                ssim_array[i, j] = ssim(gt_normalized, pred_normalized)
                nmse_array[i, j] = nmse(gt_normalized, pred_normalized)

    # 3D data is evaluated independently for each slice.
    elif gt_recon.ndim == 3:   # gt_recon: [nx,ny,nz]
        psnr_array = np.zeros((1, gt_recon.shape[-1]))
        ssim_array = np.zeros((1, gt_recon.shape[-1]))
        nmse_array = np.zeros((1, gt_recon.shape[-1]))

        for i in range(gt_recon.shape[-1]):
            pred, gt = np.squeeze(pred_recon)[:, :, i], gt_recon[:, :, i]
            # Normalize each 2D slice before metrics for comparable scaling.
            if norm_scheme == 'percentile':
                pred_normalized = normalize_percentile(pred, 99.5)
                gt_normalized = normalize_percentile(gt, 99.5)
            elif norm_scheme == 'maxval':
                pred_normalized = normalize_maxval(pred)
                gt_normalized = normalize_maxval(gt)
            elif norm_scheme == 'std':
                pred_normalized = normalize_std(pred)
                gt_normalized = normalize_std(gt)

            psnr_array[0,i] = psnr(gt_normalized, pred_normalized)
            ssim_array[0,i] = ssim(gt_normalized, pred_normalized)
            nmse_array[0,i] = nmse(gt_normalized, pred_normalized)

    # 2D data has a single metric value for the whole image.
    else:  # gt_recon: [nx,ny]
        pred, gt = np.squeeze(pred_recon), gt_recon
        # Normalize before metrics for comparable scaling across methods.
        if norm_scheme == 'percentile':
            pred_normalized = normalize_percentile(pred, 99.5)
            gt_normalized = normalize_percentile(gt, 99.5)
        elif norm_scheme == 'maxval':
            pred_normalized = normalize_maxval(pred)
            gt_normalized = normalize_maxval(gt)
        elif norm_scheme == 'std':
            pred_normalized = normalize_std(pred)
            gt_normalized = normalize_std(gt)

        psnr_array = psnr(gt_normalized, pred_normalized)
        ssim_array = ssim(gt_normalized, pred_normalized)
        nmse_array = nmse(gt_normalized, pred_normalized)

    return psnr_array, ssim_array, nmse_array


def get_mixedimage_toshow(gt_recon, zf_recon, method_1, method_2, method_3, method_4, method_5, method_6, imageshowcrop=True, imageshownorm=True):
    """
    Select matched 2D images from GT, zero-filled, and six method outputs.

    For 4D and 3D inputs, a random slice/time index is selected for display.
    Optional display cropping and max normalization are applied consistently to
    all returned images.

    Args:
        gt_recon: Ground-truth reconstruction shaped ``[nx, ny, nz, nt]``,
            ``[nx, ny, nz]``, or ``[nx, ny]``.
        zf_recon: Zero-filled reconstruction with a compatible shape.
        method_1: First reconstruction method output.
        method_2: Second reconstruction method output.
        method_3: Third reconstruction method output.
        method_4: Fourth reconstruction method output.
        method_5: Fifth reconstruction method output.
        method_6: Sixth reconstruction method output.
        imageshowcrop: If true, center-crop each selected 2D image for display.
        imageshownorm: If true, normalize each selected 2D image by its maximum.

    Returns:
        Tuple containing the selected display images in the order ``gt``, ``zf``,
        ``method_1`` through ``method_6``, followed by one-based ``index_i`` and
        ``index_j``. For 3D data ``index_j`` is zero; for 2D data both indices
        are zero.
    """
    # 4D data chooses one random slice index i and one random time index j.
    if gt_recon.ndim == 4:   # gt_recon: [nx,ny,nz,nt]
        i = random.randint(0, gt_recon.shape[-2] - 1)
        j = random.randint(0, gt_recon.shape[-1] - 1)
        if gt_recon.shape[-2] == 1:
            gt_show, zf_show = gt_recon[:, :, i, j], zf_recon[:, :, j]
        else:
            gt_show, zf_show = gt_recon[:, :, i, j], zf_recon[:, :, i, j]
        method_1_show = method_1[:, :, i, j]
        method_2_show = method_2[:, :, i, j]
        method_3_show = method_3[:, :, i, j]
        method_4_show = method_4[:, :, i, j]
        method_5_show = method_5[:, :, i, j]
        method_6_show = method_6[:, :, i, j]

        index_i = i + 1
        index_j = j + 1

    # 3D data chooses one random slice index i.
    elif gt_recon.ndim == 3:   # gt_recon: [nx,ny,nz]
        i = random.randint(0, gt_recon.shape[-1] - 1)
        gt_show, zf_show = gt_recon[:, :, i], zf_recon[:, :, i]
        method_1_show = np.squeeze(method_1)[:, :, i]
        method_2_show = np.squeeze(method_2)[:, :, i]
        method_3_show = np.squeeze(method_3)[:, :, i]
        method_4_show = np.squeeze(method_4)[:, :, i]
        method_5_show = np.squeeze(method_5)[:, :, i]
        method_6_show = np.squeeze(method_6)[:, :, i]

        index_i = i + 1
        index_j = 0

    # 2D data is already a display image, so no random index is needed.
    else:  # gt_recon: [nx,ny]
        gt_show, zf_show = gt_recon, zf_recon
        method_1_show = np.squeeze(method_1)
        method_2_show = np.squeeze(method_2)
        method_3_show = np.squeeze(method_3)
        method_4_show = np.squeeze(method_4)
        method_5_show = np.squeeze(method_5)
        method_6_show = np.squeeze(method_6)

        index_i = 0
        index_j = 0

    if imageshowcrop:
        # Apply the same display crop to every method for visual comparison.
        gt_show = image_centercrop(gt_show)
        zf_show = image_centercrop(zf_show)
        method_1_show = image_centercrop(method_1_show)
        method_2_show = image_centercrop(method_2_show)
        method_3_show = image_centercrop(method_3_show)
        method_4_show = image_centercrop(method_4_show)
        method_5_show = image_centercrop(method_5_show)
        method_6_show = image_centercrop(method_6_show)

    if imageshownorm:
        # Normalize each panel independently to [approximately] comparable display range.
        gt_show       = gt_show / (np.max(gt_show))
        zf_show       = zf_show / (np.max(zf_show))
        method_1_show = method_1_show / (np.max(method_1_show))
        method_2_show = method_2_show / (np.max(method_2_show))
        method_3_show = method_3_show / (np.max(method_3_show))
        method_4_show = method_4_show / (np.max(method_4_show))
        method_5_show = method_5_show / (np.max(method_5_show))
        method_6_show = method_6_show / (np.max(method_6_show))

    return gt_show, zf_show, method_1_show, method_2_show, method_3_show, method_4_show, method_5_show, method_6_show, index_i, index_j


def get_errormap_toshow(gt_show, zf_show, *methods_show):
    """
    Build absolute-error maps for zero-filled and method display images.

    Args:
        gt_show: Ground-truth 2D display image.
        zf_show: Zero-filled 2D display image.
        *methods_show: Any number of method display images to compare against
            ``gt_show``.

    Returns:
        Tuple containing a zero-valued GT error map, the zero-filled absolute
        error map, and one absolute error map for each method input.
    """
    def compute_errormap(img):
        # Error maps are absolute pixel-wise differences from the display GT.
        errormap = np.abs(img - gt_show)
        return errormap
    
    gt_err = np.zeros_like(gt_show)
    zf_err = compute_errormap(zf_show)
    methods_err = [compute_errormap(m) for m in methods_show]
    
    return (gt_err, zf_err, *methods_err)
    
