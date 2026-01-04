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
    """ Crop the image tensor to the desired size (..., nx) on the last dimension."""
    start_x = (image.shape[0] - nx) // 2
    return image[start_x:start_x + nx, ...]


def image_halfcropnx(sosimage):
    '''
    load sos image 
    input image shape [nx,ny,nz,nt] or [nx,ny,nz] or [nx,ny]
    return image shape [nx/2,ny,nz,nt] or [nx/2,ny,nz] or [nx/2,ny]
    '''
    nx_halfcrop = sosimage.shape[0] // 2  # nx half crop to remove 2 times oversampling (reduce training burden)
    sosimage_crop = crop_firstnx(sosimage, nx_halfcrop)
    return sosimage_crop


def image_centercrop(sosimage):
    '''
    load sos image 
    input image shape [nx,ny]
    return image shape [min(nx/2, ny),min(nx/2, ny)]
    '''

    nx, ny = sosimage.shape
    crop_size = int(min(nx / 2, ny))

    x_start = (nx - crop_size) // 2
    y_start = (ny - crop_size) // 2

    sosimage_crop = sosimage[x_start:x_start + crop_size, y_start:y_start + crop_size]

    return sosimage_crop


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if maxval is None:
        maxval = gt.max()
    return structural_similarity(gt, pred, data_range=maxval)


def normalize_percentile_clip(array, percentile=99.5):
    """Normalize an array using a specified percentile."""
    norm_array = array / np.percentile(array, percentile)  # normalize using 99.5th percentile value
    norm_clip_array = np.clip(norm_array, 0, 1)  # value larger than 1 will be truncated to 1
    return norm_clip_array


def normalize_percentile(array, percentile=99.5):
    """Normalize an array using a specified percentile."""
    norm_array = array / np.percentile(array, percentile)  # normalize using 99.5th percentile value
    return norm_array


def normalize_maxval(array):
    """Normalize an array using a specified percentile."""
    norm_array = array / array.max()  # normalize using 99.5th percentile value
    return norm_array


def normalize_std(array):
    """Normalize an array using mean and standard deviation."""
    mean = np.mean(array)
    std = np.std(array) + 1e-8
    return (array - mean) / std


def cal_objcriteria(pred_recon, gt_recon, norm_scheme):
    # gt_recon: [nx,ny,nz,nt] or [nx,ny,nz] or [nx,ny]
    if gt_recon.ndim == 4:   # gt_recon: [nx,ny,nz,nt]
        psnr_array = np.zeros((gt_recon.shape[-2], gt_recon.shape[-1]))
        ssim_array = np.zeros((gt_recon.shape[-2], gt_recon.shape[-1]))
        nmse_array = np.zeros((gt_recon.shape[-2], gt_recon.shape[-1]))

        for i in range(gt_recon.shape[-2]):
            for j in range(gt_recon.shape[-1]):
                pred, gt = pred_recon[:, :, i, j], gt_recon[:, :, i, j]
                # use normlization to hold a more stable way
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

    elif gt_recon.ndim == 3:   # gt_recon: [nx,ny,nz]
        psnr_array = np.zeros((1, gt_recon.shape[-1]))
        ssim_array = np.zeros((1, gt_recon.shape[-1]))
        nmse_array = np.zeros((1, gt_recon.shape[-1]))

        for i in range(gt_recon.shape[-1]):
            pred, gt = np.squeeze(pred_recon)[:, :, i], gt_recon[:, :, i]
            # use normlization to hold a more stable way
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

    else:  # gt_recon: [nx,ny]
        pred, gt = np.squeeze(pred_recon), gt_recon
        # use normlization to hold a more stable way
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
    # gt_recon: [nx,ny,nz,nt] or [nx,ny,nz] or [nx,ny]
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
        gt_show = image_centercrop(gt_show)
        zf_show = image_centercrop(zf_show)
        method_1_show = image_centercrop(method_1_show)
        method_2_show = image_centercrop(method_2_show)
        method_3_show = image_centercrop(method_3_show)
        method_4_show = image_centercrop(method_4_show)
        method_5_show = image_centercrop(method_5_show)
        method_6_show = image_centercrop(method_6_show)

    if imageshownorm:
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
    def compute_errormap(img):
        errormap = np.abs(img - gt_show)
        return errormap
    
    gt_err = np.zeros_like(gt_show)
    zf_err = compute_errormap(zf_show)
    methods_err = [compute_errormap(m) for m in methods_show]
    
    return (gt_err, zf_err, *methods_err)
    