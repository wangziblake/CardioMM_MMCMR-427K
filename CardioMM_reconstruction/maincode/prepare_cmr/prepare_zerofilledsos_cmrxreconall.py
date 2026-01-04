"""
Zero-filled SOS reconstruction in advance for fast evaluaion - pytorch - CMRxReconAll
Created on 2025/07/25
@author: Zi Wang
Email: Zi Wang (zi.wang@imperial.ac.uk)
If you want to use this code, please cite our relevant papers in the GitHub page.
"""

import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.absolute())))

import argparse
import scipy.io as sio
import glob
import torch
from utils import load_kdata_compatible, load_maskdata, ifft2c
from os.path import join
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from evaluate_utils import image_halfcropnx


def replace_mask_to_find_data(mask_filename):
    if "_mask_" in mask_filename:
        base, ext = mask_filename.rsplit(".", 1)
        base = base.rsplit("_mask_", 1)[0]
        data_filename = f"{base}.{ext}"
    else:
        raise NotImplementedError("The filename does not contain '_mask_'")  # Raise an error if "_mask_" is missing
    return data_filename


def get_zfdata(fname, task):
    # here, input kspace from .mat: 5D-[nt, nz, nc, ny, nx] or 4D-[nz, nc, ny, nx] or 3D-[nc, ny, nx]
    data_fname = replace_mask_to_find_data(fname.replace(f'Mask_{task}', 'FullSample'))
    kspace = load_kdata_compatible(data_fname)
    if len(kspace.shape) != 5:
        kspace = np.expand_dims(kspace, axis=0)  # make sure its shape is [1, nz, nc, ny, nx]
        if len(kspace.shape) != 5:
            kspace = np.expand_dims(kspace, axis=0)  # make sure its shape is [1, nz, nc, ny, nx]
    num_t = kspace.shape[0]
    num_slices = kspace.shape[1]
    # here, input mask from .mat: 3D-[nt, ny, nx] or 2D-[ny, nx]
    mask = load_maskdata(fname)
    if len(mask.shape) == 3:
        mask = np.expand_dims(np.expand_dims(mask, axis=1),axis=2)  # make sure its shape is [nt, 1, 1, ny, nx]
    elif len(mask.shape) == 2:
        mask = np.expand_dims(np.expand_dims(np.expand_dims(mask, axis=0), axis=1), axis=2)  # make sure its shape is [1, 1, 1, ny, nx]
        mask = np.tile(mask, reps=(num_t, 1, 1, 1, 1))  # make sure its shape is [nt, 1, 1, ny, nx]
    else:
        raise NotImplementedError("The mask shape should be 2D(k) or 3D(k-t).")
    mask = np.tile(mask, reps=(1, num_slices, 1, 1, 1))  # make sure its shape is [nt, nz, 1, ny, nx]

    masked_kspace = kspace * mask
    zfimage_coil = ifft2c(torch.tensor(masked_kspace))
    zfimage = (zfimage_coil.abs()**2).sum(-3)**0.5  # [nt, nz, ny, nx]
    zfimagesos = zfimage.cpu().numpy()
    return np.squeeze(zfimagesos.transpose(3, 2, 1, 0))


def ZFpredict(f, center_crop=False, input_dir='', output_dir='', task='TaskAll', image_scale=1, plot_image=False):
    # 1. predict
    for ff in tqdm(f, desc='files'):
        print('-- processing --', ff)
        # If ff is '/path/to/your/cine.mat', then save_path will be '/path/to/your/'
        save_path = ff.replace('Mask', 'ZFSOS').replace(input_dir, output_dir)
        if os.path.isfile(save_path):  # check if the .mat file already exists
            continue
        elif not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        recimage = np.squeeze(get_zfdata(ff, task))  # recimage after squeeze: [nx,ny,nz,nt] or [nx,ny,nz] or [nx,ny]
        print(f"{recimage.shape}")

        if center_crop:
            recimage = image_halfcropnx(recimage)  # recimage after crop: [nx/2,ny,nz,nt] or [nx/2,ny,nz] or [nx/2,ny]

        sosimage_savemat = recimage
        sio.savemat(save_path, {'zfsosimage': sosimage_savemat})
        print('-- saving --', save_path)


if __name__ == '__main__':
    argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    parser.add_argument('--center_crop', action='store_true', default=False, help='Enable center cropping')
    parser.add_argument('--evaluate_set', type=str, default="TestSet", help='Choose the evaluation set: TestSet')
    parser.add_argument('--task', type=str, default='TaskAll', help='Choose to inference on which type of task')
    parser.add_argument('--modality', type=str, default='All', help='Choose to inference on which type of data')
    parser.add_argument('--undersample', type=str, default='', help='Choose the undersampling pattern (Uniform8, ktGaussian16, ktRadial24)')
    parser.add_argument('--exact_mask_filename', type=str, default=None, help='exact filename to test')
    # exact_mask_filename example:
    # /SSDHome/home/Raw_data/MICCAIChallengeAll/ChallengeData/MultiCoil/Cine/TestSet/Mask_TaskAll/Center015/Siemens_30T_Vida/P301/cine_sax_mask_ktGaussian16.mat
    parser.add_argument('--image_scale', type=float, default=1, help='scale the recon image for better visualization')
    parser.add_argument('--plot_image', type=bool, default=False, help='plot the recon image')

    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    center_crop = args.center_crop
    evaluate_set = args.evaluate_set
    task = args.task
    modality = args.modality
    undersample = args.undersample
    exact_mask_filename = args.exact_mask_filename
    image_scale = args.image_scale
    plot_image = args.plot_image

    print("Input data store in:", input_dir)
    print("Output data store in:", output_dir)

    if exact_mask_filename is not None:
        f = [exact_mask_filename]
        print('##############')
        print("Exact recon filename:", exact_mask_filename)
        print(f'Total files: {len(f)}')
        print('##############')

    elif exact_mask_filename is None:
        # get input file list
        # TODO: Need to be changed according to the TestSet !!!
        modalities = {
            'Cine': 'cine*.mat',
            'Mapping': '*map*.mat',
            'Aorta': 'aorta*.mat',
            'Tagging': 'tagging*.mat',
            'Flow2d': 'flow2d*.mat',
            'BlackBlood': 'blackblood*.mat',
            'LGE': 'lge*.mat',
            'Perfusion': 'perfusion*.mat',
            'T1rho': 'T1rho*.mat',
            'T1w': 'T1w*.mat',
            'T2w': 'T2w*.mat',
        }
        file_dict = {m: [] for m in modalities}

        for modal, pattern in modalities.items():
            if modality == modal or modality == 'All':
                file_dict[modal] = sorted([
                    file for file in glob.glob(join(input_dir, f'**/{pattern}'), recursive=True)
                    if all(x in file for x in ['Mask', modal, task, evaluate_set, undersample])
                ])
        f = sum(file_dict.values(), [])
        print('##############')
        for modal, files in file_dict.items():
            print(f'{modal} files: {len(files)}')
        print(f'Total files: {len(f)}')
        print('##############')

    # main function: reconstruct zf and save files
    ZFpredict(f, center_crop=center_crop, input_dir=input_dir, output_dir=output_dir, task=task, image_scale=image_scale, plot_image=plot_image)
