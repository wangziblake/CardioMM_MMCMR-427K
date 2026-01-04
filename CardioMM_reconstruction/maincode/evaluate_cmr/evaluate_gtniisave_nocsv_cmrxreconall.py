"""
Save reconstructed gt sos images to .nii.gz for subsequent analysis (segmentation, classification) - pytorch - CMRxReconAll
Created on 2025/07/17
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
from os.path import join
import numpy as np
import nibabel as nib
import re
import pandas as pd
from tqdm import tqdm
from evaluate_utils import image_halfcropnx


def remove_metakeys_inmat(matdata):
    # Remove metadata keys that are not needed
    metadata_keys = ['__header__', '__version__', '__globals__']
    for key in metadata_keys:
        matdata.pop(key, None)
    return matdata


def load_firstkeyvalue_inmat(matdata):
    matdata2 = remove_metakeys_inmat(matdata)
    first_value_key = next(iter(matdata2))
    recimage = matdata[first_value_key]  # recimage: [nx,ny,nz,nt] or [nx,ny,nz] or [nx,ny]
    return recimage


def saveto_nii_nocsv(full_save_path, recimage):
    affine = np.diag([1, 1, 1, 1])  # default affine is np.eye(4)
    resnx, resny, resnz, resnt = 1, 1, 1, 1  # default voxel size
    print(f"resnx: {resnx}, resny: {resny}, resnz: {resnz}, resnt: {resnt}")

    nii_img = nib.Nifti1Image(recimage, affine)
    # Set voxel size nx, ny, nz in header
    header = nii_img.header
    header.set_zooms((resnx, resny, resnz, resnt))

    nib.save(nii_img, full_save_path)
    return


def gtsaveimage2nii(f, center_crop=False, input_dir='', output_dir='', image_scale=1, plot_image=False):
    # 1. transform and save .nii.gz
    for ff in tqdm(f, desc='files'):
        print('-- processing --', ff)
        # If ff is '/path/to/your/cine.mat', then save_path will be '/path/to/your/'
        save_path = os.path.dirname(ff).replace('GTSOS', 'GTSOS_NII').replace(input_dir, output_dir)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        filename = os.path.basename(ff).replace('.mat', '')  # If ff is '/path/to/your/cine.mat', then filename will be 'cine'
        csvname = ff.replace('GTSOS', 'FullSample').replace('.mat', '_info.csv')  # load the corresponding .csv file
        
        if os.path.isfile(f"{save_path}/{filename}.nii.gz"):  # check if the .nii.gz file already exists
            continue
        recimage = load_firstkeyvalue_inmat(sio.loadmat(ff))  # recimage: [nx,ny,nz,nt] or [nx,ny,nz] or [nx,ny]
        
        if any(keyword in ff for keyword in {'Cine', 'LGE', 'Mapping', 'Aorta', 'Flow2d', 'Tagging', 'Perfusion', 'T1rho'}):
            if recimage.ndim == 3:
                recimage = np.expand_dims(recimage, axis=-2)  # nx, ny, 1, nt
        print(f"{recimage.shape}")
        
        if not os.path.isfile(csvname):
            csvname = ff.replace('GTSOS', 'FullSample').replace('.mat', '.csv')  # load the corresponding .csv file for some cases (Center007/Siemens_055T_Freemax)

        if center_crop:
            recimage = image_halfcropnx(recimage)  # recimage after crop: [nx/2,ny,nz,nt] or [nx/2,ny,nz] or [nx/2,ny]

        if recimage.ndim == 4:
            nx, ny, nslice, nt = recimage.shape  # nx, ny, nz, nt
            recimage_save = abs(recimage)
            percentiles = np.percentile(recimage_save, 99.5, axis=(0, 1, 3), keepdims=True)  # normalize using 99.5th percentile value, not using nz
            recimage_save_norm = recimage_save / percentiles
            filename_save = f"{filename}.nii.gz"
            full_save_path = os.path.join(save_path, filename_save)
            saveto_nii_nocsv(full_save_path, recimage_save_norm)
        elif recimage.ndim == 3:
            recimage = np.expand_dims(recimage, axis=-1)  # nx, ny, nz, 1
            nx, ny, nslice, _ = recimage.shape  # nx, ny, nz, 1
            recimage_save = abs(recimage)
            percentiles = np.percentile(recimage_save, 99.5, axis=(0, 1, 3), keepdims=True)  # normalize using 99.5th percentile value, not using nz
            recimage_save_norm = recimage_save / percentiles
            filename_save = f"{filename}.nii.gz"
            full_save_path = os.path.join(save_path, filename_save)
            saveto_nii_nocsv(full_save_path, recimage_save_norm)
        else:
            recimage = np.expand_dims(recimage, axis=-1)
            recimage = np.expand_dims(recimage, axis=-1)  # nx, ny, 1, 1
            nx, ny, _, _ = recimage.shape
            recimage_save = abs(recimage)
            percentiles = np.percentile(recimage_save, 99.5, axis=(0, 1, 3), keepdims=True)  # normalize using 99.5th percentile value, not using nz
            recimage_save_norm = recimage_save / percentiles
            filename_save = f"{filename}.nii.gz"
            full_save_path = os.path.join(save_path, filename_save)
            saveto_nii_nocsv(full_save_path, recimage_save_norm)
        print('-- saving --', save_path)


if __name__ == '__main__':
    argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    parser.add_argument('--center_crop', action='store_true', default=False, help='Enable center cropping')
    parser.add_argument('--evaluate_set', type=str, default="TestSet", help='Choose the evaluation set: TestSet')
    parser.add_argument('--modality', type=str, default='All', help='Choose to inference on which type of data')
    parser.add_argument('--exact_filename', type=str, default=None, help='exact filename to test')
    # exact_filename example:
    # /SSDHome/home/Raw_data/MICCAIChallengeAll/ChallengeData/MultiCoil/Cine/TestSet/GTSOS/Center015/Siemens_30T_Vida/P301/cine_sax.mat
    parser.add_argument('--image_scale', type=float, default=1, help='scale the recon image for better visualization')
    parser.add_argument('--plot_image', type=bool, default=False, help='plot the recon image')

    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    center_crop = args.center_crop
    evaluate_set = args.evaluate_set
    modality = args.modality
    exact_filename = args.exact_filename
    image_scale = args.image_scale
    plot_image = args.plot_image

    print("Input data store in:", input_dir)
    print("Output data store in:", output_dir)

    if exact_filename is not None:
        f = [exact_filename]
        print('##############')
        print("Exact recon filename:", exact_filename)
        print(f'Total files: {len(f)}')
        print('##############')

    elif exact_filename is None:
        # get input file list
        # TODO: Need to be changed according to the TestSet !!!
        modalities = {
            'Cine': 'cine*.mat',
            'LGE': 'lge*.mat',
            'Mapping': '*map*.mat',

            'Aorta': 'aorta*.mat',
            'Tagging': 'tagging*.mat',
            'Flow2d': 'flow2d*.mat',
            'BlackBlood': 'blackblood*.mat',
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
                    if all(x in file for x in ['GTSOS', modal, evaluate_set])
                ])
        f = sum(file_dict.values(), [])
        print('##############')
        for modal, files in file_dict.items():
            print(f'{modal} files: {len(files)}')
        print(f'Total files: {len(f)}')
        print('##############')

    # main function: load and save files
    gtsaveimage2nii(f, center_crop=center_crop, input_dir=input_dir, output_dir=output_dir, image_scale=image_scale, plot_image=plot_image)
