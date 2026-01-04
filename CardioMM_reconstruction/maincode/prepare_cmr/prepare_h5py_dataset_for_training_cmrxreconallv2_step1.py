"""
Prepare .h5 datasets for training - from CMRxReconAll dataset .mat format - revised issues for train/val splitting
- no medical condition
Created on 2025/04/28
@author: Zi Wang
Email: Zi Wang (zi.wang@imperial.ac.uk)
If you want to use this code, please cite our relevant papers in the GitHub page.
"""

import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.absolute())))

import shutil
import argparse
import numpy as np
from utils import zf_recon_4D5D
import h5py
import glob
from os.path import join
from tqdm import tqdm
from datamapping import datamapping_from_filename


if __name__ == '__main__':
    argv = sys.argv
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--data_path",
            type=str,
            default="/SSDHome/home/Raw_data/MICCAIChallengeAll/ChallengeData/MultiCoil",
            help="Path to the multi-coil MATLAB folder",
        )

    parser.add_argument(
        "--newsave_path",
        type=str,
        default="/SSDHome/home/wangz/CMRData/MICCAIChallengeAll/ChallengeData/MultiCoil",
        help="Path to the multi-coil h5py folder",
    )

    parser.add_argument(
        "--h5py_folder",
        type=str,
        default="h5_FullSample",
        help="the folder name to save the h5py files",
    )

    args = parser.parse_args()
    data_path = args.data_path
    save_folder_name = args.h5py_folder
    newsave_path = args.newsave_path

    # 0. Get input folder and file list
    folders = [
        "Cine", "Mapping", "Aorta", "Tagging", "Flow2d",
        "BlackBlood", "LGE", "Perfusion", "T1rho", "T1w", "T2w"
    ]
    # folders = ["Flow2d"]  # TODO: fast debug
    data_paths = {}  # folder path
    data_files = {}  # file path
    for folder in folders:
        folder_path = join(data_path, f"{folder}/TrainingSet/FullSample")
        # files = sorted(glob.glob(join(folder_path, '**/*.mat'), recursive=True))
        files = sorted(glob.glob(join(folder_path, '**/*.mat'), recursive=True))  # TODO: fast debug
        data_paths[folder] = folder_path  # fully_cine_matlab_folder = data_paths["Cine"]
        data_files[folder] = files  # f_cine = data_files["Cine"]
        num_centers = len(os.listdir(folder_path))
        num_files = len(files)
        print(f"{folder} centers: {num_centers}, {folder} files: {num_files}")

    f = []
    for folder in folders:
        f += data_files[folder]

    # 1. Save as fastMRI style h5py files
    for ff in tqdm(f):
        save_path = ff.replace('FullSample', save_folder_name).replace('.mat', '.h5').replace(data_path, newsave_path)
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        filename = os.path.basename(ff)  # If ff is '/path/to/your/cine.mat', then filename will be 'cine.mat'
        kdata, image = zf_recon_4D5D(ff)  # kdata: [nt,nz,nc,ny,nx] or [nz,nc,ny,nx], image: [nt,nz,ny,nx] or [nz,ny,nx]
        print(f"{filename}, {kdata.shape}")

        # Open the HDF5 file in write mode
        file = h5py.File(save_path, 'w')

        # Create a dataset
        # we need to reshape and transpose it to (nt*nz, nc, nx=FE, ny=PE) as 'kspace' for fastMRI style
        if kdata.ndim == 4 or kdata.ndim == 5:
            save_kdata = kdata.reshape(-1, kdata.shape[-3], kdata.shape[-2], kdata.shape[-1]).transpose(0, 1, 3, 2)
        else:  # kdata.ndim == 3
            save_kdata = kdata.transpose(0, 2, 1)
            save_kdata = np.expand_dims(save_kdata, axis=0)  # make sure 4D [1, nc, nx, ny]
        file.create_dataset('kspace', data=save_kdata)

        # we need to reshape and transpose it to (nt*nz, nx=FE, ny=PE) as 'reconstruction_rss' for fastMRI style
        if image.ndim == 3 or image.ndim == 4:
            save_image = image.reshape(-1, image.shape[-2], image.shape[-1]).transpose(0, 2, 1)
        else:  # image.ndim == 2
            save_image = image.transpose(2, 1)
            save_image = np.expand_dims(save_image, axis=0)  # ensure 3D [1, nx, ny]
        file.create_dataset('reconstruction_rss', data=save_image)
        file.attrs['max'] = image.max()
        file.attrs['norm'] = np.linalg.norm(image)

        # Add attributes to the dataset
        (file.attrs['center'], file.attrs['vendor'], file.attrs['field'], file.attrs['scanner'],
         file.attrs['modality'], file.attrs['view'],
         file.attrs['medcon'], file.attrs['lifespan']) = datamapping_from_filename(save_path)

        file.attrs['medcon'] = 'unknown'  # reset all medcon to unknown
        file.attrs['patient_id'] = save_path.split('ChallengeData/')[-1]
        file.attrs['shape'] = kdata.shape
        file.attrs['padding_left'] = 0
        file.attrs['padding_right'] = save_kdata.shape[3]
        file.attrs['encoding_size'] = (save_kdata.shape[2], save_kdata.shape[3], 1)
        file.attrs['recon_size'] = (save_kdata.shape[2], save_kdata.shape[3], 1)

        # Close the file
        file.close()

