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
# Make repository-local modules importable when launching from the nested prepare_cmr folder.
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
    """
    Convert CMRxReconAll FullSample MATLAB files into fastMRI-style HDF5 files.

    ``data_path`` points to the source CMRxReconAll MATLAB root,
    ``newsave_path`` points to the destination root, and ``h5py_folder`` replaces
    the ``FullSample`` folder name in generated save paths.
    """
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
    # Main modality folders scanned under each TrainingSet/FullSample directory.
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
        # Recursively collect all MATLAB files for this modality.
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
    # Main conversion loop: read MATLAB k-space, compute RSS target, and write HDF5.
    for ff in tqdm(f):
        # Replace FullSample with the chosen HDF5 folder, .mat with .h5, and source root with destination root.
        save_path = ff.replace('FullSample', save_folder_name).replace('.mat', '.h5').replace(data_path, newsave_path)
        if not os.path.isdir(os.path.dirname(save_path)):
            # Create the mirrored destination directory before writing the HDF5 file.
            os.makedirs(os.path.dirname(save_path))

        filename = os.path.basename(ff)  # If ff is '/path/to/your/cine.mat', then filename will be 'cine.mat'
        # Load complex k-space and direct IFFT + RSS reconstruction target.
        kdata, image = zf_recon_4D5D(ff)  # kdata: [nt,nz,nc,ny,nx] or [nz,nc,ny,nx], image: [nt,nz,ny,nx] or [nz,ny,nx]
        print(f"{filename}, {kdata.shape}")

        # Open the HDF5 file in write mode
        file = h5py.File(save_path, 'w')

        # Create a dataset
        # we need to reshape and transpose it to (nt*nz, nc, nx=FE, ny=PE) as 'kspace' for fastMRI style
        if kdata.ndim == 4 or kdata.ndim == 5:
            # Flatten time/slice dimensions and swap spatial axes from [ny, nx] to [nx, ny].
            save_kdata = kdata.reshape(-1, kdata.shape[-3], kdata.shape[-2], kdata.shape[-1]).transpose(0, 1, 3, 2)
        else:  # kdata.ndim == 3
            save_kdata = kdata.transpose(0, 2, 1)
            save_kdata = np.expand_dims(save_kdata, axis=0)  # make sure 4D [1, nc, nx, ny]
        file.create_dataset('kspace', data=save_kdata)

        # we need to reshape and transpose it to (nt*nz, nx=FE, ny=PE) as 'reconstruction_rss' for fastMRI style
        if image.ndim == 3 or image.ndim == 4:
            # Flatten time/slice dimensions and swap spatial axes from [ny, nx] to [nx, ny].
            save_image = image.reshape(-1, image.shape[-2], image.shape[-1]).transpose(0, 2, 1)
        else:  # image.ndim == 2
            save_image = image.transpose(2, 1)
            save_image = np.expand_dims(save_image, axis=0)  # ensure 3D [1, nx, ny]
        file.create_dataset('reconstruction_rss', data=save_image)
        # Store target scaling information used by downstream transforms/losses.
        file.attrs['max'] = image.max()
        file.attrs['norm'] = np.linalg.norm(image)

        # Add attributes to the dataset
        # Infer scanner/modality metadata from the generated file path.
        (file.attrs['center'], file.attrs['vendor'], file.attrs['field'], file.attrs['scanner'],
         file.attrs['modality'], file.attrs['view'],
         file.attrs['medcon'], file.attrs['lifespan']) = datamapping_from_filename(save_path)

        # Medical condition is intentionally removed for this training split.
        file.attrs['medcon'] = 'unknown'  # reset all medcon to unknown
        # Keep identifiers and geometry metadata needed by dataset transforms.
        file.attrs['patient_id'] = save_path.split('ChallengeData/')[-1]
        file.attrs['shape'] = kdata.shape
        file.attrs['padding_left'] = 0
        file.attrs['padding_right'] = save_kdata.shape[3]
        file.attrs['encoding_size'] = (save_kdata.shape[2], save_kdata.shape[3], 1)
        file.attrs['recon_size'] = (save_kdata.shape[2], save_kdata.shape[3], 1)

        # Close the file
        # Explicit close ensures all datasets and attrs are flushed to disk.
        file.close()

