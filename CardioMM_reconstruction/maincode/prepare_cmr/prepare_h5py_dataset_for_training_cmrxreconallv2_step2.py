"""
Prepare .h5 datasets for training - from CMRxReconAll dataset .mat format - revised issues for train/val splitting
- Step2: for train/val splitting only
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
import glob
from os.path import join


def split_train_val(h5_folder="../Cine/TrainingSet/h5_FullSample"):
    """
    Split converted HDF5 patient folders into train and validation folders.

    Args:
        h5_folder: Converted modality folder containing patient directories.

    Behavior:
        Creates ``train`` and ``val`` subfolders, recursively finds patient
        folders matching ``P[0-9][0-9][0-9]``, moves the first ~90% in sorted
        order into train and the remaining ~10% into validation, and renames
        moved folders to sequential ``P0001``, ``P0002``, etc. Existing target
        names are skipped by incrementing the sequence number.
    """
    train_folder = join(h5_folder, 'train')
    val_folder = join(h5_folder, 'val')

    if not os.path.exists(train_folder):
        # Create the training split folder if step2 has not been run before.
        os.makedirs(train_folder)

    if not os.path.exists(val_folder):
        # Create the validation split folder if step2 has not been run before.
        os.makedirs(val_folder)

    if os.path.exists(h5_folder):
        # Discover patient folders in deterministic sorted order before splitting.
        AllPfolder = sorted(glob.glob(join(h5_folder, "**/P[0-9][0-9][0-9]"), recursive=True))
        max_Pnum = len(AllPfolder)
        # Current split rule keeps approximately 90% for training.
        train_Pnum = round(0.9 * max_Pnum)  # ~90% training, ~10% validation
        print(f"CasePfiles in total: {max_Pnum}, {train_Pnum} for train, {max_Pnum-train_Pnum} for validation")

        for i, Pfolder in enumerate(AllPfolder, start=1):
            # First train_Pnum sorted folders go to train; the rest go to validation.
            if i <= train_Pnum:
                target_folder = train_folder
            else:
                target_folder = val_folder
            # Rename moved folders into a sequential P0001, P0002, ... namespace.
            new_folder_name = f"P{i:04d}"  # P0001, P0002, ...
            new_path = join(target_folder, new_folder_name)
            while os.path.exists(new_path):
                i = i + 1
                new_folder_name = f"P{i:04d}"
                new_path = join(target_folder, new_folder_name)
            # Move the original patient folder into the selected split folder.
            shutil.move(Pfolder, new_path)


if __name__ == '__main__':
    """
    Split step1-generated HDF5 datasets into train/validation patient folders.

    ``data_path`` points to the source MATLAB root used for locating modalities,
    ``newsave_path`` points to the converted HDF5 root, and ``h5py_folder`` is
    the converted folder name replacing ``FullSample``. This step only moves
    existing converted patient folders; it assumes step1 has already created
    the HDF5 files.
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
    # Main modality folders used for reporting and deriving converted HDF5 paths.
    folders = [
        "Cine", "Mapping", "Aorta", "Tagging", "Flow2d",
        "BlackBlood", "LGE", "Perfusion", "T1rho", "T1w", "T2w"
    ]
    # folders = ["LGE"]  # TODO: fast debug
    data_paths = {}  # folder path
    data_files = {}  # file path
    for folder in folders:
        folder_path = join(data_path, f"{folder}/TrainingSet/FullSample")
        # files = sorted(glob.glob(join(folder_path, '**/*.mat'), recursive=True))
        # Recursively collect source MATLAB files only for reporting/counting.
        files = sorted(glob.glob(join(folder_path, '**/*.mat'), recursive=True))  # TODO: fast debug
        data_paths[folder] = folder_path  # fully_cine_matlab_folder = data_paths["Cine"]
        data_files[folder] = files  # f_cine = data_files["Cine"]
        num_centers = len(os.listdir(folder_path))
        num_files = len(files)
        print(f"{folder} centers: {num_centers}, {folder} files: {num_files}")

    # 2. Split first ~90% cases as net training set and the rest ~10% cases as net validation set
    h5_data_paths = {}
    for folder in folders:
        # Derive each converted HDF5 modality path from the corresponding FullSample source path.
        h5_data_paths[folder] = data_paths[folder].replace('FullSample', save_folder_name).replace(data_path, newsave_path)

    for folder in folders:
        # Move patient folders for each modality into train/val splits.
        split_train_val(h5_data_paths[folder])
        print(f"{folder} is split for training set and validations set.")

