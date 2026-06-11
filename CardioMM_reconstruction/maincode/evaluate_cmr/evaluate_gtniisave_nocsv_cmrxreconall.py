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
# Make repository-local modules importable when this script is launched from
# the nested maincode/evaluate_cmr directory.
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
    """Remove scipy.io MATLAB metadata keys from a loaded ``.mat`` dictionary.

    Args:
        matdata: Dictionary returned by ``scipy.io.loadmat``.

    Returns:
        The same dictionary object with MATLAB bookkeeping keys removed.
    """
    # Remove metadata keys that are not needed
    metadata_keys = ['__header__', '__version__', '__globals__']
    for key in metadata_keys:
        # ``pop(..., None)`` keeps files valid even if a metadata key is absent.
        matdata.pop(key, None)
    return matdata


def load_firstkeyvalue_inmat(matdata):
    """Load the first image-like value from a MATLAB dictionary.

    Ground-truth SOS files are expected to contain one image array plus MATLAB
    metadata keys. After metadata removal, the first remaining key is treated as
    the reconstruction image.

    Args:
        matdata: Dictionary returned by ``scipy.io.loadmat``.

    Returns:
        The reconstruction image, typically ``[nx, ny, nz, nt]``,
        ``[nx, ny, nz]``, or ``[nx, ny]``.
    """
    matdata2 = remove_metakeys_inmat(matdata)
    first_value_key = next(iter(matdata2))
    recimage = matdata[first_value_key]  # recimage: [nx,ny,nz,nt] or [nx,ny,nz] or [nx,ny]
    return recimage


def saveto_nii_nocsv(full_save_path, recimage):
    """Save a normalized reconstruction image as ``.nii.gz`` without CSV metadata.

    This no-CSV variant intentionally writes a unit diagonal affine and unit
    voxel/time spacing. It is useful when scanner CSV metadata are unavailable
    or should not be used for downstream analysis.

    Args:
        full_save_path: Destination ``.nii.gz`` path.
        recimage: 4D image array in ``[nx, ny, nz, nt]`` layout.
    """
    # Without scanner CSV metadata, use identity-like spatial geometry.
    affine = np.diag([1, 1, 1, 1])  # default affine is np.eye(4)
    resnx, resny, resnz, resnt = 1, 1, 1, 1  # default voxel size
    print(f"resnx: {resnx}, resny: {resny}, resnz: {resnz}, resnt: {resnt}")

    nii_img = nib.Nifti1Image(recimage, affine)
    # Set voxel size nx, ny, nz in header
    header = nii_img.header
    # NIfTI zooms still require four entries for [x, y, z, time].
    header.set_zooms((resnx, resny, resnz, resnt))

    nib.save(nii_img, full_save_path)
    return


def gtsaveimage2nii(f, center_crop=False, input_dir='', output_dir='', image_scale=1, plot_image=False):
    """Convert ground-truth SOS ``.mat`` files into normalized NIfTI images.

    Args:
        f: Iterable of ``GTSOS`` ``.mat`` files.
        center_crop: If ``True``, remove 2x readout oversampling with
            ``image_halfcropnx`` before saving.
        input_dir: Source root used when mirroring paths into ``output_dir``.
        output_dir: Destination root for the mirrored ``GTSOS_NII`` tree.
        image_scale: Compatibility argument for older visualization workflows;
            the current saving path does not apply it.
        plot_image: Compatibility argument retained by the CLI; no plotting is
            performed in the current implementation.

    Saves:
        One ``.nii.gz`` file per input under a mirrored ``GTSOS_NII`` directory.
        Images are saved as nonnegative magnitudes normalized by the 99.5th
        percentile computed over x, y, and time, while preserving slice-specific
        scaling. Spatial and temporal NIfTI spacing are set to one because this
        variant does not read CSV metadata.
    """
    # 1. transform and save .nii.gz
    for ff in tqdm(f, desc='files'):
        print('-- processing --', ff)
        # If ff is '/path/to/your/cine.mat', then save_path will be '/path/to/your/'
        # Mirror the input tree while replacing GTSOS folders with GTSOS_NII.
        save_path = os.path.dirname(ff).replace('GTSOS', 'GTSOS_NII').replace(input_dir, output_dir)
        if not os.path.isdir(save_path):
            # Create the mirrored output folder before saving the NIfTI file.
            os.makedirs(save_path)
        filename = os.path.basename(ff).replace('.mat', '')  # If ff is '/path/to/your/cine.mat', then filename will be 'cine'
        # The CSV path is retained for parity with the CSV-aware script, but this
        # no-CSV variant does not read scanner parameters before saving.
        csvname = ff.replace('GTSOS', 'FullSample').replace('.mat', '_info.csv')  # load the corresponding .csv file
        
        if os.path.isfile(f"{save_path}/{filename}.nii.gz"):  # check if the .nii.gz file already exists
            continue
        recimage = load_firstkeyvalue_inmat(sio.loadmat(ff))  # recimage: [nx,ny,nz,nt] or [nx,ny,nz] or [nx,ny]
        
        if any(keyword in ff for keyword in {'Cine', 'LGE', 'Mapping', 'Aorta', 'Flow2d', 'Tagging', 'Perfusion', 'T1rho'}):
            if recimage.ndim == 3:
                # Dynamic single-slice modalities are interpreted as [nx, ny, 1, nt].
                recimage = np.expand_dims(recimage, axis=-2)  # nx, ny, 1, nt
        print(f"{recimage.shape}")
        
        if not os.path.isfile(csvname):
            # Keep the same fallback path derivation as the CSV-aware script,
            # although the current no-CSV path does not load this file.
            csvname = ff.replace('GTSOS', 'FullSample').replace('.mat', '.csv')  # load the corresponding .csv file for some cases (Center007/Siemens_055T_Freemax)

        if center_crop:
            # Optional readout oversampling removal is applied before NIfTI save.
            recimage = image_halfcropnx(recimage)  # recimage after crop: [nx/2,ny,nz,nt] or [nx/2,ny,nz] or [nx/2,ny]

        if recimage.ndim == 4:
            nx, ny, nslice, nt = recimage.shape  # nx, ny, nz, nt
            recimage_save = abs(recimage)
            # Normalize per slice using the high percentile over x, y, and time.
            percentiles = np.percentile(recimage_save, 99.5, axis=(0, 1, 3), keepdims=True)  # normalize using 99.5th percentile value, not using nz
            recimage_save_norm = recimage_save / percentiles
            filename_save = f"{filename}.nii.gz"
            full_save_path = os.path.join(save_path, filename_save)
            saveto_nii_nocsv(full_save_path, recimage_save_norm)
        elif recimage.ndim == 3:
            # Static 3D volumes are promoted to a singleton time dimension.
            recimage = np.expand_dims(recimage, axis=-1)  # nx, ny, nz, 1
            nx, ny, nslice, _ = recimage.shape  # nx, ny, nz, 1
            recimage_save = abs(recimage)
            percentiles = np.percentile(recimage_save, 99.5, axis=(0, 1, 3), keepdims=True)  # normalize using 99.5th percentile value, not using nz
            recimage_save_norm = recimage_save / percentiles
            filename_save = f"{filename}.nii.gz"
            full_save_path = os.path.join(save_path, filename_save)
            saveto_nii_nocsv(full_save_path, recimage_save_norm)
        else:
            # 2D images are promoted to [nx, ny, 1, 1] for NIfTI zoom handling.
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
    """CLI entrypoint for converting CMRxReconAll GTSOS files to NIfTI.

    The script scans ``GTSOS`` ``.mat`` files under ``--input`` unless
    ``--exact_filename`` is provided. Files are filtered by modality and
    evaluation set, converted to normalized 4D NIfTI images, and saved under a
    mirrored ``GTSOS_NII`` tree rooted at ``--output`` with unit spacing.
    """
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
        # A single explicit GTSOS file bypasses the modality glob search.
        f = [exact_filename]
        print('##############')
        print("Exact recon filename:", exact_filename)
        print(f'Total files: {len(f)}')
        print('##############')

    elif exact_filename is None:
        # get input file list
        # TODO: Need to be changed according to the TestSet !!!
        # Modality-specific glob patterns match the CMRxReconAll file names.
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
                # Keep only GTSOS files matching the requested modality and split.
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
