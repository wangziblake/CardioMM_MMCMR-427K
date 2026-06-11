"""
Save reconstructed images from different methods to .nii.gz for subsequent analysis (segmentation, classification) - pytorch - CMRxReconAll
Created on 2025/07/07
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

    Reconstruction files are expected to contain one data array plus MATLAB
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


def replace_mask_to_find_data(mask_filename):
    """Convert a mask-pattern reconstruction filename to its base data filename.

    Files with names like ``cine_sax_mask_Uniform8.mat`` map to
    ``cine_sax.mat``. This is used before replacing the reconstruction task
    folder with ``FullSample`` to find the paired scanner-parameter CSV.

    Args:
        mask_filename: Path containing a ``_mask_`` suffix before ``.mat``.

    Returns:
        Path with the ``_mask_...`` suffix removed.

    Raises:
        NotImplementedError: If ``_mask_`` is not present.
    """
    if "_mask_" in mask_filename:
        base, ext = mask_filename.rsplit(".", 1)
        # Drop the mask-pattern suffix while preserving the original extension.
        base = base.rsplit("_mask_", 1)[0]
        data_filename = f"{base}.{ext}"
    else:
        raise NotImplementedError("The filename does not contain '_mask_'")  # Raise an error if "_mask_" is missing
    return data_filename


def read_mri_parameters_fromcsv(file_path):
    """Read scanner geometry/timing parameters from a CMRxReconAll CSV file.

    The CSV is expected to contain a ``Parameter`` column and either ``Value``
    or ``Value_1``. Parenthesized units are stripped from parameter names, and
    numeric-looking values are converted to ``int`` or ``float`` where possible.

    Args:
        file_path: Path to the ``*_info.csv`` or fallback ``.csv`` file paired
            with the corresponding ``FullSample`` acquisition.

    Returns:
        A dictionary mapping normalized parameter names, such as ``FOVx``,
        ``FOVy``, ``ReconMatrix_X``, ``ReconMatrix_Y``, ``SliceThickness``, and
        ``TR``, to parsed values.
    """
    params = {}
    df = pd.read_csv(file_path)

    for _, row in df.iterrows():
        param_name = row['Parameter']
        # Some exported CSVs use Value_1 instead of Value for the parameter value.
        value = row.get('Value', row.get('Value_1'))
        if value is None:
            raise KeyError("No 'Value' or 'Value_1' found in the CSV file.")

        # Remove unit annotations such as "(mm)" so downstream keys are stable.
        std_name = re.sub(r'\s*\(.*?\)\s*', '', param_name)

        try:
            float_val = float(value)
            if float_val.is_integer():
                params[std_name] = int(float_val)
            else:
                params[std_name] = float_val
        except (ValueError, TypeError):
            params[std_name] = value.strip()
    return params


def get_affine(ff, nx, ny, params):
    """Build a NIfTI affine and voxel spacing from scanner parameters.

    Siemens files generally use reconstruction matrix fields for in-plane
    spacing, while UIH and Center007 files use the current image matrix size.
    Missing or ``NaN`` slice thickness and in-plane resolution values are
    replaced with the current script defaults.

    Args:
        ff: Source reconstruction file path, used for vendor-specific handling.
        nx: Current first image dimension after any optional crop.
        ny: Current second image dimension after any optional crop.
        params: Parsed scanner parameter dictionary.

    Returns:
        ``(affine, resnx, resny, resnz, resnt)`` where ``affine`` is a diagonal
        4x4 NIfTI affine and the remaining values are voxel/time spacings.
    """
    if 'Siemens' in ff and 'Center007' not in ff:
        resnx = params.get('FOVx') / params.get('ReconMatrix_X')
        resny = params.get('FOVy') / params.get('ReconMatrix_Y')
        resnz = params.get('SliceThickness')
        resnt = params.get('TR') / 1000  # Convert TR from ms to seconds
        resnz = 8 if (resnz == 'NaN' or resnz is None or np.isnan(resnz)) else resnz
    elif 'UIH' in ff or 'Center007' in ff:
        resnx = params.get('FOVx') / nx
        resny = params.get('FOVy') / ny
        resnz = params.get('SliceThickness')
        resnt = params.get('TR') / 1000  # Convert TR from ms to seconds
        resnz = 8 if (resnz == 'NaN' or resnz is None or np.isnan(resnz)) else resnz
    else:  # Default case for other vendors, need to check if the parameters are available
        resnx = params.get('FOVx') / params.get('ReconMatrix_X')
        resny = params.get('FOVy') / params.get('ReconMatrix_Y')
        resnz = params.get('SliceThickness')
        resnt = params.get('TR') / 1000  # Convert TR from ms to seconds
        resnz = 8 if (resnz == 'NaN' or resnz is None or np.isnan(resnz)) else resnz
    
    # If res is not provided, set it to 1 by default
    # The current script uses 2 mm as the fallback in-plane spacing.
    resnx = 2 if (resnx == 'NaN' or resnx is None or np.isnan(resnx)) else resnx
    resny = 2 if (resny == 'NaN' or resny is None or np.isnan(resny)) else resny

    affine = np.diag([resnx, resny, resnz, 1])  # default affine is np.eye(4)
    return affine, resnx, resny, resnz, resnt


def saveto_nii(ff, nx, ny, params, full_save_path, recimage):
    """Save a normalized reconstruction image as ``.nii.gz``.

    Args:
        ff: Source reconstruction filename, used only for vendor-specific affine
            calculation.
        nx: Current image x dimension.
        ny: Current image y dimension.
        params: Parsed scanner parameter dictionary.
        full_save_path: Destination ``.nii.gz`` path.
        recimage: 4D image array in ``[nx, ny, nz, nt]`` layout.
    """
    affine, resnx, resny, resnz, resnt = get_affine(ff, nx, ny, params)
    print(f"resnx: {resnx}, resny: {resny}, resnz: {resnz}, resnt: {resnt}")

    nii_img = nib.Nifti1Image(recimage, affine)
    # Set voxel size nx, ny, nz in header
    header = nii_img.header
    # NIfTI zooms store the spatial voxel spacing and temporal spacing.
    header.set_zooms((resnx, resny, resnz, resnt))

    nib.save(nii_img, full_save_path)
    return


def reconsaveimage2nii(f, center_crop=False, input_dir='', output_dir='', csv_dir='', task='TaskAll', image_scale=1, plot_image=False):
    """Convert reconstruction-result ``.mat`` files into normalized NIfTI images.

    Args:
        f: Iterable of method reconstruction ``.mat`` files.
        center_crop: If ``True``, remove 2x readout oversampling with
            ``image_halfcropnx`` before saving.
        input_dir: Method-specific ``ReconResult`` root.
        output_dir: Method-specific ``ImageNII`` output root.
        csv_dir: Root of the original CMRxReconAll data tree containing
            ``FullSample`` scan-parameter CSV files.
        task: Reconstruction task folder name used when mapping to
            ``FullSample`` CSV paths.
        image_scale: Compatibility argument for older visualization workflows;
            the current saving path does not apply it.
        plot_image: Compatibility argument retained by the CLI; no plotting is
            performed in the current implementation.

    Saves:
        One ``.nii.gz`` file per reconstruction under a mirrored output
        directory. Images are saved as nonnegative magnitudes normalized by the
        99.5th percentile over x, y, and time, while preserving slice-specific
        scaling.
    """
    # 1. transform and save .nii.gz
    for ff in tqdm(f, desc='files'):
        print('-- processing --', ff)
        # If ff is '/path/to/your/cine.mat', then save_path will be '/path/to/your/'
        # Mirror the reconstruction tree from ReconResult into ImageNII.
        save_path = os.path.dirname(ff).replace(input_dir, output_dir)
        if not os.path.isdir(save_path):
            # Create the mirrored output directory before saving the NIfTI file.
            os.makedirs(save_path)
        filename = os.path.basename(ff).replace('.mat', '')  # If ff is '/path/to/your/cine.mat', then filename will be 'cine'
        # Remove the mask suffix and map task output back to the paired FullSample CSV.
        csvname = replace_mask_to_find_data(ff).replace(input_dir, csv_dir).replace(task, 'FullSample').replace('.mat', '_info.csv')  # load the corresponding .csv file

        if os.path.isfile(f"{save_path}/{filename}.nii.gz"):  # check if the .nii.gz file already exists
            continue
        recimage = np.squeeze(load_firstkeyvalue_inmat(sio.loadmat(ff)))  # recimage after squeeze: [nx,ny,nz,nt] or [nx,ny,nz] or [nx,ny]
        
        if any(keyword in ff for keyword in {'Cine', 'LGE', 'Mapping', 'Aorta', 'Flow2d', 'Tagging', 'Perfusion', 'T1rho'}):
            if recimage.ndim == 3:
                # Dynamic single-slice modalities are interpreted as [nx, ny, 1, nt].
                recimage = np.expand_dims(recimage, axis=-2)  # nx, ny, 1, nt
        print(f"{recimage.shape}")

        if not os.path.isfile(csvname):
            # Some cases store scan parameters as .csv instead of *_info.csv.
            csvname = csvname.replace('_info.csv', '.csv')  # load the corresponding .csv file for some cases (Center007/Siemens_055T_Freemax)
        params = read_mri_parameters_fromcsv(csvname)  # scanparams: FOVx, FOVy, SliceThickness
        
        if center_crop:
            # Optional readout oversampling removal is applied before affine spacing.
            recimage = image_halfcropnx(recimage)  # recimage after crop: [nx/2,ny,nz,nt] or [nx/2,ny,nz] or [nx/2,ny]

        if recimage.ndim == 4:
            nx, ny, nslice, nt = recimage.shape  # nx, ny, nz, nt
            recimage_save = abs(recimage)
            # Normalize per slice using the high percentile over x, y, and time.
            percentiles = np.percentile(recimage_save, 99.5, axis=(0, 1, 3), keepdims=True)  # normalize using 99.5th percentile value, not using nz
            recimage_save_norm = recimage_save / percentiles
            filename_save = f"{filename}.nii.gz"
            full_save_path = os.path.join(save_path, filename_save)
            saveto_nii(ff, nx, ny, params, full_save_path, recimage_save_norm)
        elif recimage.ndim == 3:
            # Static 3D volumes are promoted to a singleton time dimension.
            recimage = np.expand_dims(recimage, axis=-1)  # nx, ny, nz, 1
            nx, ny, nslice, _ = recimage.shape  # nx, ny, nz, 1
            recimage_save = abs(recimage)
            percentiles = np.percentile(recimage_save, 99.5, axis=(0, 1, 3), keepdims=True)  # normalize using 99.5th percentile value, not using nz
            recimage_save_norm = recimage_save / percentiles
            filename_save = f"{filename}.nii.gz"
            full_save_path = os.path.join(save_path, filename_save)
            saveto_nii(ff, nx, ny, params, full_save_path, recimage_save_norm)
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
            saveto_nii(ff, nx, ny, params, full_save_path, recimage_save_norm)
        print('-- saving --', save_path)


if __name__ == '__main__':
    """CLI entrypoint for exporting method reconstructions as NIfTI files.

    The script scans ``<input>/<method>/ReconResult`` unless
    ``--exact_filename`` is provided. Files are filtered by modality,
    evaluation split, task, and undersampling substring, converted to normalized
    4D NIfTI images using scanner CSV metadata from ``--csv_dir``, and saved
    under ``<output>/<method>/ImageNII``.
    """
    argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    parser.add_argument('--csv_dir', type=str, nargs='?', default='/csv_dir', help='metadata csv directory')
    parser.add_argument('--center_crop', action='store_true', default=False, help='Enable center cropping')
    parser.add_argument('--evaluate_set', type=str, default="TestSet", help='Choose the evaluation set: TestSet')
    parser.add_argument('--task', type=str, default='TaskAll', help='Choose to inference on which type of task')
    parser.add_argument('--modality', type=str, default='All', help='Choose to inference on which type of data')
    parser.add_argument('--method', type=str, default="CardioMM", help='Choose the reconstruction method')
    parser.add_argument('--undersample', type=str, default='', help='Choose the undersampling pattern (Uniform8, ktGaussian16, ktRadial24)')
    parser.add_argument('--exact_filename', type=str, default=None, help='exact filename to test')
    # exact_filename example:
    # /SSDHome/home/wangz/CMRData/Results/CardioMM/ReconResult/Cine/TestSet/TaskAll/Center015/Siemens_30T_Vida/P301/cine_sax_mask_Uniform8.mat
    parser.add_argument('--image_scale', type=float, default=1, help='scale the recon image for better visualization')
    parser.add_argument('--plot_image', type=bool, default=False, help='plot the recon image')

    args = parser.parse_args()
    center_crop = args.center_crop
    evaluate_set = args.evaluate_set
    task = args.task
    modality = args.modality
    method = args.method
    undersample = args.undersample
    exact_filename = args.exact_filename
    image_scale = args.image_scale
    plot_image = args.plot_image

    # Reconstruction inputs and NIfTI outputs are grouped by method name.
    input_dir = f"{args.input}/{method}/ReconResult"
    output_dir = f"{args.output}/{method}/ImageNII"
    csv_dir = args.csv_dir  # xxx/MICCAIChallengeAll/ChallengeData/MultiCoil

    print("Input data store in:", input_dir)
    print("Output data store in:", output_dir)

    if exact_filename is not None:
        # A single explicit reconstruction file bypasses the modality glob search.
        f = [exact_filename]
        print('##############')
        print("Exact recon filename:", exact_filename)
        print(f'Total files: {len(f)}')
        print('##############')

    elif exact_filename is None:
        # get input file list
        # TODO: Need to be changed according to the TestSet !!!
        # Modality-specific glob patterns match the CMRxReconAll result names.
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
                # Empty CLI filters are substrings of every path, preserving optional filters.
                file_dict[modal] = sorted([
                    file for file in glob.glob(join(input_dir, f'**/{pattern}'), recursive=True)
                    if all(x in file for x in [method, modal, evaluate_set, task, undersample])
                ])
        f = sum(file_dict.values(), [])
        print('##############')
        for modal, files in file_dict.items():
            print(f'{modal} files: {len(files)}')
        print(f'Total files: {len(f)}')
        print('##############')

    # main function: load and save files
    reconsaveimage2nii(f, center_crop=center_crop, input_dir=input_dir, output_dir=output_dir, csv_dir=csv_dir, task=task, image_scale=image_scale, plot_image=plot_image)
