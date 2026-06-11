"""
Save reconstructed gt sos images to .png for visualization - pytorch - CMRxReconAll
Created on 2025/04/28
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
from tqdm import tqdm
import matplotlib.pyplot as plt
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


def gtsaveimage2png(f, center_crop=False, input_dir='', output_dir='', image_scale=1, plot_image=False):
    """Convert ground-truth SOS ``.mat`` files into per-slice PNG images.

    Args:
        f: Iterable of ``GTSOS`` ``.mat`` files.
        center_crop: If ``True``, remove 2x readout oversampling with
            ``image_halfcropnx`` before PNG export.
        input_dir: Source root used when mirroring paths into ``output_dir``.
        output_dir: Destination root for the mirrored ``GTSOS_PNG`` tree.
        image_scale: Display multiplier applied after percentile normalization.
        plot_image: If ``True``, display each exported image with matplotlib.

    Saves:
        PNG files under a mirrored ``GTSOS_PNG`` directory. 4D inputs are saved
        as ``<name>_<slice>_<time>.png``; 3D and 2D inputs are saved as
        ``<name>_<index>.png``. Each PNG is normalized by its own 99.5th
        percentile and clipped to ``[0, 1]`` for grayscale display.
    """
    # 1. transform and save .png
    for ff in tqdm(f, desc='files'):
        print('-- processing --', ff)
        # If ff is '/path/to/your/cine.mat', then save_path will be '/path/to/your/'
        # Mirror the input tree while replacing GTSOS folders with GTSOS_PNG.
        save_path = os.path.dirname(ff).replace('GTSOS', 'GTSOS_PNG').replace(input_dir, output_dir)
        if not os.path.isdir(save_path):
            # Create the mirrored output folder before saving the PNG files.
            os.makedirs(save_path)
        filename = os.path.basename(ff).replace('.mat', '')  # If ff is '/path/to/your/cine.mat', then filename will be 'cine'
        
        if os.path.isfile(f"{save_path}/{filename}_1_1.png") or os.path.isfile(f"{save_path}/{filename}_1.png"):  # check if the .png file already exists
            # Either naming pattern indicates this volume was already exported.
            continue
        recimage = load_firstkeyvalue_inmat(sio.loadmat(ff))  # recimage: [nx,ny,nz,nt] or [nx,ny,nz] or [nx,ny]
        # print(f"{recimage.shape}")

        if center_crop:
            # Optional readout oversampling removal keeps the exported PNG width consistent.
            recimage = image_halfcropnx(recimage)  # recimage after crop: [nx/2,ny,nz,nt] or [nx/2,ny,nz] or [nx/2,ny]

        if recimage.ndim == 4:
            nx, ny, nslice, nt = recimage.shape  # nx, ny, nz, nt
            # 4D data are exported for every slice and time frame.
            for n in range(1, nslice + 1):
                for i in range(1, nt + 1):
                    recimage_save = abs(np.squeeze(recimage[:,:,n-1,i-1]))
                    # Normalize each displayed frame independently for visualization.
                    recimage_save_norm = recimage_save / np.percentile(recimage_save, 99.5)  # normalize using 99.5th percentile value
                    filename_save = f"{filename}_{n}_{i}.png"
                    full_save_path = os.path.join(save_path, filename_save)
                    # Values above one are saturated in the saved grayscale PNG.
                    plt.imsave(full_save_path, np.clip(recimage_save_norm*image_scale, 0, 1), cmap='gray')  # value larger than 1 will be truncated to 1
                    if plot_image:
                        plt.imshow(recimage_save_norm*image_scale, cmap='gray')
                        plt.axis('off')
                        plt.show()
        elif recimage.ndim == 3:
            nx, ny, nslice_nt = recimage.shape  # nx, ny, nz
            # 3D inputs may represent slices or time frames depending on modality.
            for n in range(1, nslice_nt + 1):
                recimage_save = abs(np.squeeze(recimage[:,:,n-1]))
                recimage_save_norm = recimage_save / np.percentile(recimage_save, 99.5)  # normalize using 99.5th percentile value
                filename_save = f"{filename}_{n}.png"
                full_save_path = os.path.join(save_path, filename_save)
                plt.imsave(full_save_path, np.clip(recimage_save_norm*image_scale, 0, 1), cmap='gray')  # value larger than 1 will be truncated to 1
                if plot_image:
                    plt.imshow(recimage_save_norm*image_scale, cmap='gray')
                    plt.axis('off')
                    plt.show()
        else:
            nx, ny = recimage.shape  # nx, ny
            nslice_nt = 1
            # 2D inputs produce a single PNG with the same suffix style as 3D data.
            for n in range(1, nslice_nt + 1):
                recimage_save = abs(np.squeeze(recimage))
                recimage_save_norm = recimage_save / np.percentile(recimage_save, 99.5)  # normalize using 99.5th percentile value
                filename_save = f"{filename}_{n}.png"
                full_save_path = os.path.join(save_path, filename_save)
                plt.imsave(full_save_path, np.clip(recimage_save_norm * image_scale, 0, 1), cmap='gray')  # value larger than 1 will be truncated to 1
                if plot_image:
                    plt.imshow(recimage_save_norm * image_scale, cmap='gray')
                    plt.axis('off')
                    plt.show()
        print('-- saving --', save_path)


if __name__ == '__main__':
    """CLI entrypoint for exporting CMRxReconAll GTSOS files as PNGs.

    The script scans ``GTSOS`` ``.mat`` files under ``--input`` unless
    ``--exact_filename`` is provided. Files are filtered by modality and
    evaluation set, converted into grayscale slice/time PNGs, and saved under a
    mirrored ``GTSOS_PNG`` tree rooted at ``--output``.
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
    gtsaveimage2png(f, center_crop=center_crop, input_dir=input_dir, output_dir=output_dir, image_scale=image_scale, plot_image=plot_image)
