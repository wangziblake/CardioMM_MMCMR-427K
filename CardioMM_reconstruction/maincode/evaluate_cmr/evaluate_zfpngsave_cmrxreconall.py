"""
Save reconstructed zero-filled sos images to .png for visualization - pytorch - CMRxReconAll
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
import torch
from utils import load_kdata_compatible, load_maskdata, ifft2c
from os.path import join
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from evaluate_utils import image_halfcropnx


def replace_mask_to_find_data(mask_filename):
    """Convert a mask filename into the corresponding full-sample data filename.

    CMRxReconAll mask files include an undersampling suffix such as
    ``_mask_ktGaussian16`` before the extension. The full-sample file uses the
    same prefix without the ``_mask_...`` portion, after the caller has already
    replaced the ``Mask_<task>`` folder with ``FullSample``.

    Args:
        mask_filename: Path to a mask-derived ``.mat`` filename.

    Returns:
        The matching full-sample ``.mat`` filename.

    Raises:
        NotImplementedError: If the input filename does not contain ``_mask_``.
    """
    if "_mask_" in mask_filename:
        base, ext = mask_filename.rsplit(".", 1)
        # Drop the mask-pattern suffix while preserving the original extension.
        base = base.rsplit("_mask_", 1)[0]
        data_filename = f"{base}.{ext}"
    else:
        raise NotImplementedError("The filename does not contain '_mask_'")  # Raise an error if "_mask_" is missing
    return data_filename


def get_zfdata(fname, task):
    """Load masked data and compute a zero-filled SOS/RSS reconstruction.

    The input ``fname`` is a mask file. The matching full-sample k-space file is
    inferred by replacing ``Mask_<task>`` with ``FullSample`` and removing the
    ``_mask_...`` suffix from the filename. K-space is normalized to
    ``[nt, nz, nc, ny, nx]`` before masking. The mask may be either 2D
    ``[ny, nx]`` and reused across time, or 3D ``[nt, ny, nx]`` for kt masks.

    Args:
        fname: Path to the mask ``.mat`` file.
        task: Task folder suffix used in the ``Mask_<task>`` path component.

    Returns:
        A squeezed NumPy array in MATLAB-style image layout
        ``[nx, ny, nz, nt]``, ``[nx, ny, nz]``, or ``[nx, ny]``.
    """
    # The data file lives beside the mask path after folder/name conversion.
    data_fname = replace_mask_to_find_data(fname.replace(f'Mask_{task}', 'FullSample'))
    kspace = load_kdata_compatible(data_fname)
    if len(kspace.shape) != 5:
        # Promote 4D data to [1, nz, nc, ny, nx].
        kspace = np.expand_dims(kspace, axis=0)  # make sure its shape is [1, nz, nc, ny, nx]
        if len(kspace.shape) != 5:
            # Promote 3D data to [1, 1, nc, ny, nx].
            kspace = np.expand_dims(kspace, axis=0)  # make sure its shape is [1, nz, nc, ny, nx]
    num_t = kspace.shape[0]
    num_slices = kspace.shape[1]
    # here, input mask from .mat: 3D-[nt, ny, nx] or 2D-[ny, nx]
    mask = load_maskdata(fname)
    if len(mask.shape) == 3:
        # kt masks already contain a time dimension and share one mask per slice/coil.
        mask = np.expand_dims(np.expand_dims(mask, axis=1),axis=2)  # make sure its shape is [nt, 1, 1, ny, nx]
    elif len(mask.shape) == 2:
        # Static 2D masks are reused for every time frame.
        mask = np.expand_dims(np.expand_dims(np.expand_dims(mask, axis=0), axis=1), axis=2)  # make sure its shape is [1, 1, 1, ny, nx]
        mask = np.tile(mask, reps=(num_t, 1, 1, 1, 1))  # make sure its shape is [nt, 1, 1, ny, nx]
    else:
        raise NotImplementedError("The mask shape should be 2D(k) or 3D(k-t).")
    # Reuse the same sampling pattern across all slices in the volume.
    mask = np.tile(mask, reps=(1, num_slices, 1, 1, 1))  # make sure its shape is [nt, nz, 1, ny, nx]

    masked_kspace = kspace * mask
    # Centered inverse FFT reconstructs one complex image per coil.
    zfimage_coil = ifft2c(torch.tensor(masked_kspace))
    # Root-sum-of-squares combines coils into a zero-filled SOS magnitude image.
    zfimage = (zfimage_coil.abs()**2).sum(-3)**0.5  # [nt, nz, ny, nx]
    zfimagesos = zfimage.cpu().numpy()
    # Save-facing layout follows MATLAB/image convention [nx, ny, nz, nt].
    return np.squeeze(zfimagesos.transpose(3, 2, 1, 0))


def zfsaveimage2png(f, center_crop=False, input_dir='', output_dir='', task='TaskAll', image_scale=1, plot_image=False):
    """Reconstruct zero-filled SOS images from masks and export PNG files.

    Args:
        f: Iterable of CMRxReconAll mask ``.mat`` files.
        center_crop: If ``True``, remove 2x readout oversampling with
            ``image_halfcropnx`` before PNG export.
        input_dir: Source root used when mirroring paths into ``output_dir``.
        output_dir: Destination root for the mirrored ``ZFSOS_PNG`` tree.
        task: Task folder suffix used to map ``Mask_<task>`` to ``FullSample``.
        image_scale: Display multiplier applied after percentile normalization.
        plot_image: If ``True``, display each exported image with matplotlib.

    Saves:
        PNG files under a mirrored ``ZFSOS_PNG`` directory. 4D inputs are saved
        as ``<name>_<slice>_<time>.png``; 3D and 2D inputs are saved as
        ``<name>_<index>.png``. Each PNG is normalized by its own 99.5th
        percentile and clipped to ``[0, 1]`` for grayscale display.
    """
    # 1. transform and save .png
    for ff in tqdm(f, desc='files'):
        print('-- processing --', ff)
        # If ff is '/path/to/your/cine.mat', then save_path will be '/path/to/your/'
        # Mirror the input tree while replacing Mask folders with ZFSOS_PNG.
        save_path = os.path.dirname(ff).replace('Mask', 'ZFSOS_PNG').replace(input_dir, output_dir)
        if not os.path.isdir(save_path):
            # Create the mirrored output folder before saving the PNG files.
            os.makedirs(save_path)
        filename = os.path.basename(ff).replace('.mat', '')  # If ff is '/path/to/your/cine.mat', then filename will be 'cine'
        
        if os.path.isfile(f"{save_path}/{filename}_1_1.png") or os.path.isfile(f"{save_path}/{filename}_1.png"):  # check if the .png file already exists
            # Either naming pattern indicates this zero-filled volume was already exported.
            continue
        recimage = np.squeeze(get_zfdata(ff, task))  # recimage after squeeze: [nx,ny,nz,nt] or [nx,ny,nz] or [nx,ny]
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
                plt.imsave(full_save_path, np.clip(recimage_save_norm*image_scale, 0, 1), cmap='gray')  # value larger than 1 will be truncated to 1
                if plot_image:
                    plt.imshow(recimage_save_norm*image_scale, cmap='gray')
                    plt.axis('off')
                    plt.show()
        print('-- saving --', save_path)


if __name__ == '__main__':
    """CLI entrypoint for exporting zero-filled SOS reconstructions as PNGs.

    The script scans CMRxReconAll mask files under ``--input`` unless
    ``--exact_mask_filename`` is provided. Matching files are filtered by
    modality, task, and evaluation set, reconstructed by applying each mask to
    the paired ``FullSample`` k-space, and saved under a mirrored
    ``ZFSOS_PNG`` tree rooted at ``--output``.
    """
    argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    parser.add_argument('--center_crop', action='store_true', default=False, help='Enable center cropping')
    parser.add_argument('--evaluate_set', type=str, default="TestSet", help='Choose the evaluation set: TestSet')
    parser.add_argument('--task', type=str, default='TaskAll', help='Choose to inference on which type of task')
    parser.add_argument('--modality', type=str, default='All', help='Choose to inference on which type of data')
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
    exact_mask_filename = args.exact_mask_filename
    image_scale = args.image_scale
    plot_image = args.plot_image

    print("Input data store in:", input_dir)
    print("Output data store in:", output_dir)

    if exact_mask_filename is not None:
        # A single explicit mask file bypasses the modality glob search.
        f = [exact_mask_filename]
        print('##############')
        print("Exact recon filename:", exact_mask_filename)
        print(f'Total files: {len(f)}')
        print('##############')

    elif exact_mask_filename is None:
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
                # Keep only mask files matching the selected modality/task/split.
                file_dict[modal] = sorted([
                    file for file in glob.glob(join(input_dir, f'**/{pattern}'), recursive=True)
                    if all(x in file for x in ['Mask', modal, task, evaluate_set])
                ])
        f = sum(file_dict.values(), [])
        print('##############')
        for modal, files in file_dict.items():
            print(f'{modal} files: {len(files)}')
        print(f'Total files: {len(f)}')
        print('##############')

    # main function: reconstruct zf and save files
    zfsaveimage2png(f, center_crop=center_crop, input_dir=input_dir, output_dir=output_dir, task=task, image_scale=image_scale, plot_image=plot_image)
