"""
Save reconstructed images from different methods to .png for visualization - pytorch - CMRxReconAll
Created on 2025/04/28
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
from tqdm import tqdm
import matplotlib.pyplot as plt
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


def reconsaveimage2png(f, center_crop=False, input_dir='', output_dir='', image_scale=1, plot_image=False):
    # 1. transform and save .png
    for ff in tqdm(f, desc='files'):
        print('-- processing --', ff)
        # If ff is '/path/to/your/cine.mat', then save_path will be '/path/to/your/'
        save_path = os.path.dirname(ff).replace(input_dir, output_dir)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        filename = os.path.basename(ff).replace('.mat', '')  # If ff is '/path/to/your/cine.mat', then filename will be 'cine'
        
        if os.path.isfile(f"{save_path}/{filename}_1_1.png") or os.path.isfile(f"{save_path}/{filename}_1.png"):  # check if the .png file already exists
            continue
        recimage = np.squeeze(load_firstkeyvalue_inmat(sio.loadmat(ff)))  # recimage after squeeze: [nx,ny,nz,nt] or [nx,ny,nz] or [nx,ny]
        # print(f"{recimage.shape}")
        
        if center_crop:
            recimage = image_halfcropnx(recimage)  # recimage after crop: [nx/2,ny,nz,nt] or [nx/2,ny,nz] or [nx/2,ny]

        if recimage.ndim == 4:
            nx, ny, nslice, nt = recimage.shape  # nx, ny, nz, nt
            for n in range(1, nslice + 1):
                for i in range(1, nt + 1):
                    recimage_save = abs(np.squeeze(recimage[:,:,n-1,i-1]))
                    recimage_save_norm = recimage_save / np.percentile(recimage_save, 99.5)  # normalize using 99.5th percentile value
                    filename_save = f"{filename}_{n}_{i}.png"
                    full_save_path = os.path.join(save_path, filename_save)
                    plt.imsave(full_save_path, np.clip(recimage_save_norm*image_scale, 0, 1), cmap='gray')  # value larger than 1 will be truncated to 1
                    if plot_image:
                        plt.imshow(recimage_save_norm*image_scale, cmap='gray')
                        plt.axis('off')
                        plt.show()
        elif recimage.ndim == 3:
            nx, ny, nslice_nt = recimage.shape  # nx, ny, nz
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
    argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
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

    input_dir = f"{args.input}/{method}/ReconResult"
    output_dir = f"{args.output}/{method}/ImagePNG"

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
                    if all(x in file for x in [method, modal, evaluate_set, task, undersample])
                ])
        f = sum(file_dict.values(), [])
        print('##############')
        for modal, files in file_dict.items():
            print(f'{modal} files: {len(files)}')
        print(f'Total files: {len(f)}')
        print('##############')

    # main function: load and save files
    reconsaveimage2png(f, center_crop=center_crop, input_dir=input_dir, output_dir=output_dir, image_scale=image_scale, plot_image=plot_image)
