"""
Groudtruth SOS reconstruction in advance for fast evaluaion - pytorch - CMRxReconAll
Created on 2025/04/23
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
from utils import zf_recon_4D5D_cpu


def GTpredict(f, center_crop=False, input_dir='', output_dir='', plot_image=False):
    # 1. predict
    for ff in tqdm(f, desc='files'):
        print('-- processing --', ff)
        save_path = ff.replace('FullSample', 'GTSOS').replace(input_dir, output_dir)
        if os.path.isfile(save_path):  # check if the .mat file already exists
            continue
        elif not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        _, sosimage = zf_recon_4D5D_cpu(ff)  # image: [nt,nz,ny,nx] or [nz,ny,nx] or [ny,nx]
        # print(f"{sosimage.shape}")

        if sosimage.ndim == 4:
            sosimage_savemat = sosimage.transpose(3, 2, 1, 0)  # nx, ny, nz, nt
        elif sosimage.ndim == 3:
            sosimage_savemat = sosimage.transpose(2, 1, 0)  # nx, ny, nz
        else:
            sosimage_savemat = sosimage.transpose(1, 0)  # nx, ny
        sio.savemat(save_path, {'gtsosimage': sosimage_savemat})
        print('-- saving --', save_path)


if __name__ == '__main__':
    argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    parser.add_argument('--center_crop', action='store_true', default=False, help='Enable center cropping for validation leaderboard submission')
    parser.add_argument('--evaluate_set', type=str, default="TestSet", help='Choose the evaluation set: TestSet')
    parser.add_argument('--modality', type=str, default='All', help='Choose to inference on which type of data')
    parser.add_argument('--exact_filename', type=str, default=None, help='exact filename to test')
    # exact_filename example:
    # /SSDHome/home/Raw_data/MICCAIChallengeAll/ChallengeData/MultiCoil/Cine/TestSet/FullSample/Center015/Siemens_30T_Vida/P301/cine_sax.mat
    parser.add_argument('--plot_image', type=bool, default=False, help='plot the recon image')

    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    center_crop = args.center_crop
    evaluate_set = args.evaluate_set
    modality = args.modality
    exact_filename = args.exact_filename
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
                    if all(x in file for x in ['FullSample', modal, evaluate_set])
                ])
        f = sum(file_dict.values(), [])
        print('##############')
        for modal, files in file_dict.items():
            print(f'{modal} files: {len(files)}')
        print(f'Total files: {len(f)}')
        print('##############')

    # main function: reconstruct gt and save files
    GTpredict(f, center_crop=center_crop, input_dir=input_dir, output_dir=output_dir, plot_image=plot_image)
