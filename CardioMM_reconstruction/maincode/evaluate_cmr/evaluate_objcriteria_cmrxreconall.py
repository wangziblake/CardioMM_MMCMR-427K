"""
Calculate quantatative objective criteria (NMSE, PSNR, SSIM) between reconstructed images and gt images - pytorch - CMRxReconAll
- load the .mat files (sos images are saved in advance for fast evaluation)
Created on 2025/08/14
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
from evaluate_utils import cal_objcriteria
import pandas as pd
from itertools import takewhile


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


def split_letters_digits(s):
    letters = ''.join(takewhile(str.isalpha, s))
    digits = s[len(letters):]
    return letters, digits


def replace_mask_to_find_data(mask_filename):
    if "_mask_" in mask_filename:
        base, ext = mask_filename.rsplit(".", 1)
        base = base.rsplit("_mask_", 1)[0]
        data_filename = f"{base}.{ext}"
    else:
        raise NotImplementedError("The filename does not contain '_mask_'")  # Raise an error if "_mask_" is missing
    return data_filename


def replace_mask_to_find_datatype_undersample_af(mask_filename):
    if "_mask_" in mask_filename:
        base, ext = mask_filename.rsplit(".", 1)  # 'cine_sax_mask_Uniform8', '.mat'
        datatype = base.rsplit("_mask_", 1)[0]  # 'cine_sax'
        masktype = base.rsplit("_mask_", 1)[1]  # 'Uniform8'
        mask, af = split_letters_digits(masktype)  # 'Uniform', '8'
    else:
        raise NotImplementedError("The filename does not contain '_mask_'")  # Raise an error if "_mask_" is missing
    return datatype, mask, af


def extract_attrs(save_path, medcon=None):
    # save_path: "../Cine/TestSet/TaskAll/Center015/Siemens_30T_Vida/P031/cine_sax_mask_Uniform8.mat"
    path_parts = save_path.split(os.sep)
    modality = path_parts[-7]  # 'Cine'
    center = path_parts[-4]  # 'Center001'
    scanner = path_parts[-3]  # 'Siemens_30T_Vida'
    pfolder = path_parts[-2]  # 'P031'
    datatype, mask, af = replace_mask_to_find_datatype_undersample_af(path_parts[-1])  # 'cine_sax', 'Uniform', '8'
    medcon = None

    # Split '_' to obtain 'Siemens', '30T', 'Vida'
    vendor, field_strength, _ = scanner.split('_')

    # TODO: get more disease information from patient clinical report
    if medcon is None:
        if  'Center014' in save_path or 'Center015' in save_path or 'Center007' in save_path:  # TODO: Need to check sometimes
            medcon = 'NC'
        else:
            medcon = 'unknown'
    return modality, center, vendor, field_strength, pfolder, datatype, mask, af, medcon


def add_or_update_row(ranks, new_row, check_cols, criteria_cols):
    """
    This function checks if a row with the same 'check_cols' values already exists in the DataFrame 'ranks'.
    If it exists and only the criteria are different, it updates the criteria ('PSNR', 'SSIM', 'NMSE').
    If the entire row (including criteria) is the same, it skips adding the row.
    If the row doesn't exist, it adds the new row to the DataFrame.
    
    Parameters:
    ranks (pd.DataFrame): The DataFrame containing previous records.
    new_row (dict): The new row to be added or checked for duplication.
    check_cols (list): The list of columns to check for duplicate rows (non-criteria columns).
    criteria_cols (list): The list of columns to update if they are different (criteria columns like 'PSNR', 'SSIM', 'NMSE').
    
    Returns:
    pd.DataFrame: The updated DataFrame with the new row or updated criteria.
    """
    # Check if there is any existing row that matches 'check_cols' values (excluding criteria columns)
    mask = (ranks[check_cols] == pd.Series(new_row)[check_cols]).all(axis=1)
    if mask.any():  # If a row with matching non-criteria columns is found
        idx = mask.idxmax()  # Get the index of the matching row
        # Check if the criteria columns are different
        if any(ranks.loc[idx, criteria_cols] != pd.Series(new_row)[criteria_cols]):
            # If criteria are different, update them
            for col in criteria_cols:
                ranks.loc[idx, col] = new_row[col]
        # If criteria are the same, skip adding the row (do nothing)
    else:  # If no matching row is found, add the new row
        ranks = pd.concat([ranks, pd.DataFrame([new_row])], ignore_index=True)
    return ranks


def evaluate_objcriteria(f, center_crop=False, input_dir='', output_dir='', gtinput_dir='', method='e2evarnet_cas10_acs20', task='TaskAll', evaluate_set='TestSet', medcon=None, normscheme='percentile'):
    save_path = os.path.join(output_dir, f'ObjCal_{evaluate_set}_{task}_{normscheme}_{method}.csv')
    # 0. load previous saved .csv file, if possible
    if os.path.exists(save_path):
        print(f'-- find existing CSV: {save_path}, loading --')
        ranks = pd.read_csv(save_path)
    else:
        # placeholder for ranks
        print(f'-- no existing CSV, starting fresh --')
        ranks = pd.DataFrame(columns = ['Method', 'Modality', 'Task', 'Center', 'Vendor', 'Field', 'Pfolder', 'Datatype', 'Mask', 'AF', 'Medcon', 'PSNR', 'SSIM', 'NMSE'])
   
    # columns to check for duplicates (excluding the criteria columns)
    check_cols = ['Method', 'Modality', 'Task', 'Center', 'Vendor', 'Field', 'Pfolder', 'Datatype', 'Mask', 'AF', 'Medcon']
    # criteria columns to update if different (PSNR, SSIM, NMSE)
    criteria_cols = ['PSNR', 'SSIM', 'NMSE']

    # 1. evaluate recontructed images
    for ff in tqdm(f, desc='files'):
        print('-- processing --', ff)
        # ff example: /{input_dir}/Cine/TestSet/TaskAll/Center015/Siemens_30T_Vida/P301/cine_sax_mask_Uniform8.mat
        # gtff example: /{gtinput_dir}/Cine/TestSet/GTSOS/Center015/Siemens_30T_Vida/P301/cine_sax.mat
        gtff = replace_mask_to_find_data(ff).replace(input_dir, gtinput_dir).replace(task, 'GTSOS')
        modality, center, vendor, field_strength, pfolder, datatype, mask, af, medcon = extract_attrs(ff, medcon)  # get the attributes from the filename
        
        # load data for evaluation
        recimage = load_firstkeyvalue_inmat(sio.loadmat(ff))  # recimage: [nx,ny,nz,nt] or [nx,ny,nz,1]/[nx,ny,1,nt] or [nx,ny,1,1]
        gtimage = load_firstkeyvalue_inmat(sio.loadmat(gtff))  # gtimage: [nx,ny,nz,nt] or [nx,ny,nz]/[nx,ny,nt] or [nx,ny]

        print('-- start calculating --', ff)
        # calculate the objective criteria for each .mat file
        # norm_scheme: percentile, maxval, std
        psnr_array, ssim_array, nmse_array = cal_objcriteria(recimage, gtimage, norm_scheme=normscheme)
        # take mean for each .mat file
        psnr_filemean = round(np.nanmean(psnr_array), 4)
        ssim_filemean = round(np.nanmean(ssim_array), 4)
        nmse_filemean = round(np.nanmean(nmse_array), 4)

        # save the evaluation results to the pandas frame
        new_row = {'Method': method, 'Modality': modality, 'Task': task, 'Center': center, 'Vendor': vendor, 'Field': field_strength, 
                    'Pfolder': pfolder, 'Datatype': datatype, 'Mask': mask, 'AF': af, 'Medcon': medcon, 
                    'PSNR': psnr_filemean, 'SSIM': ssim_filemean, 'NMSE': nmse_filemean}
        # Call the function to either add the row or update the metrics if only they are different
        ranks = add_or_update_row(ranks, new_row, check_cols, criteria_cols)
        print('-- end calculating --', ff)

    # 2. save results to .csv
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    ranks.to_csv(save_path, index=False)
    print('-- saving --', output_dir)


if __name__ == '__main__':
    argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='recon input directory')
    parser.add_argument('--gtinput', type=str, nargs='?', default='/gtinput', help='gt input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    parser.add_argument('--center_crop', action='store_true', default=False, help='Enable center cropping')
    parser.add_argument('--evaluate_set', type=str, default="TestSet", help='Choose the evaluation set: TestSet')
    parser.add_argument('--task', type=str, default='TaskAll', help='Choose to inference on which type of task')
    parser.add_argument('--modality', type=str, default='All', help='Choose to inference on which type of data')
    parser.add_argument('--method', type=str, default="CardioMM", help='Choose the reconstruction method')
    parser.add_argument('--undersample', type=str, default='', help='Choose the undersampling pattern (Uniform8, ktGaussian16, ktRadial24)')
    parser.add_argument('--center', type=str, default='', help='Choose the center (e.g., Center00X)')
    parser.add_argument('--vendor', type=str, default='', help='Choose the vendor (Siemens, UIH, GE, Philips, Neusoft)')
    parser.add_argument('--field', type=str, default='', help='Choose the magnetic field strength (15T, 30T, 50T, 055T)')
    parser.add_argument('--medcon', type=str, default='', help='Choose the patient medical condition (e.g., NC: Normal control)')
    parser.add_argument('--normscheme', type=str, default='percentile', help='Choose the data norm scheme for calculation (e.g., percentile, maxval, std)')
    parser.add_argument('--exact_filename', type=str, default=None, help='exact filename to test')
    # exact_filename example:
    # /SSDHome/home/wangz/CMRData/Results/Results/CardioMM/ReconResult/Cine/TestSet/TaskAll/Center015/Siemens_30T_Vida/P301/cine_sax_mask_Uniform8.mat

    args = parser.parse_args()
    center_crop = args.center_crop
    evaluate_set = args.evaluate_set
    task = args.task
    modality = args.modality
    method = args.method
    undersample = args.undersample
    center = args.center
    vendor = args.vendor
    field = args.field
    medcon = args.medcon
    normscheme = args.normscheme
    exact_filename = args.exact_filename

    input_dir = f"{args.input}/{method}/ReconResult"
    output_dir = f"{args.output}/{method}/ReconResult_Criteria"
    gtinput_dir = args.gtinput

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
                    if all(x in file for x in [method, modal, evaluate_set, task, undersample, center, vendor, field, medcon])
                ])
        f = sum(file_dict.values(), [])
        print('##############')
        for modal, files in file_dict.items():
            print(f'{modal} files: {len(files)}')
        print(f'Total files: {len(f)}')
        print('##############')

    # main function: calculate and save files
    evaluate_objcriteria(f, center_crop=center_crop, input_dir=input_dir, output_dir=output_dir, gtinput_dir=gtinput_dir, method=method, task=task, evaluate_set=evaluate_set, medcon=medcon, normscheme=normscheme)
