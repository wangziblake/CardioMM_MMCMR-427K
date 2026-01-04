"""
Calculate the wall thickness from recon sax .nii.gz - pytorch - CMRxReconAll
Created on 2025/08/25
@author: Zi Wang
Modified from Wenjia Bai's code (https://github.com/baiwenjia/ukbb_cardiac)
Email: Zi Wang (zi.wang@imperial.ac.uk)
If you want to use this code, please cite our relevant papers in the GitHub page.
"""

import os
import nibabel as nib
import numpy as np
import glob
import pandas as pd
from tqdm import tqdm
from cardiac_utils import sa_pass_quality_control, evaluate_wall_thickness
from itertools import takewhile


def split_letters_digits(s):
    letters = ''.join(takewhile(str.isalpha, s))
    digits = s[len(letters):]
    return letters, digits


def replace_mask_to_find_datatype_undersample_af(mask_filename):
    if "_mask_" in mask_filename:
        base, ext = mask_filename.rsplit(".", 1)  # 'cine_sax_mask_Uniform8_label_ED.nii', '.gz'
        datatype = base.rsplit("_mask_", 1)[0]  # 'cine_sax'
        masktype = base.rsplit("_mask_", 1)[1].replace('_label_ED.nii', '')  # 'Uniform8_label_ED.nii' -> 'Uniform8'
        mask, af = split_letters_digits(masktype)  # 'Uniform', '8'
    else:
        raise NotImplementedError("The filename does not contain '_mask_'")  # Raise an error if "_mask_" is missing
    return datatype, mask, af


def extract_attrs(save_path, medcon=None):
    # save_path: "../Cine/TestSet/TaskAll/Center015/Siemens_30T_Vida/P031/cine_sax_mask_Uniform8_label_ED.nii.gz"
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
        if 'Center014' in save_path or 'Center015' in save_path or 'Center007' in save_path:  # TODO: Need to check sometimes
            medcon = 'NC'
        else:
            medcon = 'unknown'
    return modality, center, vendor, field_strength, pfolder, datatype, mask, af, medcon


def add_or_update_row(ranks, new_row, check_cols, criteria_cols):
    """
    This function checks if a row with the same 'check_cols' values already exists in the DataFrame 'ranks'.
    If it exists and only the criteria are different, it updates the criteria.
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
    new_df = pd.DataFrame([new_row])
    for col in check_cols:
        if col in ranks.columns and col in new_df.columns:
            ranks[col] = ranks[col].astype(str)
            new_df[col] = new_df[col].astype(str)

    mask = pd.Series(True, index=ranks.index)
    for col in check_cols:
        mask &= (ranks[col] == new_row[col])

    if mask.any():
        idx = mask.idxmax()
        for col in criteria_cols:
            old_val = ranks.loc[idx, col]
            new_val = new_row[col]
            if pd.isna(old_val) or old_val != new_val:
                ranks.loc[idx, col] = new_val
    else:  # If no matching row is found, add the new row
        ranks = pd.concat([ranks, new_df], ignore_index=True)
    return ranks


def process_case(f, RootDir, method, evaluate_set, medcon):
    csvdir = RootDir.replace('SegNII', 'CalClinicalMeasure')
    save_path = os.path.join(csvdir, f'CliCal_{evaluate_set}_WallThickness_{method}.csv')
    save_path_max = os.path.join(csvdir, f'CliCal_{evaluate_set}_WallThicknessMax_{method}.csv')
    # 0. load previous saved .csv file, if possible
    if os.path.exists(save_path) and os.path.exists(save_path_max):
        print(f'-- find existing CSV: {save_path}, loading --')
        print(f'-- find existing CSV_Max: {save_path_max}, loading --')
        ranks = pd.read_csv(save_path)
        ranks_max = pd.read_csv(save_path_max)
    else:
        # placeholder for ranks
        print(f'-- no existing CSV, starting fresh --')
        ranks = pd.DataFrame(
            columns=['Method', 'Modality', 'Task', 'Center', 'Vendor', 'Field', 'Pfolder', 'Datatype',
                     'Mask', 'AF', 'Medcon',
                     'WT_AHA_1 (mm)', 'WT_AHA_2 (mm)', 'WT_AHA_3 (mm)',
                     'WT_AHA_4 (mm)', 'WT_AHA_5 (mm)', 'WT_AHA_6 (mm)',
                     'WT_AHA_7 (mm)', 'WT_AHA_8 (mm)', 'WT_AHA_9 (mm)',
                     'WT_AHA_10 (mm)', 'WT_AHA_11 (mm)', 'WT_AHA_12 (mm)',
                     'WT_AHA_13 (mm)', 'WT_AHA_14 (mm)', 'WT_AHA_15 (mm)', 'WT_AHA_16 (mm)',
                     'WT_Global (mm)'])
        ranks_max = pd.DataFrame(
            columns=['Method', 'Modality', 'Task', 'Center', 'Vendor', 'Field', 'Pfolder', 'Datatype',
                     'Mask', 'AF', 'Medcon',
                     'WT_Max_AHA_1 (mm)', 'WT_Max_AHA_2 (mm)', 'WT_Max_AHA_3 (mm)',
                     'WT_Max_AHA_4 (mm)', 'WT_Max_AHA_5 (mm)', 'WT_Max_AHA_6 (mm)',
                     'WT_Max_AHA_7 (mm)', 'WT_Max_AHA_8 (mm)', 'WT_Max_AHA_9 (mm)',
                     'WT_Max_AHA_10 (mm)', 'WT_Max_AHA_11 (mm)', 'WT_Max_AHA_12 (mm)',
                     'WT_Max_AHA_13 (mm)', 'WT_Max_AHA_14 (mm)', 'WT_Max_AHA_15 (mm)', 'WT_Max_AHA_16 (mm)',
                     'WT_Max_Global (mm)'])

    # columns to check for duplicates (excluding the criteria columns)
    check_cols = ['Method', 'Modality', 'Task', 'Center', 'Vendor', 'Field', 'Pfolder', 'Datatype', 'Mask',
                  'AF', 'Medcon']
    # criteria columns to update if different
    criteria_cols = ['WT_AHA_1 (mm)', 'WT_AHA_2 (mm)', 'WT_AHA_3 (mm)',
                     'WT_AHA_4 (mm)', 'WT_AHA_5 (mm)', 'WT_AHA_6 (mm)',
                     'WT_AHA_7 (mm)', 'WT_AHA_8 (mm)', 'WT_AHA_9 (mm)',
                     'WT_AHA_10 (mm)', 'WT_AHA_11 (mm)', 'WT_AHA_12 (mm)',
                     'WT_AHA_13 (mm)', 'WT_AHA_14 (mm)', 'WT_AHA_15 (mm)', 'WT_AHA_16 (mm)',
                     'WT_Global (mm)']
    criteria_cols_max = ['WT_Max_AHA_1 (mm)', 'WT_Max_AHA_2 (mm)', 'WT_Max_AHA_3 (mm)',
                         'WT_Max_AHA_4 (mm)', 'WT_Max_AHA_5 (mm)', 'WT_Max_AHA_6 (mm)',
                         'WT_Max_AHA_7 (mm)', 'WT_Max_AHA_8 (mm)', 'WT_Max_AHA_9 (mm)',
                         'WT_Max_AHA_10 (mm)', 'WT_Max_AHA_11 (mm)', 'WT_Max_AHA_12 (mm)',
                         'WT_Max_AHA_13 (mm)', 'WT_Max_AHA_14 (mm)', 'WT_Max_AHA_15 (mm)', 'WT_Max_AHA_16 (mm)',
                         'WT_Max_Global (mm)']

    # 1. evaluate segmentation images
    for ff in tqdm(f, desc='files'):
        print('-- processing --', ff)
        # ff example: /{input_dir}/Cine/TestSet/TaskAll/Center015/Siemens_30T_Vida/P301/cine_sax_mask_Uniform8_label_ED.nii.gz
        modality, center, vendor, field_strength, pfolder, datatype, mask, af, medcon = extract_attrs(ff, medcon)  # get the attributes from the filename

        # load data for evaluation
        seg_ED_name = ff
        if not os.path.exists(seg_ED_name):
            continue
        if not sa_pass_quality_control(seg_ED_name):
            continue

        # Evaluate myocardial wall thickness
        print('-- start calculating --', ff)
        temp_WT_ED_vtk = ff.replace('_label_ED.nii.gz', '_label_ED_WT')
        evaluate_wall_thickness(ff, temp_WT_ED_vtk)

        # Record data
        if os.path.exists('{0}.csv'.format(temp_WT_ED_vtk)):
            df = pd.read_csv('{0}.csv'.format(temp_WT_ED_vtk), index_col=0)
            line = df['Thickness'].values
            val = {}
            for i in range(16):
                val[f'WT_AHA_{i+1}'] = line[i]
            val['WT_Global'] = line[16]

        if os.path.exists('{0}_max.csv'.format(temp_WT_ED_vtk)):
            df = pd.read_csv('{0}_max.csv'.format(temp_WT_ED_vtk), index_col=0)
            line = df['Thickness_Max'].values
            val_max = {}
            for i in range(16):
                val_max[f'WT_Max_AHA_{i+1}'] = line[i]
            val_max['WT_Max_Global'] = line[16]

        # save the evaluation results to the pandas frame
        new_row = {'Method': method, 'Modality': modality, 'Task': 'TaskAll', 'Center': center, 'Vendor': vendor,
                   'Field': field_strength, 'Pfolder': pfolder, 'Datatype': datatype, 'Mask': mask, 'AF': af, 'Medcon': medcon,
                   'WT_AHA_1 (mm)': val['WT_AHA_1'], 'WT_AHA_2 (mm)': val['WT_AHA_2'], 'WT_AHA_3 (mm)': val['WT_AHA_3'],
                   'WT_AHA_4 (mm)': val['WT_AHA_4'], 'WT_AHA_5 (mm)': val['WT_AHA_5'], 'WT_AHA_6 (mm)': val['WT_AHA_6'],
                   'WT_AHA_7 (mm)': val['WT_AHA_7'], 'WT_AHA_8 (mm)': val['WT_AHA_8'], 'WT_AHA_9 (mm)': val['WT_AHA_9'],
                   'WT_AHA_10 (mm)': val['WT_AHA_10'], 'WT_AHA_11 (mm)': val['WT_AHA_11'], 'WT_AHA_12 (mm)': val['WT_AHA_12'],
                   'WT_AHA_13 (mm)': val['WT_AHA_13'], 'WT_AHA_14 (mm)': val['WT_AHA_14'], 'WT_AHA_15 (mm)': val['WT_AHA_15'], 'WT_AHA_16 (mm)': val['WT_AHA_16'],
                   'WT_Global (mm)': val['WT_Global']}
        new_row_max = {'Method': method, 'Modality': modality, 'Task': 'TaskAll', 'Center': center, 'Vendor': vendor,
                       'Field': field_strength, 'Pfolder': pfolder, 'Datatype': datatype, 'Mask': mask, 'AF': af, 'Medcon': medcon,
                       'WT_Max_AHA_1 (mm)': val_max['WT_Max_AHA_1'], 'WT_Max_AHA_2 (mm)': val_max['WT_Max_AHA_2'], 'WT_Max_AHA_3 (mm)': val_max['WT_Max_AHA_3'],
                       'WT_Max_AHA_4 (mm)': val_max['WT_Max_AHA_4'], 'WT_Max_AHA_5 (mm)': val_max['WT_Max_AHA_5'], 'WT_Max_AHA_6 (mm)': val_max['WT_Max_AHA_6'],
                       'WT_Max_AHA_7 (mm)': val_max['WT_Max_AHA_7'], 'WT_Max_AHA_8 (mm)': val_max['WT_Max_AHA_8'], 'WT_Max_AHA_9 (mm)': val_max['WT_Max_AHA_9'],
                       'WT_Max_AHA_10 (mm)': val_max['WT_Max_AHA_10'], 'WT_Max_AHA_11 (mm)': val_max['WT_Max_AHA_11'], 'WT_Max_AHA_12 (mm)': val_max['WT_Max_AHA_12'],
                       'WT_Max_AHA_13 (mm)': val_max['WT_Max_AHA_13'], 'WT_Max_AHA_14 (mm)': val_max['WT_Max_AHA_14'], 'WT_Max_AHA_15 (mm)': val_max['WT_Max_AHA_15'], 'WT_Max_AHA_16 (mm)': val_max['WT_Max_AHA_16'],
                       'WT_Max_Global (mm)': val_max['WT_Max_Global']}

        # Call the function to either add the row or update the metrics if only they are different
        ranks = add_or_update_row(ranks, new_row, check_cols, criteria_cols)
        ranks_max = add_or_update_row(ranks_max, new_row_max, check_cols, criteria_cols_max)
        print('-- end calculating --', ff)

    # 2. save results to .csv
    if not os.path.isdir(csvdir):
        os.makedirs(csvdir)
    ranks.to_csv(save_path, index=False)
    ranks_max.to_csv(save_path_max, index=False)
    print('-- saving --', csvdir)


if __name__ == "__main__":
    method = 'CardioMM'
    # SENSE
    # CardioMM

    RootDir = '/mnt/nas/nas3/openData/MMCMR_427K/' \
              'Results_h5_FullSamplev2_Trained/' \
              f'{method}/' \
              'SegNII/'
    modality = 'Cine'
    evaluate_set = 'TestSet'
    task = 'TaskAll'
    undersample = 'Uniform8'
    # Uniform8, ktGaussian16, ktRadial24

    EXCLUDED_KEYWORDS = ['Center010', 'Center007', 'Center012', '055T', '50T']  # Exclude specific centers (pediatric) or scanners (low/ultra high-field)

    modalities = {
        'Cine': 'cine_sax*_label_ED.nii.gz',
    }
    file_dict = {m: [] for m in modalities}

    for modal, pattern in modalities.items():
        if modality == modal:
            file_dict[modal] = sorted([
                file for file in glob.glob(os.path.join(RootDir, f'**/{pattern}'), recursive=True)
                if all(x in file for x in [task, modal, evaluate_set, undersample])
                and not any(excluded in file for excluded in EXCLUDED_KEYWORDS)
            ])
    f = sum(file_dict.values(), [])
    print('##############')
    for modal, files in file_dict.items():
        print(f'{modal} files: {len(files)}')
    print(f'Total files: {len(f)}')
    print('##############')

    process_case(f, RootDir, method, evaluate_set, medcon='')
