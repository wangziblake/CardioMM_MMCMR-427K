"""
Calculate the ventricular volume from recon sax .nii.gz - pytorch - CMRxReconAll
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
from itertools import takewhile


def split_letters_digits(s):
    letters = ''.join(takewhile(str.isalpha, s))
    digits = s[len(letters):]
    return letters, digits


def replace_mask_to_find_datatype_undersample_af(mask_filename):
    if "_mask_" in mask_filename:
        base, ext = mask_filename.rsplit(".", 1)  # 'cine_sax_mask_Uniform8_label.nii', '.gz'
        datatype = base.rsplit("_mask_", 1)[0]  # 'cine_sax'
        masktype = base.rsplit("_mask_", 1)[1].replace('_label.nii', '')  # 'Uniform8_label.nii' -> 'Uniform8'
        mask, af = split_letters_digits(masktype)  # 'Uniform', '8'
    else:
        raise NotImplementedError("The filename does not contain '_mask_'")  # Raise an error if "_mask_" is missing
    return datatype, mask, af


def extract_attrs(save_path, medcon=None):
    # save_path: "../Cine/TestSet/TaskAll/Center015/Siemens_30T_Vida/P031/cine_sax_mask_Uniform8_label.nii.gz"
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
    save_path = os.path.join(csvdir, f'CliCal_{evaluate_set}_VentricularVolume_{method}.csv')
    # 0. load previous saved .csv file, if possible
    if os.path.exists(save_path):
        print(f'-- find existing CSV: {save_path}, loading --')
        ranks = pd.read_csv(save_path)
    else:
        # placeholder for ranks
        print(f'-- no existing CSV, starting fresh --')
        ranks = pd.DataFrame(
            columns=['Method', 'Modality', 'Task', 'Center', 'Vendor', 'Field', 'Pfolder', 'Datatype',
                     'Mask', 'AF', 'Medcon',
                     'LVEDV (mL)', 'LVESV (mL)', 'LVSV (mL)', 'LVEF (%)', 'LVCO (L/min)', 'LVM (g)',
                     'RVEDV (mL)', 'RVESV (mL)', 'RVSV (mL)', 'RVEF (%)'])

    # columns to check for duplicates (excluding the criteria columns)
    check_cols = ['Method', 'Modality', 'Task', 'Center', 'Vendor', 'Field', 'Pfolder', 'Datatype', 'Mask',
                  'AF', 'Medcon']
    # criteria columns to update if different
    criteria_cols = ['LVEDV (mL)', 'LVESV (mL)', 'LVSV (mL)', 'LVEF (%)', 'LVCO (L/min)', 'LVM (g)',
                     'RVEDV (mL)', 'RVESV (mL)', 'RVSV (mL)', 'RVEF (%)']

    # 1. evaluate segmentation images
    for ff in tqdm(f, desc='files'):
        print('-- processing --', ff)
        # ff example: /{input_dir}/Cine/TestSet/TaskAll/Center015/Siemens_30T_Vida/P301/cine_sax_mask_Uniform8_label.nii.gz
        modality, center, vendor, field_strength, pfolder, datatype, mask, af, medcon = extract_attrs(ff, medcon)  # get the attributes from the filename

        # load data for evaluation
        seg4D_name = ff
        seg_ED_name = ff.replace('_label.nii.gz', '_label_ED.nii.gz')
        seg_ES_name = ff.replace('_label.nii.gz', '_label_ES.nii.gz')

        # data shape
        seg4D = nib.load(seg4D_name)
        pixdim = seg4D.header['pixdim'][1:4]
        volume_per_pix = pixdim[0] * pixdim[1] * pixdim[2] * 1e-3
        density = 1.05

        # heart rate
        if 'UIH' in ff:
            duration_per_cycle = seg4D.header['dim'][4] * (seg4D.header['pixdim'][4] * 10)  # TODO: some mistakes in temporal resolution of UIH data, so use *10 here
            heart_rate = 60.0 / duration_per_cycle
        else:
            duration_per_cycle = seg4D.header['dim'][4] * seg4D.header['pixdim'][4]
            heart_rate = 60.0 / duration_per_cycle

        # segmentation - ED ES
        seg_ED_data = nib.load(seg_ED_name).get_fdata()
        seg_ES_data = nib.load(seg_ES_name).get_fdata()

        print('-- start calculating --', ff)
        val = {}
        # Clinical measures - ED
        val['LVEDV'] = np.sum(seg_ED_data == 1) * volume_per_pix
        val['LVEDM'] = np.sum(seg_ED_data == 2) * volume_per_pix * density
        val['RVEDV'] = np.sum(seg_ED_data == 3) * volume_per_pix

        # Clinical measures - ES
        val['LVESV'] = np.sum(seg_ES_data == 1) * volume_per_pix
        val['LVESM'] = np.sum(seg_ES_data == 2) * volume_per_pix * density
        val['RVESV'] = np.sum(seg_ES_data == 3) * volume_per_pix

        val['LVSV'] = val['LVEDV'] - val['LVESV']
        val['LVCO'] = val['LVSV'] * heart_rate * 1e-3
        val['LVEF'] = val['LVSV'] / val['LVEDV'] * 100

        val['RVSV'] = val['RVEDV'] - val['RVESV']
        val['RVCO'] = val['RVSV'] * heart_rate * 1e-3
        val['RVEF'] = val['RVSV'] / val['RVEDV'] * 100

        # save the evaluation results to the pandas frame
        new_row = {'Method': method, 'Modality': modality, 'Task': 'TaskAll', 'Center': center, 'Vendor': vendor,
                   'Field': field_strength,
                   'Pfolder': pfolder, 'Datatype': datatype, 'Mask': mask, 'AF': af, 'Medcon': medcon,
                   'LVEDV (mL)': val['LVEDV'], 'LVESV (mL)': val['LVESV'], 'LVSV (mL)': val['LVSV'],
                   'LVEF (%)': val['LVEF'], 'LVCO (L/min)': val['LVCO'], 'LVM (g)': val['LVEDM'],
                   'RVEDV (mL)': val['RVEDV'], 'RVESV (mL)': val['RVESV'], 'RVSV (mL)': val['RVSV'], 'RVEF (%)': val['RVEF']}

        # Call the function to either add the row or update the metrics if only they are different
        ranks = add_or_update_row(ranks, new_row, check_cols, criteria_cols)
        print('-- end calculating --', ff)

    # 2. save results to .csv
    if not os.path.isdir(csvdir):
        os.makedirs(csvdir)
    ranks.to_csv(save_path, index=False)
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
        'Cine': 'cine_sax*_label.nii.gz',
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
