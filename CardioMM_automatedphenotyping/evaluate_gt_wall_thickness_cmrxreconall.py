"""
Calculate myocardial wall thickness from ground-truth SAX segmentations.

This script scans pre-extracted ED short-axis segmentation files, runs
short-axis segmentation quality control, calls the shared wall-thickness
estimator, and aggregates mean/max AHA-segment wall thickness values into CSV
tables. The expected segmentation label convention is LV cavity = 1,
myocardium = 2, and RV cavity = 3.

Created on 2025/08/17
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


def replace_mask_to_find_datatype_undersample_af(mask_filename):
    """Infer datatype, mask type, and acceleration factor from a label filename.

    Ground-truth SOS segmentations are treated as fully sampled data, so the
    mask and acceleration factor are fixed to ``"Full"`` and ``"1"``.

    Args:
        mask_filename (str): Segmentation filename such as
            ``"cine_sax_label_ED.nii.gz"``.

    Returns:
        tuple: ``(datatype, mask, af)`` parsed from the filename and GT setting.
    """
    base, ext = mask_filename.rsplit(".", 1)  # 'cine_sax_label_ED.nii', '.gz'
    datatype = base.rsplit("_label", 1)[0]  # 'cine_sax'
    mask, af = 'Full', '1'
    return datatype, mask, af


def extract_attrs(save_path, medcon=None):
    """Extract metadata fields encoded in a CMRxReconAll ED segmentation path.

    The path layout is assumed to follow
    ``.../<Modality>/<Set>/GTSOS_NII/<Center>/<Scanner>/<Patient>/<File>``.
    Scanner names are expected to contain vendor, field strength, and scanner
    model separated by underscores.

    Args:
        save_path (str): Full path to an ED segmentation NIfTI file.
        medcon (str, optional): Medical condition override. If None, a simple
            centre-based fallback is used.

    Returns:
        tuple: ``(modality, center, vendor, field_strength, pfolder, datatype,
        mask, af, medcon)`` for CSV reporting.
    """
    # save_path: "../Cine/TestSet/GTSOS_NII/Center015/Siemens_30T_Vida/P031/cine_sax_label_ED.nii.gz"
    path_parts = save_path.split(os.sep)
    modality = path_parts[-7]  # 'Cine'
    center = path_parts[-4]  # 'Center001'
    scanner = path_parts[-3]  # 'Siemens_30T_Vida'
    pfolder = path_parts[-2]  # 'P031'
    datatype, mask, af = replace_mask_to_find_datatype_undersample_af(path_parts[-1])  # 'cine_sax', 'Full', '1'
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
    """Add a new result row or update existing metric columns in-place.

    Rows are considered the same case/method entry when all columns in
    ``check_cols`` match. If such a row already exists, only the wall-thickness
    metric columns in ``criteria_cols`` are refreshed; otherwise the row is
    appended.

    Args:
        ranks (pd.DataFrame): Existing result table.
        new_row (dict): Candidate row containing metadata and metric values.
        check_cols (list): Metadata columns used to identify duplicate records.
        criteria_cols (list): Metric columns to update for an existing record.

    Returns:
        pd.DataFrame: Updated result table.
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


def process_case(f, RootDir, evaluate_set, medcon):
    """Compute mean and maximum wall thickness metrics for ED segmentations.

    For each ED SAX segmentation, this function validates the segmentation,
    calls ``evaluate_wall_thickness`` to create temporary per-case VTK/CSV
    outputs, reads the mean and max AHA-segment thickness CSVs, and updates the
    aggregate result tables under ``CalClinicalMeasure``.

    Args:
        f (list): ED segmentation file paths to process.
        RootDir (str): Root segmentation directory used to derive the CSV
            output directory.
        evaluate_set (str): Dataset split name included in the output filename.
        medcon (str): Medical condition value passed to ``extract_attrs``.

    Returns:
        None: Results are written to CSV files.
    """
    csvdir = RootDir.replace('SegNII', 'CalClinicalMeasure')
    save_path = os.path.join(csvdir, f'CliCal_{evaluate_set}_WallThickness_GTSOS.csv')
    save_path_max = os.path.join(csvdir, f'CliCal_{evaluate_set}_WallThicknessMax_GTSOS.csv')
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

    # 1. Evaluate segmentation images.
    for ff in tqdm(f, desc='files'):
        print('-- processing --', ff)
        # gtff example: /{gtinput_dir}/Cine/TestSet/GTSOS/Center015/Siemens_30T_Vida/P301/cine_sax_label_ED.nii.gz
        modality, center, vendor, field_strength, pfolder, datatype, mask, af, medcon = extract_attrs(ff, medcon)  # get the attributes from the filename

        # Load data for evaluation and skip missing or poor-quality ED masks.
        seg_ED_name = ff
        if not os.path.exists(seg_ED_name):
            continue
        if not sa_pass_quality_control(seg_ED_name):
            continue

        # Evaluate myocardial wall thickness. The helper writes per-case VTK,
        # mean-thickness CSV, and max-thickness CSV files using this stem.
        print('-- start calculating --', ff)
        temp_WT_ED_vtk = ff.replace('_label_ED.nii.gz', '_label_ED_WT')
        evaluate_wall_thickness(ff, temp_WT_ED_vtk)

        # Record mean wall thickness for AHA segments 1-16 plus the global row.
        if os.path.exists('{0}.csv'.format(temp_WT_ED_vtk)):
            df = pd.read_csv('{0}.csv'.format(temp_WT_ED_vtk), index_col=0)
            line = df['Thickness'].values
            val = {}
            for i in range(16):
                val[f'WT_AHA_{i+1}'] = line[i]
            val['WT_Global'] = line[16]

        # Record maximum wall thickness for AHA segments 1-16 plus the global
        # row.
        if os.path.exists('{0}_max.csv'.format(temp_WT_ED_vtk)):
            df = pd.read_csv('{0}_max.csv'.format(temp_WT_ED_vtk), index_col=0)
            line = df['Thickness_Max'].values
            val_max = {}
            for i in range(16):
                val_max[f'WT_Max_AHA_{i+1}'] = line[i]
            val_max['WT_Max_Global'] = line[16]

        # Save the evaluation results to the pandas frames.
        new_row = {'Method': 'GTSOS', 'Modality': modality, 'Task': 'TaskAll', 'Center': center, 'Vendor': vendor,
                   'Field': field_strength, 'Pfolder': pfolder, 'Datatype': datatype, 'Mask': mask, 'AF': af, 'Medcon': medcon,
                   'WT_AHA_1 (mm)': val['WT_AHA_1'], 'WT_AHA_2 (mm)': val['WT_AHA_2'], 'WT_AHA_3 (mm)': val['WT_AHA_3'],
                   'WT_AHA_4 (mm)': val['WT_AHA_4'], 'WT_AHA_5 (mm)': val['WT_AHA_5'], 'WT_AHA_6 (mm)': val['WT_AHA_6'],
                   'WT_AHA_7 (mm)': val['WT_AHA_7'], 'WT_AHA_8 (mm)': val['WT_AHA_8'], 'WT_AHA_9 (mm)': val['WT_AHA_9'],
                   'WT_AHA_10 (mm)': val['WT_AHA_10'], 'WT_AHA_11 (mm)': val['WT_AHA_11'], 'WT_AHA_12 (mm)': val['WT_AHA_12'],
                   'WT_AHA_13 (mm)': val['WT_AHA_13'], 'WT_AHA_14 (mm)': val['WT_AHA_14'], 'WT_AHA_15 (mm)': val['WT_AHA_15'], 'WT_AHA_16 (mm)': val['WT_AHA_16'],
                   'WT_Global (mm)': val['WT_Global']}
        new_row_max = {'Method': 'GTSOS', 'Modality': modality, 'Task': 'TaskAll', 'Center': center, 'Vendor': vendor,
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

    # 2. Save results to .csv.
    if not os.path.isdir(csvdir):
        os.makedirs(csvdir)
    ranks.to_csv(save_path, index=False)
    ranks_max.to_csv(save_path_max, index=False)
    print('-- saving --', csvdir)


if __name__ == "__main__":
    RootDir = '/mnt/nas/nas3/openData/MMCMR_427K/AllData/SegNII/'
    # '/mnt/nas/nas3/openData/MMCMR_427K/AllData/SegNII/'
    modality = 'Cine'
    evaluate_set = 'TestSet'

    EXCLUDED_KEYWORDS = ['Center010', 'Center007', 'Center012', '055T', '50T']  # Exclude specific centers (pediatric) or scanners (low/ultra high-field)

    # Map each supported modality to the ED segmentation filename pattern used
    # for recursive discovery. Only the selected modality is populated below.
    modalities = {
        'Cine': 'cine_sax*_label_ED.nii.gz',
    }
    file_dict = {m: [] for m in modalities}

    for modal, pattern in modalities.items():
        if modality == modal:
            # Keep only GT SOS NIfTI files from the requested evaluation split,
            # while excluding centres/scanners that should not enter this run.
            file_dict[modal] = sorted([
                file for file in glob.glob(os.path.join(RootDir, f'**/{pattern}'), recursive=True)
                if all(x in file for x in ['GTSOS_NII', modal, evaluate_set])
                and not any(excluded in file for excluded in EXCLUDED_KEYWORDS)
            ])
    f = sum(file_dict.values(), [])
    print('##############')
    for modal, files in file_dict.items():
        print(f'{modal} files: {len(files)}')
    print(f'Total files: {len(f)}')
    print('##############')

    # Process all discovered ED files and update the wall-thickness CSVs.
    process_case(f, RootDir, evaluate_set, medcon='')
