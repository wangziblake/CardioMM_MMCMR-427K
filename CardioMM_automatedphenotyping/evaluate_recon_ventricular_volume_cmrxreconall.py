"""
Calculate ventricular clinical measures from reconstructed SAX segmentations.

This script scans reconstructed short-axis CMRxReconAll segmentation files,
loads the pre-extracted ED/ES labels, computes LV/RV volumes and derived
ventricular metrics, and stores the results in a method-specific CSV file. The
expected segmentation label convention is LV cavity = 1, myocardium = 2, and
RV cavity = 3.

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
    """Split a compact mask string into alphabetic mask name and numeric AF.

    Args:
        s (str): String such as ``"Uniform8"`` or ``"ktGaussian16"``.

    Returns:
        tuple: ``(letters, digits)`` where the leading alphabetic portion is
        separated from the remaining suffix.
    """
    letters = ''.join(takewhile(str.isalpha, s))
    digits = s[len(letters):]
    return letters, digits


def replace_mask_to_find_datatype_undersample_af(mask_filename):
    """Parse datatype, undersampling mask, and acceleration factor from filename.

    Reconstructed segmentation filenames are expected to contain ``"_mask_"``,
    for example ``"cine_sax_mask_Uniform8_label.nii.gz"``. The datatype is the
    prefix before ``"_mask_"`` and the mask/AF are split from the mask token.

    Args:
        mask_filename (str): Reconstructed segmentation filename.

    Returns:
        tuple: ``(datatype, mask, af)`` parsed from the filename.

    Raises:
        NotImplementedError: If the filename does not contain ``"_mask_"``.
    """
    if "_mask_" in mask_filename:
        base, ext = mask_filename.rsplit(".", 1)  # 'cine_sax_mask_Uniform8_label.nii', '.gz'
        datatype = base.rsplit("_mask_", 1)[0]  # 'cine_sax'
        masktype = base.rsplit("_mask_", 1)[1].replace('_label.nii', '')  # 'Uniform8_label.nii' -> 'Uniform8'
        mask, af = split_letters_digits(masktype)  # 'Uniform', '8'
    else:
        raise NotImplementedError("The filename does not contain '_mask_'")  # Raise an error if "_mask_" is missing
    return datatype, mask, af


def extract_attrs(save_path, medcon=None):
    """Extract metadata fields encoded in a reconstructed segmentation path.

    The path layout is assumed to follow
    ``.../<Modality>/<Set>/<Task>/<Center>/<Scanner>/<Patient>/<File>``.
    Scanner names are expected to contain vendor, field strength, and scanner
    model separated by underscores.

    Args:
        save_path (str): Full path to a reconstructed segmentation NIfTI file.
        medcon (str, optional): Medical condition override. If None, a simple
            centre-based fallback is used.

    Returns:
        tuple: ``(modality, center, vendor, field_strength, pfolder, datatype,
        mask, af, medcon)`` for CSV reporting.
    """
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
    """Add a new result row or update existing metric columns in-place.

    Rows are considered the same case/method entry when all columns in
    ``check_cols`` match. If such a row already exists, only the clinical metric
    columns in ``criteria_cols`` are refreshed; otherwise the row is appended.

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


def process_case(f, RootDir, method, evaluate_set, medcon):
    """Compute ventricular metrics for reconstructed segmentation files.

    For each 4D SAX segmentation, the function loads the matching ED and ES
    segmentation files, calculates voxel-volume-based LV/RV metrics, and writes
    or updates a method-specific CSV file under ``CalClinicalMeasure``.

    Args:
        f (list): Reconstructed segmentation file paths to process.
        RootDir (str): Root segmentation directory used to derive the CSV
            output directory.
        method (str): Reconstruction method name written to the result table.
        evaluate_set (str): Dataset split name included in the output filename.
        medcon (str): Medical condition value passed to ``extract_attrs``.

    Returns:
        None: Results are written to a CSV file.
    """
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

    # 1. Evaluate segmentation images.
    for ff in tqdm(f, desc='files'):
        print('-- processing --', ff)
        # ff example: /{input_dir}/Cine/TestSet/TaskAll/Center015/Siemens_30T_Vida/P301/cine_sax_mask_Uniform8_label.nii.gz
        modality, center, vendor, field_strength, pfolder, datatype, mask, af, medcon = extract_attrs(ff, medcon)  # get the attributes from the filename

        # Load the 4D segmentation and the pre-extracted ED/ES segmentation
        # volumes generated by the ED/ES extraction step.
        seg4D_name = ff
        seg_ED_name = ff.replace('_label.nii.gz', '_label_ED.nii.gz')
        seg_ES_name = ff.replace('_label.nii.gz', '_label_ES.nii.gz')

        # Voxel volume is converted from mm^3 to mL. Myocardial mass uses the
        # conventional myocardial density of 1.05 g/mL.
        seg4D = nib.load(seg4D_name)
        pixdim = seg4D.header['pixdim'][1:4]
        volume_per_pix = pixdim[0] * pixdim[1] * pixdim[2] * 1e-3
        density = 1.05

        # Heart rate is derived from the temporal spacing and number of frames.
        # UIH data uses a correction factor noted in the original pipeline.
        if 'UIH' in ff:
            duration_per_cycle = seg4D.header['dim'][4] * (seg4D.header['pixdim'][4] * 10)  # TODO: some mistakes in temporal resolution of UIH data, so use *10 here
            heart_rate = 60.0 / duration_per_cycle
        else:
            duration_per_cycle = seg4D.header['dim'][4] * seg4D.header['pixdim'][4]
            heart_rate = 60.0 / duration_per_cycle

        # Load the pre-extracted ED and ES segmentation volumes.
        seg_ED_data = nib.load(seg_ED_name).get_fdata()
        seg_ES_data = nib.load(seg_ES_name).get_fdata()

        print('-- start calculating --', ff)
        val = {}
        # Clinical measures at ED: LV cavity volume, LV myocardial mass, and RV
        # cavity volume.
        val['LVEDV'] = np.sum(seg_ED_data == 1) * volume_per_pix
        val['LVEDM'] = np.sum(seg_ED_data == 2) * volume_per_pix * density
        val['RVEDV'] = np.sum(seg_ED_data == 3) * volume_per_pix

        # Clinical measures at ES using the same label convention.
        val['LVESV'] = np.sum(seg_ES_data == 1) * volume_per_pix
        val['LVESM'] = np.sum(seg_ES_data == 2) * volume_per_pix * density
        val['RVESV'] = np.sum(seg_ES_data == 3) * volume_per_pix

        # Derived LV measures: stroke volume, cardiac output, and ejection
        # fraction.
        val['LVSV'] = val['LVEDV'] - val['LVESV']
        val['LVCO'] = val['LVSV'] * heart_rate * 1e-3
        val['LVEF'] = val['LVSV'] / val['LVEDV'] * 100

        # Derived RV measures: stroke volume, cardiac output, and ejection
        # fraction. RVCO is calculated for completeness but is not saved below.
        val['RVSV'] = val['RVEDV'] - val['RVESV']
        val['RVCO'] = val['RVSV'] * heart_rate * 1e-3
        val['RVEF'] = val['RVSV'] / val['RVEDV'] * 100

        # Save the evaluation results to the pandas frame.
        new_row = {'Method': method, 'Modality': modality, 'Task': 'TaskAll', 'Center': center, 'Vendor': vendor,
                   'Field': field_strength,
                   'Pfolder': pfolder, 'Datatype': datatype, 'Mask': mask, 'AF': af, 'Medcon': medcon,
                   'LVEDV (mL)': val['LVEDV'], 'LVESV (mL)': val['LVESV'], 'LVSV (mL)': val['LVSV'],
                   'LVEF (%)': val['LVEF'], 'LVCO (L/min)': val['LVCO'], 'LVM (g)': val['LVEDM'],
                   'RVEDV (mL)': val['RVEDV'], 'RVESV (mL)': val['RVESV'], 'RVSV (mL)': val['RVSV'], 'RVEF (%)': val['RVEF']}

        # Call the function to either add the row or update the metrics if only they are different
        ranks = add_or_update_row(ranks, new_row, check_cols, criteria_cols)
        print('-- end calculating --', ff)

    # 2. Save results to .csv.
    if not os.path.isdir(csvdir):
        os.makedirs(csvdir)
    ranks.to_csv(save_path, index=False)
    print('-- saving --', csvdir)


if __name__ == "__main__":
    # Reconstruction method whose segmentation results should be evaluated.
    method = 'CardioMM'
    # SENSE
    # CardioMM

    # Reconstructed segmentation root. The clinical-measure output directory is
    # derived from this path by replacing SegNII with CalClinicalMeasure.
    RootDir = '/mnt/nas/nas3/openData/MMCMR_427K/' \
              'Results_h5_FullSamplev2_Trained/' \
              f'{method}/' \
              'SegNII/'
    modality = 'Cine'
    evaluate_set = 'TestSet'
    task = 'TaskAll'
    # Undersampling pattern/acceleration subset to evaluate.
    undersample = 'Uniform8'
    # Uniform8, ktGaussian16, ktRadial24

    EXCLUDED_KEYWORDS = ['Center010', 'Center007', 'Center012', '055T', '50T']  # Exclude specific centers (pediatric) or scanners (low/ultra high-field)

    # Map each supported modality to the reconstructed segmentation filename
    # pattern used for recursive discovery. Only the selected modality is used.
    modalities = {
        'Cine': 'cine_sax*_label.nii.gz',
    }
    file_dict = {m: [] for m in modalities}

    for modal, pattern in modalities.items():
        if modality == modal:
            # Keep only files matching the requested task, modality, split, and
            # undersampling setting.
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

    # Process all discovered segmentation files and update the result CSV.
    process_case(f, RootDir, method, evaluate_set, medcon='')
