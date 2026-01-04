"""
Extract ED ES images and segmented labels from recon .nii.gz - pytorch - CMRxReconAll
Created on 2025/08/25
@author: Zi Wang
Email: Zi Wang (zi.wang@imperial.ac.uk)
If you want to use this code, please cite our relevant papers in the GitHub page.
"""

import os
import nibabel as nib
import numpy as np
import subprocess
import glob
from concurrent.futures import ProcessPoolExecutor

max_workers = 5  # adjust to number of parallel processes you want


def get_EDES_frames(casenii_name, segcase_name):
    img = nib.load(casenii_name)
    img_data = img.get_fdata()  # nx, ny, nz, nt
    seg = nib.load(segcase_name)
    seg_data = seg.get_fdata()  # nx, ny, nz, nt

    pixdim = img.header['pixdim'][1:4]
    volume_per_pix = pixdim[0] * pixdim[1] * pixdim[2] * 1e-3

    vol_t = np.sum(seg_data == 1, axis=(0, 1, 2)) * volume_per_pix

    ED_index = np.argmax(vol_t)  # index of the frame with maximum volume for ED
    ES_index = np.argmin(vol_t)  # index of the frame with minimum volume for ES

    return ED_index, ES_index, seg_data, img_data, img.affine, img.header.get_zooms()[:3]


# ─── Per-case processing ──────────────────────────────────────────────────────
def process_case(ff):
    try:
        # --- prepare dirs ---
        casenii = ff  # Image4D.nii.gz
        segcase = ff.replace('ImageNII', 'SegNII').replace('.nii.gz', '_label.nii.gz')  # Seg4D.nii.gz

        # --- load each 4D .nii.gz and save ED and ES frames ---

        if 'ED' in casenii or 'ES' in casenii:
            return  # skip if already processed ED or ES
        elif not os.path.exists(segcase):
            print(f"Segmentation file {segcase} does not exist, skipping {ff}")
            return
        else:
            # skip if already done
            if os.path.exists(segcase.replace('.nii.gz', f'_ED.nii.gz')) and os.path.exists(segcase.replace('.nii.gz', f'_ES.nii.gz')):
                print(f"{ff} already done, skipping")
            else:
                ED_index, ES_index, seg, img, affine, zooms = get_EDES_frames(casenii, segcase)
                print(f"[{ff}] ED index: {ED_index}, ES index: {ES_index}")
                
                # Save ED frames
                imgnii_ED = nib.Nifti1Image(img[:, :, :, ED_index], affine)
                imgnii_ED.header.set_zooms(zooms)
                nib.save(imgnii_ED, casenii.replace('.nii.gz', f'_ED.nii.gz'))

                segnii_ED = nib.Nifti1Image(seg[:, :, :, ED_index], affine)
                segnii_ED.header.set_zooms(zooms)
                nib.save(segnii_ED, segcase.replace('.nii.gz', f'_ED.nii.gz'))

                # Save ES frames
                imgnii_ES = nib.Nifti1Image(img[:, :, :, ES_index], affine)
                imgnii_ES.header.set_zooms(zooms)
                nib.save(imgnii_ES, casenii.replace('.nii.gz', f'_ES.nii.gz'))

                segnii_ES = nib.Nifti1Image(seg[:, :, :, ES_index], affine)
                segnii_ES.header.set_zooms(zooms)
                nib.save(segnii_ES, segcase.replace('.nii.gz', f'_ES.nii.gz'))
        print(f"[{ff}] done")
    except Exception as e:
        print(f"[{ff}] ERROR: {e}")


# ─── Main: parallel dispatch ───────────────────────────────────────────────────
if __name__ == "__main__":
    method = 'CardioMM'
    # SENSE
    # CardioMM

    RootDir = '/mnt/nas/nas3/openData/MMCMR_427K/' \
              'Results_h5_FullSamplev2_Trained/' \
              f'{method}/' \
              'ImageNII/'
    modality = 'Cine'
    evaluate_set = 'TestSet'
    task = 'TaskAll'
    undersample = 'Uniform8'
    # Uniform8, ktGaussian16, ktRadial24

    EXCLUDED_KEYWORDS = ['Center010', 'Center007', 'Center012', '055T', '50T']  # Exclude specific centers (pediatric) or scanners (low/ultra high-field)

    modalities = {
        'Cine': 'cine*.nii.gz',
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

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        pool.map(process_case, f)
