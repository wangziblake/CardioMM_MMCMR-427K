"""
Extract ED/ES images and segmentation labels from reconstructed 4D NIfTI files.

This script scans reconstructed CMRxReconAll Cine image volumes, finds the
matching segmentation volumes, identifies end-diastolic (ED) and end-systolic
(ES) frames from the LV cavity volume curve, and writes the selected 3D image
and label frames as separate NIfTI files. The output frames preserve the
original affine transform and voxel spacing.

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
    """Load a 4D reconstructed image/segmentation pair and identify ED/ES.

    ED is defined as the time frame with the largest LV cavity volume, and ES
    is defined as the time frame with the smallest LV cavity volume. The LV
    cavity is expected to use label value 1 in the segmentation.

    Args:
        casenii_name (str): Path to the 4D reconstructed image NIfTI file.
        segcase_name (str): Path to the matching 4D segmentation NIfTI file.

    Returns:
        tuple: ``(ED_index, ES_index, seg_data, img_data, affine, zooms)``,
        where indices are integer time-frame positions, ``seg_data`` and
        ``img_data`` are 4D arrays, ``affine`` is the image affine, and
        ``zooms`` contains the spatial voxel spacing.
    """
    img = nib.load(casenii_name)
    img_data = img.get_fdata()  # nx, ny, nz, nt
    seg = nib.load(segcase_name)
    seg_data = seg.get_fdata()  # nx, ny, nz, nt

    pixdim = img.header['pixdim'][1:4]
    # Convert voxel volume from mm^3 to mL: 1 mL = 1000 mm^3.
    volume_per_pix = pixdim[0] * pixdim[1] * pixdim[2] * 1e-3

    # Build the LV cavity volume curve across time using label value 1.
    vol_t = np.sum(seg_data == 1, axis=(0, 1, 2)) * volume_per_pix

    ED_index = np.argmax(vol_t)  # index of the frame with maximum volume for ED
    ES_index = np.argmin(vol_t)  # index of the frame with minimum volume for ES

    return ED_index, ES_index, seg_data, img_data, img.affine, img.header.get_zooms()[:3]


# ─── Per-case processing ──────────────────────────────────────────────────────
def process_case(ff):
    """Extract and save ED/ES image and segmentation frames for one recon case.

    The input path is expected to point to a reconstructed image file under an
    ``ImageNII`` directory. The matching segmentation path is inferred by
    replacing ``ImageNII`` with ``SegNII`` and adding the ``_label`` suffix
    before the file extension.

    Args:
        ff (str): Path to one reconstructed 4D image NIfTI file.

    Returns:
        None: ED/ES image and segmentation files are written to disk.
    """
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
            # Skip cases whose segmentation ED/ES outputs already exist.
            if os.path.exists(segcase.replace('.nii.gz', f'_ED.nii.gz')) and os.path.exists(segcase.replace('.nii.gz', f'_ES.nii.gz')):
                print(f"{ff} already done, skipping")
            else:
                ED_index, ES_index, seg, img, affine, zooms = get_EDES_frames(casenii, segcase)
                print(f"[{ff}] ED index: {ED_index}, ES index: {ES_index}")
                
                # Save ED image and segmentation frames as 3D NIfTI volumes.
                imgnii_ED = nib.Nifti1Image(img[:, :, :, ED_index], affine)
                imgnii_ED.header.set_zooms(zooms)
                nib.save(imgnii_ED, casenii.replace('.nii.gz', f'_ED.nii.gz'))

                segnii_ED = nib.Nifti1Image(seg[:, :, :, ED_index], affine)
                segnii_ED.header.set_zooms(zooms)
                nib.save(segnii_ED, segcase.replace('.nii.gz', f'_ED.nii.gz'))

                # Save ES image and segmentation frames as 3D NIfTI volumes.
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
    # Reconstruction method whose results should be processed.
    method = 'CardioMM'
    # SENSE
    # CardioMM

    # Reconstructed image root. Segmentation and ED/ES outputs are derived from
    # the same folder structure by replacing ImageNII with SegNII where needed.
    RootDir = '/mnt/nas/nas3/openData/MMCMR_427K/' \
              'Results_h5_FullSamplev2_Trained/' \
              f'{method}/' \
              'ImageNII/'
    modality = 'Cine'
    evaluate_set = 'TestSet'
    task = 'TaskAll'
    # Undersampling pattern/acceleration subset to evaluate.
    undersample = 'Uniform8'
    # Uniform8, ktGaussian16, ktRadial24

    EXCLUDED_KEYWORDS = ['Center010', 'Center007', 'Center012', '055T', '50T']  # Exclude specific centers (pediatric) or scanners (low/ultra high-field)

    # Map each supported modality to the reconstructed image filename pattern
    # used for recursive discovery. Only the selected modality is populated.
    modalities = {
        'Cine': 'cine*.nii.gz',
    }
    file_dict = {m: [] for m in modalities}

    for modal, pattern in modalities.items():
        if modality == modal:
            # Keep only reconstructed NIfTI files matching the requested task,
            # modality, split, and undersampling setting.
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

    # Extract ED/ES frames for independent files in parallel.
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        pool.map(process_case, f)
