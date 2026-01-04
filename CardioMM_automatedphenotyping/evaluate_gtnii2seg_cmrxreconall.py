"""
Segment reconstructed gt sos .nii.gz images for subsequent analysis using nnunetv2 - pytorch - CMRxReconAll
Created on 2025/07/10
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
import shutil

# ─── Configuration ─────────────────────────────────────────────────────────────
WORKSPACE_DIR = '../nnUNet_related'  # TODO: need to check when using in different servers

os.environ['nnUNet_raw'] = os.path.join(WORKSPACE_DIR, 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = os.path.join(WORKSPACE_DIR, 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = os.path.join(WORKSPACE_DIR, 'nnUNet_results')

GPU_num = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_num
device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'

view_names = ['sax', '2ch', '3ch', '4ch']
model_ds = ['100', '101', '102', '103']  # matches view_names

max_workers = 5  # adjust to number of parallel processes you want


# ─── Per-case processing ──────────────────────────────────────────────────────
def process_case(fdir):
    try:
        # --- prepare dirs ---
        caseinputdir = fdir.replace('ImageNII', 'Inputs')  # Image2Dtemp for nnunet
        outputdir = fdir.replace('ImageNII', 'Outputs')  # Seg2Dtemp from nnunet
        caseniidir = fdir  # Image4D
        segcasedir = fdir.replace('ImageNII', 'SegNII')  # Seg4D

        os.makedirs(caseinputdir, exist_ok=True)
        os.makedirs(segcasedir, exist_ok=True)

        # --- split each 4D .nii.gz into slice and time ---
        files = sorted(f for f in os.listdir(caseniidir) if f.endswith('.nii.gz'))
        for view, ds in zip(view_names, model_ds):
            view_in = os.path.join(caseinputdir, view)
            os.makedirs(view_in, exist_ok=True)

            matched = [f for f in files if view in f]
            print(f"Processing {fdir} for view {view}, found {len(matched)} files")
            for fn in matched:
                img = nib.load(os.path.join(caseniidir, fn))
                data = img.get_fdata()  # nx, ny, nz, nt
                affine = img.affine
                zooms = img.header.get_zooms()[:3]

                if data.ndim != 4:
                    raise ValueError(f"{fn} not 4D (shape {data.shape})")

                prefix = fn.replace('.nii.gz', '')
                if os.path.exists(os.path.join(view_in, f"{prefix}__z0__t0_0000.nii.gz")):
                    print(f"{fdir} already done, skipping")
                else:
                    for z in range(data.shape[-2]):
                        for t in range(data.shape[-1]):
                            vol = data[:, :, z, t]
                            vol = np.expand_dims(vol, axis=-1)  # nx, ny, 1
                            nii = nib.Nifti1Image(vol, affine)
                            nii.header.set_zooms(zooms)
                            nib.save(nii, os.path.join(view_in, f"{prefix}__z{z}__t{t}_0000.nii.gz"))

            # --- run prediction if needed ---
            view_out = os.path.join(outputdir, view)
            os.makedirs(view_out, exist_ok=True)
            # skip if already done (allowing for 3 auxiliary files)
            if len(os.listdir(view_in)) + 3 == len(os.listdir(view_out)):
                print(f"{fdir} already done, skipping")
            else:
                print(f"{fdir} running prediction")
                subprocess.run([
                    "nnUNetv2_predict",
                    "-i", view_in,
                    "-o", view_out,
                    "-d", ds,
                    "-device", device,
                    "-c", "2d",
                    "--disable_progress_bar"
                ], check=True)

        # --- collect & reassemble all labels ---
        for view in view_names:
            view_out = os.path.join(outputdir, view)

            preds = sorted(f for f in os.listdir(view_out) if f.endswith('.nii.gz') and view in f)
            # if f={prefix}__z{z}__t{t}.nii.gz, get {prefix}
            prefixes = sorted({f.split('__')[0] for f in preds})

            for prefix in prefixes:
                if os.path.exists(os.path.join(segcasedir, prefix + '_label.nii.gz')):
                    print(f"{fdir} already done for {prefix}, skipping")
                else:
                    parts = [f for f in preds if f.startswith(prefix+'__')]
                    parts.sort(key=lambda x: (
                        int(x.split('__')[1][1:]),  # get z from __z{z}
                        int(x.split('__')[2].split('.')[0][1:])  # get t from __t{t}
                    ))  # sort by z# and t#

                    z_list = sorted(list(set(int(f.split('__')[1][1:]) for f in parts)))
                    t_list = sorted(list(set(int(f.split('__')[2].split('.')[0][1:]) for f in parts)))

                    sample_img = nib.load(os.path.join(view_out, parts[0]))  # load first image to get shape and dtype
                    data_shape = sample_img.shape  # 3D shape

                    label4d = np.zeros((data_shape[0], data_shape[1], len(z_list), len(t_list)), dtype=sample_img.get_data_dtype())
                    for f in parts:
                        z = int(f.split('__')[1][1:])
                        t = int(f.split('__')[2].split('.')[0][1:])
                        label = nib.load(os.path.join(view_out, f))
                        label4d[:, :, z, t] = np.squeeze(label.get_fdata(), axis=-1)  # nx, ny, 1 -> nx, ny

                    # Save .nii.gz
                    img4d = nib.load(os.path.join(caseniidir, prefix + '.nii.gz'))
                    seg4d = nib.Nifti1Image(label4d, img4d.affine, img4d.header)
                    # if the label4d.shape and img4d.get_fdata().shape are not the same, raise an error
                    if label4d.shape == img4d.get_fdata().shape:
                        nib.save(seg4d, os.path.join(segcasedir, prefix + '_label.nii.gz'))
                    else:
                        raise ValueError(f"{fdir}, the shape of seg4d and img4d are not the same")
        print(f"[{fdir}] done")
    except Exception as e:
        print(f"[{fdir}] ERROR: {e}")


# ─── Main: parallel dispatch ───────────────────────────────────────────────────
if __name__ == "__main__":
    RootDir = '/mnt/nas/nas3/openData/MMCMR_427K/AllData/ImageNII/'
    # '/mnt/nas/nas3/openData/MMCMR_427K/AllData/ImageNII/'
    modality = 'Cine'
    evaluate_set = 'TestSet'

    EXCLUDED_KEYWORDS = ['Center010', 'Center007', 'Center012', '055T', '50T']  # Exclude specific centers (pediatric) or scanners (low/ultra high-field)

    modalities = {
        'Cine': 'cine*.nii.gz',
    }
    file_dict = {m: [] for m in modalities}

    for modal, pattern in modalities.items():
        if modality == modal:
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
    fdir = sorted(set(os.path.dirname(p) for p in f))

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        pool.map(process_case, fdir)

    # Clean up temporary directories
    print("Cleaning up temporary directories...")
    Rootinputdir = RootDir.replace('ImageNII', 'Inputs')  # Root Image2Dtemp for nnunet
    Rootoutputdir = RootDir.replace('ImageNII', 'Outputs')  # Root Seg2Dtemp from nnunet
    if os.path.exists(Rootinputdir):
        shutil.rmtree(Rootinputdir)
        print("Temp input Folder deleted successfully.")
    if os.path.exists(Rootoutputdir):
        shutil.rmtree(Rootoutputdir)
        print("Temp output Folder deleted successfully.")
    else:
        print("Folder does not exist.")
