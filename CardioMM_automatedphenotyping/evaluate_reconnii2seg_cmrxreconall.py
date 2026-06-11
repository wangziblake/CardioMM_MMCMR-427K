"""
Segment reconstructed SOS NIfTI images with nnUNetv2 for downstream analysis.

This script prepares reconstructed CMRxReconAll 4D Cine image volumes for
nnUNetv2 inference by splitting each selected volume into per-slice/per-time
2D NIfTI inputs. It then runs the view-specific nnUNet models and reassembles
the predicted 2D labels back into 4D segmentation volumes that match the
original reconstructed image geometry.

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
from functools import partial
import shutil

# ─── Configuration ─────────────────────────────────────────────────────────────
# nnUNet workspace configuration. These environment variables are read by the
# nnUNetv2 command-line tools.
WORKSPACE_DIR = '/media/ssd/wangzi/nnUNet_related'  # TODO: need to check when using in different servers

os.environ['nnUNet_raw'] = os.path.join(WORKSPACE_DIR, 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = os.path.join(WORKSPACE_DIR, 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = os.path.join(WORKSPACE_DIR, 'nnUNet_results')

GPU_num = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_num
device = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'

# Each cardiac view is paired with the corresponding nnUNet dataset ID.
view_names = ['sax', '2ch', '3ch', '4ch']
model_ds = ['100', '101', '102', '103']  # matches view_names

max_workers = 5  # adjust to number of parallel processes you want


# ─── Per-case processing ──────────────────────────────────────────────────────
def process_case(fdir, undersample):
    """Run nnUNet segmentation for reconstructed files in one case directory.

    The function creates temporary 2D nnUNet input folders, runs the
    view-specific nnUNetv2 predictors, and reassembles predicted 2D labels into
    4D segmentation files saved under the matching ``SegNII`` directory. Only
    files whose names contain the requested undersampling token are processed.

    Args:
        fdir (str): Directory containing source reconstructed 4D image NIfTI
            files for one case.
        undersample (str): Undersampling token to select, such as
            ``"Uniform8"``.

    Returns:
        None: Segmentation files are written to disk.
    """
    try:
        # --- prepare dirs ---
        caseinputdir = fdir.replace('ImageNII', 'Inputs')  # Image2Dtemp for nnunet
        outputdir = fdir.replace('ImageNII', 'Outputs')  # Seg2Dtemp from nnunet
        caseniidir = fdir  # Image4D
        segcasedir = fdir.replace('ImageNII', 'SegNII')  # Seg4D

        os.makedirs(caseinputdir, exist_ok=True)
        os.makedirs(segcasedir, exist_ok=True)

        # --- split each 4D .nii.gz into slice and time ---
        files = sorted(f for f in os.listdir(caseniidir) if f.endswith('.nii.gz') and undersample in f)
        for view, ds in zip(view_names, model_ds):
            view_in = os.path.join(caseinputdir, view)
            os.makedirs(view_in, exist_ok=True)

            # Select files for the current anatomical view before converting
            # each 4D volume into nnUNet's 2D input naming convention.
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
                    # nnUNet 2D inference expects one spatial slice per file.
                    # The channel suffix ``_0000`` denotes the first image
                    # channel.
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
            # Skip if prediction outputs already appear complete. The +3 allows
            # for nnUNet auxiliary files written into the output directory.
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

            preds = sorted(f for f in os.listdir(view_out) if f.endswith('.nii.gz') and view in f and undersample in f)
            # if f={prefix}__z{z}__t{t}.nii.gz, get {prefix}
            prefixes = sorted({f.split('__')[0] for f in preds})

            for prefix in prefixes:
                if os.path.exists(os.path.join(segcasedir, prefix + '_label.nii.gz')):
                    print(f"{fdir} already done for {prefix}, skipping")
                else:
                    # Group all 2D predictions belonging to the same original
                    # 4D image and sort them in deterministic z-then-t order.
                    parts = [f for f in preds if f.startswith(prefix+'__')]
                    parts.sort(key=lambda x: (
                        int(x.split('__')[1][1:]),  # get z from __z{z}
                        int(x.split('__')[2].split('.')[0][1:])  # get t from __t{t}
                    ))  # sort by z# and t#

                    z_list = sorted(list(set(int(f.split('__')[1][1:]) for f in parts)))
                    t_list = sorted(list(set(int(f.split('__')[2].split('.')[0][1:]) for f in parts)))

                    sample_img = nib.load(os.path.join(view_out, parts[0]))  # load first image to get shape and dtype
                    data_shape = sample_img.shape  # 3D shape

                    # Reassemble the predicted labels into the original
                    # ``(nx, ny, nz, nt)`` layout.
                    label4d = np.zeros((data_shape[0], data_shape[1], len(z_list), len(t_list)), dtype=sample_img.get_data_dtype())
                    for f in parts:
                        z = int(f.split('__')[1][1:])
                        t = int(f.split('__')[2].split('.')[0][1:])
                        label = nib.load(os.path.join(view_out, f))
                        label4d[:, :, z, t] = np.squeeze(label.get_fdata(), axis=-1)  # nx, ny, 1 -> nx, ny

                    # Save .nii.gz
                    img4d = nib.load(os.path.join(caseniidir, prefix + '.nii.gz'))
                    seg4d = nib.Nifti1Image(label4d, img4d.affine, img4d.header)
                    # Preserve the original image geometry only when the
                    # reconstructed label shape matches the source image shape.
                    if label4d.shape == img4d.get_fdata().shape:
                        nib.save(seg4d, os.path.join(segcasedir, prefix + '_label.nii.gz'))
                    else:
                        raise ValueError(f"{fdir}, the shape of seg4d and img4d are not the same")
        print(f"[{fdir}] done")
    except Exception as e:
        print(f"[{fdir}] ERROR: {e}")


# ─── Main: parallel dispatch ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Reconstruction method whose images should be segmented.
    method = 'CardioMM'
    # SENSE
    # CardioMM

    # Reconstructed image root. Temporary nnUNet input/output folders and final
    # SegNII outputs are derived from this folder structure.
    RootDir = '/mnt/nas/nas3/openData/MMCMR_427K/' \
              'Results_h5_FullSamplev2_Trained/' \
              f'{method}/' \
              'ImageNII/'
    modality = 'Cine'
    evaluate_set = 'TestSet'
    task = 'TaskAll'
    # Undersampling pattern/acceleration subset to segment.
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
    # Process each case directory once, even if it contains multiple view files.
    fdir = sorted(set(os.path.dirname(p) for p in f))

    # Bind the undersampling token for each parallel case-processing call.
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        pool.map(partial(process_case, undersample=undersample), fdir)

    # Clean up temporary 2D nnUNet input/output directories after all cases
    # have been reassembled into final 4D segmentations.
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
    
