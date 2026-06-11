"""
Testing the conventional SENSE with CSM estimation using ESPIRiT - pytorch - CMRxReconAll
- find mask first then match data
Created on 2025/08/29
@author: Zi Wang
Email: Zi Wang (zi.wang@imperial.ac.uk)
If you want to use this code, please cite our relevant papers in the GitHub page.
"""

import os
import sys
import pathlib
# Make repository-local modules importable when launching from the nested recon_cmr folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.absolute())))

import argparse
import torch
import scipy.io as sio
import glob
from os.path import join
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import sigpy as sp
import sigpy.mri as mr
import cupy as cp

import fastmri.data.transforms as T
from utils import load_kdata_compatible, load_maskdata


def np_double2complex(x):  # double [b, c, h, w, 2] --> complex [b, c, h, w]
    """
    Convert a real/imag-last NumPy array to a complex NumPy array.

    Args:
        x: Array shaped ``[batch, coils, height, width, 2]`` with real and
            imaginary values in the final dimension.

    Returns:
        Complex array shaped ``[batch, coils, height, width]``.
    """
    b, c, h, w, two = x.shape
    assert two == 2
    x_real, x_imag = x[..., 0], x[..., 1]
    x_complex = x_real + 1j * x_imag
    return x_complex


def np_complex2double(x):  # complex [b, c, h, w] --> double [b, c, h, w, 2]
    """
    Convert a complex NumPy array to real/imag-last representation.

    Args:
        x: Complex array shaped ``[batch, coils, height, width]``.

    Returns:
        Real-valued array shaped ``[batch, coils, height, width, 2]``.
    """
    x_real, x_imag = np.real(x), np.imag(x)
    x_double = np.stack((x_real, x_imag), axis=-1)
    return x_double


def get_frames_indices_stage(dataslice, num_slices_in_volume, num_t_in_volume=None):
    """
    Map a flattened frame index back to original time/slice coordinates.

    During reconstruction, time ``t`` and slice ``z`` are flattened into one
    leading axis. This helper recovers the original coordinates and returns the
    flattened indices to read. The current implementation returns only the
    current/central slice.

    Args:
        dataslice: Flattened ``t,z`` index.
        num_slices_in_volume: Number of z-slices per time point.
        num_t_in_volume: Number of time points in the volume.

    Returns:
        List of flattened indices to load from k-space and mask arrays.
    """
    ti = dataslice//num_slices_in_volume
    zi = dataslice - ti*num_slices_in_volume

    zi_idx_list = [zi]

    ti_idx_list = [ti % num_t_in_volume]
    output_list = []

    for zz in zi_idx_list:
        for tt in ti_idx_list:
            output_list.append(tt*num_slices_in_volume + zz)

    return output_list


def replace_mask_to_find_data(mask_filename):
    """
    Convert a mask filename into its matching full-sample data filename.

    Args:
        mask_filename: Path containing ``"_mask_"`` before the undersampling
            pattern suffix.

    Returns:
        Data filename with the ``"_mask_..."`` suffix removed.

    Raises:
        NotImplementedError: If the filename does not contain ``"_mask_"``.
    """
    if "_mask_" in mask_filename:
        base, ext = mask_filename.rsplit(".", 1)
        base = base.rsplit("_mask_", 1)[0]
        data_filename = f"{base}.{ext}"
    else:
        raise NotImplementedError("The filename does not contain '_mask_'")  # Raise an error if "_mask_" is missing
    return data_filename


class stage_dataset(torch.utils.data.Dataset):
    """
    Dataset for reconstructing one masked file with conventional SENSE.

    The dataset starts from a mask ``.mat`` filename, finds the corresponding
    full-sample k-space file, normalizes k-space and mask layouts to flattened
    ``[nt * nz, ...]`` arrays, and returns tensors consumed by the SENSE loop.
    """

    def __init__(self, fname, task):
        """
        Args:
            fname: Mask ``.mat`` filename used as the reconstruction driver.
            task: Task folder name used when mapping ``Mask_<task>`` to
                ``FullSample``.
        """
        self.fname = fname
        self.task = task
        # here, input kspace from .mat: 5D-[nt, nz, nc, ny, nx] or 4D-[nz, nc, ny, nx] or 3D-[nc, ny, nx]
        # Derive the matching fully sampled k-space filename from the mask filename.
        data_fname = replace_mask_to_find_data(fname).replace(f'Mask_{self.task}', 'FullSample')
        self.kspace = load_kdata_compatible(data_fname)
        if len(self.kspace.shape) != 5:
            # Expand 4D k-space to [1, nz, nc, ny, nx].
            self.kspace = np.expand_dims(self.kspace, axis=0)  # make sure its shape is [1, nz, nc, ny, nx]
            if len(self.kspace.shape) != 5:
                # Expand 3D k-space to [1, 1, nc, ny, nx].
                self.kspace = np.expand_dims(self.kspace, axis=0)  # make sure its shape is [1, nz, nc, ny, nx]
        self.num_t = self.kspace.shape[0]
        self.num_slices = self.kspace.shape[1]
        # Flatten time/slice and transpose spatial axes from [ny, nx] to [nx, ny].
        self.kspace = self.kspace.reshape(-1, self.kspace.shape[2], self.kspace.shape[3], self.kspace.shape[4]).transpose(0, 1, 3, 2)  # --> [nt*nz, nc, nx, ny]

        # here, input mask from .mat: 3D-[nt, ny, nx] or 2D-[ny, nx]
        self.mask = load_maskdata(fname)
        if len(self.mask.shape) == 3:
            # kt masks vary over time and are expanded to [nt, 1, 1, ny, nx].
            self.mask = np.expand_dims(np.expand_dims(self.mask, axis=1),axis=2)  # make sure its shape is [nt, 1, 1, ny, nx]
        elif len(self.mask.shape) == 2:
            # 2D masks are reused across all time points.
            self.mask = np.expand_dims(np.expand_dims(np.expand_dims(self.mask, axis=0), axis=1), axis=2)  # make sure its shape is [1, 1, 1, ny, nx]
            self.mask = np.tile(self.mask, reps=(self.num_t, 1, 1, 1, 1))  # make sure its shape is [nt, 1, 1, ny, nx]
        else:
            raise NotImplementedError("The mask shape should be 2D(k) or 3D(k-t).")
        # Tile masks across slices, flatten [nt, nz], then transpose to [nx, ny].
        self.mask = np.tile(self.mask, reps=(1, self.num_slices, 1, 1, 1))  # make sure its shape is [nt, nz, 1, ny, nx]
        self.mask = self.mask.reshape(-1, self.mask.shape[2], self.mask.shape[3], self.mask.shape[4]).transpose(0, 1, 3, 2)  # --> [nt*nz, 1, nx, ny]

        self.num_files = self.kspace.shape[0]
        self.maskfunc = None

    def __getitem__(self, dataslice):
        """
        Load one flattened frame and build SENSE inputs.

        Args:
            dataslice: Flattened time/slice index.

        Returns:
            Tuple ``(masked_kspace, mask, dataslice)``.
        """
        slice_idx_list = get_frames_indices_stage(dataslice, self.num_slices, self.num_t)
        _input = []
        for slc_i in slice_idx_list:
            _input.append(self.kspace[slc_i])
        # Concatenate selected slice(s) along the coil/channel axis.
        _input = np.concatenate(_input, axis=0) #.transpose(0,2,1)
        kspace_torch = T.to_tensor(_input)  # [nt*nz, nc, nx, ny, 2]
        masked_kspace = kspace_torch.to(torch.float32)

        _input_mask = []
        for slc_i in slice_idx_list:
            _input_mask.append(self.mask[slc_i])
        _input_mask = np.concatenate(_input_mask, axis=0) #.transpose(0,2,1)
        _input_mask = np.expand_dims(_input_mask, axis=-1)
        mask_torch = torch.from_numpy(_input_mask.astype(np.float32))  # [nt*nz, nc, nx, ny, 1]
        mask_torch = mask_torch.to(torch.bool)

        return masked_kspace * mask_torch, mask_torch, dataslice

    def __len__(self):
        """Return the number of flattened time/slice frames in this file."""
        return self.num_files


def predict(f, num_low_frequencies, bs1 = 1, center_crop=False, num_works = 2, input_dir='', output_dir='', task='TaskAll'):
    """
    Run ESPIRiT-calibrated SENSE reconstruction and save ``.mat`` outputs.

    Args:
        f: List of mask filenames to reconstruct.
        num_low_frequencies: ACS size used for ESPIRiT calibration.
        bs1: DataLoader batch size.
        center_crop: Unused compatibility flag retained in the signature.
        num_works: Number of DataLoader worker processes.
        input_dir: Input root used when deriving output paths.
        output_dir: Output root where ``ReconResult`` files are saved.
        task: Task name used in mask/data/output path replacement.
    """
    # 0. config
    # Retained string device setting; SigPy device below determines actual reconstruction backend.
    device = 'cuda:0'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(os.getcwd())

    # SigPy uses GPU 0 when CUDA is available, otherwise CPU.
    device = sp.Device(0) if torch.cuda.is_available() else sp.Device(-1)

    # 1. predict
    with torch.no_grad():
        for ff in tqdm(f, desc='files'):
            print('-- processing --', ff)
            # Save reconstructions by replacing mask task folders with result task folders.
            save_path = ff.replace(f'Mask_{task}', f'{task}').replace(input_dir, join(output_dir, 'ReconResult'))
            if os.path.isfile(save_path):  # check if the .mat file already exists
                continue
            elif not os.path.isdir(os.path.dirname(save_path)):
                # Create the nested output directory that mirrors the input structure.
                os.makedirs(os.path.dirname(save_path))

            dataset = stage_dataset(ff, task)
            dataloader = DataLoader(dataset, batch_size=bs1, shuffle=False, num_workers=num_works, pin_memory=True, drop_last=False)
            pred_stage = []

            for masked_kspace, mask, dataslice in tqdm(dataloader, desc='stage1'):

                bs = masked_kspace.shape[0]
                # num_low_frequencies = 20  # TODO: need to check under different undersampling scenarios, cannot be 0

                # SigPy reconstruction expects CuPy complex arrays.
                masked_kspace_cp = cp.asarray(np_double2complex(masked_kspace.cpu().numpy()))  # (bs, nc, nx, ny, 2) -> complex (bs, nc, nx, ny)
                mask_cp = cp.asarray(mask.cpu().numpy())  # (bs, 1, nx, ny, 1)

                for i in range(bs):
                    masked_kspace_complex = masked_kspace_cp[i]  # complex (nc, nx, ny)
                    mask_float = cp.squeeze(mask_cp[i]).astype(cp.float32)  # (nx, ny)

                    # ESPIRiT needs a sufficiently large ACS region for stable CSM estimation.
                    assert int(num_low_frequencies) > 15, "num_low_frequencies must be larger than 15, too small ACS cannot obtain the proper CSMs"
                    calib_width = min(int(num_low_frequencies), 24)  # Get ACS region size for calibration

                    CSM = mr.app.EspiritCalib(masked_kspace_complex, calib_width=calib_width, crop=0.00, device=device, show_pbar=False).run()  # input k should be [num_coils, n_ndim, …, n_1]

                    # Solve conventional SENSE reconstruction with the estimated CSMs.
                    recon = mr.app.SenseRecon(masked_kspace_complex, CSM, lamda=0.01, device=device, show_pbar=False).run()  # SENSE is better
                    # recon = mr.app.L1WaveletRecon(masked_kspace_complex, CSM, lamda=0.001, device=device, show_pbar=False).run()  # L1-wavelet SENSE
                    output = cp.abs(recon)

                    img_torch = torch.from_numpy(cp.asnumpy(output.astype(cp.float32)))[None, ...]  # [1, nx, ny]
                    pred_stage.append((dataslice[i], img_torch))
            # Restore flattened frame order before reshaping back to [nt, nz, nx, ny].
            pred_stage = torch.cat([out for _, out in sorted(pred_stage)], dim=0).cpu()

            pred_stage_final = pred_stage.cpu().numpy().transpose(0, 2, 1).reshape(dataset.num_t, dataset.num_slices, pred_stage.shape[2], pred_stage.shape[1])

            # MATLAB submission layout is [nx, ny, nz, nt].
            save_mat = pred_stage_final.transpose(3, 2, 1, 0)  # nx, ny, nz, nt
            sio.savemat(save_path, {'reconimage': save_mat})
            print('-- saving --', save_path)


if __name__ == '__main__':
    argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    parser.add_argument('--center_crop', action='store_true', default=False, help='Enable center cropping for validation leaderboard submission')
    parser.add_argument('--evaluate_set', type=str, default='TestSet', help='Choose the evaluation set')
    parser.add_argument('--task', type=str, default='TaskAll', help='Choose to inference on which type of task')
    parser.add_argument('--modality', type=str, default='All', help='Choose to inference on which type of data')
    parser.add_argument('--undersample', type=str, default='', help='Choose the undersampling pattern (Uniform8, ktGaussian16, ktRadial24)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the model.')
    parser.add_argument('--num_works', type=int, default=2, help='num of processors to load data.')
    parser.add_argument('--num_low_frequencies', type=int, default=20, help='num of acs size.')
    parser.add_argument('--exact_mask_filename', type=str, default=None, help='exact mask_filename to test')
    # exact_mask_filename example:
    # /SSDHome/home/Raw_data/MICCAIChallengeAll/ChallengeData/MultiCoil/Cine/TestSet/Mask_TaskAll/Center015/Siemens_30T_Vida/P301/cine_sax_mask_ktGaussian16.mat


    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    center_crop = args.center_crop
    evaluate_set = args.evaluate_set
    task = args.task
    modality = args.modality
    undersample = args.undersample
    bs1 = args.batch_size
    num_works = args.num_works
    num_low_frequencies = args.num_low_frequencies
    exact_mask_filename = args.exact_mask_filename

    print("Input data store in:", input_dir)
    print("Output data store in:", output_dir)

    if exact_mask_filename is not None:
        # Exact filename mode bypasses modality/task glob discovery.
        f = [exact_mask_filename]
        print('##############')
        print("Exact recon mask filename:", exact_mask_filename)
        print(f'Total files: {len(f)}')
        print('##############')

    elif exact_mask_filename is None:
        # get input file list
        # TODO: Need to be changed according to the TestSet !!!
        modalities = {
            'Cine': 'cine_*.mat',
            'Mapping': '*map*_*.mat',
            'Aorta': 'aorta_*.mat',
            'Tagging': 'tagging_*.mat',
            'Flow2d': 'flow2d_*.mat',
            'BlackBlood': 'blackblood_*.mat',
            'LGE': 'lge_*.mat',
            'Perfusion': 'perfusion_*.mat',
            'T1rho': 'T1rho_*.mat',
            'T1w': 'T1w_*.mat',
            'T2w': 'T2w_*.mat',
        }
        file_dict = {m: [] for m in modalities}

        for modal, pattern in modalities.items():
            if modality == modal or modality == 'All':
                # Keep only mask files matching modality, task, evaluation split, and undersampling pattern.
                file_dict[modal] = sorted([
                    file for file in glob.glob(join(input_dir, f'**/{pattern}'), recursive=True)
                    if all(x in file for x in ['Mask', modal, task, evaluate_set, undersample])
                ])
        f = sum(file_dict.values(), [])
        print('##############')
        for modal, files in file_dict.items():
            print(f'{modal} files: {len(files)}')
        print(f'Total files: {len(f)}')
        print('##############')

    # main function: reconstruct and save files
    predict(f, num_low_frequencies, bs1=bs1, center_crop=center_crop, num_works=num_works, input_dir=input_dir, output_dir=output_dir, task=task)
