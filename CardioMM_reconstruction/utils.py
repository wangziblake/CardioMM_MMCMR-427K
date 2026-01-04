"""
Created on 2025/08/29
@author: Zi Wang
Email: Zi Wang (zi.wang@imperial.ac.uk)
If you want to use this code, please cite our relevant papers in the GitHub page.

Some codes are modified based on https://arxiv.org/abs/2309.13839
"""
import h5py
import math
import torch
import numpy as np
import scipy.io
from typing import Tuple

############### metric function
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from fastmri.data import transforms


def double2complex(x):  # double [b, c, h, w, 2] --> complex [b, c, h, w]
    b, c, h, w, two = x.shape
    assert two == 2
    x_real, x_imag = x[..., 0], x[..., 1]
    x_complex = torch.complex(x_real, x_imag)
    return x_complex


def complex2double(x):  # complex [b, c, h, w] --> double [b, c, h, w, 2]
    x_real, x_imag = torch.real(x), torch.imag(x)
    x_double = torch.stack((x_real, x_imag), dim=-1)
    return x_double


def count_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params


def ifft2c(kdata_tensor, dim=(-2,-1), norm='ortho'):
    """
    ifft2c -  ifft2 from centered kspace data tensor
    """
    kdata_tensor_uncentered = torch.fft.ifftshift(kdata_tensor,dim=dim)
    image_uncentered = torch.fft.ifft2(kdata_tensor_uncentered,dim=dim, norm=norm)
    image = torch.fft.fftshift(image_uncentered,dim=dim)
    return image


def fft2c(image_tensor, dim=(-2,-1), norm='ortho'):
    """
    fft2c -  fft2 from image data tensor
    """
    image_tensor_uncentered = torch.fft.ifftshift(image_tensor,dim=dim)
    kdata_uncentered = torch.fft.fft2(image_tensor_uncentered,dim=dim, norm=norm)
    kdata = torch.fft.fftshift(kdata_uncentered,dim=dim)
    return kdata


def zf_recon(filename):
    '''
    load kdata and direct IFFT + RSS recon
    return kdata shape [t,z,c,y,x]
    return image shape [t,z,y,x]
    '''
    kdata = load_kdata(filename)
    kdata_tensor = torch.tensor(kdata).cuda()
    image_coil = ifft2c(kdata_tensor)
    image = (image_coil.abs()**2).sum(2)**0.5
    image_np = image.cpu().numpy()
    return kdata, image_np


def zf_recon_4D5D(filename):
    '''
    load kdata and direct IFFT + RSS recon
    return kdata shape [t,z,c,y,x] or [z,c,y,x]
    return image shape [t,z,y,x] or [z,y,x]
    '''
    kdata = load_kdata_compatible(filename)
    # print(kdata.dtype)
    # print(kdata.shape)
    kdata_tensor = torch.tensor(kdata).cuda()
    image_coil = ifft2c(kdata_tensor)
    image = (image_coil.abs()**2).sum(-3)**0.5
    image_np = image.cpu().numpy()
    return kdata, image_np


def zf_recon_4D5D_cpu(filename):
    '''
    load kdata and direct IFFT + RSS recon
    return kdata shape [t,z,c,y,x] or [z,c,y,x]
    return image shape [t,z,y,x] or [z,y,x]
    '''
    kdata = load_kdata_compatible(filename)
    # print(kdata.dtype)
    # print(kdata.shape)
    kdata_tensor = torch.tensor(kdata)
    image_coil = ifft2c(kdata_tensor)
    image = (image_coil.abs()**2).sum(-3)**0.5
    image_np = image.cpu().numpy()
    return kdata, image_np


def crop_lastnx(image, nx):
    """ Crop the image tensor to the desired size (..., nx) on the last dimension."""
    start_x = (image.shape[-1] - nx) // 2
    return image[..., start_x:start_x + nx]


def zf_recon_4D5D_halfcropnx(filename):
    '''
    load kdata and direct IFFT + RSS recon
    return kdata shape [t,z,c,y,x/2] or [z,c,y,x/2]
    return image shape [t,z,y,x/2] or [z,y,x/2]
    '''
    kdata = load_kdata_compatible(filename)
    # print(kdata.dtype)
    # print(kdata.shape)
    kdata_tensor = torch.tensor(kdata)
    image_coil = ifft2c(kdata_tensor)
    nx_halfcrop = image_coil.shape[-1] // 2  # nx half crop to remove 2 times oversampling (reduce training burden)
    image_coil_crop = crop_lastnx(image_coil, nx_halfcrop)
    image_crop = (image_coil_crop.abs() ** 2).sum(-3) ** 0.5
    image_np = image_crop.cpu().numpy()
    kdata_crop_np = fft2c(image_coil_crop).cpu().numpy()
    return kdata_crop_np, image_np


def extract_number(filename):
    '''
    extract number from filename
    '''
    return ''.join(filter(str.isdigit, filename))


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim.item() / gt.shape[0]


def ssim_4d(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 4:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    metric = np.array([0])
    for t_num in range(gt.shape[0]):
        metric = metric + ssim(
            gt[t_num], pred[t_num], maxval=maxval
        )

    return metric.item() / gt.shape[0]


def cal_metric(gt, pred):
    # metric_rmse = mse(gt,pred)**0.5
    metric_nmse = nmse(gt,pred)
    metric_psnr = psnr(gt,pred)
    metric_ssim_4d = ssim_4d(gt,pred)
    # if is_print:
    #     print('mse: {metric_mse:.4f}, nmse: {metric_nmse:.4f}, psnr: {metric_psnr:.4f}, ssim: {metric_ssim_4d:.4f}')
    return metric_nmse, metric_psnr, metric_ssim_4d


def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) if model is not None else 0


def count_trainable_parameters(model):
    return (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        if model is not None
        else 0
    )


def count_untrainable_parameters(model):
    return (
        sum(p.numel() for p in model.parameters() if not p.requires_grad)
        if model is not None
        else 0
    )


def loadmat(filename):
    """
    Load Matlab v7.3 format .mat file using h5py.
    """
    with h5py.File(filename, 'r') as f:
        data = {}
        for k, v in f.items():
            if isinstance(v, h5py.Dataset):
                data[k] = v[()]
            elif isinstance(v, h5py.Group):
                data[k] = loadmat_group(v)
    return data


def loadmat_compatible(filename):
    """
    Load Matlab v7.3 format .mat file using h5py.
    If that fails, try to load it as an older version .mat file using scipy.io.loadmat.
    """
    try:
        # Try to load the file as an HDF5 (v7.3) file
        with h5py.File(filename, 'r') as f:
            data = {}
            for k, v in f.items():
                if isinstance(v, h5py.Dataset):
                    data[k] = v[()]
                elif isinstance(v, h5py.Group):
                    data[k] = loadmat_group(v)
            return data, 'h5py'
    except (OSError, ValueError) as e:
        print(f"Error loading file with h5py: {e}")
        # If h5py fails, try to load the file using scipy.io.loadmat
        data = convert_to_v73(filename)
        result = {}
        for k, v in data.items():
            if isinstance(v, dict):
                result[k] = loadmat_group(v)
            else:
                result[k] = v
        print(f"Successfully loading file with loadmat")
        return result, 'loadmat'


def convert_to_v73(filename):
    """
    Convert a .mat file to the v7.3 format.
    This function attempts to read the input .mat file. If it's an older version,
    it uses scipy.io.loadmat. Then it saves the data into a new .mat file in v7.3 format.
    """
    # Try to load the .mat file using scipy.io.loadmat for older versions
    data = scipy.io.loadmat(filename)
    # Remove metadata keys that are not needed in v7.3 format
    metadata_keys = ['__header__', '__version__', '__globals__']
    for key in metadata_keys:
        data.pop(key, None)
    return data


def loadmat_group(group):
    """
    Load a group in Matlab v7.3 format .mat file using h5py.
    """
    data = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            data[k] = v[()]
        elif isinstance(v, h5py.Group):
            data[k] = loadmat_group(v)
    return data


def load_kdata(filename):
    '''
    load kdata from .mat file
    return shape: [t,nz,nc,ny,nx]
    '''
    data = loadmat(filename)
    keys = list(data.keys())[0]
    kdata = data[keys]
    kdata = kdata['real'] + 1j*kdata['imag']
    return kdata


def load_kdata_compatible(filename):
    '''
    load kdata from .mat file
    return shape: [t,nz,nc,ny,nx] or [nz,nc,ny,nx] or [nc,ny,nx]
    '''
    data, case = loadmat_compatible(filename)
    keys = list(data.keys())[0]
    kdata = data[keys]
    # print(case)
    if case == 'h5py':
        kdata = kdata['real'] + 1j * kdata['imag']
    elif case == 'loadmat':
        if kdata.ndim == 5:
            kdata = kdata.transpose(4,3,2,1,0)
        elif kdata.ndim == 4:
            kdata = kdata.transpose(3,2,1,0)
        else:
            kdata = kdata.transpose(2,1,0)
    return kdata


def load_maskdata(filename):
    '''
    load kdata from .mat file 3D-[nt, ny, nx] or 2D-[ny, nx]
    return shape: same to input
    '''
    data = loadmat(filename)
    keys = list(data.keys())[0]
    maskdata = data[keys]
    return maskdata


class KspaceACSExtractor:
    '''
    Extract ACS lines from k-space data
    '''

    def __init__(self, mask_center):
        self.mask_center = mask_center
        self.low_mask_dict = {}  # avoid repeated calculation

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        get the padding size and number of low frequencies for the center mask. For fastmri and cmrxrecon dataset
        '''
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2
        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def __call__(self, masked_kspace: torch.Tensor,
                 mask: torch.Tensor,
                 num_low_frequencies: Optional[int] = None,
                 mask_type: Optional[str] = "random",
                 ) -> torch.Tensor:
        if self.mask_center:
            # assume the same type in a batch: "uniform", "random", "radial"
            mask_type = "2D" if mask_type in ["radial"] else "1D"
            if mask_type == "1D":  # 1D undersampling
                pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                    mask, num_low_frequencies
                )
                masked_kspace_acs = transforms.batched_mask_center(
                    masked_kspace, pad, pad + num_low_freqs
                )
            elif mask_type == "2D":  # 2D undersampling
                mask_low = torch.zeros_like(mask)
                b, adj_nc, h, w, two = mask.shape
                h_left = (h - num_low_frequencies + 1) // 2
                w_left = (w - num_low_frequencies + 1) // 2
                mask_low[:, :, h_left:h_left+num_low_frequencies, w_left:w_left+num_low_frequencies, :] \
                    = mask[:, :, h_left:h_left+num_low_frequencies, w_left:w_left+num_low_frequencies, :]
                masked_kspace_acs = masked_kspace * mask_low
            else:
                raise ValueError("mask_type should be 1D or 2D undersampling")
            return masked_kspace_acs
        else:
            return masked_kspace


############# help[ function #############
def matlab_round(n):
    if n > 0:
        return int(n + 0.5)
    else:
        return int(n - 0.5)


def _crop(a, crop_shape):
    indices = [
        (math.floor(dim/2) + math.ceil(-crop_dim/2),
         math.floor(dim/2) + math.ceil(crop_dim/2))
        for dim, crop_dim in zip(a.shape, crop_shape)
    ]
    return a[indices[0][0]:indices[0][1], indices[1][0]:indices[1][1], indices[2][0]:indices[2][1], indices[3][0]:indices[3][1]]