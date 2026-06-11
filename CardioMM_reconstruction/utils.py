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
    """
    Convert a real-valued tensor with a final complex dimension to complex dtype.

    Args:
        x: Tensor shaped ``[batch, coils/channels, height, width, 2]`` where the
            final dimension stores real and imaginary components.

    Returns:
        Complex tensor shaped ``[batch, coils/channels, height, width]``.
    """
    b, c, h, w, two = x.shape
    assert two == 2
    x_real, x_imag = x[..., 0], x[..., 1]
    x_complex = torch.complex(x_real, x_imag)
    return x_complex


def complex2double(x):  # complex [b, c, h, w] --> double [b, c, h, w, 2]
    """
    Convert a complex tensor to a real-valued tensor with real/imag channels.

    Args:
        x: Complex tensor shaped ``[batch, coils/channels, height, width]``.

    Returns:
        Real-valued tensor shaped ``[batch, coils/channels, height, width, 2]``.
    """
    x_real, x_imag = torch.real(x), torch.imag(x)
    x_double = torch.stack((x_real, x_imag), dim=-1)
    return x_double


def count_params(model):
    """Return the number of trainable parameters in a PyTorch module."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params


def ifft2c(kdata_tensor, dim=(-2,-1), norm='ortho'):
    """
    Apply a centered 2D inverse FFT to k-space data.

    Args:
        kdata_tensor: Complex k-space tensor with the zero-frequency component
            centered along ``dim``.
        dim: Spatial dimensions used for the 2D transform. By default these are
            the last two dimensions.
        norm: Normalization mode passed to ``torch.fft.ifft2``.

    Returns:
        Complex image-space tensor with the same shape as ``kdata_tensor``.
    """
    # Move the k-space center to the FFT origin before applying PyTorch's IFFT.
    kdata_tensor_uncentered = torch.fft.ifftshift(kdata_tensor,dim=dim)
    image_uncentered = torch.fft.ifft2(kdata_tensor_uncentered,dim=dim, norm=norm)
    # Shift back so image-space data follows the centered convention used here.
    image = torch.fft.fftshift(image_uncentered,dim=dim)
    return image


def fft2c(image_tensor, dim=(-2,-1), norm='ortho'):
    """
    Apply a centered 2D FFT to image-space data.

    Args:
        image_tensor: Complex image tensor with centered spatial dimensions.
        dim: Spatial dimensions used for the 2D transform. By default these are
            the last two dimensions.
        norm: Normalization mode passed to ``torch.fft.fft2``.

    Returns:
        Complex k-space tensor with the zero-frequency component centered.
    """
    # Convert from centered image convention to PyTorch FFT origin convention.
    image_tensor_uncentered = torch.fft.ifftshift(image_tensor,dim=dim)
    kdata_uncentered = torch.fft.fft2(image_tensor_uncentered,dim=dim, norm=norm)
    # Return centered k-space to match MRI reconstruction conventions.
    kdata = torch.fft.fftshift(kdata_uncentered,dim=dim)
    return kdata


def zf_recon(filename):
    """
    Load k-space data and perform zero-filled reconstruction on GPU.

    The function assumes MATLAB v7.3 complex k-space saved as separate ``real``
    and ``imag`` datasets. Coil images are combined with root-sum-of-squares
    (RSS) over the coil dimension.

    Args:
        filename: Path to a MATLAB ``.mat`` file containing k-space data.

    Returns:
        Tuple ``(kdata, image_np)`` where ``kdata`` has shape
        ``[time, slice, coil, height, width]`` and ``image_np`` has shape
        ``[time, slice, height, width]``.
    """
    kdata = load_kdata(filename)
    kdata_tensor = torch.tensor(kdata).cuda()
    image_coil = ifft2c(kdata_tensor)
    # RSS combines coil images into a magnitude image after direct IFFT.
    image = (image_coil.abs()**2).sum(2)**0.5
    image_np = image.cpu().numpy()
    return kdata, image_np


def zf_recon_4D5D(filename):
    """
    Load 4D/5D k-space data and perform zero-filled reconstruction on GPU.

    Args:
        filename: Path to a MATLAB ``.mat`` file. Both HDF5 MATLAB v7.3 files
            and older MATLAB formats are supported by ``load_kdata_compatible``.

    Returns:
        Tuple ``(kdata, image_np)``. ``kdata`` is shaped
        ``[time, slice, coil, height, width]`` or ``[slice, coil, height, width]``;
        ``image_np`` is the corresponding RSS image without the coil dimension.
    """
    kdata = load_kdata_compatible(filename)
    # print(kdata.dtype)
    # print(kdata.shape)
    kdata_tensor = torch.tensor(kdata).cuda()
    image_coil = ifft2c(kdata_tensor)
    # The coil dimension is always third from the end for 4D and 5D layouts.
    image = (image_coil.abs()**2).sum(-3)**0.5
    image_np = image.cpu().numpy()
    return kdata, image_np


def zf_recon_4D5D_cpu(filename):
    """
    Load 4D/5D k-space data and perform zero-filled reconstruction on CPU.

    This is the CPU equivalent of ``zf_recon_4D5D`` and is useful when CUDA is
    unavailable or when reconstruction is used in lightweight preprocessing.

    Args:
        filename: Path to a compatible MATLAB ``.mat`` k-space file.

    Returns:
        Tuple ``(kdata, image_np)`` with RSS-combined image data. The input
        k-space may be ``[time, slice, coil, height, width]`` or
        ``[slice, coil, height, width]``.
    """
    kdata = load_kdata_compatible(filename)
    # print(kdata.dtype)
    # print(kdata.shape)
    kdata_tensor = torch.tensor(kdata)
    image_coil = ifft2c(kdata_tensor)
    image = (image_coil.abs()**2).sum(-3)**0.5
    image_np = image.cpu().numpy()
    return kdata, image_np


def crop_lastnx(image, nx):
    """
    Center-crop the last dimension of an image or k-space tensor.

    Args:
        image: Tensor or array whose final dimension is the readout/x dimension.
        nx: Target size for the last dimension.

    Returns:
        Tensor/array with shape ``[..., nx]``.
    """
    start_x = (image.shape[-1] - nx) // 2
    return image[..., start_x:start_x + nx]


def zf_recon_4D5D_halfcropnx(filename):
    """
    Load k-space, remove 2x readout oversampling, and reconstruct RSS images.

    The crop is performed in image space along the final x/readout dimension,
    then the cropped coil images are transformed back to k-space.

    Args:
        filename: Path to a compatible MATLAB ``.mat`` k-space file.

    Returns:
        Tuple ``(kdata_crop_np, image_np)`` where the final readout dimension is
        half of the original size.
    """
    kdata = load_kdata_compatible(filename)
    # print(kdata.dtype)
    # print(kdata.shape)
    kdata_tensor = torch.tensor(kdata)
    image_coil = ifft2c(kdata_tensor)
    # Half crop removes 2x readout oversampling and reduces training cost.
    nx_halfcrop = image_coil.shape[-1] // 2
    image_coil_crop = crop_lastnx(image_coil, nx_halfcrop)
    image_crop = (image_coil_crop.abs() ** 2).sum(-3) ** 0.5
    image_np = image_crop.cpu().numpy()
    kdata_crop_np = fft2c(image_coil_crop).cpu().numpy()
    return kdata_crop_np, image_np


def extract_number(filename):
    """Extract all digits from a filename and return them as one string."""
    return ''.join(filter(str.isdigit, filename))


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Compute mean squared error between ground truth and prediction arrays.

    Args:
        gt: Ground-truth image array.
        pred: Predicted image array with the same shape as ``gt``.

    Returns:
        Scalar MSE value.
    """
    return np.mean((gt - pred) ** 2)


def nmse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Compute normalized mean squared error using the ground-truth energy.

    Args:
        gt: Ground-truth image array.
        pred: Predicted image array with the same shape as ``gt``.

    Returns:
        Scalar NMSE value as a NumPy array scalar.
    """
    return np.array(np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2)


def psnr(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """
    Compute peak signal-to-noise ratio (PSNR).

    Args:
        gt: Ground-truth image array.
        pred: Predicted image array with the same shape as ``gt``.
        maxval: Data range used for PSNR. If omitted, ``gt.max()`` is used.

    Returns:
        Scalar PSNR value.
    """
    if maxval is None:
        maxval = gt.max()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """
    Compute mean structural similarity (SSIM) over a 3D image stack.

    Args:
        gt: Ground-truth array shaped ``[slice, height, width]``.
        pred: Predicted array with the same number of dimensions as ``gt``.
        maxval: Data range used for SSIM. If omitted, ``gt.max()`` is used.

    Returns:
        Average SSIM across slices.
    """
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = np.array([0])
    # skimage SSIM is computed slice-by-slice for volumetric data.
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim.item() / gt.shape[0]


def ssim_4d(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """
    Compute mean SSIM over a 4D dynamic image stack.

    Args:
        gt: Ground-truth array shaped ``[time, slice, height, width]``.
        pred: Predicted array with the same number of dimensions as ``gt``.
        maxval: Data range used for SSIM. If omitted, ``gt.max()`` is used.

    Returns:
        Average SSIM across time points, each computed by ``ssim`` over slices.
    """
    if not gt.ndim == 4:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    metric = np.array([0])
    # Average the per-volume SSIM across the temporal dimension.
    for t_num in range(gt.shape[0]):
        metric = metric + ssim(
            gt[t_num], pred[t_num], maxval=maxval
        )

    return metric.item() / gt.shape[0]


def cal_metric(gt, pred):
    """
    Compute the reconstruction metrics used by evaluation scripts.

    Args:
        gt: Ground-truth 4D image array.
        pred: Predicted 4D image array.

    Returns:
        Tuple ``(nmse, psnr, ssim_4d)``.
    """
    # metric_rmse = mse(gt,pred)**0.5
    metric_nmse = nmse(gt,pred)
    metric_psnr = psnr(gt,pred)
    metric_ssim_4d = ssim_4d(gt,pred)
    # if is_print:
    #     print('mse: {metric_mse:.4f}, nmse: {metric_nmse:.4f}, psnr: {metric_psnr:.4f}, ssim: {metric_ssim_4d:.4f}')
    return metric_nmse, metric_psnr, metric_ssim_4d


def count_parameters(model):
    """Return all parameters in ``model``, or zero when ``model`` is ``None``."""
    return sum(p.numel() for p in model.parameters()) if model is not None else 0


def count_trainable_parameters(model):
    """Return trainable parameters in ``model``, or zero when ``model`` is ``None``."""
    return (
        sum(p.numel() for p in model.parameters() if p.requires_grad)
        if model is not None
        else 0
    )


def count_untrainable_parameters(model):
    """Return frozen parameters in ``model``, or zero when ``model`` is ``None``."""
    return (
        sum(p.numel() for p in model.parameters() if not p.requires_grad)
        if model is not None
        else 0
    )


def loadmat(filename):
    """
    Load a MATLAB v7.3 ``.mat`` file with ``h5py``.

    MATLAB v7.3 files are HDF5 containers. This helper recursively converts
    top-level datasets and groups into Python dictionaries and NumPy arrays.

    Args:
        filename: Path to a MATLAB v7.3 ``.mat`` file.

    Returns:
        Dictionary containing all top-level variables from the file.
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
    Load a MATLAB ``.mat`` file from either v7.3 or older formats.

    The function first tries the HDF5/v7.3 path via ``h5py``. If that fails, it
    falls back to ``scipy.io.loadmat`` for older MATLAB files and removes MATLAB
    metadata keys.

    Args:
        filename: Path to a MATLAB ``.mat`` file.

    Returns:
        Tuple ``(data, case)`` where ``case`` is ``"h5py"`` for v7.3/HDF5 files
        or ``"loadmat"`` for older files loaded with SciPy.
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
        # Older MATLAB files are not HDF5, so SciPy handles the fallback path.
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
    Read an older MATLAB ``.mat`` file and remove MATLAB metadata keys.

    Despite the historical function name, this helper does not write a converted
    file to disk. It returns the SciPy-loaded content in a dictionary compatible
    with downstream loaders.

    Args:
        filename: Path to a non-HDF5 MATLAB ``.mat`` file.

    Returns:
        Dictionary of MATLAB variables with ``__header__``, ``__version__``, and
        ``__globals__`` removed.
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
    Recursively load a MATLAB v7.3/HDF5 group.

    Args:
        group: ``h5py.Group`` or dict-like object containing datasets/groups.

    Returns:
        Nested dictionary containing arrays for datasets and dictionaries for
        child groups.
    """
    data = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            data[k] = v[()]
        elif isinstance(v, h5py.Group):
            data[k] = loadmat_group(v)
    return data


def load_kdata(filename):
    """
    Load complex k-space data from a MATLAB v7.3 ``.mat`` file.

    The first top-level variable is expected to store separate ``real`` and
    ``imag`` fields, which are combined into a complex NumPy array.

    Args:
        filename: Path to a MATLAB v7.3 ``.mat`` k-space file.

    Returns:
        Complex k-space array shaped ``[time, slice, coil, height, width]``.
    """
    data = loadmat(filename)
    keys = list(data.keys())[0]
    kdata = data[keys]
    # MATLAB/HDF5 complex numbers are stored as separate real and imaginary arrays.
    kdata = kdata['real'] + 1j*kdata['imag']
    return kdata


def load_kdata_compatible(filename):
    """
    Load complex k-space data from v7.3 or older MATLAB ``.mat`` files.

    For HDF5/v7.3 files, real and imaginary fields are combined directly. For
    older SciPy-loaded files, axes are transposed from MATLAB order into the
    repository convention with coil and spatial dimensions at the end.

    Args:
        filename: Path to a MATLAB ``.mat`` k-space file.

    Returns:
        Complex k-space array shaped ``[time, slice, coil, height, width]``,
        ``[slice, coil, height, width]``, or ``[coil, height, width]``.
    """
    data, case = loadmat_compatible(filename)
    keys = list(data.keys())[0]
    kdata = data[keys]
    # print(case)
    if case == 'h5py':
        # HDF5 MATLAB complex values are represented with named real/imag fields.
        kdata = kdata['real'] + 1j * kdata['imag']
    elif case == 'loadmat':
        # SciPy reads MATLAB arrays in reversed axis order relative to this codebase.
        if kdata.ndim == 5:
            kdata = kdata.transpose(4,3,2,1,0)
        elif kdata.ndim == 4:
            kdata = kdata.transpose(3,2,1,0)
        else:
            kdata = kdata.transpose(2,1,0)
    return kdata


def load_maskdata(filename):
    """
    Load sampling mask data from a MATLAB v7.3 ``.mat`` file.

    Args:
        filename: Path to a MATLAB ``.mat`` mask file.

    Returns:
        Mask array with the same stored shape, typically ``[time, height, width]``
        or ``[height, width]``.
    """
    data = loadmat(filename)
    keys = list(data.keys())[0]
    maskdata = data[keys]
    return maskdata


class KspaceACSExtractor:
    """
    Extract auto-calibration signal (ACS) samples from masked k-space.

    ACS data is used by the sensitivity map estimation network. When
    ``mask_center`` is true, only the central low-frequency region is retained.
    When false, the input masked k-space is returned unchanged.
    """

    def __init__(self, mask_center):
        """
        Args:
            mask_center: If true, keep only the central ACS region before
                sensitivity estimation; otherwise bypass ACS extraction.
        """
        self.mask_center = mask_center
        self.low_mask_dict = {}  # avoid repeated calculation

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ACS start positions and widths for 1D undersampling masks.

        Args:
            mask: Sampling mask shaped like fastMRI/CardioMM k-space masks,
                typically ``[batch, 1 or coils, height, width, 1]``.
            num_low_frequencies: Optional explicit ACS width. If ``None`` or
                zero, the width is inferred from contiguous center samples.

        Returns:
            Tuple ``(pad, num_low_frequencies_tensor)``. ``pad`` is the left
            index of the centered ACS band along the phase-encoding dimension.
        """
        if num_low_frequencies is None or num_low_frequencies == 0:
            # Work on one representative coil/channel because the mask is shared.
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # argmin on flipped/forward halves finds distance to the first zero.
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
        """
        Return k-space restricted to the ACS region when requested.

        Args:
            masked_kspace: Complex-valued k-space tensor, usually shaped
                ``[batch, adjacent_coils, height, width, 2]`` with real/imag in
                the final dimension.
            mask: Sampling mask broadcastable to ``masked_kspace``.
            num_low_frequencies: ACS size. For 1D masks it may be inferred when
                omitted; for 2D/radial masks it defines the square center window.
            mask_type: Original mask family. ``"radial"`` is treated as 2D ACS;
                other current mask types are treated as 1D ACS.

        Returns:
            ACS-masked k-space if ``self.mask_center`` is true; otherwise the
            original ``masked_kspace`` tensor.
        """
        if self.mask_center:
            # assume the same type in a batch: "uniform", "random", "radial"
            mask_type = "2D" if mask_type in ["radial"] else "1D"
            if mask_type == "1D":  # 1D undersampling
                pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                    mask, num_low_frequencies
                )
                # fastMRI helper keeps the centered phase-encoding ACS band.
                masked_kspace_acs = transforms.batched_mask_center(
                    masked_kspace, pad, pad + num_low_freqs
                )
            elif mask_type == "2D":  # 2D undersampling
                mask_low = torch.zeros_like(mask)
                b, adj_nc, h, w, two = mask.shape
                # Radial/2D masks use a square low-frequency ACS crop.
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
    """
    Round a scalar using MATLAB's half-away-from-zero convention.

    Args:
        n: Numeric scalar to round.

    Returns:
        Integer rounded as MATLAB would round positive and negative half values.
    """
    if n > 0:
        return int(n + 0.5)
    else:
        return int(n - 0.5)


def _crop(a, crop_shape):
    """
    Center-crop a 4D array to ``crop_shape``.

    Args:
        a: Input array with four dimensions.
        crop_shape: Target crop size for each of the four dimensions.

    Returns:
        Center-cropped view of ``a``.
    """
    # Compute inclusive/exclusive crop bounds around the center of each axis.
    indices = [
        (math.floor(dim/2) + math.ceil(-crop_dim/2),
         math.floor(dim/2) + math.ceil(crop_dim/2))
        for dim, crop_dim in zip(a.shape, crop_shape)
    ]
    return a[indices[0][0]:indices[0][1], indices[1][0]:indices[1][1], indices[2][0]:indices[2][1], indices[3][0]:indices[3][1]]
