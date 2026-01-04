"""
Testing the CardioMM model with CSM estimation using SEMnetwork - pytorch - CMRxReconAll - two lmhead for metadata and undersampling
- find mask first then match data - using CLIP text encoder
- Modified KspaceACSExtractor for 1D and 2D undersampling
- no medical condition injection
- Modified Data consistency
- Modified Text information injection
Created on 2025/10/15
@author: Zi Wang
Email: Zi Wang (zi.wang@imperial.ac.uk)
If you want to use this code, please cite our relevant papers in the GitHub page.

Some codes are modified based on https://arxiv.org/abs/2309.13839
"""

import os
import sys
import pathlib
sys.path.insert(0, os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.absolute())))

import argparse
import torch
import scipy.io as sio
import glob
from os.path import join
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import re

import fastmri.data.transforms as T
from models.CardioMM_model import CardioMM_SEM
from text.text_models import LanguageModel, LMHead, LMHead2
from utils import load_kdata_compatible, load_maskdata
from utils import count_parameters
from datamapping import get_metadata_attribute_from_filename
language_model = LanguageModel(llm_model_name="../clip-vit-base-patch16")


def get_frames_indices_stage(dataslice, num_slices_in_volume, num_t_in_volume=None):
    '''
    when we reshape t, z to one axis in preprocessing, we need to get the indices of the slices in the original t, z axis;
    then find the adjacent slices in the original z axis
    '''
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
    if "_mask_" in mask_filename:
        base, ext = mask_filename.rsplit(".", 1)
        base = base.rsplit("_mask_", 1)[0]
        data_filename = f"{base}.{ext}"
    else:
        raise NotImplementedError("The filename does not contain '_mask_'")  # Raise an error if "_mask_" is missing
    return data_filename


class stage_dataset(torch.utils.data.Dataset):
    def __init__(self, fname, task):
        self.fname = fname
        self.task = task
        # here, input kspace from .mat: 5D-[nt, nz, nc, ny, nx] or 4D-[nz, nc, ny, nx] or 3D-[nc, ny, nx]
        data_fname = replace_mask_to_find_data(fname).replace(f'Mask_{self.task}', 'FullSample')
        self.kspace = load_kdata_compatible(data_fname)
        if len(self.kspace.shape) != 5:
            self.kspace = np.expand_dims(self.kspace, axis=0)  # make sure its shape is [1, nz, nc, ny, nx]
            if len(self.kspace.shape) != 5:
                self.kspace = np.expand_dims(self.kspace, axis=0)  # make sure its shape is [1, nz, nc, ny, nx]
        self.num_t = self.kspace.shape[0]
        self.num_slices = self.kspace.shape[1]
        self.kspace = self.kspace.reshape(-1, self.kspace.shape[2], self.kspace.shape[3], self.kspace.shape[4]).transpose(0, 1, 3, 2)  # --> [nt*nz, nc, nx, ny]

        # here, input mask from .mat: 3D-[nt, ny, nx] or 2D-[ny, nx]
        self.mask = load_maskdata(fname)
        if len(self.mask.shape) == 3:
            self.mask = np.expand_dims(np.expand_dims(self.mask, axis=1),axis=2)  # make sure its shape is [nt, 1, 1, ny, nx]
        elif len(self.mask.shape) == 2:
            self.mask = np.expand_dims(np.expand_dims(np.expand_dims(self.mask, axis=0), axis=1), axis=2)  # make sure its shape is [1, 1, 1, ny, nx]
            self.mask = np.tile(self.mask, reps=(self.num_t, 1, 1, 1, 1))  # make sure its shape is [nt, 1, 1, ny, nx]
        else:
            raise NotImplementedError("The mask shape should be 2D(k) or 3D(k-t).")
        self.mask = np.tile(self.mask, reps=(1, self.num_slices, 1, 1, 1))  # make sure its shape is [nt, nz, 1, ny, nx]
        self.mask = self.mask.reshape(-1, self.mask.shape[2], self.mask.shape[3], self.mask.shape[4]).transpose(0, 1, 3, 2)  # --> [nt*nz, 1, nx, ny]

        self.num_files = self.kspace.shape[0]
        self.maskfunc = None
        self.attrs = get_metadata_attribute_from_filename(fname)  # TODO: get metadata, check sometimes

    def __getitem__(self, dataslice):
        slice_idx_list = get_frames_indices_stage(dataslice, self.num_slices, self.num_t)
        _input = []
        for slc_i in slice_idx_list:
            _input.append(self.kspace[slc_i])
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

        # TODO: add more metadata from dataset attributes !!!!!!
        # Metadata: Default value should be unknown
        # -- Lifespan
        # -- Reconstruct
        # -- Center
        # -- Field strength, Vendor
        # -- Modality, View
        # -- Healthy, Patients with diseases
        metadata_lifespan = self.attrs['lifespan']  # Adult, Pediatric, Fetal
        metadata_task = "cardiac MRI reconstruction"
        metadata_center = self.attrs['center']
        metadata_modality, metadata_view = self.attrs['modality'], self.attrs['view']
        metadata_field, metadata_vendor, metadata_scanner = self.attrs['field'], self.attrs['vendor'], self.attrs['scanner']
        # metadata_medcon = self.attrs['medcon']  # [healthy, HCM, ..., or unknown]
        ori_metadata = f"{metadata_lifespan} {metadata_task}." \
                       f"Vendor: {metadata_field}, {metadata_vendor}, {metadata_scanner}. " \
                       f"Modality: {metadata_modality}, {metadata_view}. " \
                       # f"Disease: {metadata_medcon}."
        metadata_tuple = re.sub(r'\s+', ' ', ori_metadata.strip()).replace("\n", "")
        # print(metadata_tuple)  # Text debug
        metadata_list = [metadata_tuple]  # tuple -> list as input
        lm_embd_init = language_model(metadata_list)  # text including metadata for input

        # USdata: Default value should be unknown
        # -- Undersampling pattern, AF
        metadata_af = self.attrs['acceleration'] + "x"
        metadata_us = self.attrs['mask_type']
        ori_usdata = f"Undersampling: {metadata_af} {metadata_us}."
        usdata_tuple = re.sub(r'\s+', ' ', ori_usdata.strip()).replace("\n", "")
        # print(usdata_tuple)  # Text debug
        usdata_list = [usdata_tuple]  # tuple -> list as input
        lm_embd_init2 = language_model(usdata_list)  # text including usdata for tensor input

        return masked_kspace * mask_torch, mask_torch, dataslice, lm_embd_init, lm_embd_init2, metadata_us

    def __len__(self):
        return self.num_files


def predict(f, num_low_frequencies, num_cascades=12, model_path = '', bs1 = 1, center_crop=False, num_works = 2, input_dir='', output_dir='', task='TaskAll'):
    # 0. config
    device = 'cuda:0'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(os.getcwd())

    # 1. load model
    model1 = CardioMM_SEM(
            num_cascades=num_cascades,  # number of unrolled iterations
            num_adj_slices=1,  # number of adjacent slices

            n_feat0=48,  # number of top-level channels for PromptUnet
            feature_dim = [72, 96, 120],
            prompt_dim = [24, 48, 72],

            sens_n_feat0=24,
            sens_feature_dim=[36, 48, 60],
            sens_prompt_dim=[12, 24, 36],

            len_prompt=[3, 3, 3],
            prompt_size=[64, 32, 16],
            n_enc_cab=[2, 3, 3],
            n_dec_cab=[2, 2, 3],
            n_skip_cab=[1, 1, 1],
            n_bottleneck_cab=3,
            no_use_ca = False,
    )

    state_dict = torch.load(model_path)['state_dict']
    state_dict.pop('loss.w')
    state_dict1 = {k: v for k, v in state_dict.items() if k.startswith('metausunetlight.')}
    state_dict1 = {k.replace('metausunetlight.', ''): v for k, v in state_dict1.items()}
    model1.load_state_dict(state_dict1)
    model1.eval()
    model1.to(device)

    lm_head_meta = LMHead(llm_model_dim=512, llm_embd_dim=256, llm_nclasses=3)
    state_dict2 = {k: v for k, v in state_dict.items() if k.startswith('lm_head_meta.')}
    state_dict2 = {k.replace('lm_head_meta.', ''): v for k, v in state_dict2.items()}
    lm_head_meta.load_state_dict(state_dict2)
    lm_head_meta.eval()
    lm_head_meta.to(device)

    lm_head_us = LMHead2(llm_model_dim=512, llm_embd_dim=256, llm_nclasses=3)
    state_dict3 = {k: v for k, v in state_dict.items() if k.startswith('lm_head_us.')}
    state_dict3 = {k.replace('lm_head_us.', ''): v for k, v in state_dict3.items()}
    lm_head_us.load_state_dict(state_dict3)
    lm_head_us.eval()
    lm_head_us.to(device)

    print(f'model:\ntotal param: {count_parameters(model1)+count_parameters(language_model)+count_parameters(lm_head_meta)+count_parameters(lm_head_us)}\n##############')
    print(f'model:\ntrainable param: {count_parameters(model1)+count_parameters(lm_head_meta)+count_parameters(lm_head_us)}\n##############')

    # 1. predict
    with torch.no_grad():
        for ff in tqdm(f, desc='files'):
            print('-- processing --', ff)
            save_path = ff.replace(f'Mask_{task}', f'{task}').replace(input_dir, join(output_dir, 'ReconResult'))
            if os.path.isfile(save_path):  # check if the .mat file already exists
                continue
            elif not os.path.isdir(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            dataset = stage_dataset(ff, task)
            dataloader = DataLoader(dataset, batch_size=bs1, shuffle=False, num_workers=num_works, pin_memory=True, drop_last=False)
            pred_stage = []

            for masked_kspace, mask, dataslice, metadata, usdata, mask_type in tqdm(dataloader, desc='stage1'):

                bs = masked_kspace.shape[0]
                # num_low_frequencies = 20  # TODO: need to check under different undersampling scenarios, cannot be 0
                lm_embd_init = metadata
                lm_embd_init2 = usdata
                lm_embd_adapt, _ = lm_head_meta(lm_embd_init.to(device))  # meta text embadding after projection in MRI reconstruction tasks
                lm_embd_adapt2, _ = lm_head_us(lm_embd_init2.to(device))  # us text embadding after projection in MRI reconstruction tasks
                output = model1(masked_kspace.to(device), mask.to(device), num_low_frequencies, lm_embd_adapt.to(device), lm_embd_adapt2.to(device), mask_type)
                for i in range(bs):
                    pred_stage.append((dataslice[i], output[i:i+1]))
            pred_stage = torch.cat([out for _, out in sorted(pred_stage)], dim=0).cpu()

            pred_stage_final = pred_stage.cpu().numpy().transpose(0, 2, 1).reshape(dataset.num_t, dataset.num_slices, pred_stage.shape[2], pred_stage.shape[1])

            save_mat = pred_stage_final.transpose(3, 2, 1, 0)  # nx, ny, nz, nt
            sio.savemat(save_path, {'reconimage': save_mat})
            print('-- saving --', save_path)


if __name__ == '__main__':
    argv = sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    parser.add_argument('--model_path', type=str, nargs='?', default='/model', help='model path')
    parser.add_argument('--center_crop', action='store_true', default=False, help='Enable center cropping for validation leaderboard submission')
    parser.add_argument('--evaluate_set', type=str, default='TestSet', help='Choose the evaluation set')
    parser.add_argument('--task', type=str, default='TaskAll', help='Choose to inference on which type of task')
    parser.add_argument('--modality', type=str, default='All', help='Choose to inference on which type of data')
    parser.add_argument('--undersample', type=str, default='', help='Choose the undersampling pattern (Uniform8, ktGaussian16, ktRadial24)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the model.')
    parser.add_argument('--num_works', type=int, default=2, help='num of processors to load data.')
    parser.add_argument('--num_cascades', type=int, default=10, help='num of cascades of the unrolled model.')
    parser.add_argument('--num_low_frequencies', type=int, default=20, help='num of acs size.')
    parser.add_argument('--exact_mask_filename', type=str, default=None, help='exact mask_filename to test')
    # exact_mask_filename example:
    # /SSDHome/home/Raw_data/MICCAIChallengeAll/ChallengeData/MultiCoil/Cine/TestSet/Mask_TaskAll/Center015/Siemens_30T_Vida/P301/cine_sax_mask_ktGaussian16.mat

    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output
    model_path = args.model_path
    center_crop = args.center_crop
    evaluate_set = args.evaluate_set
    task = args.task
    modality = args.modality
    undersample = args.undersample
    bs1 = args.batch_size
    num_works = args.num_works
    num_cascades = args.num_cascades
    num_low_frequencies = args.num_low_frequencies
    exact_mask_filename = args.exact_mask_filename

    print("Input data store in:", input_dir)
    print("Output data store in:", output_dir)

    if exact_mask_filename is not None:
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
    predict(f, num_low_frequencies, num_cascades, model_path=model_path, bs1=bs1, center_crop=center_crop, num_works=num_works, input_dir=input_dir, output_dir=output_dir, task=task)
