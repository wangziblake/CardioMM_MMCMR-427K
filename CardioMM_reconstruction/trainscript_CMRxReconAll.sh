#### Version: 2025/09/27
#### @author: Zi Wang (zi.wang@imperial.ac.uk)


#### Acticate python envs environment first
source activate torch2.0


# 2. Training commands
## train CardioMM-10cascades model
CUDA_VISIBLE_DEVICES=0,1,2,3 python maincode/train_cmr/train_CardioMM.py \
--center_numbers 20 \
--accelerations 4 8 16 24 \
--challenge multicoil \
--mask_type random_equispaced_radial_fixed \
--data_path /mnt/nas/nas3/openData/MMCMR_427K/TrainData/MultiCoil \
--h5py_folder h5_FullSamplev2 \
--combine_train_val \
--exp_name CardioMM \
--num_cascades 10 \
--num_gpus 4 \
--nworkers 4