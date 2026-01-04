#### Version: 2025/08/29
#### @author: Zi Wang (zi.wang@imperial.ac.uk)


#### Acticate python envs environment first
source activate torch2.0


# 1. Training from scratch
## Step1: Preprocess the training dataset from CMRxReconAll
python maincode/prepare_cmr/prepare_h5py_dataset_for_training_cmrxreconallv2_step1.py \
--data_path /mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil \
--newsave_path /mnt/nas/nas3/openData/MMCMR_427K/TrainData/MultiCoil \
--h5py_folder h5_FullSamplev2
## Step2: Split for training set and validations set
python maincode/prepare_cmr/prepare_h5py_dataset_for_training_cmrxreconallv2_step2.py \
--data_path /mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil \
--newsave_path /mnt/nas/nas3/openData/MMCMR_427K/TrainData/MultiCoil \
--h5py_folder h5_FullSamplev2


## Groudtruth SOS reconstruction in advance for fast evaluaion
python maincode/prepare_cmr/prepare_groundtruthsos_cmrxreconall.py \
--input /mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil \
--output /mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil \
--evaluate_set TestSet \
--modality All


## Zero-filled SOS reconstruction in advance for fast evaluaion (A6000-fdu-gpu-node1)
python maincode/prepare_cmr/prepare_zerofilledsos_cmrxreconall.py \
--input /mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil \
--output /mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil \
--evaluate_set TestSet \
--task TaskAll \
--modality All \
--undersample Uniform8
