#### Version: 2025/10/15
#### @author: Zi Wang (zi.wang@imperial.ac.uk)


#### Acticate python envs environment first
source activate torch2.0


# 3. Recon commands
## test CardioMM-10cascades model
CUDA_VISIBLE_DEVICES=0 python maincode/recon_cmr/recon_CardioMM_cmrxreconall_findmask.py \
--input /mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil \
--output /mnt/nas/nas3/openData/MMCMR_427K/Results_h5_FullSamplev2_Trained/CardioMM \
--model_path pretrained/h5_FullSamplev2/CardioMM/epochbest.ckpt \
--evaluate_set TestSet \
--task TaskAll \
--modality All \
--batch_size 4 \
--num_works 2 \
--num_cascades 10 \
--num_low_frequencies 20 \
--undersample Uniform8


## test SENSE method
CUDA_VISIBLE_DEVICES=0 python maincode/recon_cmr/recon_SENSE_cmrxreconall_findmask.py \
--input /mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil \
--output /mnt/nas/nas3/openData/MMCMR_427K/Results_h5_FullSamplev2_Trained/SENSE \
--evaluate_set TestSet \
--task TaskAll \
--modality All \
--batch_size 4 \
--num_works 2 \
--num_low_frequencies 20 \
--undersample Uniform8