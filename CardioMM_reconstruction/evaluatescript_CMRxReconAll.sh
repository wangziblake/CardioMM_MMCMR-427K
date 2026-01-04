#### Version: 2025/10/18
#### @author: Zi Wang (zi.wang@imperial.ac.uk)


#### Acticate python envs environment first
source activate torch2.0


# 4. Evaluation
## Save reconstructed gt sos images to .png for visualization
python maincode/evaluate_cmr/evaluate_gtpngsave_cmrxreconall.py \
--input /mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil \
--output /mnt/nas/nas3/openData/MMCMR_427K/AllData/ImagePNG \
--evaluate_set TestSet \
--modality All
--center_crop


## Save reconstructed gt sos images to .nii.gz for analysis
python maincode/evaluate_cmr/evaluate_gtniisave_cmrxreconall.py \
--input /mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil \
--output /mnt/nas/nas3/openData/MMCMR_427K/AllData/ImageNII \
--evaluate_set TestSet \
--modality Cine

python maincode/evaluate_cmr/evaluate_gtniisave_nocsv_cmrxreconall.py \
--input /mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil \
--output /mnt/nas/nas3/openData/MMCMR_427K/AllData/ImageNII \
--evaluate_set TestSet \
--modality Cine


## Save reconstructed zero-filled sos images to .png for visualization
python maincode/evaluate_cmr/evaluate_zfpngsave_cmrxreconall.py \
--input /mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil \
--output /mnt/nas/nas3/openData/MMCMR_427K/AllData/ImagePNG \
--evaluate_set TestSet \
--modality All
--center_crop


## Save reconstructed images from different methods to .png for visualization
python maincode/evaluate_cmr/evaluate_reconpngsave_cmrxreconall.py \
--input /mnt/nas/nas3/openData/MMCMR_427K/Results_h5_FullSamplev2_Trained \
--output /mnt/nas/nas3/openData/MMCMR_427K/Results_h5_FullSamplev2_Trained \
--evaluate_set TestSet \
--task TaskAll \
--modality Cine \
--method CardioMM \
--undersample Uniform8 \
--center_crop


## Save reconstructed images from different methods to .nii.gz for analysis
python maincode/evaluate_cmr/evaluate_reconniisave_cmrxreconall.py \
--input /mnt/nas/nas3/openData/MMCMR_427K/Results_h5_FullSamplev2_Trained \
--output /mnt/nas/nas3/openData/MMCMR_427K/Results_h5_FullSamplev2_Trained \
--csv_dir /mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil \
--evaluate_set TestSet \
--task TaskAll \
--modality Cine \
--method CardioMM

python maincode/evaluate_cmr/evaluate_reconniisave_nocsv_cmrxreconall.py \
--input /mnt/nas/nas3/openData/MMCMR_427K/Results_h5_FullSamplev2_Trained \
--output /mnt/nas/nas3/openData/MMCMR_427K/Results_h5_FullSamplev2_Trained \
--evaluate_set TestSet \
--task TaskAll \
--modality Cine \
--method CardioMM \
--undersample Uniform8


## Calculate quantatative objective criteria (NMSE, PSNR, SSIM) between reconstructed images and gt images
python maincode/evaluate_cmr/evaluate_objcriteria_cmrxreconall.py \
--input /mnt/nas/nas3/openData/MMCMR_427K/Results_h5_FullSamplev2_Trained \
--output /mnt/nas/nas3/openData/MMCMR_427K/Results_h5_FullSamplev2_Trained \
--gtinput /mnt/nas/nas3/openData/MMCMR_427K/AllData/MultiCoil \
--evaluate_set TestSet \
--task TaskAll \
--modality All \
--undersample Uniform8 \
--normscheme percentile \
--method CardioMM