# Enabling Ultra-Fast Cardiovascular Imaging Across Heterogeneous Clinical Environments with a Generalist Foundation Model and Multimodal Database
***Motivation:*** Multimodal cardiovascular magnetic resonance (CMR) imaging offers comprehensive and non-invasive insights into cardiovascular disease (CVD) diagnosis and underlying mechanisms, but its widespread real-world adoption remains constrained by prolonged scan times and heterogeneity across medical environments. **This underscores the urgent need to address a largely unexplored gap: a generalist reconstruction foundation model for ultra-fast CMR imaging**—one capable of adapting across diverse imaging scenarios and **serving as the essential substrate for all downstream analyses**.

***Database:*** To enable this goal, we curate **MMCMR-427K, the largest and most comprehensive multimodal CMR k-space database** to date. It comprises 427,465 multi-coil k-space data paired with structured metadata (approximately 3.5 TB), from 6,120 scans of 1,504 participants, spanning 13 international centers (four public repositories and nine clinical centers), 15 scanners (four vendors from low-field to ultra-high-field strengths), 12 CMR modalities, and 17 CVD categories across three populations (Asian, European, and North American).

![MMCMR-427K_Database](https://github.com/wangziblake/CardioMM_MMCMR-427K/blob/main/Figure/MMCMR-427K_Database.png)

***Model:*** Leveraging this unprecedented resource, we introduce **CardioMM, the first generalist reconstruction foundation model capable of dynamically adapting to heterogeneous ultra-fast CMR imaging scenarios**. CardioMM unifies semantic contextual understanding with physics-informed data consistency to deliver robust reconstructions across varied scanners, protocols, and patient presentations. It establishes a foundation for scalable, generalizable, and high-throughput multimodal cardiovascular imaging.

***Validation:*** Comprehensive evaluations demonstrate that CardioMM achieves state-of-the-art performance in the internal centers and **exhibits strong zero-shot generalization to unseen external settings**. For clinical applicability, **highly accelerated reconstructions (8×–24×) reliably preserve 11 key cardiac phenotypes, three quantitative myocardial biomarkers, and diagnostic image quality**, ensuring dependable diagnostic support for medical professionals.

![CardioMM](https://github.com/wangziblake/CardioMM_MMCMR-427K/blob/main/Figure/CardioMM.png)

***Complementarity:*** Notably, **our framework does not compete with existing CMR analytical foundation models** that operate on post-reconstruction images; rather, **it complements them by filling a critical upstream gap in the multimodal CMR computational pipeline**. By delivering higher-quality and more diverse image reconstructions, CardioMM provides a more robust and reliable foundation for downstream segmentation, phenotyping, diagnosis, and beyond.

***Engagement:*** Remarkably, **in our organized CMRxRecon challenge series (2023–2025), more than 11,000 active participants from 125 countries and regions have engaged**, reflecting the broad and growing global interest in next-generation CMR technologies.

***Impact:*** This work provides **the first evidence** that a generalist reconstruction foundation model can be successfully applied to and fundamentally reshape fast CMR imaging paradigms, **revealing substantial untapped acceleration potential and strong heterogeneous generalization**. 

***Outlook:*** We anticipate that **CardioMM will become a foundational component of next-generation CMR workflows, enabling fast, consistent, and clinically accessible image reconstruction and analysis across heterogeneous imaging environments**. More broadly, this study outlines a clear direction for developing clinically deployable and reliable reconstruction foundation models, charting a decisive step toward the real-world integration of generalist models in medical imaging.

The preprint paper can be seen at (coming soon after the Christmas break ...).

Email: Dr. Zi Wang (zi.wang@imperial.ac.uk); Dr. Chengyan Wang (wangcy@fudan.edu.cn) 


## MMCMR-427K database (still under construction, coming soon after the Christmas break ...)
All clinical CMR datasets from our collection are publicly available at (coming soon after the Christmas break ...). Besides, all used public datasets are available on their websites, including https://github.com/CmrxRecon, https://ocmr.info, and https://www.ukbiobank.ac.uk. For UK Biobank, the imaging data and non-imaging participant characteristics are available to approved researchers via a standard application process at http://www.ukbiobank.ac.uk/register-apply. 

## CardioMM framework (still under construction, coming soon after the Christmas break ...)
The training, testing, visualization, and automated analysis codes of CardioMM framework are released here. Besides, we also share the implementation of the conventional SENSE method to facilitate the integration of more conventional reconstruction techniques within this pipeline, including both parallel imaging and compressed sensing approaches.

Install running environment, see:
```
Environment_install.txt
Remember to download the necessities in /clip-vit-base-patch16. 
```
Run the main code for training subset re-organization, follow:
```
/CardioMM_reconstruction/preparescript_CMRxReconAll.sh
```
Run the main code for testing subset undersampling, run MATLAB script (We also welcome code contributions, especially re-implementations of this process in Python):
```
/CardioMM_dataundersampling/MaskGeneration_TestAnalysisSet_Fast.m
```
Run the main code for training, follow:
```
/CardioMM_reconstruction/trainscript_CMRxReconAll.sh
```
Run the main code for reconstruction, follow:
```
/CardioMM_reconstruction/reconscript_CMRxReconAll.sh
```
Run the main code for evaluation and visualization, follow:
```
/CardioMM_reconstruction/evaluatescript_CMRxReconAll.sh
```
Run the main code for automated analysis, see Python scripts in:
```
/CardioMM_automatedphenotyping
Remember to download and configure the necessities in /nnUNet_related/nnUNet_results first. 
```

Python environment: python=3.8, pytorch=2.0.1, pytorch_lightning=1.9.0

Implementation tips: If you want to test on your own collected data, they should be stored in the same format as the data we provided.

Note: The software is used for academic only, and cannot be used commercially.


## Citation
**If you want to use the database and codes, please cite the following paper:<br>**
**Zi Wang et al., Enabling ultra-fast cardiovascular imaging across heterogeneous clinical environments with a generalist foundation model and multimodal database, arXiv preprint, 2025.**

You are also welcome to cite these highly related papers:<br>
1.  Zi Wang et al., CMRxRecon2024: A multimodality, multiview k-space dataset boosting universal machine learning for accelerated cardiac MRI, ***Radiology: Artificial Intelligence***, 7: e240443, 2025.<br>
2.  Fanwen Wang et al., Towards modality- and sampling-universal learning strategies for accelerating cardiovascular imaging: Summary of the CMRxRecon2024 challenge, ***IEEE Transactions on Medical Imaging***, DOI: 10.1109/TMI.2025.3641610, 2025.<br>
3.  Chengyan Wang et al., CMRxRecon: A publicly available k-space dataset and benchmark to advance deep learning for cardiac MRI, ***Scientific Data***, 11: 687, 2024.<br>
4.  Jun Lyu et al., The state-of-the-art in cardiac MRI reconstruction: Results of the CMRxRecon challenge in MICCAI 2023, ***Medical Image Analysis***, 101: 103485, 2025.


## Acknowledgement
The authors thank Drs. Jure Zbontar, Anuroop Sriram, Bingyu Xin, Ilya Sutskever, Fabian Isensee, and Devran Ugurlu for sharing their codes online. 


![Visitors](https://visitor-badge.laobi.icu/badge?page_id=<wangziblake>.<CardioMM_MMCMR-427K>)
