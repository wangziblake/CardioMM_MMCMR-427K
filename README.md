# A Generalist Foundation Model and Database for Accelerated Multimodal Cardiovascular Image Reconstruction and Analysis
***Motivation:*** Multimodal cardiovascular magnetic resonance (CMR) imaging offers comprehensive and non-invasive insights into cardiovascular disease (CVD) diagnosis and research, but its real-world adoption remains constrained by prolonged scan times and heterogeneity across medical environments. **This underscores the urgent need to address a largely unexplored gap: a generalist foundation model for accelerated CMR image reconstruction**—capable of adapting across diverse imaging scenarios and **serving as the essential substrate for all downstream analyses**.

***Database:*** To enable such a model, we curate **MMCMR-427K, the largest and most comprehensive multimodal CMR k-space database** to date. It comprises 427,465 multi-coil k-space data paired with structured metadata (approximately 3.5 TB), from 6,120 scans of 1,504 participants, spanning 13 worldwide centers (four public repositories and nine clinical centers), 15 scanners (four vendors from low-field to ultra-high-field strengths), 12 CMR modalities, and 17 CVD categories across three populations (Asian, European, and North American).

![MMCMR-427K](https://github.com/wangziblake/CardioMM_MMCMR-427K/blob/main/Figure/MMCMR-427K.png)

***Model:*** Leveraging this unprecedented infrastructure, we propose **CardioMM, the first generalist reconstruction foundation model capable of dynamically adapting to heterogeneous highly accelerated CMR scenarios**. CardioMM unifies semantic contextual understanding with physics-informed data consistency to deliver robust reconstructions across varied scanners, protocols, and patient presentations. It establishes a foundation for scalable, generalizable, and high-throughput multimodal cardiovascular imaging.

***Validation:*** Comprehensive evaluations demonstrate that CardioMM achieves state-of-the-art performance in the internal centers and **exhibits strong zero-shot generalization to unseen external settings**. For clinical applicability, **highly accelerated reconstructions (8×–24×) reliably preserve 11 key cardiac phenotypes, three quantitative myocardial biomarkers, and diagnostic image quality**, ensuring dependable diagnostic support for medical professionals.

![CardioMM](https://github.com/wangziblake/CardioMM_MMCMR-427K/blob/main/Figure/CardioMM.png)

***Complementarity:*** Notably, **our framework does not compete with existing CMR analytical foundation models** that operate on post-reconstruction images; rather, **it complements them by filling a critical upstream gap in the multimodal CMR computational pipeline**. By delivering higher-quality and more diverse image reconstructions, CardioMM provides a more robust and reliable foundation for downstream segmentation, phenotyping, diagnosis, and beyond.

***Engagement:*** Remarkably, in our organized CMRxRecon challenge series (2023–2025), more than 11,000 active participants from 125 countries and regions have engaged, reflecting the broad and growing global interest in next-generation CMR technologies.

***Impact:*** This work provides **the first evidence** that a generalist reconstruction foundation model can be successfully applied to and fundamentally reshape accelerated CMR paradigms, **revealing substantial untapped acceleration potential and strong heterogeneous generalization**. 

***Outlook:*** We anticipate that **CardioMM will become a foundational component of next-generation CMR workflows, enabling fast, consistent, and clinically accessible image reconstruction and analysis across heterogeneous imaging environments**. More broadly, this study outlines a clear direction for developing clinically deployable and reliable reconstruction foundation models, charting a decisive step toward the real-world integration of generalist models in medical imaging.

The preprint paper can be seen at (coming soon ...).

Email: Dr Zi Wang (zi.wang@imperial.ac.uk); Dr Chengyan Wang (wangcy@fudan.edu.cn) 


## MMCMR-427K database (still under construction ...)
Data availability: All used public datasets are available on their websites, including https://github.com/CmrxRecon, https://ocmr.info, and https://www.ukbiobank.ac.uk. For UK Biobank, the imaging data and non-imaging participant characteristics are available to approved researchers via a standard application process at http://www.ukbiobank.ac.uk/register-apply. Besides, most of the real-world CMR datasets from our collection are publicly available now. A small portion of real-world CMR datasets cannot be fully released publicly due to data agreements and privacy regulations and may be requested from the last corresponding author upon reasonable inquiry.

## CardioMM framework (still under construction ...)
The training and testing codes of CardioMM are released here.

Install conda environment:
```bash
conda env create -f environment
```
Run the main code for training subset re-organization:
```python
xxx.py
```
Run the main code for training:
```python
xxx.py
```
Run the main code for testing:
```python
xxx.py
```

Python environment: python=3.8, pytorch=2.0.1, pytorch_lightning=1.9.0

Implementation tips: If you want to test on your own collected data, they should be stored in the same format as the data we provided.

Note: The software is used for academic only, and cannot be used commercially.


## Citation
If you want to use the database and codes, please cite the following paper:

Zi Wang et al., A Generalist Foundation Model and Database for Accelerated Multimodal Cardiovascular Image Reconstruction and Analysis
, arXiv preprint, 2025.


## Acknowledgement
The authors thank Drs. Jure Zbontar, Anuroop Sriram, Bingyu Xin, Ilya Sutskever, Fabian Isensee, and Devran Ugurlu for sharing their codes online. 
