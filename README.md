# 🔬 KERA: Knowledge-Enhanced Representation Learning with Retrieval-Augmented Multimodal Fusion for Survival Prediction

<b>Knowledge-Enhanced Representation Learning with Retrieval-Augmented Multimodal Fusion for Survival Prediction</b>, MICCAI 2026.
<br><em>Zeyu Zhang, Puria Azadi, Behnam Maneshgar, Ali Khajegili Mirabadi, Nikolay Alabi, Hossein Farahani, and Ali Bashashati</em></br>

## 📃 Introduction
This repository contains the official implementation of the paper: "*Knowledge-Enhanced Representation Learning with Retrieval-Augmented Multimodal Fusion for Survival Prediction*". In this work, we propose **KERA**, a **K**nowledge-**E**nhanced **R**etrieval-**A**ugmented framework integrating a pan-cancer pretraining paradigm **(ARPnet)** and a retrieval-augmented fusion module **(RAFnet)** for cancer survival prediction.

The paper has been provisionally accepted to MICCAI 2026. The manuscript and additional documentation will be publicly released after the review process is completed.

## 💻 Environment Setup
### Platform
- Linux (tested on Ubuntu 18.04.2 LTS)  
- NVIDIA GPU (tested on Nvidia GeForce RTX 3090) with CUDA 11.8

### Prerequisites
```bash
python==3.8.20
torch==2.4.1+cu118
torchvision==0.19.1+cu118
torch-scatter==2.1.2+pt24cu118
nystrom-attention==0.0.14
faiss-gpu==1.7.2
scikit-survival==0.17.2
lifelines==0.27.8
h5py==3.2.1
einops==0.8.1
timm==1.0.19
```
You can also install the required packages using:

```bash
pip install -r requirements.txt
```

## 🧬 Data Preparation
### WSI
Whole-slide pathological images are collected from the public cancer cohort [The Cancer Genome Atlas (TCGA)](https://portal.gdc.cancer.gov/). Each histopathological image is partitioned into non-overlapping 224×224 image patches at 20× magnification. The pathology foundation model [UNI](https://github.com/mahmoodlab/UNI) is then adopted to extract pathological feature embeddings.

### RNA
Gene expression data are collected from [TCGA](https://portal.gdc.cancer.gov/) and grouped into six functional categories: (1) Tumor Suppression, (2) Oncogenesis, (3) Protein Kinases, (4) Cellular Differentiation, (5) Transcription, and (6) Cytokines & Growth, following previous works including [MCAT](https://github.com/mahmoodlab/MCAT) and [CMTA](https://github.com/FT-ZHOU-ZZZ/CMTA).

### Clinical Data
Disease-specific survival (DSS) data from the [UCSC Xena](https://xenabrowser.net/datapages/) database are collected to obtain a more accurate assessment of prognostic outcomes.

## 🧪 Running Experiments
### Training-Validation Splits

- In the pan-cancer pretraining stage, an integrated pan-cancer cohort containing 24 cancer types is utilized for the pretraining of ARPnet.
- In the downstream fine-tuning stage, the pretrained ARPnet is combined with RAFnet for survival prediction. Specifically, five cancer cohorts excluded from the pan-cancer pretraining stage are utilized for downstream evaluation. Each cohort is randomly split for 5-fold cross-validation, and the data split files can be found in `KERA/splits`.

### ARPnet Pretraining
Please specify the directories for the WSI and gene embeddings, along with the related hyperparameters. The experiments can then be conducted using the following command:
```bash
python ARPnet_pretraining.py \
    --rna_embed_dir <RNA_EMBED_DIR> \
    --wsi_embed_dir <WSI_EMBED_DIR> \
    --batch_size 32 \
    --alignment_factor 0.2 \
    --wsi_dis_retent_factor 0.3 \
    --gene_dis_retent_factor 0.1
```

### RAFnet Finetuning
Please specify the directories for clinical information, cancer cohort and retrieval number, and other related hyperparameters. Experiments can then be conducted using the following command:
```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python RAFnet_finetuning.py \
    --rna_embed_dir <RNA_EMBED_DIR> \
    --wsi_embed_dir <WSI_EMBED_DIR> \
    --clinic_info_dir <CLINIC_INFO_DIR> \
    --split_file_dir <SPLIT_DIR> \
    --retrieval_count 10 \
    --cohort BLCA
```

## 🙏 Acknowledgements
This repository makes use of the following open-source projects:
- [CLIP](https://github.com/mlfoundations/open_clip)
- [KEP](https://github.com/MAGIC-AI4Med/KEP)
- [SurvPath](https://github.com/mahmoodlab/SurvPath)


## 📑 License & Citation
Citation information will be updated after the paper is officially published.