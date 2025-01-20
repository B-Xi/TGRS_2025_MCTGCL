# MCTGCL
This repo is the implementation of the following paper:

**MCTGCL: Mixed CNN-Transformer for Mars Hyperspectral Image Classification With Graph Contrastive Learning** (TGRS 2024), [[paper]](DOI:10.1109/TGRS.2025.3529996)

## Abstract
Hyperspectral image (HSI) classification has been extensively studied in the context of Earth observation. However, its application in Mars exploration remains limited. Although convolutional neural networks (CNNs) have proven effective in HSI processing, their local receptive fields hinder their ability to capture long-range features.  Transformers excel in global modeling and perform well in HSI classification, but they often neglect the effective representation of local spectral and spatial features and tend to be more complex. To address these challenges, we propose a mixed CNN-Transformer network for Mars HSI classification incorporating graph contrastive learning (MCTGCL) to enhance classification performance. Specifically, we introduce an information-enhanced attention module (IEAM) designed to aggregate attention features from multiple perspectives. Additionally, we develop a lightweight dual-branch CNN-Transformer (LDCT) network that efficiently extracts both local and global spectral-spatial features with lower complexity. To improve the discrimination of inter-class features, we apply graph contrastive learning to the topological structure of labeled samples. Furthermore, we annotated three Mars HSI datasets, referred to as HyMars, to validate the effectiveness of our proposed MCTGCL. Comprehensive experimental results across different amounts of labeled samples consistently demonstrate the superiority of the method. 
The source code is available at https://github.com/B-Xi/TGRS_2025_MCTGCL.

## Training and Test Process
1. Prepare the training and test data as operated in the paper.
2. Run the 'my_Nili_train.py' to reproduce the MCTGCL results on Nili Fossae data set.