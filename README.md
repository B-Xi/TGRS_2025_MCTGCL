MCTGCL: Mixed CNN-Transformer for Mars Hyperspectral Image Classification With Graph Contrastive Learning, TGRS, 2025.
==
[Bobo Xi](https://b-xi.github.io/), [Yun Zhang](https://ieeexplore.ieee.org/author/37087032130)[Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), Tie Zheng, Xunfeng Zhao, Haitao Xu, Changbin Xue, [Yunsong Li](https://dblp.uni-trier.de/pid/87/5840.html), and [Jocelyn Chanussot](https://jocelyn-chanussot.net/).
***

Code for the paper: [MCTGCL: Mixed CNN-Transformer for Mars Hyperspectral Image Classification With Graph Contrastive Learning](https://ieeexplore.ieee.org/document/10843260).

<div align=center><img src="/Overall.png" width="80%" height="80%"></div>
Fig. 1: The architecture of the proposed MCTGCL for HSIC.

## Abstract
Hyperspectral image (HSI) classification has been extensively studied in the context of Earth observation. However, its application in Mars exploration remains limited. Although convolutional neural networks (CNNs) have proven effective in HSI processing, their local receptive fields hinder their ability to capture long-range features.  Transformers excel in global modeling and perform well in HSI classification, but they often neglect the effective representation of local spectral and spatial features and tend to be more complex. To address these challenges, we propose a mixed CNN-Transformer network for Mars HSI classification incorporating graph contrastive learning (MCTGCL) to enhance classification performance. Specifically, we introduce an information-enhanced attention module (IEAM) designed to aggregate attention features from multiple perspectives. Additionally, we develop a lightweight dual-branch CNN-Transformer (LDCT) network that efficiently extracts both local and global spectral-spatial features with lower complexity. To improve the discrimination of inter-class features, we apply graph contrastive learning to the topological structure of labeled samples. Furthermore, we annotated three Mars HSI datasets, referred to as HyMars, to validate the effectiveness of our proposed MCTGCL. Comprehensive experimental results across different amounts of labeled samples consistently demonstrate the superiority of the method. 
The source code is available at https://github.com/B-Xi/TGRS_2025_MCTGCL.

## Training and Test Process
1. Prepare the training and test data as operated in the paper.
2. Run the 'my_Nili_train.py' to reproduce the MCTGCL results on the Nili Fossae data set.

## DataSet Download: [HyMars: Mars Hyperspectral Image Classification Benchmark Datasets](https://www.scidb.cn/en/detail?dataSetId=4ff0774d45464f239a73f37796f7a786)
--
If you find this dataset helpful, please kindly cite:

[1] Bobo Xi, Yun Zhang, Jiaojiao Li, et al. HyMars: Mars Hyperspectral Image Classification Benchmark Datasets[DS/OL]. V2. Science Data Bank, 2025[2025-01-20]. https://doi.org/10.57760/sciencedb.19732. DOI:10.57760/sciencedb.19732.

## References
--
If you find this code helpful, please kindly cite:

[1] B. Xi, Y. Zhang, J. Li, T. Zheng, X. Zhao, H. Xu, C. Xue, Y. Li, and J. Chanussot, "MCTGCL: Mixed CNN-Transformer for Mars Hyperspectral Image Classification With Graph Contrastive Learning," in IEEE Transactions on Geoscience and Remote Sensing, 2025, doi: 10.1109/TGRS.2025.3529996.

Citation Details
--
BibTeX entry:
```
@ARTICLE{TGRS_2025_MCTGCL,
  author={Xi, Bobo and Zhang, Yun and Li, Jiaojiao and Zheng, Tie and Zhao, Xunfeng and Xu, Haitao and Xue, Changbin and Li, Yunsong and Chanussot, Jocelyn},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={MCTGCL: Mixed CNN-Transformer for Mars Hyperspectral Image Classification With Graph Contrastive Learning}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Transformers;Feature extraction;Mars;Minerals;Contrastive learning;Hyperspectral imaging;Convolutional neural networks;Training;Earth;Data mining;Mars exploration;information-enhanced attention;lightweight;graph contrastive learning},
  doi={10.1109/TGRS.2025.3529996}}
```
 
## Licensing
--
Copyright (C) 2025 Bobo Xi

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
