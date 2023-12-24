# CCAN
A cell cycle-aware network for data integration of scRNA-seq and scATAC-seq data

## Introduction
CCAN is a a Cell Cycle-Aware Network (CCAN) to remove cell cycle effects from the integrated single-cell multi-omics data while keeping the cell type-specific variations. (Figure 1). CCPE is applied to several downstream analyses and applications to demonstrate its ability to accurately estimate the cell cycle pseudotime and stages.<br/>
![image](https://github.com/LiuJJ0327/CCAN/blob/main/images/Fig%201.png)<br/>

## Quick Start<br/>
```bash
wget https://github.com/LiuJJ0327/CCPE/archive/refs/heads/main.zip
unzip CCAN-main.zip
cd CCAN/
python train_ca_mmd_classifier.py
