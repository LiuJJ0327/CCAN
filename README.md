# CCAN
A cell cycle-aware network for data integration of scRNA-seq and scATAC-seq data

## Introduction
CCAN is a a Cell Cycle-Aware Network (CCAN) to remove cell cycle effects from the integrated single-cell multi-omics data while keeping the cell type-specific variations. (Figure 1).CCAN is based on a domain separation network, adding a periodic activation function to the private decoder to simulate the dynamic process of the cell cycle, and projecting single-cell data from different platforms or modalities into a common low-dimensional space through shared projection. The distribution constraint function and the class alignment loss function are added to the shared embedding space to make the distribution of different data as similar as possible and the difference between different types of data to be maximized. In addition to single-cell data integration, CCAN enables cell type prediction of scATAC-seq data via transferring the cell type annotation information of scRNA-seq data to scATAC-seq data. <br/>
![image](https://github.com/LiuJJ0327/CCAN/blob/main/images/Fig%201.png)<br/>

## Datasets<br/>
Datasets in CCAN can be downloaded from [Download here](https://ccsm.uth.edu/GUatlas/CCAN_data.zip)<br>

## Quick Start<br/>
```bash
git clone https://github.com/LiuJJ0327/CCAN.git
unzip CCAN-main.zip
cd CCAN/code/
python train_ca_mmd_classifier.py
