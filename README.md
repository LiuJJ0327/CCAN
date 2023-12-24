# CCAN
A cell cycle-aware network for data integration of scRNA-seq and scATAC-seq data

## Introduction
CCAN is a a Cell Cycle-Aware Network (CCAN) to remove cell cycle effects from the integrated single-cell multi-omics data while keeping the cell type-specific variations. (Figure 1).CCAN is based on a domain separation network, adding a periodic activation function to the private decoder to simulate the dynamic process of the cell cycle, and projecting single-cell data from different platforms or modalities into a common low-dimensional space through shared projection. The distribution constraint function and the class alignment loss function are added to the shared embedding space to make the distribution of different data as similar as possible and the difference between different types of data to be maximized. In addition to single-cell data integration, CCAN enables cell type prediction of scATAC-seq data via transferring the cell type annotation information of scRNA-seq data to scATAC-seq data. Validations based on multiple sets of data prove that CCAN can not only eliminate the batch effect between scRNA-seq data from different platforms, but also integrate paired and unpaired scRNA-seq data and scATAC-seq data well in the embedding space. Integration of unpaired data enables accurate cell type prediction for scATAC-seq data. Furthermore, CCAN can maintain cell differentiation trajectories when integrating single-cell differentiation data.<br/>
![image](https://github.com/LiuJJ0327/CCAN/blob/main/images/Fig%201.png)<br/>

## Quick Start<br/>
```bash
git clone https://github.com/LiuJJ0327/CCAN.git
unzip CCAN-main.zip
cd CCAN/code/
python train_ca_mmd_classifier.py
