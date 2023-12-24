import os
#VIA-hema data
#hema_folder = "/data/jliu25/MyProject/data/VIA_hema/processed/"
#hema_rna_file = os.path.join(hema_folder, 'rna_hvg.csv')
#hema_atac_file = os.path.join(hema_folder, 'atac.csv')
#hema_rna_label_file = os.path.join(hema_folder, 'rna_two_labels.csv')
#hema_atac_label_file = os.path.join(hema_folder, 'label_atac.csv')
hema_folder = "/data/jliu25/MyProject/data/VIA_hema/overlapped/"
hema_rna_file = os.path.join(hema_folder, 'rna_overlapped.csv')
hema_rna_label_file = os.path.join(hema_folder, 'label_rna.csv')
hema_atac_file_1 = os.path.join(hema_folder, 'atac_6celltype_overlapped.csv')
hema_atac_label_file_1 = os.path.join(hema_folder, 'label_atac_6celltype.csv')
hema_atac_file_2 = os.path.join(hema_folder, 'atac_6celltype_withUNK_overlapped.csv')
hema_atac_label_file_2 = os.path.join(hema_folder, 'label_atac_6celltype_withUNK.csv')

#scMVP 10x_pbmc
pbmc_folder = "/data/jliu25/MyProject/data/scMVP_10x_pbmc/"
pbmc_rna_file = os.path.join(pbmc_folder, 'rna.csv')
pbmc_atac_file = os.path.join(pbmc_folder, 'atac_processed.csv')
pbmc_rna_label_file = os.path.join(pbmc_folder, 'label_rna.csv')
pbmc_atac_label_file = os.path.join(pbmc_folder, 'label_atac.csv')

#Seurat pbmc 10x
seurat_folder = "/data/jliu25/MyProject/data/seurat_pbmc_10x/"
seurat_rna_file = os.path.join(seurat_folder, 'rna_processed.csv')
seurat_atac_file = os.path.join(seurat_folder, 'atac_processed.csv')
seurat_rna_label_file = os.path.join(seurat_folder, 'cell_label.csv')
seurat_atac_label_file = os.path.join(seurat_folder, 'cell_label.csv')

#scJoint CITE-seq(RNA) & ASAP-seq(ATAC)
asap_cite_folder = "/data/jliu25/MyProject/data/scJoint_asap_cite/"
cite_file = os.path.join(asap_cite_folder, 'cite_rna.csv')
asap_file = os.path.join(asap_cite_folder, 'asap_atac_7celltypes.csv')
cite_normalized_file = os.path.join(asap_cite_folder, 'cite_rna_normalized.csv')
asap_normalized_file = os.path.join(asap_cite_folder, 'asap_atac_7celltypes_normalized.csv')
cite_label_file = os.path.join(asap_cite_folder, 'label_cite.csv')
asap_label_file = os.path.join(asap_cite_folder, 'label_asap.csv')

#harmony two scRNA-seq data (pbmc_6k & pbmc_8k)
harmony_folder = "/data/jliu25/MyProject/data/harmony_pbmc_6k_8k/"
pbmc_6k_nor_file = os.path.join(harmony_folder, 'normalized/pbmc6k_normalized.csv')
pbmc_8k_nor_file = os.path.join(harmony_folder, 'normalized/pbmc8k_normalized.csv')
pbmc_6k_label_file = os.path.join(harmony_folder, 'label_pbmc6k_id.csv')
pbmc_8k_label_file = os.path.join(harmony_folder, 'label_pbmc8k_id.csv')

#snare-seq adBrain
snare_adbrain_folder = "/data/jliu25/MyProject/data/snare_seq/snare_adbrain/CCAN_format/"
snare_adbrain_rna_file = os.path.join(snare_adbrain_folder, 'rna_processed.csv')
snare_adbrain_atac_file = os.path.join(snare_adbrain_folder, 'atac_processed.csv')
snare_adbrain_rna_label_file = os.path.join(snare_adbrain_folder, 'rna_label.csv')
snare_adbrain_atac_label_file = os.path.join(snare_adbrain_folder, 'atac_label.csv')