from rapidfuzz import fuzz, process
import pickle as pkl
import math
import harmonypy as hm
import seaborn as sns
import matplotlib.pyplot as plt
import anndata as ad
import scvi
import scanpy.external as sce
from numpy import mean, savetxt, shape, unique
import scipy as sp
import scanpy as sc
import numpy as np
import pandas as pd  # normally necessary: pandas==1.3.X
import importlib.util
import json
import os
import glob
from scipy.stats import median_abs_deviation
from unittest import result
from sklearn.cluster import AgglomerativeClustering
import sys
from ray import get
proj_dir = '/rwthfs/rz/cluster/home/zd775156/sclabel'
sys.path.append(os.path.abspath(proj_dir+"/code/utils"))
import mlp
import warnings
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*Reordering categories will always return a new Categorical object.*")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*is_categorical_dtype is deprecated and will be removed in a future version.*")

data_dir = '/rwthfs/rz/cluster/work/zd775156/heart/'
raw_data_dir = "/hpcwork/zd775156/heart/"

studies = ['Reichart 2022', 'Chaffin 2022', 'Litvinukova 2020', 'Kuppe 2022']
study_filenames = [study.replace(" ", "_").lower() for study in studies]

## Load raw adata in backed mode
adata = sc.read_h5ad(raw_data_dir+"scVI_Heart_atlas_TB_Subclustering.h5ad", backed='r')
adata.obs['Study'].unique()

## Split adata in 4 datasets: 1 per study, and save them
for ind, study in enumerate(studies):
    
    adata_study = adata[adata.obs['Study'] == study].copy(
        filename=raw_data_dir+"/raw_"+study_filenames[ind]+".h5ad")
adata.file.close()

## Find the 2000 HVGs for each study
gene_ls = []
for study in study_filenames:
    # study = study_filenames[0]
    adata = sc.read_h5ad(raw_data_dir+"/raw_"+study+".h5ad")
    
    ## Filter out tissues that are not the left ventricle
    adata = adata[adata.obs["Tissue"].isin(['Left Ventricle', 'heart left ventricle'])]
    
    ## Set raw counts as expression matrix X
    adata = adata.raw.to_adata()
    ## Remove adata.uns["log1p"] if it exists
    if "log1p" in adata.uns.keys():
        del adata.uns["log1p"]
    
    # ## Check that a random submatrix of adata.X has integer values
    # random_slice_x = np.random.randint(0, adata.X.shape[0], 100)
    # random_slice_y = np.random.randint(0, adata.X.shape[1], 100)
    # tmp = adata.X[random_slice_x, random_slice_y].A
    # # Apply function is_integer to all value of tmp
    # assert np.all(np.vectorize(lambda x: x.is_integer())(tmp))
    
    # ## Get all non-zero values from adata.X
    # non_zero_values = adata.X[adata.X.nonzero()].A.flatten()
    # ## Get the number of unique non-zero values, including Inf and NaN
    # unique_non_zero_values = np.unique(non_zero_values)
    # print(min(unique_non_zero_values), max(unique_non_zero_values))
    # ## Check whether there are infinite values in non_zero_values
    # assert np.isinf(unique_non_zero_values).sum() == 0
    
    ## Sum of values of adata.X.toarray() per row
    row_sum = adata.X.sum(axis=1).A.flatten()
    
    ## Select 2000 highly variable genes using the seurat_v3 method
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3', subset=False)
    gene_ls.append(adata.var_names[adata.var['highly_variable']])
    
    adata.write(raw_data_dir+"/temp_"+study+".h5ad")

# Take the union of HVGs
gene_ls = [set(ls) for ls in gene_ls]
gene_union = gene_ls[0].union(*gene_ls[1:])
gene_union = list(gene_union)

## Subset each study to the union of HVGs and normalize them
for ind, study in enumerate(study_filenames):
    adata = sc.read_h5ad(raw_data_dir+"/temp_"+study+".h5ad")
    
    # Remove adata.uns["log1p"] if it exists
    if "log1p" in adata.uns.keys():
        del adata.uns["log1p"]
    
    ## Subset to union of HVGs
    adata = adata[:, gene_union].copy()
    
    assert adata.var_names.tolist() == gene_union
    
    # Normalize and log1p transform data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Save dataset:
    adata.write(raw_data_dir+"/temp_"+study+".h5ad")

## Concatenate adatas
adata = ad.concat([sc.read_h5ad(raw_data_dir+"/temp_"+study+".h5ad")
                  for study in study_filenames], label='study')
assert adata.shape[1] == 3516

## Rename obs columns using pd.DataFrame.rename
del adata.obs["study"]
adata.obs = adata.obs.rename(columns={"Study": "study"})
adata.obs = adata.obs.rename(columns={"Annotation_1": "cell_type"})

## rename Tissue to tissue, Study to study, Annotation_1 to cell_type
# if "tissue" not in adata.obs.columns and "study" not in adata.obs.columns and "cell_type" not in adata.obs.columns:
    # adata.obs = adata.obs.rename(columns={"Tissue": "tissue", "Study": "study", "Annotation_1": "cell_type"})
# End rename obs columns

# Save final dataset for cell_type
# adata.write(data_dir+"tmp_heart_clean_subset_4_union_hvgs.h5ad")
# adata.write(data_dir+"heart_clean_subset_4_union_hvgs_cell_type.h5ad")


# -------------------------------------------
# Defining cell main types and subtypes -----
# -------------------------------------------

# -------------------------------------------
# Option1: Define cell subtypes here -----
# -------------------------------------------

# Load graph object
with open("/home/zd775156/sclabel/code/utils/cell_ontology.pkl", "rb") as f:
    graph = pkl.load(f)
node_id = list(graph.nodes())
id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
name_to_id = {v: k for k, v in id_to_name.items()}
node_name = [data.get('name') for id_, data in graph.nodes(data=True)]

# ## Relaod adata
result_dir, data_dir, adata, studies = mlp.load_dataset("heart", use_union_hvgs=True, filtering=False, label="cell_type")

## Define cell subtypes 
cell_subtype_dict = {'Adipocytes #1': 'adipocyte',
                     'Adipocytes #2': 'adipocyte',
                     'B cells': 'B cell',
                     'Cardiomyocytes #1': 'cardiac muscle cell',
                     'Cardiomyocytes #2': 'cardiac muscle cell',
                     'Cardiomyocytes (LINGO1)': 'smooth muscle cell',
                     'Cardiomyocytes (NPPA/NPPB)': 'cardiac muscle cell',
                     'Cardiomyocytes (THBS1)': 'cardiac muscle cell',
                     'Dendritic cells': 'dendritic cell',
                     'EC, Angiogenic': 'circulating angiogenic cell',
                     'EC, Aterial': 'endothelial cell',
                     'EC, Capilllary': 'endothelial cell',
                     'EC, IFN': 'endothelial cell',
                     'EC, Proliferation': 'endothelial cell',
                     'EC, Venous': 'vein endothelial cell',
                     'Endocardial (NRG1)': 'endothelial cell',
                     'Endocardial (NRG3)': 'endothelial cell',
                     'Endocardial EC': 'endothelial cell',
                     'Fibroblast (ABCA9, ACSM1)': 'fibroblast',
                     'Fibroblast (APOD/E)': 'fibroblast',
                     'Fibroblast (COL15A1, C7)': 'fibroblast',
                     'Fibroblast (PCOLCE2,SCARA5)': 'fibroblast',
                     'Fibroblast, Activated (FAP POSTN)': 'fibroblast',
                     'Fibroblast, Perivascular (PLA2G2A, THBS1)': 'fibroblast',
                     'Lymphatic EC': 'endothelial cell',
                     'Macrophage #1': 'macrophage',
                     'Macrophage #2': 'macrophage',
                     'Macrophage #3': 'macrophage',
                     'Macrophages (SPP1)': 'macrophage',
                     'Mast cells': 'mast cell',
                     'Megakaryocyte': 'megakaryocyte',
                     'Monocytes (CD14)': 'monocyte',
                     'Monocytes (CD16)': 'monocyte',
                     'Monocytes, Proliferating': 'monocyte',
                     'Myofibroblast (ACTA2, SMAD6)': 'myofibroblast cell',
                     'Myofibroblast, pro-inflammtory (FAP,CCL2)': 'fibroblast',
                     'Natural Killer (GLNY)': 'natural killer cell',
                     'Natural Killer (KLRC1)': 'natural killer cell',
                     'Neuronal': 'neural cell',
                     'Pericytes (ACTB)': 'pericyte',
                     'Pericytes (APOD)': 'pericyte',
                     'Plasma cells': 'plasma cell',
                     'Plasma cells / NK (GZMB, ACM)': 'plasma cell',
                     'Proliferating Lymphocytes': 'lymphocyte',
                     'T cells (CD4, CD40LG)': 'T cell',
                     'T cells (CD8, CCL5)': 'CD8-positive, alpha-beta regulatory T cell',
                     'T cells, regulatory (FOXP3)': 'T cell',
                     'VSMC (DSCAML1)': 'smooth muscle cell',
                     'VSMC (LGR6)': 'smooth muscle cell',
                     'VSMC (SULF1)': 'smooth muscle cell'}

## Print cell label "Subclustering" which will be converted to NaNs:
## It is the labels in adata.obs["Subclustering"] that are not in cell_subtype_dict
set(adata.obs["Subclustering"].unique()) - set(cell_subtype_dict.keys())
adata.obs["Subclustering"].nunique()

## Map Subclutering variable to a new variable: cell_subtype, using cell_subtype_dict
adata.obs["cell_subtype"] = adata.obs["Subclustering"].map(cell_subtype_dict)
adata.obs["cell_subtype"].isna().value_counts()

print("Number of cell subtypes after mapping to cell_subtype:",
        adata.obs["cell_subtype"].nunique())

adata.write(data_dir+"heart_clean_subset_4_union_hvgs_jul_16.h5ad")


# -------------------------------------------
# Option 2: Reuse cell subtypes defined -----
# for heart without union_of_jvgs -----------
# -------------------------------------------

## Con: some cell types are removed compared to 
## heart_clean_subset_4.h5ad

## Consequence: we have two datasets with different nb of cell types
# heart_clean_subset_4_union_hvgs_cell_type.h5ad 
## heart_clean_subset_4_union_hvgs_cell_subtype.h5ad 

## relaod old dataset
adata2 = sc.read(data_dir+"heart_clean_subset_4.h5ad")
adata = sc.read(data_dir+"14_jul_heart_clean_subset_4_union_hvgs.h5ad")

adata.obs["Annotation_1"].nunique()

adata2.obs["cell_type"].nunique()
adata2.obs["cell_subtype"].nunique()
adata2.obs["cell_subtype"].unique().tolist()
adata.obs["cell_type"].value_counts()

## Compare adata2.obs["cell_subtype"] and adata.obs["Cell_type"]
adata2.obs["cell_subtype"].nunique()
adata.obs["Cell_type"].nunique()

## Compute crosstab
pd.crosstab(adata2.obs["cell_subtype"], adata.obs["Cell_type"])

## Compare adata2.obs["cell_subtype"] and adata2.obs["Subclustering"]
adata2.obs["cell_subtype"].nunique()

adata.obs["Subclustering"].nunique()
adata2.obs["Subclustering"].nunique()

## Perform inner join between adata.obs and obs_df using the index as key
tmp = adata2.obs.loc[:, ["cell_subtype"]]
new_df = adata.obs.join(tmp, how="inner")

adata2.obs.shape
adata.obs.shape
adata.obs.head().index
new_df.head().index

print(
    f"Number of cells removed when using cell_subtype: {str(adata.obs.shape[0] - new_df.shape[0])}")

## Subset adata to the cells in new_df
# new = adata.obs.index.tolist()
# old = new_df.index.tolist()
# ## intersection of new and old
# intersection = list(set(new).intersection(set(old)))
# len(intersection)
# len(new)
# len(old)
# len(new) == len(set(new))

adata.obs.head().index

adata = adata[new_df.index, :].copy()

## Modify obs of new adata
adata.obs = new_df

print("Number of cell types in new adata:",
      adata.obs["cell_subtype"].nunique())

## Check that all all studies have 21 cell subtypes
## Divide adata by value of "study"
adata_ls = [adata[adata.obs["study"] == study].copy() for study in studies]
n_cell_subtypes_ls = [adata.obs["cell_subtype"].nunique() for adata in adata_ls]
assert n_cell_subtypes_ls == [21, 21, 21, 21]
del adata_ls

adata.n_obs

## Merge adata with heart_clean_susbet_4_union_hvgs, which does not have obs "cell_subtype"
result_dir, data_dir, adata_cst, studies = mlp.load_dataset("heart", True,
                                                            filtering=True, label="cell_subtype")

# Check that all cells in adata_cst are in adata
adata_cells = adata.obs.index
adata_cst_cells = adata_cst.obs.index

len(adata_cells)
len(adata_cst_cells)

assert set(adata_cst_cells).issubset(set(adata_cells))

# Get the intersection of the cells in adata and adata_cst
adata_intersec = adata[adata_cst_cells].copy()
# Check that they have the same sparsity pattern
assert adata_intersec.X.nnz == adata_cst.X.nnz


cst_df = adata_cst.obs.loc[:, ["cell_subtype"]]
## join dataframes using index as key
new_df = adata.obs.join(cst_df, how="outer")
all(new_df.columns == adata_cst.obs.columns)


adata.shape[0] == new_df.shape[0]
new_df["cell_subtype"].value_counts(dropna=False)

adata.obs = new_df

tmp = adata.obs
## remove all NaN values in cell_subtype
tmp = tmp.dropna(subset=["cell_subtype"])

adata = adata[adata.obs.dropna(subset=["cell_subtype"]).index, :].copy()

# Save file
# adata.write(data_dir+"14_jul_heart_clean_subset_4_union_hvgs.h5ad")
# adata.write(data_dir+"heart_clean_subset_4_union_hvgs.h5ad")
print("Saved file: heart_clean_subset_4_union_hvgs.h5ad")


# -------------------------------------------
# Filtering cell subtypes -------------------
# -------------------------------------------

_, _, adata, studies = mlp.load_dataset("heart", use_union_hvgs=True,
                                        filtering=False, label="cell_subtype")

## Compute rare cell types:
mlp.rare_cell_types(adata, label="cell_subtype", filtering_type="soft")
# 21 cell subtype remain after soft filtering out of 21

mlp.rare_cell_types(adata, label="cell_subtype", filtering_type="hard")
# 18 cell subtype remain after hard filtering out of 21
# {'B cell', 'lymphocyte', 'megakaryocyte'}


# -------------------------------------------
#  Save soft and hard filtering -------------
# -------------------------------------------

organ = "heart"
## Initiate dict
cts_to_rm_union_hvgs = dict.fromkeys(["cts_hf", "cts_sf", "csts_hf", "csts_sf"], [])

## Populate dict
for label, dict_key in zip(["cell_type", "cell_subtype"], ["cts", "csts"]):
    _, data_dir, adata, _ = mlp.load_dataset(organ, use_union_hvgs=True,
                                            filtering=False, label=label)
    cts_to_rm_union_hvgs[dict_key+"_sf"] = list(mlp.rare_cell_types(
        adata, label=label, filtering_type="soft"))
    cts_to_rm_union_hvgs[dict_key+"_hf"] = list(mlp.rare_cell_types(
        adata, label=label, filtering_type="hard"))
    
print(cts_to_rm_union_hvgs)

# data_dir = "/rwthfs/rz/cluster/work/zd775156/breast/"

## Save dict
import json
with open(data_dir+"cell_types_to_rm_union_hvgs.json", "w") as f:
    json.dump(cts_to_rm_union_hvgs, f)
    
# ## load dict
# with open(data_dir+"cell_types_to_rm_union_hvgs.json", "r") as f:
#     cts_to_rm_union_hvgs = json.load(f)



# -------------------------------------------
# Remove temporary files -------------------
# -------------------------------------------

for f in glob.glob(raw_data_dir+"/temp_*.h5ad"):
    os.remove(f)
    
    
## Check that the final dataset has the same number of cells as the previous iteration of the preproc pipeline
## TODO: inspect this: there are a few more cells in the new dataset than in the previous on

# tmp1 = sc.read_h5ad(data_dir+"heart_clean_subset_4_union_hvgs.h5ad")
# tmp2 = sc.read_h5ad(data_dir+"heart_clean_subset_4.h5ad")
# print("Shape of tmp1 and tmp2:")
# print(tmp1.shape)
# print(tmp2.shape)

# tmp2.obs["Tissue"].value_counts(dropna=False)

# tmp1.obs["Tissue"].value_counts(dropna=False)

# tmp1.obs.columns
# tmp1.obs["Annotation_1"].value_counts(dropna=False)
# tmp2.obs["cell_type"].value_counts(dropna=False)



