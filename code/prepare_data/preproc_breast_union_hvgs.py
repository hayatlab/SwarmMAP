# General utility
import scipy as sp
import pickle
from tkinter import font
import seaborn as sn
import math
from scipy.stats import median_abs_deviation
import os
import glob
# Computation
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import json
import scvi
# Graphics
import matplotlib.pyplot as plt
# import venn
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys  
import warnings
import pickle as pkl

proj_dir = '/rwthfs/rz/cluster/home/zd775156/sclabel/'
sys.path.append(os.path.abspath(proj_dir+"/code/utils"))
import mlp

warnings.filterwarnings("ignore", category=FutureWarning, message=r".*Reordering categories will always return a new Categorical object.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*is_categorical_dtype is deprecated and will be removed in a future version.*")

# Define project directory
data_dir = '/rwthfs/rz/cluster/work/zd775156/breast/'
# raw_data_dir = "/hpcwork/zd775156/heart_atlas/"


# -------------------------------------------
# 1. Subset studies -------------------------
# -------------------------------------------

## Find the datasets to use
# adata = sc.read_h5ad(proj_dir+"data/breast/c769bb55-b211-46a3-9b5c-7ee6e3a697d1.h5ad", 
#                      backed="r")
# obs_df = adata.obs.copy()
# ## Close connection to adata
# adata.file.close()
# obs_df.columns
# for col in ["dataset", "cell_type", "batch", "tissue", "suspension_type", "organism", "is_primary_data", "disease"]:
#     print(obs_df[col].value_counts())
    
# ## Get the datasets with the highest number of cells which are primary data
# studies_to_use = obs_df[obs_df["is_primary_data"]].groupby("dataset").size().sort_values(ascending=False).index[:4].tolist()
# save studies_to_use as txt
### studies_to_use = ['Murrow', 'Nee', 'Pal', 'Twigger']
# with open(data_dir+"studies_to_use.txt", "w") as f:
#     f.write("\n".join(studies_to_use))

# Load Subset_4 data
adata = sc.read_h5ad(data_dir+"breast_subset_4.h5ad")

# Rename dataset.obs.dataset to dataset.obs.study
adata.obs.rename(columns={'dataset': 'study'}, inplace=True)

studies = sorted(adata.obs["study"].unique())

# -------------------------------------------
# 2. Filter tissue --------------------------
# -------------------------------------------

assert adata.obs['tissue'].nunique() == 1

# -------------------------------------------
# Independently for each study  -------------
# -------------------------------------------

adatas = [None] * len(studies)
gene_ls = [None] * len(studies)

for ind in range(len(studies)):

    adata_study = adata[adata.obs['study'] == studies[ind]].copy()

    assert adata_study.obs['study'].nunique() == 1

    # -------------------------------------------
    # 3. Set raw counts as expression -----------
    # -------------------------------------------

    # Use raw counts as expression matrix X
    adata_study.X[0:10, 0:10].toarray()  # not raw counts
    adata_study.raw.X[0:10, 0:10].toarray()  # raw counts
    # adata_study.X = adata_study.raw.X
    # Set raw counts as expression matrix X
    adata_study = adata_study.raw.to_adata()

    # -------------------------------------------
    # 4. QC: Filter out low quality cells -------
    # -------------------------------------------

    # mitochondrial genes
    adata_study.var["mt"] = adata_study.var_names.str.startswith("MT-")
    # ribosomal genes
    adata_study.var["ribo"] = adata_study.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes.
    adata_study.var["hb"] = adata_study.var_names.str.contains(("^HB[^(P)]"))
    sc.pp.calculate_qc_metrics(
        adata_study, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True)

    p1 = sns.displot(adata_study.obs["total_counts"], bins=100, kde=False)
    p2 = sc.pl.violin(adata_study, "pct_counts_mt")
    p3 = sc.pl.scatter(adata_study, "total_counts",
                       "n_genes_by_counts", color="pct_counts_mt")
    
    # p3 = sc.pl.scatter(adata_study, "total_counts","n_genes_by_counts")
    # # adata_study.obs
    # # tmp = adata_study.obs[["total_counts", "n_genes_by_counts", "pct_counts_mt"]].copy()
    # # ## filter total_counts smaller than 50000
    # # tmp = tmp[tmp["total_counts"] < 50000]
    # tmp_adata = adata_study[adata_study.obs["total_counts"] < 50000].copy()
    # p3 = sc.pl.scatter(tmp_adata, "total_counts", "n_genes_by_counts")
    # tmp_adata2 = adata_study[adata_study.obs["total_counts"] < 20000].copy()
    # p3 = sc.pl.scatter(tmp_adata2, "total_counts", "n_genes_by_counts")

    adata_study.obs["outlier"] = (
        mlp.is_outlier(adata_study, "log1p_total_counts", 5)
        | mlp.is_outlier(adata_study, "log1p_n_genes_by_counts", 5)
        | mlp.is_outlier(adata_study, "pct_counts_in_top_20_genes", 5))

    print("Number of outlier cells based on Total counts, N Genes by count, and Pct counts in top 20 genes: " +
          str(adata_study.obs["outlier"].sum()))
    print("These cells are removed.")

    adata_study.obs["mt_outlier"] = mlp.is_outlier(adata_study, "pct_counts_mt", 3) | (
        adata_study.obs["pct_counts_mt"] > 8)
    adata_study.obs.mt_outlier.value_counts()

    print("Number of outlier cells based on Mitochondrial gene expr.: " +
          str(adata_study.obs["mt_outlier"].sum()))
    print("These cells are removed.")

    print(f"Total number of cells: {adata_study.n_obs}")
    adata_study = adata_study[(~adata_study.obs.outlier) & (~adata_study.obs.mt_outlier)].copy()
    print(
        f"Number of cells after filtering of low quality cells: {adata_study.n_obs}")

    # -------------------------------------------
    # 5. Compute 2000 highly variable genes -----
    # -------------------------------------------
    #  Using the seurat_v3 method
    sc.pp.highly_variable_genes(adata_study, n_top_genes=2000, flavor='seurat_v3')
    gene_ls[ind] = adata_study.var_names[adata_study.var['highly_variable']]

    # Save adata_study
    adatas[ind] = adata_study

# -------------------------------------------
# 6. Take the union of HVGs -----------------
# -------------------------------------------

gene_ls = [set(ls) for ls in gene_ls]
gene_union = gene_ls[0].union(*gene_ls[1:])
gene_union = list(gene_union)
# Save gene_ls and gene_union to one pickle file
with open(os.path.join(data_dir, "gene_ls_union_hvgs.pkl"), "wb") as f:
    pkl.dump([gene_ls, gene_union], f)

for ind in range(len(studies)):
    adata_study = adatas[ind]

    # -------------------------------------------
    # 7. Subset to union of HVGs ----------------
    # -------------------------------------------
    adata_study = adata_study[:, gene_union].copy()

    # -------------------------------------------
    # 8. Normalize the expression data
    # -------------------------------------------

    adata_study.X[0:10, 0:10].toarray()
    
    print(studies[ind])
    tmp = sp.sparse.csr_matrix(adata_study.X[0:10, ]).sum(axis=1)
    print("Sum of counts before normalization: " + str(tmp.min()) + ", "+str(tmp.max()))
    sc.pp.normalize_total(adata_study, target_sum=1e4)
    tmp = sp.sparse.csr_matrix(adata_study.X[0:10, ]).sum(axis=1)
    print("Sum of counts before normalization: " +
          str(tmp.min()) + ", "+str(tmp.max()) + "\n")
    
    sc.pp.log1p(adata_study)

    # Save if to adatas object
    adatas[ind] = adata_study

# Check that 10.000 normalization is done correctly
for studies in adata.obs["study"].unique():
    adata_tmp = adata[adata.obs["study"] == studies].copy()
    tmp = sp.sparse.csr_matrix(np.expm1(adata_tmp.X)).sum(axis=1)
    print(studies + ": min: " + str(tmp.min()), "max:" + str(tmp.max()))
    
# -------------------------------------------
# 9. Merge adatas ---------------------------
# -------------------------------------------
adatas_dict = dict(zip(studies, adatas))
adata = ad.concat(adatas_dict, label='study')
del adatas_dict

# -------------------------------------------
# 10. Rename cell_type to cell_subtype ------
# -------------------------------------------

adata.obs["cell_subtype"] = adata.obs["cell_type"]

# -------------------------------------------
# 11. Cell subtype filtering ----------------
# -------------------------------------------

csts_to_rm_sf = mlp.rare_cell_types(adata, "cell_subtype", "soft")
csts_to_rm_hf = mlp.rare_cell_types(adata, "cell_subtype", "hard")

# Filter out rare cell subtypes using hard filtering
adata = adata[~adata.obs["cell_subtype"].isin(csts_to_rm_hf)].copy()

## Remove all rows which have cell_type == "unknown"
adata = adata[adata.obs["cell_type"] != "unknown"].copy()

# -------------------------------------------
# 12. Defining main cell types --------------
# -------------------------------------------

ct = sorted(adata.obs["cell_subtype"].unique().tolist())
print(len(ct))

with open("/home/zd775156/sclabel/code/utils/cell_ontology.pkl", "rb") as f:
    graph = pkl.load(f)
node_id = list(graph.nodes())
id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
name_to_id = {v: k for k, v in id_to_name.items()}

# Get all nodes in the graph
node_id = list(graph.nodes())
id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
node_name = [id_to_name[x] for x in node_id]
name_to_id = {v: k for k, v in id_to_name.items()}

# Define sets of cell types to combine
ct_to_combine = [
    ["effector memory CD8-positive, alpha-beta T cell", "CD8-positive, alpha-beta memory T cell",
        "mature NK T cell", "CD4-positive helper T cell", "Tc1 cell"],
    ["unswitched memory B cell", "naive B cell", "class switched memory B cell"],
    ["vein endothelial cell", "endothelial cell of artery"],
    ["luminal adaptive secretory precursor cell of mammary gland",
        "mammary gland epithelial cell"]
]

# Get the common ancestor
ct_common_ancestor = [mlp.get_lowest_common_ancestor(
    graph, x, id_to_name, name_to_id) for x in ct_to_combine]
# Unpack ct_to_combine into a single list
ct_to_combine_unpacked = [x for sublist in ct_to_combine for x in sublist]

ct_to_combine_len = [len(x) for x in ct_to_combine]
# Unpack ct_common_ancestor by repeating its values n times, where n in given by ct_to_combine_len
ct_common_ancestor_unpacked = [x for x, n in zip(
    ct_common_ancestor, ct_to_combine_len) for _ in range(n)]

# Create dict from ct_to_combine_unpacked and ct_common_ancestor
ct_dict = dict(zip(ct_to_combine_unpacked, ct_common_ancestor_unpacked))

# Define a new variable "main_cell_type" in adata.obs by mapping the cell types to the main cell types
adata.obs["cell_type"] = adata.obs["cell_subtype"].map(
    ct_dict).fillna(adata.obs["cell_subtype"]).astype("category")

adata.obs["cell_type"].nunique()
adata.obs["cell_subtype"].nunique()

# Recheck that 10.000 normalization is done correctly
for studies in adata.obs["study"].unique():
    adata_tmp = adata[adata.obs["study"] == studies].copy()
    tmp = sp.sparse.csr_matrix(np.expm1(adata_tmp.X)).sum(axis=1)
    print(studies + ": min: " + str(tmp.min()), "max:" + str(tmp.max()))

# Save adata
# adata.write(data_dir+"breast_clean_subset_4_union_hvgs.h5ad")
adata.write(data_dir+"breast_clean_subset_4_union_hvgs_renorm.h5ad")
# print("Saved adata at file: "+data_dir+"breast_clean_subset_4_union_hvgs.h5ad")

adata = sc.read_h5ad(data_dir+"breast_clean_subset_4_union_hvgs_renorm.h5ad")

# _, _, adata, _ = mlp.load_dataset('lung', use_union_hvgs=True,
#                                   filtering=False, label="cell_type", renorm=True)

# for studies in adata.obs["study"].unique():
#     adata_tmp = adata[adata.obs["study"] == studies].copy()
#     tmp = sp.sparse.csr_matrix(np.expm1(adata_tmp.X)).sum(axis=1)
#     print(studies + ": min: " + str(tmp.min()), "max:" + str(tmp.max()))


# -------------------------------------------
# Compute PCA and scVI representations ------
# -------------------------------------------
n_comps = 100
import sys
sys.path.append(os.path.abspath(proj_dir+"/code/utils"))
import mlp
adata.obsm["X_pca"] = mlp.compute_representation_separately(adata, "pca", "study", n_comps)
adata.uns["pca_indep"] = True

adata.obsm["X_scVI"] = mlp.compute_representation_separately(adata, "scVI", "study", n_comps)
adata.uns["scVI_indep"] = True

# Save adata
adata.write(data_dir+"breast_clean_subset_4_union_hvgs.h5ad")
print("Saved adata at file: "+data_dir+"breast_clean_subset_4_union_hvgs.h5ad")


# -------------------------------------------
#  Save soft and hard filtering -------------
# -------------------------------------------

organ = "breast"
# Initiate dict
cts_to_rm_union_hvgs = dict.fromkeys(
    ["cts_hf", "cts_sf", "csts_hf", "csts_sf"])

# Populate dict
for label, dict_key in zip(["cell_type", "cell_subtype"], ["cts", "csts"]):
    _, data_dir, adata, _ = mlp.load_dataset(organ, use_union_hvgs=True,
                                             filtering=False, label=label)
    cts_to_rm_union_hvgs[dict_key+"_sf"] = list(mlp.rare_cell_types(
        adata, label=label, filtering_type="soft"))
    cts_to_rm_union_hvgs[dict_key+"_hf"] = list(mlp.rare_cell_types(
        adata, label=label, filtering_type="hard"))

print(cts_to_rm_union_hvgs)

# Save dict
with open(data_dir+"cell_types_to_rm_union_hvgs.json", "w") as f:
    json.dump(cts_to_rm_union_hvgs, f)


# -------------------------------------------
# Plot dendrograms --------------------------
# -------------------------------------------

# result_dir = "/home/zd775156/sclabel/results/local/dendrograms/"

# # Compute dendrogram of cell subtypes
# if "dendrogram_cell_type" in adata.uns:
#     del adata.uns["dendrogram_cell_type"]
# sc.tl.dendrogram(adata, groupby="cell_subtype", use_rep="X_scvi_integrated")

# # Plot dendogram of cell types
# fig, ax = plt.subplots()
# ax.set_title(
#     f"Breast -- cell subtypes ({adata.obs['cell_subtype'].nunique()} types)")
# sc.pl.dendrogram(adata, groupby="cell_subtype", orientation="left", ax=ax)
# fig.tight_layout()
# # Save figue
# fig.savefig(result_dir+"breast_cell_subtype.png")

# # Compute dendrogram of main cell types
# if "dendrogram_main_cell_type" in adata.uns:
#     del adata.uns["dendrogram_main_cell_type"]
# sc.tl.dendrogram(adata, groupby="cell_type", use_rep="X_scvi_integrated")

# # Plot dendogram of main cell types
# fig, ax = plt.subplots()
# ax.set_title(
#     f"Breast -- cell main types ({adata.obs['main_cell_type'].nunique()} types)")
# sc.pl.dendrogram(adata, groupby="main_cell_type", orientation="left", ax=ax)
# fig.tight_layout()
# # Save figure
# fig.savefig(result_dir+"breast_cell_main_type.png")



