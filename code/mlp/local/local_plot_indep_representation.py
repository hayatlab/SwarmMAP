import os

# Define project directory
proj_dir = '/rwthfs/rz/cluster/home/zd775156/sclabel'

# General utility
import sys
sys.path.append(os.path.abspath(proj_dir+"/code/utils"))
import mlp
import gc
import pickle
import itertools
import json
import time
import pathlib
import glob
# Graphics
from matplotlib import legend
import matplotlib.pyplot as plt
# Computation 
import pandas as pd
import numpy as np
import scipy as sp
import scanpy as sc
import scvi
# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*Reordering categories will always return a new Categorical object.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*is_categorical_dtype is deprecated and will be removed in a future version.*")

result_dir = "/rwthfs/rz/cluster/home/zd775156/sclabel/results/local/indep_umap/"
data_dir = "/hpcwork/zd775156/"
organs = ["heart", "lung", "breast"]
organ = "heart"


# ---------------------------------------------------
# --- 1. Compute UMAP representation ----------------
# ---------------------------------------------------

# studies = adata.obs["study"].cat.categories.tolist()
# adata_chaffin = adata[adata.obs["study"] == "Chaffin 2022"].copy()
# adata_chaffin.obsm["X_pca"]
# sc.pl.umap(adata_chaffin, color="cell_type", palette="tab20", title="Chaffin 2022", show=True)

for organ in organs:
    _, _, adata, studies = mlp.load_dataset(organ,
                                            use_union_hvgs=True,
                                            filtering=False,
                                            label="cell_type")

    if "X_diffmap" in adata.obsm:
        del adata.obsm["X_diffmap"]
    # adata.write(result_dir+"breast_subset_4.h5ad")

    if not os.path.exists(result_dir+"indep_representation/adata_ls.pkl"):
        
        ## Define list of adatas
        adata_ls = []
        for ind, study in enumerate(studies):
            adata_ls.append(adata[adata.obs['study'] == study].copy())
            ## Only use the top 50 PCs (to save time)
            if "X_pca" in adata_ls[ind].obsm:
                adata_ls[ind].obsm["X_pca"] = adata_ls[ind].obsm["X_pca"][:, :50]
            else:
                sc.tl.pca(adata_ls[ind], svd_solver="arpack")
                adata_ls[ind].obsm["X_pca"] = adata_ls[ind].obsm["X_pca"][:, :50]
            
        ## Run UMAPs using counts (PCA representation thereof)
        for i, study in enumerate(studies):
            if "X_diffmap" in adata_ls[i].obsm:
                del adata_ls[i].obsm["X_diffmap"]
            sc.pp.neighbors(adata_ls[i], use_rep="X_pca")
            sc.tl.umap(adata_ls[i])
            
        ## Save adata_ls with pickle
        with open(data_dir+f"{organ}_ls.pkl", "wb") as f:
            pickle.dump(adata_ls, f)


# ---------------------------------------------------
# --- 2. Plot UMAP representation colored by cell type and cell subtype ---
# ---------------------------------------------------

# for organ in organs:
        
#     ## Load adata_ls
#     with open(data_dir+f"{organ}_ls.pkl", "rb") as f:
#         adata_ls = pickle.load(f)


#     ## Define colormap manually
#     ### Define cell type labels in decreasing order of frequency
#     ct_labels = sc.concat(adata_ls, axis=0).obs["cell_type"].value_counts().sort_values(ascending=False).index.tolist()
#     ### Define colormap with the default matplotlib colormap
#     cmap = plt.cm.get_cmap("tab20", len(ct_labels))

#     ## Plot UMAP of counts for each study independently
#     fig, axs = plt.subplots(2, 2, figsize=(10, 10))
#     for i, study in enumerate(studies):
#         ax = axs[i // 2, i % 2]
#         sc.pl.umap(adata_ls[i], color="cell_type", ax=ax, show=False,
#                 #    frameon=False,
#                     palette=[cmap(ct_labels.index(x)) for x in ct_labels])
#         ax.set_title(study)
#         ax.get_legend().remove()
#     handles, labels = axs[0, 0].get_legend_handles_labels()
#     fig.suptitle("Organ: "+organ.capitalize(), fontsize=18)
#     plt.tight_layout()
#     plt.show()
#     fig.savefig(result_dir+f"{organ}_umap_counts_no_leg.png", dpi = 600)


#     # Plot UMAP of counts for each study independently
#     fig, axs = plt.subplots(2, 2, figsize=(10, 10))
#     for i, study in enumerate(studies):
#         ax = axs[i // 2, i % 2]
#         sc.pl.umap(adata_ls[i], color="cell_type", ax=ax, show=False,
#                 #    frameon=False,
#                 palette=[cmap(ct_labels.index(x)) for x in ct_labels])
#         ax.set_title(study)
#         ax.get_legend().remove()
#     handles, labels = axs[0, 0].get_legend_handles_labels()
#     if organ == "breast":
#         fig.legend(handles, labels, loc='center right',
#                 bbox_to_anchor=(1.45, 0.5), ncol=1)
#     elif organ == "heart":
#         fig.legend(handles, labels, loc='center right',
#                 bbox_to_anchor=(1.2, 0.5), ncol=1)
#     elif organ == "lung":
#         fig.legend(handles, labels, loc='center right',
#                 bbox_to_anchor=(1.39, 0.5), ncol=1)
#     fig.suptitle("Organ: "+organ.capitalize(), fontsize=18)
#     fig.tight_layout()
#     # plt.show()
#     fig.savefig(result_dir+f"{organ}_umap_counts.png",
#                 dpi=600, bbox_inches='tight')

#     ## Plot UMAPs on different plots
#     for i, study in enumerate(studies):
#         fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#         sc.pl.umap(adata_ls[i], color="cell_type", ax=ax, show=False, 
#                 #    frameon=False,
#                     palette=[cmap(ct_labels.index(x)) for x in ct_labels])
#         ax.set_title(study)
#         ax.get_legend().remove()
#         plt.tight_layout()
#         plt.show()
#         fig.savefig(result_dir+f"{organ}_umap_counts_{study}.png", dpi = 600)







## OLD
# ## Subset representation with n_comps components
# for representation in ["pca", "scVI"]:
#     adata.obsm[f"X_{representation}"] = adata.obsm[f"X_{representation}"][:, :50]

# ## Separate adata by study
# for i, study in enumerate(studies):
#     sc.pp.neighbors(adata_ls[i], use_rep="X_scVI")
#     sc.tl.umap(adata_ls[i])

# ## Plot adata scVI representation for each study independently
# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# for i, study in enumerate(studies):
#     ax = axs[i // 2, i % 2]
#     sc.pl.umap(adata_ls[i], color="cell_type", ax=ax, show=False)
#     ax.set_title(study)
#     ## remove legend from axes
#     ax.get_legend().remove()
# # Retrieve the legend
# handles, labels = axs[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.4, 0.5), ncol = 1)
# fig.suptitle(organ.capitalize()+" -- scVI representation (independent for each study)")
# plt.tight_layout()
# plt.show()
# # Save plot
# fig.savefig(result_dir+"indep_representation/"+"scVI_representation.png")

# ## Plot adata PCA representation
# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# for i, study in enumerate(studies):
#     ax = axs[i // 2, i % 2]
#     sc.pl.pca(adata_ls[i], color="cell_type", ax=ax, show=False)
#     ax.set_title(study)
#     ## remove legend from axes
#     ax.get_legend().remove()
# # Retrieve the legend
# handles, labels = axs[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.4, 0.5), ncol = 1)
# fig.suptitle("Heart -- PCA representation (independent for each study)")
# plt.tight_layout()
# plt.show()
# # Save plot
# fig.savefig(result_dir+"indep_representation/"+"pca_representation.png")

