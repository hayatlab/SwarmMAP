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
# Computation 
import pandas as pd
import numpy as np
import scipy as sp
import scanpy as sc
# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*Reordering categories will always return a new Categorical object.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*is_categorical_dtype is deprecated and will be removed in a future version.*")

organ = "breast"
result_dir, data_dir, adata, studies = mlp.load_dataset(organ)

cts = dict({"cell_type": sorted(adata.obs["cell_type"].unique().tolist()),
               "cell_subtype": sorted(adata.obs["cell_subtype"].unique().tolist())})


## Load result_df
result_df = pd.read_csv(result_dir+"results_all.csv")
## Subset result_df with representation = counts
result_df = result_df[result_df["representation"] == "counts"].copy()

result_df[["representation", "label"]].value_counts()

result_df["y_pred"] = result_df["y_pred"].apply(eval)
result_df["y_test"] = result_df["y_test"].apply(eval)

## Subset result_df according to the number of train studies
y_preds = dict({"cell_type": [], "cell_subtype": []})
y_tests = dict({"cell_type": [], "cell_subtype": []})
conf_mats = dict({"cell_type": [], "cell_subtype": []})
for label in ["cell_type", "cell_subtype"]:
    for train_on in range(1, 4):
        result_df_train_on = result_df[result_df["train_on_n_studies"] == train_on].copy()
        y_test_train_on = np.concatenate(result_df_train_on["y_test"].values)
        y_pred_train_on = np.concatenate(result_df_train_on["y_pred"].values)    
        y_tests[label].append(y_test_train_on)
        y_preds[label].append(y_pred_train_on)
        ## Create confusion matrix
        conf_mats[label].append(confusion_matrix(y_test_train_on, y_pred_train_on, labels = cts[label], normalize='true'))

## Plot confusion matrix as seaborn heatmap
for label in ["cell_type", "cell_subtype"]:

    fig, axes = plt.subplots(1, 3, figsize=(30, 10), sharey=True)
    plt.suptitle(organ.capitalize(), fontsize=20)
    for i, conf_mat in enumerate(conf_mats[label]):
        ax = axes[i]
        sns.heatmap(conf_mat, cmap='viridis', cbar=False,
                    xticklabels=sorted(cts[label]),
                    yticklabels=sorted(cts[label]), vmin=0, vmax=1, ax=ax)
        # Annotate most frequent values
        for j in range(conf_mat.shape[0]):
            for k in range(conf_mat.shape[1]):
                if conf_mat[j, k] > 0.1:  # Adjust the threshold as needed
                    ax.text(k + 0.5, j + 0.5, f'{conf_mat[j, k]:.2f}',
                                ha='center', va='center', color='white' if conf_mat[j, k] < 0.5 else 'black')
        ax.set_xlabel('Predicted', fontsize=15)
        if i == 0:
            ax.set_ylabel('Actual', fontsize=15)
        ax.set_title(f'Train on {i + 1} study')
        ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment='right', fontsize=10)
        ax.set_yticklabels(ax.get_xticklabels(), horizontalalignment='right', fontsize=10)
    ## Add colorbar to figure
    plt.subplots_adjust(wspace=0.05, hspace=0.1, right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(ax.collections[0], cax=cbar_ax)
    fig.show()
    ## Save plot
    fig.savefig(result_dir+f"confusion_matrix_{label}.png")


result_dir, data_dir, adata, studies = mlp.load_dataset('breast', use_union_hvgs=True, filtering=False, label="cell_type")

adata.obs[['cell_type', 'study']].value_counts().sort_values()

tmp = adata.obs[['cell_type', 'study']].value_counts().sort_index().reset_index()
## filter cell_type== "respiratory basal cell"
tmp[tmp["cell_type"] == "respiratory basal cell"]

## subset adata per study
adata_ls = [adata[adata.obs["study"] == study].copy() for study in studies]

for adata_study in adata_ls:
    print(adata_study.obs["cell_type"].value_counts())
    
    
    