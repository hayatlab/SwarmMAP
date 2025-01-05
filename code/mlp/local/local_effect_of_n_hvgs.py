import os
from unittest import result
# Define project directory
proj_dir = '/rwthfs/rz/cluster/home/zd775156/sclabel'

## Load Packages
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
# Computation 
import pandas as pd
import numpy as np
import scipy as sp
import scanpy as sc
import scvi
# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import argparse
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*Reordering categories will always return a new Categorical object.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*is_categorical_dtype is deprecated and will be removed in a future version.*")

# GPU usage
usecuda = torch.cuda.is_available()
device = torch.device("cuda" if usecuda else "cpu")

# Experiment
n_epochs = 100
batch_size = 256

parser = argparse.ArgumentParser()

parser.add_argument("--representation", nargs="?", default="counts", required=True,
                    type=str,
                    choices=["counts", "pca", "scVI"])
parser.add_argument("--label", nargs="?", default="cell_type", required=True,
                    type=str,
                    choices=["cell_type", "cell_subtype"])
parser.add_argument("--use_union_hvgs", nargs="?", required=True,
                    type=bool,
                    default="True", choices=[True, False])
parser.add_argument("--organ", nargs="?", default="heart", required=True,
                    type=str,
                    choices=["heart", "breast", "lung"])
parser.add_argument("--n_hvgs", nargs="?", required=True,
                    type=int,
                    choices=[200, 500, 1000, 2000])
args, unknown = parser.parse_known_args()

representation = args.representation
label = args.label
use_union_hvgs = False
organ = args.organ
n_hvgs = args.n_hvgs

result_dir, data_dir, adata, studies = mlp.load_dataset(organ, use_union_hvgs=False, filtering=True, renorm=True)

## Compute HVGs rank
adata.var["highly_variable_rank"] = adata.var["highly_variable"].astype(bool)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")

hvgs_ls = adata.var["highly_variable_rank"].astype(int)

gene_list = hvgs_ls[hvgs_ls <= n_hvgs - 1].index

## Subset HVGs
adata = adata[:, gene_list]

sim_setting_df = mlp.create_sim_setting(len(studies))
sim_setting_df["representation"] = representation
sim_setting_df["label"] = label
sim_setting_df["use_union_hvgs"] = use_union_hvgs
sim_setting_df["organ"] = organ
sim_setting_df["n_hvgs"] = n_hvgs

## Initiate empty lists to store results
y_pred_ls = []
y_test_ls = []
training_time_ls = []

start_time = time.time()
for ind_design in range(sim_setting_df.shape[0]):
# for ind_design in range(2):

    test_ind = sim_setting_df.test_ind.iloc[ind_design]
    train_ind = sim_setting_df.train_ind.iloc[ind_design]

    train = adata[adata.obs['study'].isin([studies[ind] for ind in train_ind])].copy()
    test = adata[adata.obs['study'] == studies[sim_setting_df.test_ind.iloc[ind_design]]].copy()

    train_ct = train.obs[label].unique()
    test_ct = test.obs[label].unique()
    set(train_ct) == set(test_ct)

    if representation == "counts":
        X_test = test.X.toarray()
        X_train = train.X.toarray()
    elif representation == "pca":
        X_test = test.obsm["X_pca"]
        X_train = train.obsm["X_pca"]
    elif representation == "scVI":
        X_test = test.obsm["X_scVI"]
        X_train = train.obsm["X_scVI"]

    y_test = test.obs[label]
    y_train = train.obs[label]

    # Apply OHE
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder_data = pd.DataFrame(sorted(train.obs[label].unique())).values.reshape(-1, 1)
    ohe.fit(encoder_data)
    y_test_transformed = ohe.transform(y_test.values.reshape(-1, 1))
    y_train_transformed = ohe.transform(y_train.values.reshape(-1, 1))

    ## Convert data to tensor
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test_transformed = torch.tensor(y_test_transformed, dtype=torch.float32)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train_transformed = torch.tensor(y_train_transformed, dtype=torch.float32)

    # Train model
    train_start = time.time()
    model = mlp.train_mlp(
        X_train, y_train_transformed, device, 
        n_epochs, batch_size)
    training_time = time.time() - train_start

    # Test model
    y_pred = mlp.test_mlp(model, X_test, y_test_transformed, device, ohe)
    y_test_transformed = y_test_transformed.cpu().detach().numpy()
    y_pred_labels = np.take(ohe.categories_, np.argmax(y_pred, axis=1))

    y_pred_ls.append(y_pred_labels.tolist())
    y_test_ls.append(y_test.tolist())
    training_time_ls.append(training_time)
    
    print(f"Design {ind_design} done.")

print("--- %s seconds ---" % (time.time() - start_time))

## Save sim_setting_df
sim_setting_df["y_pred"] = y_pred_ls
sim_setting_df["y_test"] = y_test_ls
sim_setting_df["training_time"] = training_time_ls

sim_setting_df.to_csv(result_dir+f"results_{representation}_{label}_n_hvgs_{n_hvgs}.csv", index=False)
print("Saved file: ",result_dir+f"results_{representation}_{label}_n_hvgs_{n_hvgs}.csv")

# ## Load all 3 sim_setting_df if they exist and concatenate them
# # Read all files in the directory result_dir matching the regex "results_*.csv"
# file_list = glob.glob(result_dir + "results_*.csv")
# ## Remove element from file_list if it contains the string "all"
# file_list = [file for file in file_list if "all" not in file]
# for elem in file_list:
#     print(elem)

# # Iterate over the file list and concatenate the dataframes
# dfs = []
# for file in file_list:
#     df = pd.read_csv(file)
#     df["representation"] = file.split("_")[-3].split(".")[0]
#     print(file.split("_")[-3].split(".")[0])
#     dfs.append(df)
    

# # Concatenate the dataframes into a single dataframe
# sim_setting_df = pd.concat(dfs, ignore_index=True)

# # Convert the "y_pred" and "y_test" columns from string to list
# sim_setting_df["y_pred"] = sim_setting_df["y_pred"].apply(eval)
# sim_setting_df["y_test"] = sim_setting_df["y_test"].apply(eval)

# ## Save sim_setting_df
# sim_setting_df.to_csv(result_dir+"results_all.csv", index=False)
# print("Saved file: ", result_dir+"results_all.csv")

# # ## add one column to old files
# old_files = [
#     "results_counts_cell_type.csv",
#     "results_pca_cell_type.csv",
#     "results_scVI_cell_type.csv"
# ]

# for file in old_files:
#     df = pd.read_csv(result_dir+file)
#     df["label"] = "cell_type"
#     df.to_csv(result_dir+file, index=False)
#     print("Saved file: ", result_dir+file)


