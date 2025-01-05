import os
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
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*Reordering categories will always return a new Categorical object.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*is_categorical_dtype is deprecated and will be removed in a future version.*")

# GPU usage
usecuda = torch.cuda.is_available()
device = torch.device("cuda" if usecuda else "cpu")


organ = "breast"
if organ == "breast":
    # Result_dir
    result_dir = "/home/zd775156/sclabel/results/breast/subset_4/local/"

    # Load data
    data_dir = '/rwthfs/rz/cluster/work/zd775156/breast/'
    adata = sc.read_h5ad(os.path.join(data_dir,'breast_clean_subset_4.h5ad'))
    studies = sorted(adata.obs["study"].unique())

    # No filtering, but we remove ischemic cells and epicardium
    adata = adata[~adata.obs["cell_type"].isin(["Ischemic cells (MI)", "Epicardium"])]
    
elif organ == "lung":
    # Result_dir
    result_dir = "/home/zd775156/sclabel/results/lung/subset_4/local/"

    # Load data
    data_dir = '/rwthfs/rz/cluster/work/zd775156/lung/core_subset/'
    adata = sc.read_h5ad(os.path.join(data_dir, 'lung_clean_subset_4.h5ad'))
    studies = sorted(adata.obs['study'].unique())

    # Filtering
    with open(os.path.join(data_dir, 'cell_types_to_rm.json'), 'rb') as f:
        cell_types_to_rm = json.load(f)
    adata.obs["cell_type"].nunique()
    adata = adata[~adata.obs["cell_type"].isin(cell_types_to_rm["cts_hf"])]
    adata.obs["cell_type"].nunique()

# Experiment
n_epochs = 100
batch_size = 256

## Define variable from argv: dropout_rate
dropout_rate = sys.argv[1]
if dropout_rate.isdigit():
    dropout_rate = float(dropout_rate)
else:
    raise ValueError("The dropout rate must be a float.")
if dropout_rate < 0 or dropout_rate > 1:
    raise ValueError("The dropout rate must be between 0 and 1.")

sim_setting_df = mlp.create_sim_setting(len(studies))
representation = "counts"
sim_setting_df[representation] = representation

## Initiate empty lists to store results
y_pred_ls = []
y_test_ls = []
training_time_ls = []

start_time = time.time()
for ind_design in range(sim_setting_df.shape[0]):

    test_ind = sim_setting_df.test_ind.iloc[ind_design]
    train_ind = sim_setting_df.train_ind.iloc[ind_design]

    train = adata[adata.obs['study'].isin([studies[ind] for ind in train_ind])].copy()
    test = adata[adata.obs['study'] == studies[sim_setting_df.test_ind.iloc[ind_design]]].copy()

    train_ct = train.obs['cell_type'].unique()
    test_ct = test.obs['cell_type'].unique()
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

    y_test = test.obs['cell_type']
    y_train = train.obs['cell_type']

    # Apply OHE
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder_data = pd.DataFrame(sorted(train.obs['cell_type'].unique())).values.reshape(-1, 1)
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
        n_epochs, batch_size, dropout_rate = dropout_rate, lr=1e-3 / dropout_rate, weight_decay=1e-3)
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

sim_setting_df.to_csv(result_dir+f"results_{representation}.csv", index=False)

print("--- %s seconds ---" % (time.time() - start_time))

# ## Load all 3 sim_setting_df if they exist and concatenate them
# # Read all files in the directory result_dir matching the regex "results_*.csv"
# file_list = glob.glob(result_dir + "results_*.csv")
# ## Remove element from file_list if it contains the string "all"
# file_list = [file for file in file_list if "all" not in file]

# # Iterate over the file list and concatenate the dataframes
# dfs = []
# for file in file_list:
#     df = pd.read_csv(file)
#     df["representation"] = file.split("_")[-1].split(".")[0]
#     dfs.append(df)
    

# # Concatenate the dataframes into a single dataframe
# sim_setting_df = pd.concat(dfs, ignore_index=True)

# # Convert the "y_pred" and "y_test" columns from string to list
# sim_setting_df["y_pred"] = sim_setting_df["y_pred"].apply(eval)
# sim_setting_df["y_test"] = sim_setting_df["y_test"].apply(lambda x: eval(x))

# ## Save sim_setting_df
# sim_setting_df.to_csv(result_dir+"results_all.csv", index=False)
