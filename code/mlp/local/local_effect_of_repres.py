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
import argparse
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*Reordering categories will always return a new Categorical object.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*is_categorical_dtype is deprecated and will be removed in a future version.*")

result_dir = "/home/zd775156/sclabel/results/local/effect_of_repres/"

# GPU usage
usecuda = torch.cuda.is_available()
device = torch.device("cuda" if usecuda else "cpu")

# Experiment
n_epochs = 100
batch_size = 256

## Read arguments --------------------------------------------------------------
parser = argparse.ArgumentParser(description='Process representation and number of components.')
parser.add_argument('--representation', type=str, choices=['counts', 'pca', 'scVI'], 
                    default="counts",   
                    help='Type of representation to use: counts, pca, or scVI')
parser.add_argument('--n_comps', type=int, help='Number of components to use',
                    default="50")
parser.add_argument('--organ', type=str, help='Organ to use',
                    default="heart")

args = parser.parse_args()
representation = args.representation
n_comps = args.n_comps
organ = args.organ

## Load data -------------------------------------------------------------------
_, data_dir, adata, studies = mlp.load_dataset(organ, use_union_hvgs=False, filtering=True, renorm=True)

## Compute PCA and scVI representation separately for each study ----------------
match representation:
    case "pca":
        if "pca_indep" not in adata.uns:
            adata.obsm["X_pca"] = mlp.compute_representation_separately(adata, "pca", "study", n_comps)
            adata.uns["pca_indep"] = True
    case "scVI":
        if "scVI_indep" not in adata.uns:
            adata.obsm["X_scVI"] = mlp.compute_representation_separately(adata, "scVI", "study", n_comps)
            adata.uns["scVI_indep"] = True
            
# Save adata
# adata.write(data_dir + "breast_clean_subset_4.h5ad")

## Subset representation with n_comps components
if representation != "counts":
    adata.obsm[f"X_{representation}"] = adata.obsm[f"X_{representation}"][:, :n_comps]

sim_setting_df = mlp.create_sim_setting(len(studies))
sim_setting_df["representation"] = representation
sim_setting_df["n_comps"] = n_comps

## Initiate empty lists to store results
y_pred_ls = []
y_test_ls = []
training_time_ls = []

start_time = time.time()
ct_match = [None] * sim_setting_df.shape[0]

# for ind_design in range(2):
for ind_design in range(sim_setting_df.shape[0]):

    test_ind = sim_setting_df.test_ind.iloc[ind_design]
    train_ind = sim_setting_df.train_ind.iloc[ind_design]

    train = adata[adata.obs['study'].isin([studies[ind] for ind in train_ind])].copy()
    test = adata[adata.obs['study'] == studies[sim_setting_df.test_ind.iloc[ind_design]]].copy()

    train_ct = train.obs['cell_type'].unique()
    test_ct = test.obs['cell_type'].unique()
    ct_match[ind_design] = set(train_ct) == set(test_ct)

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

## Save sim_setting_df ---------------------------------------------------------
sim_setting_df["y_pred"] = y_pred_ls
sim_setting_df["y_test"] = y_test_ls
sim_setting_df["organ"] = organ
sim_setting_df["training_time"] = training_time_ls


sim_setting_df.to_csv(result_dir+f"results_{organ}_{representation}_{n_comps}.csv", index=False)
print("Saved file: ", result_dir +
      f"results_{organ}_{representation}_{n_comps}.csv")





## Post-processing: merge results -----------------------------------------------

## List all files in results_dir
res_file_list = glob.glob(result_dir + "results*.csv")

## concatenate the dataframes in res_file_list
dfs = []
for file in res_file_list:
    df = pd.read_csv(file)
    dfs.append(df)

## Concatenate the dataframes into a single dataframe
sim_setting_df = pd.concat(dfs, ignore_index=True)
sim_setting_df.shape

# Convert the "y_pred" and "y_test" columns from string to list
sim_setting_df["y_pred"] = sim_setting_df["y_pred"].apply(eval)
sim_setting_df["y_test"] = sim_setting_df["y_test"].apply(eval)

## Save sim_setting_df
sim_setting_df.to_csv(result_dir+"results_all.csv", index=False)

