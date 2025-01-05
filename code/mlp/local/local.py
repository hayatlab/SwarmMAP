import os
# Define project directory
proj_dir = '/rwthfs/rz/cluster/home/zd775156/sclabel'


# ---------------------------------------------------------------------
## Load Packages
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# GPU usage
# ---------------------------------------------------------------------

usecuda = torch.cuda.is_available()
device = torch.device("cuda" if usecuda else "cpu")


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

n_epochs = 100
batch_size = 128

parser = argparse.ArgumentParser()
parser.add_argument("--representation", nargs="?", default="counts", choices=["counts", "pca", "scVI"])
parser.add_argument("--label", nargs="?", default="cell_type", choices=["cell_type", "cell_subtype"])
parser.add_argument("--use_union_hvgs", nargs="?",
                    default="True", choices=["True", "False"])
parser.add_argument("--organ", nargs="?", default="heart",
                    choices=["heart", "breast", "lung"])
parser.add_argument("--renorm", nargs="?", default="False",
                    choices=["True", "False"])
parser.add_argument("--new_order", nargs="?", default="False",
                    choices=["True", "False"])
args, unknown = parser.parse_known_args()

representation = args.representation
label = args.label
use_union_hvgs = args.use_union_hvgs == "True"
organ = args.organ
renorm = args.renorm == "True"
new_order= args.new_order == "True"

# label = "cell_subtype"
result_dir, data_dir, adata, studies = mlp.load_dataset(organ, use_union_hvgs,
                                                        filtering=False, label=label, 
                                                        renorm=renorm, new_order=new_order)


# ---------------------------------------------------------------------
## Start computation
# ---------------------------------------------------------------------

print("---------------------------------------------------------------------------------")
print("Start of job with parameters: ")
print("organ: ", organ)
print("label: ", label)
print("representation: ", representation)
print("use_union_hvgs: ", use_union_hvgs)
print("---------------------------------------------------------------------------------")

sim_setting_df = mlp.create_sim_setting(len(studies))
sim_setting_df["representation"] = representation
sim_setting_df["label"] = label

## ONLY USE THE LAST 4 DESIGNS
sim_setting_df = sim_setting_df.iloc[-4:]
## TODO: REMOVE EVENTUALLY

## Initiate empty lists to store results
y_pred_ls = []
y_test_ls = []
training_time_ls = []
model_ls = []
ohe_ls = []

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
    ## Load y_categories from pickle
    with open("/work/zd775156/y_categories.pkl", "rb") as f:
        y_categories = pickle.load(f)
    encoder_data = pd.DataFrame(y_categories[organ+"_"+label]).values.reshape(-1, 1)
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
    model_ls.append(model.to("cpu"))
    ohe_ls.append(ohe)
    
    ## Save torch model
    # torch.save(model, result_dir+f"<model_{ind_design}_{representation}_{label}.pt")
    
    ## Save torch model for debugging new test
    renorm_str = "_renorm" if renorm else ""
    torch.save(model, 
               f'/home/zd775156/sclabel/debugging_new_test/scratch/local/saved_model_{organ}_{label}_{ind_design}{renorm_str}.pt')
    
    print(f"Design {ind_design} done.")

print("--- %s seconds ---" % (time.time() - start_time))


# ---------------------------------------------------------------------
## Save sim_setting_df
# ---------------------------------------------------------------------

sim_setting_df["y_pred"] = y_pred_ls
sim_setting_df["y_test"] = y_test_ls
sim_setting_df["training_time"] = training_time_ls
# sim_setting_df["model"] = model_ls
sim_setting_df["ohe"] = ohe_ls
sim_setting_df["representation"] = representation
sim_setting_df["label"] = label
sim_setting_df["organ"] = organ
sim_setting_df["renorm"] = renorm

## Save sim_setting_df to pickle -------------------------------------
sim_setting_df.to_pickle(result_dir+f"results_{representation}_{label}{renorm_str}.pkl")

print("Saved file: ", result_dir +
      f"results_{representation}_{label}{renorm_str}.pkl")


# model = model.to("cpu")
# ## save model to pkl
# with open("/home/zd775156/temp_model.pkl", "wb") as f:
#     pickle.dump(model, f)
# ## Read model from pkl
# with open("/home/zd775156/temp_model.pkl", "rb") as f:
#     model = pickle.load(f)

# model_df = pd.DataFrame(model_ls)

# ## Save model_df to pickle -------------------------------------
# model_df.to_pickle("/home/zd775156/temp_model_df.pkl")
# ## Read model_df from pickle
# model_df = pd.read_pickle("/home/zd775156/temp_model_df.pkl")
