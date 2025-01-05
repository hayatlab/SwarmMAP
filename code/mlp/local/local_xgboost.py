import argparse
import warnings
import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import scvi
import scanpy as sc
import scipy as sp
import numpy as np
import pandas as pd
import glob
import pathlib
import time
import json
import itertools
import pickle
import gc
import sys
import os
sys.path.append(os.path.abspath("/home/zd775156/sclabel/code/utils"))
import mlp
import xgboost as xgb

# Define project directory
proj_dir = '/rwthfs/rz/cluster/home/zd775156/sclabel'
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*Reordering categories will always return a new Categorical object.*")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*is_categorical_dtype is deprecated and will be removed in a future version.*")

# GPU usage
usecuda = torch.cuda.is_available()
device = torch.device("cuda" if usecuda else "cpu")

# Experiment
n_epochs = 100
batch_size = 128

parser = argparse.ArgumentParser()
parser.add_argument("--representation", nargs="?",
                    default="counts", choices=["counts", "pca", "scVI"])
parser.add_argument("--label", nargs="?", default="cell_type",
                    choices=["cell_type", "cell_subtype"])
parser.add_argument("--use_union_hvgs", nargs="?",
                    default="True", choices=["True", "False"])
parser.add_argument("--organ", nargs="?", default="heart",
                    choices=["heart", "breast", "lung"])
args, unknown = parser.parse_known_args()

representation = args.representation
label = args.label
use_union_hvgs = args.use_union_hvgs == "True"
organ = args.organ

result_dir, data_dir, adata, studies = mlp.load_dataset(organ, use_union_hvgs,
                                                        filtering=False, label=label)
result_dir = result_dir.replace("local/", "local/xgboost/")


# Start computation

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

# Initiate empty lists to store results
y_pred_ls = []
y_test_ls = []
training_time_ls = []

ind_design = 0
# for ind_design in range(2):
for ind_design in range(sim_setting_df.shape[0]):

    test_ind = sim_setting_df.test_ind.iloc[ind_design]
    train_ind = sim_setting_df.train_ind.iloc[ind_design]

    train = adata[adata.obs['study'].isin([studies[ind] for ind in train_ind])].copy()
    test = adata[adata.obs['study'] == studies[sim_setting_df.test_ind.iloc[ind_design]]].copy()

    if representation == "counts":
        X_test = test.X.toarray()
        X_train = train.X.toarray()

    y_test = test.obs[label]
    y_train = train.obs[label]

    labels = sorted(list(set(train.obs[label].unique()).union(set(test.obs[label].unique()))))
    # Apply OHE
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder_data = pd.DataFrame(labels).values.reshape(-1, 1)
    ohe.fit(encoder_data)
    y_train_transformed = ohe.transform(y_train.values.reshape(-1, 1))

    # # Define model
    # model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8,
    #                       colsample_bytree=0.8, objective='binary:logistic', gamma=0, reg_alpha=0, reg_lambda=1)
    
    # # Train model
    # train_start = time.time()
    # model.fit(X_train, y_train_transformed)
    # training_time = time.time() - train_start

    # # Test model
    # y_pred = model.predict(X_test)
    # y_pred_labels = np.take(ohe.categories_, np.argmax(y_pred, axis=1))
    
    # ## Compute f1 score
    # f1 = f1_score(y_test, y_pred_labels, average='weighted')
    # print(f1)
    
    # Encode categorical data in y_train and y_test onto integers
    y_train_int = np.array([np.where(ohe.categories_[0] == el)[0][0] for el in y_train])
    
    ## Get which values in [0, ..., n_classes - 1] are present in y_train_int
    unique_values = np.unique(y_train_int)
    not_present_in_y_train = [el for el in range(len(ohe.categories_[0])) if el not in unique_values]
    if len(not_present_in_y_train) > 0:
        print(f"Values not present in y_train_int: {not_present_in_y_train}")
        ## Set one random value of y_train_int to 4
        y_train_int[0] = 4
    
    y_test_int = np.array([np.where(ohe.categories_[0] == el)[0][0] for el in y_test])
    
    ## Train model2
    model2 = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8,
                              colsample_bytree=0.8, objective='multi:softprob', gamma=0, reg_alpha=0, reg_lambda=1)
    train_start = time.time()
    model2.fit(X_train, y_train_int)
    training_time = time.time() - train_start
    
    ## Test model2
    y_pred_int = model2.predict(X_test)
    y_pred_labels = np.take(ohe.categories_, y_pred_int)
    
    ## Compute f1 score for model2
    f1_2 = f1_score(y_test, y_pred_labels, average='weighted')
    print(f1_2)

    y_pred_ls.append(y_pred_labels.tolist())
    y_test_ls.append(y_test.tolist())
    training_time_ls.append(training_time)
    
    print(f"Design {ind_design} done.")

## Save sim_setting_df
sim_setting_df["y_pred"] = y_pred_ls
sim_setting_df["y_test"] = y_test_ls
sim_setting_df["training_time"] = training_time_ls
sim_setting_df["representation"] = representation
sim_setting_df["label"] = label
sim_setting_df["organ"] = organ

sim_setting_df.to_csv(result_dir+f"results_{representation}_{label}_xgboost_multiclass.csv", index=False)

print("Saved file: ", result_dir+f"results_{representation}_{label}_xgboost_multiclass.csv")

# ## Load all 3 sim_setting_df if they exist and concatenate them
# # Read all files in the directory result_dir matching the regex "results_*.csv"
# file_list = glob.glob(result_dir + "results_*.csv")
# ## Remove element from file_list if it contains the string "all"
# file_list = [file for file in file_list if "all" not in file]

# # Iterate over the file list and concatenate the dataframes
# dfs = []
# for file in file_list:
#     df = pd.read_csv(file)
#     df["representation"] = file.split("/")[-1].split("_")[2]
#     dfs.append(df)

# # Concatenate the dataframes into a single dataframe
# sim_setting_df = pd.concat(dfs, ignore_index=True)

# # Convert the "y_pred" and "y_test" columns from string to list
# sim_setting_df["y_pred"] = sim_setting_df["y_pred"].apply(eval)
# sim_setting_df["y_test"] = sim_setting_df["y_test"].apply(lambda x: eval(x))

# ## Save sim_setting_df
# sim_setting_df.to_csv(result_dir+"results_all.csv", index=False)

# for el in file_list:
#     print(el)

# # ## add one column to old files
# file_list

# for file in file_list:
#     df = pd.read_csv(file)
#     file.split("/")[-1].split("_")[2]
#     prefix = file.split("local/")[0]+ "local/"
#     file_name = file.split("/")[-1].split(".")[0]
#     df["label"] = "cell_subtype"
#     new_file_name = prefix+file_name+"_cell_subtype.csv"
#     df.to_csv(new_file_name, index=False)
#     print("Saved file: ", new_file_name)
