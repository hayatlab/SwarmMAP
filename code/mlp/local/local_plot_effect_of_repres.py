import os

from matplotlib import legend
# Define project directory
if os.getcwd().startswith('/rwthfs'):
    proj_dir = '/rwthfs/rz/cluster/home/zd775156/sclabel'
    data_dir = '/rwthfs/rz/cluster/work/zd775156/lung/core_subset/'
else:
    proj_dir = '/Users/vgoepp/work/aachen/sclabel/'
    data_dir = os.path.join(proj_dir, 'data/lung/core_subset/')

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
# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
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

result_dir = "/home/zd775156/sclabel/results/local/effect_of_repres/"

## Compute f1_score from "results_all.csv" 
if not os.path.exists(result_dir+"results_all_f1.csv"):
    result_df = pd.read_csv(result_dir+"results_all.csv")
    
    result_df["y_pred"] = result_df["y_pred"].apply(eval)
    result_df["y_test"] = result_df["y_test"].apply(eval)

    for avg in ["micro", "macro", "weighted"]:
        result_df[f"f1_{avg}"]  = [f1_score(y_test, y_pred, average=avg) for y_test, y_pred in zip(result_df["y_test"].values, result_df["y_pred"].values)]
    result_df = result_df.drop(columns=["y_test", "y_pred"])

    tidy_res_df = pd.melt(result_df, id_vars=["organ", "train_on_n_studies", "training_time", "representation", "n_comps"],
                        value_vars=["f1_micro", "f1_macro", "f1_weighted"], var_name="f1_avg", value_name="f1_score")
    tidy_res_df.to_csv(result_dir+"f1_results_n_comps_pca.csv", index=False)

## Save f1_score results
tidy_res_df.to_csv(result_dir+"results_all_f1.csv", index=False)

## Load f1_score results
tidy_res_df = pd.read_csv(result_dir+"results_all_f1.csv")

# Convert train_on_n_studies to string
tidy_res_df["train_on_n_studies"] = tidy_res_df["train_on_n_studies"].astype(str)


# --------------------------------------------------------------------------------------------
# 1. Plot boxplot for heart --------------------------------------------------------------
# --------------------------------------------------------------------------------------------

g = sns.catplot(data=tidy_res_df, x="train_on_n_studies", y="f1_score", 
                hue="n_comps", col="f1_avg", kind="box", height=4, aspect=1, sharey=True)
g.set_axis_labels("Number of train studies", "F1 Score")
g.set(ylim=(0, 1))  # Set y-axis limits
# g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 0.65), title = "Number of components used in PCA", frameon=False)
plt.suptitle("Heart -- using PCA representation")
plt.tight_layout()
plt.show()
## Save figure
g.savefig(result_dir+"f1_score_boxplot_n_comps_pca.png")

# --------------------------------------------------------------------------------------------
# 2. Compare representations with n_comps = 50 ------------------------------------------------
# --------------------------------------------------------------------------------------------
tidy_res_df[["organ", "representation", "n_comps"]].value_counts().sort_index()
tidy_res_df_n_comps_50 = tidy_res_df[tidy_res_df["n_comps"] == 50]

## TEMP -- TODO REMOVE this
tidy_res_df_n_comps_50 = tidy_res_df_n_comps_50[tidy_res_df_n_comps_50["organ"] != "breast"]

tidy_res_df_n_comps_50[["organ", "representation", "f1_avg",
                        "n_comps", "train_on_n_studies"]].value_counts().sort_index()

# --------------------------------------------------------------------------------------------
# 2.1 Plot boxplot for heart --------------------------------------------------------------
# --------------------------------------------------------------------------------------------

g = sns.catplot(data=tidy_res_df_n_comps_50[tidy_res_df_n_comps_50["organ"] != "heart"],
                hue="train_on_n_studies",
                y="f1_score", 
                x = "representation",
                col="f1_avg",
                kind="box", height=4, aspect=1, sharey=True)
g.set_axis_labels("Representation", "F1 Score")
g.set(ylim=(0, 1))
custom_titles = ["Micro F1", "Macro F1", "Weighted F1"]  # Modify as needed
for ax, title in zip(g.axes.flat, custom_titles):
    ax.set_title(title)
sns.move_legend(g, "upper left", bbox_to_anchor=(0.98, 0.5), title = "Number of train studies", frameon=False)
plt.tight_layout()
plt.show()
g.savefig(result_dir+"f1_score_boxplot_heart_temp.png", dpi=300, bbox_inches='tight')


# --------------------------------------------------------------------------------------------
# 2.2 Plot boxplot for lung --------------------------------------------------------------
# --------------------------------------------------------------------------------------------

g = sns.catplot(data=tidy_res_df_n_comps_50[tidy_res_df_n_comps_50["organ"] != "lung"],
                hue="train_on_n_studies",
                y="f1_score",
                x="representation",
                col="f1_avg",
                kind="box", height=4, aspect=1, sharey=True)
g.set_axis_labels("Representation", "F1 Score")
g.set(ylim=(0, 1))
custom_titles = ["Micro F1", "Macro F1", "Weighted F1"]  # Modify as needed
for ax, title in zip(g.axes.flat, custom_titles):
    ax.set_title(title)
sns.move_legend(g, "upper left", bbox_to_anchor=(0.98, 0.5),
                title="Number of train studies", frameon=False)
plt.tight_layout()
plt.show()
g.savefig(result_dir+"f1_score_boxplot_lung_temp.png",
          dpi=300, bbox_inches='tight')
