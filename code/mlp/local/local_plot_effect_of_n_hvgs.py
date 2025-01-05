import os
from unittest import result
# Define project directory
proj_dir = '/rwthfs/rz/cluster/home/zd775156/sclabel'

# General utility
import sys
sys.path.append(os.path.abspath(proj_dir+"/code/utils"))
import mlp
import gc
import pickle
import itertools
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
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
import glob
from itertools import chain
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*Reordering categories will always return a new Categorical object.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*is_categorical_dtype is deprecated and will be removed in a future version.*")

organs = ["heart", "breast", "lung"]
result_dir_out = "/home/zd775156/sclabel/results/local/effect_of_n_hvgs/"
result_dirs = [f"/home/zd775156/sclabel/results/{organ}/subset_4/local/" for organ in organs]

result_files = [glob.glob(result_dir+"results_counts_cell_type_n_hvgs_*.csv") for result_dir in result_dirs]
# result_files = dict(zip(organs, result_files))
result_files = list(chain.from_iterable(result_files)) ## flatten list

## Load all files
dfs = [pd.read_csv(file) for file in result_files]

for df in dfs:
    print(df.shape)

result_df = pd.concat(dfs, axis=0)
result_df["y_pred"] = result_df["y_pred"].apply(eval)
result_df["y_test"] = result_df["y_test"].apply(eval)

result_df[["organ", "representation", "n_hvgs"]].value_counts()

## Compute f1_score from y_test and y_pred using micro, macro, and weighted average
for avg in ["micro", "macro", "weighted"]:
    result_df[f"f1_{avg}"] = [f1_score(y_test, y_pred, labels = sorted(y_test), average=avg)
                                for y_test, y_pred in zip(result_df["y_test"].values, result_df["y_pred"].values)]

# result_df = result_df.drop(columns=["y_test", "y_pred"])

tidy_res_df = pd.melt(result_df, id_vars=["organ", "train_on_n_studies", "training_time", "representation", "n_hvgs"],
                        value_vars=["f1_micro", "f1_macro", "f1_weighted"], var_name="f1_avg", value_name="f1_score")


## Generate wider dataframe from result_df
## F1-score boxplots with representation == "counts" and all organs
g = sns.catplot(data=tidy_res_df, x="n_hvgs", y="f1_score", 
                hue="train_on_n_studies", col="f1_avg", row="organ", kind="box", height=4, aspect=1, sharey=False)
g.set_axis_labels("Number of features (genes) used", "F1 Score")
# g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 0.65), title = "Number of train studies", frameon=False)
# g.figure.suptitle("F1-score boxplots for different numbers of features", y=1.0)
plt.tight_layout()
plt.show()
## Save figure
g.savefig(result_dir_out+f"f1_score_boxplot_effect_of_n_hvgs_all_organs_lq.png", dpi=300)


# F1-score boxplots for each organ separately
for organ in organs:
    g = sns.catplot(data=tidy_res_df[tidy_res_df["organ"] == organ], x="n_hvgs", y="f1_score",
                    hue="train_on_n_studies", col="f1_avg", kind="box", height=4, aspect=1, sharey=False)
    g.set_axis_labels("Number of features (genes) used", "F1 Score")
    # g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 0.65),
                    title="Number of train studies", frameon=False)
    # g.figure.suptitle("F1-score boxplots for different numbers of features", y=1.0)
    plt.tight_layout()
    custom_titles = ["Micro F1", "Macro F1", "Weighted F1"]
    for ax, title in zip(g.axes.flat, custom_titles):
        ax.set_title(title)
    plt.show()
    # Save figure
    g.savefig(result_dir_out+f"f1_score_boxplot_effect_of_n_hvgs_{organ}_lq.png", dpi=300)


## F1-score boxplots with representation == "counts"
g = sns.catplot(data=tidy_res_df, x="n_hvgs", y="f1_score", 
                hue="train_on_n_studies", col="f1_avg", kind="box", height=4, aspect=1, sharey=False)
g.set_axis_labels("Number of features (genes) used", "F1 Score")
# g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 0.65), title = "Number of train studies", frameon=False)
g.figure.suptitle("Heart -- F1-score boxplots for different numbers of features", y=1.0)
plt.tight_layout()
plt.show()
## Save figure
g.savefig(result_dir_out+f"{organ}_f1_score_boxplot_effect_of_n_hvgs.png")


## Compute f1 score per cell type
result_df["f1_score"] = [f1_score(y_test, y_pred, average=None)
                            for y_test, y_pred in zip(result_df["y_test"].values, result_df["y_pred"].values)]

result_df["f1_score_label"] = [list(set(y_test)) for y_test in result_df["y_test"]]
result_df = result_df.drop(columns=["y_test", "y_pred"])

## Explode data in f1_score and f1_score_label for each organ:
tmp = []
for ind, n_hvgs in enumerate(result_df["n_hvgs"].unique()):
    result_df_organ = result_df[result_df["n_hvgs"] == n_hvgs]
    result_df_organ = result_df_organ.explode(["f1_score", 'f1_score_label'])
    tmp.append(result_df_organ)
        
tidy_res_per_ct_df = pd.concat(tmp, axis=0)

## Plot f1-score boxplots per cell type
g = sns.catplot(data=tidy_res_per_ct_df, 
                  row="train_on_n_studies", height=5, aspect=2, hue="n_hvgs",
                  x="f1_score_label", y="f1_score", kind="box", sharex=False, sharey=True)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
g.set_xticklabels(rotation=90, fontsize=8)
g.figure.suptitle("F1-score boxplots per cell type per number of features used", y=1.0)
g.set_axis_labels("Cell type", "F1 Score")
sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.1), ncol=4, title="Number of train studies", frameon=False)
## Save figure
g.savefig(result_dir+f"{organ}_f1_score_boxplot_per_ct_counts_n_hvgs.png")
