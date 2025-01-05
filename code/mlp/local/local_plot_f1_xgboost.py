import textwrap
import ast
import os
from tkinter import font
from unittest import result

from matplotlib.pylab import f
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
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*Reordering categories will always return a new Categorical object.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*is_categorical_dtype is deprecated and will be removed in a future version.*")

result_dirs = dict({"breast": "/home/zd775156/sclabel/results/breast/subset_4/local_union_hvgs/",
                    "lung": "/home/zd775156/sclabel/results/lung/subset_4/local_union_hvgs/",
                    "heart": "/home/zd775156/sclabel/results/heart/subset_4/local_union_hvgs/"})
result_dir = "/home/zd775156/sclabel/results/local/xgboost/"
organs = ["heart", "lung", "breast"]
labels = ["cell_type", "cell_subtype"]

file1 = glob.glob(result_dir+"results_counts_cell_type.csv")
file2 = glob.glob(result_dir+"xgboost/"+"*cell_type_xgboost.csv")

## Load XGBoost results
## List all files in directories of results_dirs which contain the string "xgboost"
xgboost_files = [glob.glob(result_dirs[organ]+"*xgboost_multiclass*") for organ in organs]
##xgboost_files is a list of list: flatten it into a simple list
xgboost_files = list(itertools.chain(*xgboost_files))

## Load files in xgboost_files as pickle files and concatenate them
results_xgboost_df = [pickle.load(open(file, "rb")) for file in xgboost_files]
results_xgboost_df = pd.concat(results_xgboost_df, axis=0)
results_xgboost_df["method"] = "XGBoost"

## Load MLP results
mlp_files = [glob.glob(result_dirs[organ]+f"results_counts_{label}.pkl") for organ, label in itertools.product(organs, labels)]
mlp_files = [elem[0] for elem in mlp_files]

results_mlp_df = []
for organ, label in itertools.product(organs, labels):
    file = glob.glob(result_dirs[organ]+f"results_counts_{label}.pkl")[0]
    tmp = pickle.load(open(file, "rb"))
    tmp["organ"] = organ
    results_mlp_df.append(tmp)
    
results_mlp_df = pd.concat(results_mlp_df, axis=0)
results_mlp_df["method"] = "MLP"

# Load simulation results and concatenate them
result_df = pd.concat([results_xgboost_df, results_mlp_df], axis=0)
result_df["y_pred"] = result_df["y_pred"].apply(eval)
result_df["y_test"] = result_df["y_test"].apply(eval)

result_df[["method", "organ", "representation", "label"]].value_counts()

# ## Save result_df in home directory
# result_df.to_csv("~/tmp/result_df.csv", index=False)

# ## Load result_df from home directory
# result_df = pd.read_csv("~/tmp/result_df.csv")

f1_results = []
for organ in ["heart", "lung", "breast"]:
    for label in ["cell_type", "cell_subtype"]:

        tmp = result_df[(result_df["label"] == label) & (result_df["organ"] == organ)].copy()
        try:
            tmp["y_pred"] = tmp["y_pred"].apply(ast.literal_eval)
        except:
            pass
        try:
            tmp["y_test"] = tmp["y_test"].apply(ast.literal_eval)
        except:
            pass
        
        y_test_ls_of_ls = [list(y_test) for y_test in tmp["y_test"]]
        ct_labels = sorted(set(list(itertools.chain.from_iterable(y_test_ls_of_ls))))
        
        # ct_labels = [sorted(list(set(y_test))) for y_test in tmp["y_test"]]
        # ct_labels = sorted(list(set(itertools.chain.from_iterable(ct_labels))))
        
        # If tmp does not contain columns f1_micro, f1_macro, f1_weighted, compute them
        for avg in ["micro", "macro", "weighted"]:
            tmp[f"f1_{avg}"] = [f1_score(y_test, y_pred, labels=ct_labels, average=avg)
                                for y_test, y_pred in zip(tmp["y_test"], tmp["y_pred"])]
            
        # for ind, (y_test, y_pred) in enumerate(zip(tmp["y_test"], tmp["y_pred"])):
        #     f1_score(y_test, y_pred, labels=sorted(y_test), average=avg)
        #     print(ind)

        # Compute f1_per_ct if it does not exist
        if not all([col in tmp.columns for col in ["f1_per_ct"]]):
            tmp["f1_per_ct"] = [f1_score(y_test, y_pred, labels = ct_labels,
                                            average=None, zero_division=np.nan).tolist()
                                for y_test, y_pred in zip(tmp["y_test"], tmp["y_pred"])]
            tmp["f1_label"] = [ct_labels] * tmp.shape[0]

        # Concatenate data frame
        f1_results.append(tmp)

result_df = pd.concat(f1_results, axis=0)

## Save in result_dir as pickle
# result_df.to_pickle(result_dir+"f1_result_df_xgboost.pkl")

## load
result_df = pd.read_pickle("~/f1_result_df_xgboost.pkl")

## --------------------------------------------
## 1. Plot average f1 scores
## --------------------------------------------
result_df = result_df.drop(columns=["y_test", "y_pred"])

tidy_res_df = pd.melt(result_df, id_vars=["organ", "train_on_n_studies", "training_time", 
                                            "representation", "label", "method"],
                        value_vars=["f1_micro", "f1_macro", "f1_weighted"], var_name="f1_avg", value_name="f1_score")


## Plot F1_weighted for MLP vs XGBoost
g = sns.catplot(data=tidy_res_df[tidy_res_df["f1_avg"] == "f1_weighted"],
                x="method", y="f1_score", 
                hue="train_on_n_studies", col="organ", kind="box", height=4, aspect=1, sharey=False)
g.set_axis_labels("Method", "F1 Score")
g.figure.suptitle("Weighted F1 score for MLP vs XGBoost")
plt.subplots_adjust(top=0.85)
plt.show()
## Save figure
g.savefig(result_dir+"f1_score_boxplot_method_f1_weighted.png")

## Plot all F1 scores for MLP vs XGBoost
g = sns.catplot(data=tidy_res_df,
                x="method", y="f1_score", row = "f1_avg",
                hue="train_on_n_studies", col="organ", kind="box", height=4, aspect=1, sharey=False)
g.set_axis_labels("Method", "F1 Score")
g.figure.suptitle("F1-scores for MLP vs XGBoost")
plt.subplots_adjust(top=0.85)
plt.show()
# Save figure
g.savefig(result_dir+"f1_score_boxplot_method_f1_scores.png")

## Plot boxplot of training_time 
g = sns.catplot(data=tidy_res_df, x="method", y="training_time",
                hue="train_on_n_studies", col = "organ",
                kind="box", height=4, aspect=1, sharey=False)
g.set_axis_labels("Number of Train Studies", "Training Time")
g.set_axis_labels("Organ", "Training Time (seconds)")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 0.59))
g.figure.suptitle("Training times for MLP vs XGBoost")
plt.tight_layout()
plt.show()
g.savefig(result_dir+"training_time_boxplot_method.png")

tidy_mlp = tidy_res_df[tidy_res_df["method"] == "MLP"]
## rename training_time to training_time_mlp
tidy_mlp = tidy_mlp.rename(columns={"training_time": "training_time_mlp"})
tidy_xgboost = tidy_res_df[tidy_res_df["method"] == "XGBoost"]
# rename training_time to training_time_xgboost
tidy_xgboost = tidy_xgboost.rename(columns={"training_time": "training_time_xgboost"})

## Merge tidy_mlp and tidy_xgboost, only add the variable training_time_xgboost from tidy_xgboost,
## merge on the variables organ, train_on_n_studies, representation, label, f1_avg
tidy_res_df_ratio = pd.merge(tidy_mlp, tidy_xgboost[["organ", "train_on_n_studies", "representation", "label", "f1_avg", "f1_score", "training_time_xgboost"]],
                        on=["organ", "train_on_n_studies", "representation", "label", "f1_avg"], how="inner")

tidy_res_df_ratio["training_time_ratio"] = tidy_res_df["training_time_mlp"] / \
    tidy_res_df["training_time_xgboost"]

# Plot boxplot of training_time ratio
g = sns.catplot(data=tidy_res_df_ratio, x="method", y="training_time_ratio",
                hue="train_on_n_studies", col="organ",
                kind="box", height=4, aspect=1, sharey=False)
g.set_axis_labels("Number of Train Studies", "Training Time")
g.set_axis_labels("Organ", "Training Time (seconds)")
sns.move_legend(g, "upper left", bbox_to_anchor=(1, 0.59))
g.figure.suptitle("Training times for MLP / XGBoost")
plt.tight_layout()
plt.show()
g.savefig(result_dir+"training_time_ratio_boxplot_method.png")



## --------------------------------------------
## Plot F1 score per cell type
# --------------------------------------------

## Filter cell types
result_ct_df = result_df[result_df["label"] == "cell_type"]
## Explose per f1_per_cet and f1_label for each organ separately

f1_results_per_ct_all = []
for organ in organs:
    tmp = result_ct_df[result_ct_df["organ"] == organ]
    
    aaa = tmp["f1_label"].tolist()
    assert all([elem == aaa[0] for elem in aaa])
    aaa = list(set(itertools.chain.from_iterable(aaa)))
    aaa = sorted(set(aaa))
    
    # bbb = [aaa] * tmp.shape[0]
    # del tmp["f1_label"]
    # tmp['f1_label'] = bbb
    # Aply the replace("nan", "None") method to all elements in f1_per_ct
    # ccc = tmp["f1_per_ct"].tolist()
    # ccc.replace("nan", "None")
    # tmp["f1_per_ct"] = tmp["f1_per_ct"].tolist().apply(
    #     lambda x: x.replace("nan", "None"))
    # Apply ast.literal_eval to all elements in f1_per_ct
    # tmp['f1_per_ct'] = tmp['f1_per_ct'].apply(ast.literal_eval)

    tmp = tmp.explode(["f1_per_ct", "f1_label"])
    f1_results_per_ct_all.append(tmp)

result_per_ct_df = pd.concat(f1_results_per_ct_all, axis=0)


# result_per_ct_df = result_df.explode(["f1_per_ct", "f1_label"])

# Wrapper cell type labels
def wrap_labels(labels, width=10):
    return ['\n'.join(textwrap.wrap(label, width)) for label in labels]

## f1_label dtype set to string
result_per_ct_df["f1_label"] = result_per_ct_df["f1_label"].astype(str)

## Plot boxplot of f1_per_ct
g = sns.catplot(data=result_per_ct_df, x="f1_label", y="f1_per_ct", hue="train_on_n_studies", row="organ", kind="box",
                height=4, aspect=3, sharey=False, sharex=False)

g = sns.catplot(data=result_per_ct_df[result_per_ct_df["train_on_n_studies"] == 3],
                row="organ", height=4, aspect=2, hue="method",
                x="f1_label", y="f1_per_ct", kind="box", sharex=False, sharey=False)
g.set_axis_labels("Cell type", "F1 Score")
g.set_titles("{row_name}".capitalize())
for ax in g.axes.flat:
    # Get the current labels
    labels = ax.get_xticklabels()
    # Wrap the labels
    wrapped_labels = wrap_labels([label.get_text()
                                 for label in labels], width=25)
    # Set the new labels
    ax.set_xticklabels(wrapped_labels, rotation=45, ha='right', fontsize=8)
plt.suptitle("F1-score boxplots per cell type", y=1.0)
# add spacing between suptitle and subplots to prevent overlap
# plt.subplots_adjust(top=0.95)
g.fig.subplots_adjust(hspace=0.6, wspace=0.1)
g.savefig(result_dir+"f1_score_boxplot_method_per_ct.png")
