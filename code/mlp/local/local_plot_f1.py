from statsmodels.formula.api import ols
import statsmodels.api as sm
from statannot import add_stat_annotation
import textwrap
import matplotlib.ticker as ticker
from ast import literal_eval
import glob
import warnings
from matplotlib.pylab import f
import torch.optim as optim
import torch.nn as nn
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
import scipy as sp
import numpy as np
import pandas as pd
import itertools
import pickle
import gc
import sys
import ast
import os
from unittest import result
# Define project directory
proj_dir = '/rwthfs/rz/cluster/home/zd775156/sclabel'

sys.path.append(os.path.abspath(proj_dir+"/code/utils"))
import mlp
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*Reordering categories will always return a new Categorical object.*")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*is_categorical_dtype is deprecated and will be removed in a future version.*")


organs = ["heart", "lung", "breast"]
result_dirs_local = dict({"breast": "/home/zd775156/sclabel/results/breast/subset_4/local_union_hvgs/",
                          "lung": "/home/zd775156/sclabel/results/lung/subset_4/local_union_hvgs/",
                          "heart": "/home/zd775156/sclabel/results/heart/subset_4/local_union_hvgs/"})

result_dirs_swarm = dict({"breast": "/home/zd775156/sclabel/results/breast/subset_4/swarm_union_of_hvgs/",
                          "lung": "/home/zd775156/sclabel/results/lung/subset_4/swarm_union_of_hvgs/",
                          "heart": "/home/zd775156/sclabel/results/heart/subset_4/swarm_union_of_hvgs/"})

result_dir = "/home/zd775156/sclabel/results/local/f1_score_plots/"
result_dir_data = "/home/zd775156/sclabel/results/local/"

# --------------------------------------------
# Load and format results from Local learning
# --------------------------------------------

# ## Load f1_results_all_organs from result_dirs_local
# f1_results_local = []
# for organ in organs:
#     for label in ["cell_type", "cell_subtype"]:
#         ## Get filenames in result_dirs_local[organ] containing "counts_cell_type" or "counts_cell_subtype"
#         files = glob.glob(result_dirs_local[organ]+f"results_counts_{label}.pkl")
#         print(files)
#         ## Load files
#         assert len(files) == 1
#         tmp = pd.read_pickle(files[0])
#         ## add organ and label columns
#         tmp["organ"] = organ
#         tmp["label"] = label
#         tmp["representation"] = "counts"
#         ## Make sure y_pred and y_test are lists
#         tmp["y_pred"] = tmp["y_pred"].apply(eval)
#         tmp["y_test"] = tmp["y_test"].apply(eval)
        
                
#         ## Define cell type labels
#         ct_labels = [set(y_test) for y_test in tmp["y_test"]]
#         ## set union
#         ct_labels = sorted(list(set.union(*ct_labels)))
#         ## Remove "Ischemic cells (MI)" and "Epicardium" from ct_labels
#         # ct_labels = [elem for elem in ct_labels if elem not in ["Ischemic cells (MI)", "Epicardium"]]
        
#         # Compute f1_micro, f1_macro, f1_weighted
#         for avg in ["micro", "macro", "weighted"]:
#             tmp[f"f1_{avg}"] = [f1_score(y_test, y_pred, labels = ct_labels, average=avg)
#                                 for y_test, y_pred in zip(tmp["y_test"], tmp["y_pred"])]
        
#         # Compute f1_per_ct
#         if not all([col in tmp.columns for col in ["f1_per_ct"]]):
#             tmp["f1_per_ct"] = [f1_score(y_test, y_pred, labels = ct_labels, 
#                                          average=None, zero_division=np.nan).tolist()
#                                for y_test, y_pred in zip(tmp["y_test"], tmp["y_pred"])]
#             tmp["f1_label"] = [ct_labels] * tmp.shape[0]
        
#         ## Concatenate data frame
#         f1_results_local.append(tmp)
        
# f1_results_local = pd.concat(f1_results_local, axis=0)

# ## Modify variable 'train_on_n_studies' by adding "Local_" to the beginning
# f1_results_local["train_on_n_studies"] = ["Local_" +
#                                           str(elem) for elem in f1_results_local["train_on_n_studies"]]

# ## Rename train_on_n_studies to Training set
# f1_results_local = f1_results_local.rename(columns={"train_on_n_studies": "Training set"})

# ## assert that representation is counts, label is in ["cell_type", "cell_subtype"], and organ is in ["breast", "lung", "heart"]
# assert all(f1_results_local["representation"] == "counts")
# assert all(f1_results_local["label"].isin(["cell_type", "cell_subtype"]))
# assert all(f1_results_local["organ"].isin(["breast", "lung", "heart"]))

# f1_results_filter = f1_results_local

## Load
f1_results = pd.concat([
    pd.read_pickle(result_dir_data+"f1_results_local_swarm_cell_type.pkl"),
    pd.read_pickle(result_dir_data+"f1_results_local_swarm_cell_subtype.pkl")], axis=0)

## Filter our Swarm Learning
f1_results = f1_results[f1_results["Training set"].isin(["Local_1", "Local_2", "Local_3"])]


## --------------------------------------------
## 1. Plot average f1 scores
## --------------------------------------------

f1_results_all_ct = f1_results.melt(id_vars=["organ", "label", "representation", "Training set"],
                                             value_vars=["f1_micro", "f1_macro", "f1_weighted"],
                                             var_name="f1_avg", value_name="f1_score")
f1_weighted = f1_results_all_ct[f1_results_all_ct["f1_avg"] == "f1_weighted"]
f1_macro = f1_results_all_ct[f1_results_all_ct["f1_avg"] == "f1_macro"]
f1_micro = f1_results_all_ct[f1_results_all_ct["f1_avg"] == "f1_micro"]

# f1_results_all_ct_filter = f1_results_filter.melt(id_vars=["organ", "label", "representation", "Training set"],
#                                              value_vars=[
#                                                  "f1_micro", "f1_macro", "f1_weighted"],
#                                              var_name="f1_avg", value_name="f1_score")
# f1_weighted_filter = f1_results_all_ct_filter[f1_results_all_ct["f1_avg"] == "f1_weighted"]

# ## Compare f1_results_all_ct and f1_results_all_ct_filter using variable f1_macro
# f1_results_all_ct["f1_score"].equals(f1_results_all_ct_filter["f1_score"])
# ## Compute which indices are different
# diff = f1_results["f1_micro"].values != f1_results_filter["f1_micro"].values
# ## Compute where "Epicardium" and "Ischemic cells (MI)" are in f1_results_all_ct["y_test"]
# index = f1_results_filter["y_test"].apply(lambda x: [elem for elem in x if elem in ["Epicardium", "Ischemic cells (MI)"]])
# ## Remove "Epicardium" and "Ischemic cells (MI)" from f1_results_all_ct["y_test"]
# f1_results_filter["y_test"] = f1_results_filter["y_test"].apply(
#     lambda x: [elem for elem in x if elem not in ["Epicardium", "Ischemic cells (MI)"]])
# ## Remove "Epicardium" and "Ischemic cells (MI)" from f1_results_all_ct["y_pred"]
# f1_results_filter["y_pred"] = f1_results_filter["y_pred"].apply(
#     lambda x: [elem for elem in x if elem not in ["Epicardium", "Ischemic cells (MI)"]])

# ## Get length of f1_results_filter["y_test"] and f1_results_filter["y_pred"]
# len(f1_results_filter["y_test"].values[0]), len(f1_results_filter["y_pred"].values[0])

# ## Compute f1_micro, f1_macro, f1_weighted for f1_results_filter
# f1_results_filter["f1_micro"] = [f1_score(y_test, y_pred, labels=sorted(y_test), average="micro")
#                                     for y_test, y_pred in zip(f1_results_filter["y_test"], f1_results_filter["y_pred"])]
# f1_results_filter["f1_macro"] = [f1_score(y_test, y_pred, labels=sorted(y_test), average="macro")
#                                     for y_test, y_pred in zip(f1_results_filter["y_test"], f1_results_filter["y_pred"])]
# f1_results_filter["f1_weighted"] = [f1_score(y_test, y_pred, labels=sorted(y_test), average="weighted")
#                                     for y_test, y_pred in zip(f1_results_filter["y_test"], f1_results_filter["y_pred"])]



# --------------------------------------------
## 1. Aggregated f1 scores
# --------------------------------------------

## 1.1 Cell type, with P values
g = sns.catplot(data=f1_results_all_ct[f1_results_all_ct["label"]=="cell_type"],
                x="f1_avg", y="f1_score", hue="Training set", col="organ", kind="box",
                height=4, aspect=1, sharey=False)
g.set_axis_labels("F1 averaging method", "F1 Score")
g.set_titles("Organ: {col_name}")
plt.suptitle("Cell type")
plt.subplots_adjust(top=0.85)
## Define box_pairs: outer product of f1_micro, f1_macro, f1_weigthted and Training set
box_pairs = [(("f1_micro", "Local_1"), ("f1_micro", "Local_2")),
                (("f1_micro", "Local_2"), ("f1_micro", "Local_3")),
                (("f1_macro", "Local_1"), ("f1_macro", "Local_2")),
                (("f1_macro", "Local_2"), ("f1_macro", "Local_3")),
                (("f1_weighted", "Local_1"), ("f1_weighted", "Local_2")),
                (("f1_weighted", "Local_2"), ("f1_weighted", "Local_3"))]
for ax in g.axes.flat:
    add_stat_annotation(ax, data=f1_results_all_ct[f1_results_all_ct["label"]=="cell_type"],
                        x='f1_avg', y='f1_score', hue='Training set',
                        box_pairs=box_pairs,
                        test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
plt.show()
## Save figure
g.savefig(result_dir+"local_f1_score_boxplot_cell_type_f1_avg_pval.png", dpi=1200)

# 1.2 Cell subtype, with P values
g = sns.catplot(data=f1_results_all_ct[f1_results_all_ct["label"] == "cell_subtype"],
                x="f1_avg", y="f1_score", hue="Training set", col="organ", kind="box",
                height=4, aspect=1, sharey=False)
g.set_axis_labels("F1 averaging method", "F1 Score")
g.set_titles("Organ: {col_name}")
plt.suptitle("Cell subtype")
plt.subplots_adjust(top=0.85)
# Define box_pairs: outer product of f1_micro, f1_macro, f1_weigthted and Training set
box_pairs = [(("f1_micro", "Local_1"), ("f1_micro", "Local_2")),
             (("f1_micro", "Local_2"), ("f1_micro", "Local_3")),
             (("f1_macro", "Local_1"), ("f1_macro", "Local_2")),
             (("f1_macro", "Local_2"), ("f1_macro", "Local_3")),
             (("f1_weighted", "Local_1"), ("f1_weighted", "Local_2")),
             (("f1_weighted", "Local_2"), ("f1_weighted", "Local_3"))]
for ax in g.axes.flat:
    add_stat_annotation(ax, data=f1_results_all_ct[f1_results_all_ct["label"] == "cell_subtype"],
                        x='f1_avg', y='f1_score', hue='Training set',
                        box_pairs=box_pairs,
                        test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
plt.show()
# Save figure
g.savefig(result_dir+"local_f1_score_boxplot_cell_subtype_f1_avg_pval.png", dpi=1200)

# 1.3 Cell type, NO P values
g = sns.catplot(data=f1_results_all_ct[f1_results_all_ct["label"] == "cell_type"],
                x="f1_avg", y="f1_score", hue="Training set", col="organ", kind="box",
                height=4, aspect=1, sharey=False)
g.set_axis_labels("F1 average method", "F1 Score")
plt.suptitle("Cell subtype")
g.set_titles("Organ: {col_name}")
plt.tight_layout()
plt.show()
# Save figure
g.savefig(result_dir+"local_f1_score_boxplot_cell_type_f1_avg.png", dpi=1200)

# 1.4 Cell subtype, NO P values
g = sns.catplot(data=f1_results_all_ct[f1_results_all_ct["label"] == "cell_subtype"],
                x="f1_avg", y="f1_score", hue="Training set", col="organ", kind="box",
                height=4, aspect=1, sharey=False)
g.set_axis_labels("F1 average method", "F1 Score")
plt.suptitle("Cell subtype")
g.set_titles("Organ: {col_name}")
plt.tight_layout()
plt.show()
# Save figure
g.savefig(result_dir+"local_f1_score_boxplot_cell_subtype_f1_avg.png", dpi=1200)



# g = sns.catplot(data=f1_results_all_ct_filter[f1_results_all_ct_filter["label"] == "cell_type"],
#                 x="f1_avg", y="f1_score", hue="Training set", col="organ", kind="box",
#                 height=4, aspect=1, sharey=False)
# g.set_axis_labels("F1 averaging method", "F1 Score")
# g.set_titles("Organ: {col_name}")
# plt.suptitle("Cell type")
# plt.subplots_adjust(top=0.85)
# # Define box_pairs: outer product of f1_micro, f1_macro, f1_weigthted and Training set
# box_pairs = [(("f1_micro", "Local_1"), ("f1_micro", "Local_2")),
#              # (("f1_micro", "Local_1"), ("f1_micro", "Local_3")),
#              (("f1_micro", "Local_2"), ("f1_micro", "Local_3")),
#              (("f1_macro", "Local_1"), ("f1_macro", "Local_2")),
#              # (("f1_macro", "Local_1"), ("f1_macro", "Local_3")),
#              (("f1_macro", "Local_2"), ("f1_macro", "Local_3")),
#              (("f1_weighted", "Local_1"), ("f1_weighted", "Local_2")),
#              # (("f1_weighted", "Local_1"), ("f1_weighted", "Local_3")),
#              (("f1_weighted", "Local_2"), ("f1_weighted", "Local_3"))]
# for ax in g.axes.flat:
#     add_stat_annotation(ax, data=f1_results_all_ct_filter[f1_results_all_ct_filter["label"] == "cell_type"],
#                         x='f1_avg', y='f1_score', hue='Training set',
#                         box_pairs=box_pairs,
#                         test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
# plt.show()
# # Save figure
# g.savefig(result_dir+"local_f1_score_boxplot_cell_type_f1_avg_hf.png", dpi=600)


## 1.5 Plot only f1_weighted for cell_type and cell_subtype: COL = label
g = sns.catplot(data=f1_weighted, x="organ", y="f1_score", hue="Training set", kind="box",
                col="label",
                height=4, aspect=1, sharey=False)
g.set_axis_labels("Organ", "F1 Score")
g.set_titles("Label: {col_name}")
plt.suptitle("Weighted F1 Score")
plt.subplots_adjust(top=0.85)
## Add statistical annotation to each facet
box_pairs = [(("heart", "Local_1"), ("heart", "Local_2")),
                (("heart", "Local_2"), ("heart", "Local_3")),
                (("lung", "Local_1"), ("lung", "Local_2")),
                (("lung", "Local_2"), ("lung", "Local_3")),
                (("breast", "Local_1"), ("breast", "Local_2")),
                (("breast", "Local_2"), ("breast", "Local_3"))]
for ax in g.axes.flat:
    add_stat_annotation(ax, data=f1_weighted, x='organ', y='f1_score', hue='Training set',
                        box_pairs=box_pairs,
                        test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
plt.show()
## Save figure
g.savefig(result_dir+"local_f1_score_boxplot_f1_weighted_pval.png", dpi=800)


# 1.6 Plot only f1_weighted for cell_type and cell_subtype: COL = organ
g = sns.catplot(data=f1_weighted, x="label", y="f1_score", hue="Training set", kind="box",
                col="organ",
                height=4, aspect=1, sharey=False)
g.set_axis_labels("Label", "F1 Score")
g.set_titles("Organ: {col_name}")
plt.suptitle("Weighted F1 Score")
plt.subplots_adjust(top=0.85)
plt.show()
# Save figure
g.savefig(result_dir+"local_f1_score_boxplot_f1_weighted_by_organ.png", dpi=800)

# 1.6 Plot only f1_weighted for cell_type and cell_subtype: COL = organ
g = sns.catplot(data=f1_weighted, x="label", y="f1_score", hue="Training set", kind="box",
                col="organ",
                height=4, aspect=1, sharey=False)
g.set_axis_labels("Label", "F1 Score")
g.set_titles("Organ: {col_name}")
plt.suptitle("Weighted F1 Score")
plt.subplots_adjust(top=0.85)
box_pairs = [(("cell_type", "Local_1"), ("cell_type", "Local_2")),
             (("cell_type", "Local_2"), ("cell_type", "Local_3")),
             (("cell_subtype", "Local_1"), ("cell_subtype", "Local_2")),
             (("cell_subtype", "Local_2"), ("cell_subtype", "Local_3"))]
for ax in g.axes.flat:
    add_stat_annotation(ax, data=f1_weighted, x='label', y='f1_score', hue='Training set',
                        box_pairs=box_pairs,
                        test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
plt.show()
# Save figure
g.savefig(result_dir+"local_f1_score_boxplot_f1_weighted_by_organ_pval.png", dpi=800)

# --------------------------------------------
## Run anova and linear model to check p-values w.r.t training_set_size
# --------------------------------------------

## Define new df
anova_df = f1_weighted
## Rename variable Training set to training_set_size
anova_df = anova_df.rename(columns={"Training set": "training_set_size"})
## Remove leading "Local_" from training_set_size
anova_df["training_set_size"] = anova_df["training_set_size"].apply(lambda x: x.replace("Local_", ""))
## Convert to integer
anova_df["training_set_size"] = anova_df["training_set_size"].astype(int)

## subset organ == lung and label == cell_type
anova_df = anova_df[(anova_df["organ"] == "lung") & (anova_df["label"] == "cell_type")]

# Create the boxplot
sns.set(style="whitegrid")
boxplot = sns.catplot(x="training_set_size", y="f1_score", kind="box", data=anova_df, height=6, aspect=1.5)
# Perform ANOVA
model = ols('f1_score ~ C(training_set_size)', data=anova_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
# Extract the p-value from ANOVA table
anova_p_value = anova_table["PR(>F)"][0]
# Display the ANOVA p-value on the plot
for ax in boxplot.axes.flat:
    ax.text(0.5, 1.1, f'ANOVA p-value = {anova_p_value:.3e}', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes, fontsize=12)
# Define the box pairs for comparison
box_pairs = [((1, 2), (2, 3)), ((1, 2), (1, 3)), ((2, 3), (1, 3))]
# Add the statistical annotation
add_stat_annotation(boxplot.ax, data=anova_df, x='training_set_size', y='f1_score',
                    box_pairs=box_pairs,
                    test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
plt.show()

## Linear model of f1_score ~ training_set_size in anova_df
model = ols('f1_score ~ training_set_size', data=anova_df).fit()
print(model.summary())

# --------------------------------------------
## End run anova and LM
# --------------------------------------------

# 1.7 Plot f1_macro
g = sns.catplot(data=f1_macro, x="organ", y="f1_score", hue="Training set", kind="box",
                col="label",
                height=4, aspect=1, sharey=False)
g.set_axis_labels("Organ", "F1 Score")
g.set_titles("Label: {col_name}")
plt.suptitle("Macro F1 Score")
plt.subplots_adjust(top=0.85)
# Add statistical annotation to each facet
box_pairs = [(("heart", "Local_1"), ("heart", "Local_2")),
             (("heart", "Local_2"), ("heart", "Local_3")),
             (("lung", "Local_1"), ("lung", "Local_2")),
             (("lung", "Local_2"), ("lung", "Local_3")),
             (("breast", "Local_1"), ("breast", "Local_2")),
             (("breast", "Local_2"), ("breast", "Local_3"))]
for ax in g.axes.flat:
    add_stat_annotation(ax, data=f1_macro, x='organ', y='f1_score', hue='Training set',
                        box_pairs=box_pairs,
                        test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
plt.show()
# Save figure
g.savefig(result_dir+"local_f1_score_boxplot_f1_macro_pval.png", dpi=1200)


# ## 2. Barplots
# ## 2.1. Plot the 3 average methods for f1_score using barplots using catplot:
# g = sns.catplot(data=f1_results_all_ct[f1_results_all_ct['label']=="cell_type"],
#                 x="f1_avg", y="f1_score", hue="Training set", col="organ", kind="bar",
#                 height=4, aspect=1, sharey=False)
# g.set_axis_labels("F1 averaging", "F1 Score")
# g.set_titles("{col_name}")
# plt.suptitle("Cell type")
# ## add spacing between suptitle and subplots to prevent overlap
# ## but keep legend outside of the plot
# plt.subplots_adjust(top=0.85)
# plt.show()
# # Save figure in high resolution
# g.savefig(result_dir+"f1_score_barplot_cell_type_f1_avg.png",
#             dpi=600)

# g = sns.catplot(data=f1_results_all_ct[f1_results_all_ct['label']=="cell_subtype"],
#                 x="f1_avg", y="f1_score", hue="Training set", col="organ", kind="bar",
#                 height=4, aspect=1, sharey=False)
# g.set_axis_labels("F1 averaging", "F1 Score")
# g.set_titles("{col_name}")
# plt.suptitle("Cell subtype")
# ## add spacing between suptitle and subplots to prevent overlap
# ## but keep legend outside of the plot
# plt.subplots_adjust(top=0.85)
# plt.show()
# # Save figure in high resolution
# g.savefig(result_dir+"f1_score_barplot_cell_subtype_f1_avg.png",
#             dpi=600)


# ## 2.2. Plot only f1_weighted for cell_type and cell_subtype
# g = sns.catplot(data=f1_weighted, x="organ", y="f1_score", hue="Training set", kind="bar",
#                 col="label", height=4, aspect=1, sharey=False)
# g.set_axis_labels("Organ", "Weighted F1 Score")
# # g.set_titles("Label: {col_name.replace('_', ' ').capitalize()}")
# ## add overall title "Cell types"
# # plt.title("Cell type")
# plt.show()
# # Save figure
# g.savefig(result_dir+"f1_score_barplot_f1_weighted.png", dpi=600)


# # 2.2. Plot only f1_weighted for cell_type and cell_subtype
# g = sns.catplot(data=f1_weighted, x="organ", y="f1_score", hue="Training set", kind="box",
#                 col="label", height=4, aspect=1, sharey=False)
# g.set_axis_labels("Organ", "Weighted F1 Score")
# # g.set_titles("Label: {col_name.replace('_', ' ').capitalize()}")
# # add overall title "Cell types"
# # plt.title("Cell type")
# plt.show()
# # Save figure
# g.savefig(result_dir+"f1_score_boxplot_f1_weighted.png", dpi=600)


## --------------------------------------------
## 2. Plot f1 per cell type
## --------------------------------------------

f1_results_per_ct = f1_results[f1_results["label"] == "cell_type"]
f1_results_per_ct = f1_results_per_ct.drop(columns=["y_pred", "y_test", "f1_micro", "f1_macro", "f1_weighted", "training_time"])

f1_results_per_ct_all = []
for organ in f1_results_per_ct["organ"].unique():
    tmp = f1_results_per_ct[f1_results_per_ct["organ"] == organ]
    ## Check that unique(tmp["f1_label"]) is of length 1
    assert len(tmp["f1_label"].unique()) == 1
    aaa = tmp["f1_label"].unique().tolist()[0]
    aaa = ast.literal_eval(aaa)
    bbb = [aaa] * tmp.shape[0]
    # del tmp["f1_label"]
    tmp['f1_label'] = bbb
    ## Aply the replace("nan", "None") method to all elements in f1_per_ct
    tmp["f1_per_ct"] = tmp["f1_per_ct"].apply(lambda x: x.replace("nan", "None"))
    ## Apply ast.literal_eval to all elements in f1_per_ct
    tmp['f1_per_ct'] = tmp['f1_per_ct'].apply(ast.literal_eval)
    tmp = tmp.explode(["f1_per_ct", "f1_label"])
    f1_results_per_ct_all.append(tmp)

f1_results_per_ct = pd.concat(f1_results_per_ct_all, axis=0)
## Remove all rows where f1_per_ct is None
f1_results_per_ct = f1_results_per_ct[f1_results_per_ct["f1_per_ct"].notna()]
## Wrapper cell type labels
def wrap_labels(labels, width=10):
    return ['\n'.join(textwrap.wrap(label, width)) for label in labels]

# 1. Boxplots
def add_p_values_to_facet(facet_df, ax):
    x_labels = facet_df['f1_label'].unique()
    print(x_labels.shape[0])
    hue_labels = facet_df['Training set'].unique()
    box_pairs = []
    # Generate pairs for comparisons
    for x_label in x_labels:
        # box_pairs.append(
        #     ((x_label, hue_labels[0]), (x_label, hue_labels[-1])))
        for i in range(len(hue_labels) - 1):
            box_pairs.append(
                ((x_label, hue_labels[i]), (x_label, hue_labels[i + 1])))
    add_stat_annotation(ax, data=facet_df, x='f1_label', y='f1_per_ct', hue='Training set',
                        box_pairs=box_pairs,
                        # run t-test instead of mann-whitney
                        test='t-test_ind',
                        # test='Mann-Whitney',
                        text_format='star', loc='inside', verbose=2)


if "Unnamed: 0" in f1_results_per_ct.columns:
    del f1_results_per_ct["Unnamed: 0"]
## Set f1_per_ct as float
f1_results_per_ct["f1_per_ct"] = f1_results_per_ct["f1_per_ct"].astype(float)
desired_order = organs
# Convert 'organ' column to categorical with the specified order
f1_results_per_ct['organ'] = pd.Categorical(
    f1_results_per_ct['organ'], categories=desired_order, ordered=True)

g = sns.catplot(data=f1_results_per_ct, x="f1_label", y="f1_per_ct", hue="Training set", row="organ", kind="box",
                height=4, aspect=3, sharey=False, sharex=False)
g.set_axis_labels("Cell type", "F1 Score")
g.set_titles("{row_name}".capitalize())
for ax in g.axes.flat:
    # Get the current labels
    labels = ax.get_xticklabels()
    # Wrap the labels
    wrapped_labels = wrap_labels([label.get_text()
                                 for label in labels], width=20)
    # Set the new labels
    ax.set_xticklabels(wrapped_labels, rotation=45, ha='right')
# Loop through each facet and add the statistical annotations
plt.suptitle("Weighted F1 score per cell type", y=1.0)
plt.subplots_adjust(top=0.95)
g.fig.subplots_adjust(hspace=0.9, wspace=0.40)
# Save figure
g.savefig(result_dir+"local_f1_score_boxplot_cell_type_f1_per_ct.png", dpi=1200)


# 2. Same figures with barplots instead of boxplots
# 2.1. Plot the 3 average methods for f1_score using barplots using catplot:
g = sns.catplot(data=f1_results_per_ct, y="f1_label", x="f1_per_ct", hue="Training set", row="organ", kind="bar",
                height=4, aspect=3, sharey=False, sharex=False)
g.set_axis_labels("F1 Score", "cell type")
g.set_yticklabels(rotation=45, fontsize=12, ha="right")
g.set_titles("{row_name}".capitalize())
plt.show()
# Save figure
g.savefig(result_dir+"f1_score_barplot_cell_type_f1_per_ct.png")

# 2.1.2 Plot the 3 average methods for f1_score using barplots using catplot:
g = sns.catplot(data=f1_results_per_ct, y="f1_label", x="f1_per_ct", hue="Training set", row="organ", kind="bar",
                height=4, aspect=3, sharey=False, sharex=False, orient="h")
g.set_axis_labels("F1 Score", "Organ")
g.set_yticklabels(rotation=0)
g.set_titles("{row_name}".capitalize())
plt.show()
# Save figure
g.savefig(result_dir+"f1_score_barplot_cell_type_f1_per_ct.png")

# 2.2. Plot only f1_weighted
g = sns.catplot(data=f1_results_per_ct, x="organ", y="f1_per_ct", hue="Training set", kind="bar",
                height=4, aspect=1, sharey=False)
g.set_axis_labels("Organ", "F1 Score")
g.set_titles("F1 Score: {col_name}")
g.set_titles("F1 per cell type")
plt.show()
# Save figure
g.savefig(result_dir+"f1_score_barplot_cell_type_f1_per_ct.png")


# --------------------------------------------
# 3. Plot f1 score for "cell_subtype": averages f1 scores
# --------------------------------------------

## Load f1_results
f1_results = pd.read_pickle(result_dir_data+"f1_results_local_swarm_cell_subtype.pkl")
f1_results_all_cst = f1_results[f1_results["label"] == "cell_subtype"]
f1_results_all_cst = f1_results_all_cst.melt(id_vars=["organ", "label", "representation", "Training set"],
                                             value_vars=["f1_micro", "f1_macro", "f1_weighted"],
                                             var_name="f1_avg", value_name="f1_score")

## 1. Boxplots
## 1.1 Plot the 3 average methods for f1_score using boxplots using catplot:
g = sns.catplot(data=f1_results_all_cst, x="f1_avg", y="f1_score", hue="Training set", col="organ", kind="box",
                height=4, aspect=1, sharey=False)
g.set_axis_labels("F1 average method", "F1 Score")
g.set_titles("{col_name}".capitalize())
plt.tight_layout()
plt.show()
## Save figure
g.savefig(result_dir+"local_f1_score_boxplot_cell_subtype_f1_avg.png")

## 1.2 Plot only f1_weighted
f1_weighted = f1_results_all_cst[f1_results_all_cst["f1_avg"] == "f1_weighted"]
g = sns.catplot(data=f1_weighted, x="organ", y="f1_score", hue="Training set", kind="box",
                height=4, aspect=1, sharey=False)
g.set_axis_labels("Organ", "F1 Score")
g.set_titles("F1 Score: {col_name}")
g.set_titles("Weighted F1 score")
plt.tight_layout()
plt.show()
## Save figure
g.savefig(result_dir+"local_f1_score_boxplot_cell_subtype_f1_weighted.png")

## 2. Same figures with barplots instead of boxplots
## 2.1. Plot the 3 average methods for f1_score using barplots using catplot:
g = sns.catplot(data=f1_results_all_cst, x="f1_avg", y="f1_score", hue="Training set", col="organ", kind="bar",
                height=4, aspect=1, sharey=False)
g.set_axis_labels("Organ", "F1 Score")
g.set_titles("{col_name}".capitalize())
plt.show()
# Save figure
g.savefig(result_dir+"local_f1_score_barplot_cell_subtype_f1_avg.png")

## 2.2. Plot only f1_weighted
g = sns.catplot(data=f1_weighted, x="organ", y="f1_score", hue="Training set", kind="bar",
                height=4, aspect=1, sharey=False)
g.set_axis_labels("Organ", "F1 Score")
g.set_titles("F1 Score: {col_name}")
g.set_titles("Weighted F1 score")
plt.show()
# Save figure
g.savefig(result_dir+"local_f1_score_barplot_cell_subtype_f1_weighted.png")


# --------------------------------------------
# 4. Plot f1 score for "cell_subtype": f1 scores per cell subtypes
# --------------------------------------------


f1_results_per_cst = f1_results[(f1_results["label"] == "cell_subtype") & f1_results["Training set"].isin(["Local_1", "Local_2", "Local_3"])]
f1_results_per_cst = f1_results_per_cst.drop(
    columns=["y_pred", "y_test", "f1_micro", "f1_macro", "f1_weighted", "training_time"])

f1_results_per_cst_all = []
for organ in f1_results_per_cst["organ"].unique():
    tmp = f1_results_per_cst[f1_results_per_cst["organ"] == organ]
    # Check that unique(tmp["f1_label"]) is of length 1
    assert len(tmp["f1_label"].unique()) == 1
    aaa = tmp["f1_label"].unique().tolist()[0]
    aaa = ast.literal_eval(aaa)
    bbb = [aaa] * tmp.shape[0]
    # del tmp["f1_label"]
    tmp['f1_label'] = bbb
    # Aply the replace("nan", "None") method to all elements in f1_per_ct
    tmp["f1_per_ct"] = tmp["f1_per_ct"].apply(
        lambda x: x.replace("nan", "None"))
    # Apply ast.literal_eval to all elements in f1_per_ct
    tmp['f1_per_ct'] = tmp['f1_per_ct'].apply(ast.literal_eval)
    tmp = tmp.explode(["f1_per_ct", "f1_label"])
    f1_results_per_cst_all.append(tmp)

f1_results_per_cst = pd.concat(f1_results_per_cst_all, axis=0)
f1_results_per_cst.dtypes
## Convert f1_per_ct to float
f1_results_per_cst["f1_per_ct"] = f1_results_per_cst["f1_per_ct"].astype(float)
## Set order of organs
desired_order = organs
# Convert 'organ' column to categorical with the specified order
f1_results_per_cst['organ'] = pd.Categorical(
    f1_results_per_cst['organ'], categories=desired_order, ordered=True)

g = sns.catplot(data=f1_results_per_cst, x="f1_label", y="f1_per_ct", hue="Training set", row="organ", kind="box",
                height=4, aspect=3, sharey=False, sharex=False)
g.set_axis_labels("Cell type", "F1 Score")
g.set_titles("Organ: {row_name}")
for ax in g.axes.flat:
    # Get the current labels
    labels = ax.get_xticklabels()
    # Wrap the labels
    wrapped_labels = wrap_labels([label.get_text()
                                 for label in labels], width=20)
    # Set the new labels
    ax.set_xticklabels(wrapped_labels, rotation=45, ha='right')
# Loop through each facet and add the statistical annotations
for ax, (facet_key, facet_df) in zip(g.axes.flat, f1_results_per_cst.groupby('organ')):
    add_p_values_to_facet(facet_df, ax)
plt.suptitle("F1-scores per cell subtype", y=1.0)
# add spacing between suptitle and subplots to prevent overlap
plt.subplots_adjust(top=0.95)
g.fig.subplots_adjust(hspace=0.9, wspace=0.40)
# Save figure
g.savefig(result_dir+"local_f1_score_boxplot_cell_subtype_f1_per_ct_pval.png", dpi=800)


g = sns.catplot(data=f1_results_per_cst, x="f1_label", y="f1_per_ct", hue="Training set", row="organ", kind="box",
                height=4, aspect=3, sharey=False, sharex=False)
g.set_axis_labels("", "F1 Score")
g.set_titles("Organ: {row_name}")
for ax in g.axes.flat:
    # Get the current labels
    labels = ax.get_xticklabels()
    # Wrap the labels
    wrapped_labels = wrap_labels([label.get_text()
                                 for label in labels], width=20)
    # Set the new labels
    ax.set_xticklabels(wrapped_labels, rotation=45, ha='right', fontsize=8)
plt.subplots_adjust(top=0.95)
g.fig.subplots_adjust(hspace=0.9, wspace=0.40)
# Save figure
g.savefig(
    result_dir+"local_f1_score_boxplot_cell_subtype_f1_per_ct.png", dpi=800)


## Subset organ == "heart", label == "T cell" from f1_results_per_cst
subset = f1_results_per_cst[(f1_results_per_cst["organ"] == "heart") & (f1_results_per_cst["f1_label"] == "T cell")]

## Perform a T test between Local_1 and Local_2
from scipy.stats import ttest_ind
t_stat, p_val = ttest_ind(subset[subset["Training set"] == "Local_1"]["f1_per_ct"],
                          subset[subset["Training set"] == "Local_2"]["f1_per_ct"], equal_var=False)
print(p_val)

# Perform a T test between Local_3 and Local_2
t_stat, p_val = ttest_ind(subset[subset["Training set"] == "Local_3"]["f1_per_ct"],
                            subset[subset["Training set"] == "Local_2"]["f1_per_ct"], equal_var=False)
print(p_val)

# Compute the difference mean of f1_per_ct for every value of f1_label and organ:
means_df = f1_results_per_cst.groupby(["f1_label", "organ", "Training set"])["f1_per_ct"].mean().reset_index()
## Group by f1_label and organ and compute the difference of f1_per_ct between Local_1 and Local_2
means_df = means_df.pivot_table(index=["f1_label", "organ"], columns="Training set", values="f1_per_ct").reset_index()
means_df["diff1"] = means_df["Local_2"] - means_df["Local_1"]
means_df["diff2"] = means_df["Local_3"] - means_df["Local_2"]

differences = [means_df["diff1"].values.tolist(), means_df["diff2"].values.tolist()]
# histogram of differences
plt.hist(differences, bins=20, alpha=0.5)

f1_results[(f1_results["label"] == "cell_subtype") & (f1_results["Training set"].isin(["Local_3"])) & (f1_results["organ"] == "heart")]
        
# 1. Boxplots
# 1.1 Plot the 3 average methods for f1_score using boxplots using catplot:
g = sns.catplot(data=f1_results_per_cst, x="f1_label", y="f1_per_ct", hue="Training set", row="organ", kind="box",
                height=4, aspect=1, sharey=False)
g.set_axis_labels("Cell subtype", "F1 Score")
g.set_titles("{row_name}".capitalize())
plt.tight_layout()
plt.show()
## Save figure
g.savefig(result_dir+"local_f1_score_boxplot_cell_subtype_f1_per_cst.png")

# 1.2 Plot only f1_weighted
g = sns.catplot(data=f1_results_per_cst, x="organ", y="f1_per_cst", hue="Training set", kind="box",
                height=4, aspect=1, sharey=False)
g.set_axis_labels("Organ", "F1 Score")
g.set_titles("F1 Score: {col_name}")
g.set_titles("F1 per cell subtype")
plt.tight_layout()
plt.show()
## Save figure
g.savefig(result_dir+"local_f1_score_boxplot_cell_subtype_f1_per_cst.png")

# 2. Same figures with barplots instead of boxplots
# 2.1. Plot the 3 average methods for f1_score using barplots using catplot:
g = sns.catplot(data=f1_results_per_cst, x="f1_label", y="f1_per_cst", hue="Training set", col="organ", kind="bar",
                height=4, aspect=1, sharey=False)
g.set_axis_labels("Organ", "F1 Score")
g.set_titles("{col_name}".capitalize())
plt.show()
# Save figure
g.savefig(result_dir+"local_f1_score_barplot_cell_subtype_f1_per_cst.png")

# 2.2. Plot only f1_weighted
g = sns.catplot(data=f1_results_per_cst, x="organ", y="f1_per_cst", hue="Training set", kind="bar",
                height=4, aspect=1, sharey=False)
g.set_axis_labels("Organ", "F1 Score")
g.set_titles("F1 Score: {col_name}")
g.set_titles("F1 per cell subtype")
plt.show()
# Save figure
g.savefig(result_dir+"local_f1_score_barplot_cell_subtype_f1_per_cst.png")



