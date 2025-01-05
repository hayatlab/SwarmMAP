import os
from tkinter import Label
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
import torch.nn.functional as nnf
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*Reordering categories will always return a new Categorical object.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*is_categorical_dtype is deprecated and will be removed in a future version.*")


organs = ["heart", "lung", "breast"]
labels = ["cell_type", "cell_subtype"]

ind_design_dict = dict({'heart': 3, "breast": 3, "lung": 2})
## Indic for heart is not satisfactory: change it.

# for organ in organs:
#     for label in ["cell_type"]:

#         # Load data
#         result_dir, data_dir, adata, studies = mlp.load_dataset(organ, use_union_hvgs=True,
#                                                                 filtering=False, label=label)
        
#         ## Load result_df to use simulation settings
#         result_df = pd.read_csv(result_dir+f"results_counts_{label}.csv")
#         ## Only use representation == "counts"
#         result_df = result_df[result_df["representation"] == "counts"]
#         result_df = result_df[result_df["label"] == "cell_type"]

#         result_df_train_on_3 = result_df[result_df["train_ind"].apply(eval).apply(len) == 3]
#         y_pred_proba_ls = [None] * result_df_train_on_3.shape[0]

#         # for ind_design in range(result_df_train_on_3.shape[0]):

#         ind_design = ind_design_dict[organ]

#         # --------------------------------
#         ## Train and test model
#         # --------------------------------

#         test_ind = result_df_train_on_3.test_ind.iloc[ind_design]
#         test = adata[adata.obs['study'] == studies[result_df_train_on_3.test_ind.iloc[ind_design]]].copy()
#         train_ind = eval(result_df_train_on_3.train_ind.iloc[ind_design])
#         train = adata[adata.obs['study'].isin([studies[ind] for ind in train_ind])].copy()

#         label = result_df_train_on_3.label.iloc[ind_design]

#         # Apply OHE
#         ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
#         encoder_data = pd.DataFrame(sorted(train.obs[label].unique())).values.reshape(-1, 1)
#         ohe.fit(encoder_data)

#         ## Convert data to tensor
#         y_test = test.obs[label]
#         y_train = train.obs[label]
#         y_test_transformed = ohe.transform(y_test.values.reshape(-1, 1))
#         y_train_transformed = ohe.transform(y_train.values.reshape(-1, 1))
#         X_test = test.X.toarray()
#         X_train = train.X.toarray()
#         X_test = torch.tensor(X_test, dtype=torch.float32)
#         y_test_transformed = torch.tensor(y_test_transformed, dtype=torch.float32)
#         X_train = torch.tensor(X_train, dtype=torch.float32)
#         y_train_transformed = torch.tensor(y_train_transformed, dtype=torch.float32)

#         # Train model
#         usecuda = torch.cuda.is_available()
#         device = torch.device("cuda" if usecuda else "cpu")
#         # Experiment
#         n_epochs = 100
#         batch_size = 128
#         model = mlp.train_mlp(
#             X_train, y_train_transformed, device, 
#             n_epochs, batch_size)

#         # Test model
#         y_pred = mlp.test_mlp(model, X_test, y_test_transformed, device, ohe)
#         y_test_transformed = y_test_transformed.cpu().detach().numpy()
#         y_pred_labels = np.take(ohe.categories_, np.argmax(y_pred, axis=1))

#         # --------------------------------
#         # Save prediction score
#         # --------------------------------
        
#         y_pred_proba_ls[ind_design] = nnf.softmax(torch.tensor(y_pred), dim=1)

#         # save y_pred_proba_ls using pickle
#         with open(result_dir+f"umap/y_pred_proba_ls_{label}.pkl", "wb") as f:
#             pickle.dump(y_pred_proba_ls, f)

#         os.path.exists(result_dir+f"umap/y_pred_proba_ls_{label}.pkl")


label = labels[0]
organ = organs[2]
organ = 'lung'
organ = 'heart'
organ = 'breast'

for organ in organs:
    for label in ["cell_type"]:

        # Load data
        result_dir, data_dir, adata, studies = mlp.load_dataset(organ, use_union_hvgs=True,
                                                                filtering=False, label=label, renorm=True)

        
        # Load result_df to use simulation settings
        # result_df = pd.read_pickle(result_dir+f"results_counts_{label}.pkl")
        
        ## Same file but computed with renorm=True
        result_df = pd.read_pickle(result_dir+f"results_counts_cell_type_renorm_test.pkl")
        
        result_df = result_df[result_df["representation"] == "counts"]
        result_df = result_df[result_df["label"] == label]

        ## Only use training design with 3 training studies
        # result_df_train_on_3 = result_df[result_df["train_ind"].apply(
        #     eval).apply(len) == 3]
        # no need with the latest version of result_counts.pkl:
        result_df_train_on_3 = result_df
        y_pred_proba_ls = [None] * result_df_train_on_3.shape[0]

        ind_design = ind_design_dict[organ]
        ind_design = 2 # for heart

        # Define train and test data using ind_design_dict
        test_ind = result_df_train_on_3.test_ind.iloc[ind_design]
        test = adata[adata.obs['study'] == studies[result_df_train_on_3.test_ind.iloc[ind_design]]].copy()
        test_study = test.obs['study'].values[0].replace(" ", "_")
        print(test_study)
        
        # train_ind = eval(result_df_train_on_3.train_ind.iloc[ind_design])
        train_ind = result_df_train_on_3.train_ind.iloc[ind_design]
        train = adata[adata.obs['study'].isin(
            [studies[ind] for ind in train_ind])].copy()

        # Apply OHE
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder_data = pd.DataFrame(
            sorted(train.obs[label].unique())).values.reshape(-1, 1)
        ohe.fit(encoder_data)

# --------------------------------
# Load prediction score
# --------------------------------

with open(result_dir+f"umap/y_pred_proba_ls_{label}.pkl", "rb") as f:
    y_pred_proba_ls = pickle.load(f)
# y_pred = y_pred_proba_ls[ind_design_dict[organ]].detach().numpy()

# for "heart"
y_pred = y_pred_proba_ls[3].detach().numpy()

# --------------------------------
## Compute and plot umap
## --------------------------------
if "X_diffmap" in test.obsm.keys():
    del test.obsm["X_diffmap"]
sc.pp.pca(test)
sc.pp.neighbors(test)
sc.tl.umap(test)
sc.pl.umap(test, color=[label])
test.obs[label].value_counts()

## --------------------------------
## Plot the 2 best and 2 worst predictions for cell types
## --------------------------------

## Get the ct with best and worst F1 score
f1_df = pd.DataFrame({"label_ls": ohe.categories_[0], 
                      "f1": f1_score(test.obs[label], np.take(ohe.categories_, np.argmax(y_pred, axis=1)), average=None)})
f1_df = f1_df.sort_values("f1", ascending=False)
## Set y_test values in decreasing order from the most frequent to the least frequent
y_test_ordered = f1_df["label_ls"].tolist()

cts_to_plot = [y_test_ordered[ind] for ind in [0, -2, 1, -1]]

fig, axs = plt.subplots(len(cts_to_plot) // 2, 5, 
                        figsize=(6 * 2 * 1.2, 2.5 * len(cts_to_plot) / 1.5),
                        gridspec_kw={'width_ratios': [1, 1, 0.05, 1, 1]}
)
for ind_ct in range(len(cts_to_plot)):
    ct = cts_to_plot[ind_ct]
    ind_ct2 = list(ohe.categories_[0]).index(ct)
    ### Compute binary variable for cell type
    test.obs["is_cell_type"] = [int(val == ct) for val in test.obs["cell_type"].tolist()]
    ### Compute prediction score for cell type
    test.obs['cell_type_prediction_score'] = y_pred_proba_ls[ind_design][:, ind_ct2]
    # test.obs['cell_type_prediction_score'] = y_pred_proba_ls[0][:, ind_ct]
    ### Plot
    ax_ind1 = ind_ct // 2
    ax_ind2 = ind_ct % 2 * 2
    if ax_ind2 > 1: 
        ax_ind2 += 1
    sc.pl.umap(test, color=["is_cell_type"], wspace=0.5, legend_loc=False, 
            title="", ax=axs[ax_ind1, ax_ind2], show=False,
            vmin=0, vmax=1, colorbar_loc=None)
    sc.pl.umap(test, color=["cell_type_prediction_score"], title = "",
            wspace=0.5, legend_loc=False, ax=axs[ax_ind1, ax_ind2 + 1], show=False,
            vmin=0, vmax=1)
    ## Add "\n" after every "," in ct
    ct_label = ct.replace(",", ",\n")
    ct_label = ct_label.replace("of mammary", "\nof mammary")
    axs[ax_ind1, ax_ind2].set_ylabel(ct_label, fontsize=12, wrap=True)
    axs[ax_ind1, ax_ind2 + 1].set_ylabel("")
    ## Remove x and y axes' labels
    for ind_ax in range(2):
        axs[ax_ind1, ax_ind2 + ind_ax].set_xticks([])
        axs[ax_ind1, ax_ind2 + ind_ax].set_yticks([])
        axs[ax_ind1, ax_ind2 + ind_ax].set_xlabel("")
        axs[ax_ind1, ax_ind2 + ind_ax].set(frame_on=False)
    if ax_ind1 == 0:
        axs[ax_ind1, ax_ind2].set_title("True cell type")
        axs[ax_ind1, ax_ind2 + 1].set_title("Prediction score")
# Set the 3rd column as empty
for ax in axs[:, 2]:
    ax.axis("off")  
plt.figtext(0.24, 1.05, "Top 2 best predictions",
            va="center", ha="center", size=15)
plt.figtext(0.74, 1.05, "Top 2 worst predictions",
            va="center", ha="center", size=15)
fig.tight_layout()
## Add a common color bar
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.show()

## Save the figure
plt.savefig(os.path.join(result_dir+"umap/" f"umap_prediction_score_test_{test_study}_critical_cts_{label}_new.png"), 
            bbox_inches = 'tight', dpi=600)
print("Saved figure: ", os.path.join(result_dir+"umap/" f"umap_prediction_score_test_{test_study}_critical_cts_{label}_new.png"))

## --------------------------------
## Plot the 2 best and 2 worst predictions for cell subtypes
## --------------------------------
