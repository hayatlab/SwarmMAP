from scipy.stats import median_abs_deviation
import os
from unittest import result
from networkx import number_connected_components
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, auc, classification_report, accuracy_score
import torch
import scvi
import torch.nn as nn
import torch.optim as optim
import scanpy as sc
import json
import numpy as np
import scipy as sp
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data.sampler import WeightedRandomSampler
from PIL import Image
import itertools

def renormalize(X):
    """
    Renormalizes the input matrix X.

    Parameters:
    X (scipy.sparse.csr_matrix): Input matrix to be renormalized.

    Returns:
    scipy.sparse.csr_matrix: Renormalized matrix.
    """
    aa = X.toarray()
    aa = np.expm1(aa)
    aa = aa / aa.sum(axis=1)[:, np.newaxis] * 10000
    aa[np.where(np.isnan(aa).any(axis=1))] = 0
    aa = np.log1p(aa)
    aa = sp.sparse.csr_matrix(aa)
    return aa

class Multiclass(nn.Module):
    """
    A simple multilayer perceptron (MLP) for multiclass classification.

    Attributes:
        - input_dim (int): The dimensionality of the input features.
        - output_dim (int): The number of classes for classification.
        - dropout_rate (float): The dropout rate for regularization.
        - l1 (int): The number of neurons in the first hidden layer.
        - l2 (int): The number of neurons in the second hidden layer.
        - activation (torch.nn.Module): The activation function applied after each hidden layer.

    Layers:
        - Fully connected layer (fc1) with input_dim features and l1 neurons.
        - Dropout layer with dropout_rate.
        - Activation function.
        - Fully connected layer (fc2) with l1 neurons and l2 neurons.
        - Dropout layer with dropout_rate.
        - Activation function.
        - Fully connected layer (fc3) with l2 neurons and output_dim neurons.

    Methods:
        - forward(x): Forward pass of the neural network.
    """

    def __init__(self, input_dim, output_dim, dropout_rate, l1=128, l2=32, activation=nn.Tanh()):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, l1),
            nn.Dropout(p=dropout_rate),
            activation,
            nn.Linear(l1, l2),
            nn.Dropout(p=dropout_rate),
            activation,
            nn.Linear(l2, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)

def stats(model, X_test, y_test, output_dim, ohe, device, experiment, result_dir):
    """
    Computes and saves various statistics for the model.

    Parameters:
    model (nn.Module): The trained model.
    X_test (torch.Tensor): The test input data.
    y_test (torch.Tensor): The test target data.
    output_dim (int): The number of classes for classification.
    ohe (OneHotEncoder): The one-hot encoder used for encoding the target data.
    device (torch.device): The device (e.g., GPU or CPU) to run the model on.
    experiment (str): The name of the experiment.
    result_dir (str): The directory to save the results.
    """
    # Define file name
    file_name_prefix = experiment.replace(" ", "_")

    # Move testing data and labels to a specified device (e.g., GPU or CPU)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Set model to evaluation mode
    model.eval()

    # Predict labels from test data
    y_pred = model(X_test)

    y_test = y_test.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    y_pred_labels = np.take(ohe.categories_, np.argmax(y_pred, axis=1))
    y_test_labels = ohe.inverse_transform(y_test).flatten()

    # Classification Report
    report = classification_report(
        y_test_labels, y_pred_labels, output_dict=True)
    report = pd.DataFrame(report).transpose()

    report.to_csv(os.path.join(result_dir, file_name_prefix+
                               "_classification_report.csv")) #save
    print(report.to_string)

    # AUROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(output_dim):
        fpr[i], tpr[i], thresholds = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.style.use('default')
    colors = plt.cm.tab20c(np.linspace(0, 1, output_dim))
    plt.figure()
    for i in range(output_dim):
        plt.plot(fpr[i], tpr[i], lw=3, linestyle='dotted', color=colors[i],
            label='{0} (AUROC = {1:0.2f})'
            ''.format(ohe.categories_[0][i], roc_auc[i]))
    plt.plot(fpr["micro"], tpr["micro"],
            label='Average (AUROC = {0:0.2f})'
            ''.format(roc_auc["micro"]), linewidth=3, color='royalblue')

    plt.plot([0, 1], [0, 1], color='lightgrey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{experiment}")
    plt.legend(loc="lower left", bbox_to_anchor=(1, 0))

    plt.savefig(os.path.join(result_dir, file_name_prefix+
                             "_roc.png"), bbox_inches = 'tight') # save
    plt.show()
    plt.clf()

    # F1 Score
    categories = np.transpose(ohe.categories_).flatten()

    score = f1_score(y_test_labels, y_pred_labels,
                     labels=categories, average=None)
    score_df = pd.DataFrame({'cell_type': categories, 'score': score})

    plt.figure(figsize=(8, 8))
    sns.barplot(score_df, x='cell_type', y='score')
    plt.ylabel("F1 score")
    plt.xlabel("Cell type")
    plt.xticks(rotation=90)
    plt.title(f"{experiment}")

    plt.savefig(os.path.join(result_dir, file_name_prefix+
                             "_f1.png"), bbox_inches = 'tight') # save
    plt.show()
    plt.clf()

    # Confusion Matrix
    pred_df = pd.DataFrame(list(zip(y_pred_labels, y_test_labels)),
                             columns = ['pred_label', 'true_label'])

    df = pd.crosstab(pred_df['pred_label'], pred_df['true_label'])
    norm_df = df / df.sum(axis=0)

    plt.figure(figsize=(8, 8))
    _ = plt.pcolor(norm_df)
    _ = plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, rotation=90)
    _ = plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{experiment}")

    plt.savefig(os.path.join(result_dir, file_name_prefix+"_confusion.png"), bbox_inches = 'tight') # save
    plt.show()
    plt.clf()
    
    # Save Result Arrays  
    np.save(os.path.join(result_dir, file_name_prefix+"_y_pred.npy"), y_pred_labels)
    np.save(os.path.join(result_dir, file_name_prefix+"_y_test.npy"), y_test_labels)

def main(X_train, y_train, X_test, y_test, ohe, experiment, result_dir, n_epochs=50, batch_size=500):
    """
    Main function to train the model and compute statistics.

    Parameters:
    X_train (torch.Tensor): The training input data.
    y_train (torch.Tensor): The training target data.
    X_test (torch.Tensor): The test input data.
    y_test (torch.Tensor): The test target data.
    ohe (OneHotEncoder): The one-hot encoder used for encoding the target data.
    experiment (str): The name of the experiment.
    result_dir (str): The directory to save the results.
    n_epochs (int): The number of epochs for training (default: 5).
    batch_size (int): The batch size for training (default: 500).
    """
    # GPU usage
    usecuda = torch.cuda.is_available()
    if usecuda:
        print('CUDA is accessible')
    else:
        print('CUDA  is not accessible')
    device = torch.device("cuda" if usecuda else "cpu")

    # Define data dimension
    input_dim = X_train.size()[1]
    output_dim = y_train.size()[1]

    # Model, loss metric and optimizer
    model = Multiclass(input_dim=input_dim , output_dim=output_dim).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Data loading
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
    trainDs = torch.utils.data.TensorDataset(X_train,y_train)
    trainLoader = torch.utils.data.DataLoader(trainDs,batch_size)

    # Training loop
    for epoch in range(n_epochs):
        # Set model in training mode and run through each batch
        model.train()
        for batchIdx, (X_batch, y_batch) in enumerate(trainLoader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            # Forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            # Compute and store metrics
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            if batchIdx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}'.format(
                    epoch, n_epochs-1, batchIdx * len(X_batch), len(trainLoader.dataset),
                    100. * batchIdx / len(trainLoader), loss.item(), acc*100))

        # Set model in evaluation mode and run through the val set
        X_val, y_val = X_val.to(device), y_val.to(device)
        model.eval()
        y_pred = model(X_val)
        ce = loss_fn(y_pred, y_val)
        acc = (torch.argmax(y_pred, 1) == torch.argmax(y_val, 1)).float().mean()
        ce = float(ce)
        acc = float(acc)
        print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")

    # Run statistics
    stats(model, X_test, y_test, output_dim, ohe, device, experiment, result_dir)

class FocalLoss(nn.Module):
    """
    Focal loss for multiclass classification.

    Attributes:
        - alpha (torch.Tensor): The weight for each class (optional).
        - gamma (float): The focusing parameter (default: 2).

    Methods:
        - forward(inputs, targets): Computes the focal loss.
    """

    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss

class FocalLoss2(nn.Module):
    """
    Focal loss for multiclass classification.

    Attributes:
        - alpha (torch.Tensor): The weight for each class (optional).
        - gamma (float): The focusing parameter (default: 2).
        - reduction (str): The reduction method for the loss (default: 'none').

    Methods:
        - forward(input_tensor, target_tensor): Computes the focal loss.
    """

    def __init__(self, alpha=None, gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = torch.nn.functional.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return torch.nn.functional.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            alpha=self.alpha,
            reduction=self.reduction
        )

def get_f1_score(model, X_test, y_test, ohe, device, percent):
    """
    Computes the F1 score for the model.

    Parameters:
    model (nn.Module): The trained model.
    X_test (torch.Tensor): The test input data.
    y_test (torch.Tensor): The test target data.
    ohe (OneHotEncoder): The one-hot encoder used for encoding the target data.
    device (torch.device): The device (e.g., GPU or CPU) to run the model on.
    percent (float): The percentage of data to use for computing the F1 score.

    Returns:
    float: The F1 score.
    """

    # Move testing data and labels to a specified device (e.g., GPU or CPU)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Set model to evaluation mode
    model.eval()

    # Predict labels from test data
    y_pred = model(X_test)

    y_test = y_test.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    y_pred_labels = np.take(ohe.categories_, np.argmax(y_pred, axis=1))
    y_test_labels = ohe.inverse_transform(y_test).flatten()

    # F1 Score
    categories = np.transpose(ohe.categories_).flatten()

    score = f1_score(y_test_labels, y_pred_labels,
                     labels=categories, average=None)
    score_df = pd.DataFrame({'cell_type': categories, 'score': score,
                             'percent_ct': percent})
    return (score_df)

def train_mlp(X_train, y_train, device, n_epochs, batch_size, dropout_rate=0,
              lr=1e-3, weight_decay=0, imbalance_resample=False, use_tensorboard=False):
    
    if use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter('runs/lung')
    
    # Define data dimension
    input_dim = X_train.size()[1]
    output_dim = y_train.size()[1]

    # Model, loss metric and optimizer
    model = Multiclass(
        input_dim=input_dim, output_dim=output_dim, dropout_rate=dropout_rate).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Data loading
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.33, random_state=42)
    trainDs = torch.utils.data.TensorDataset(X_train, y_train)
    
    if imbalance_resample:
        target = torch.argmax(y_train, 1)
        class_sample_count = np.array(
            [len(np.where(target == t)[0]) for t in np.unique(target)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in target])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weigth = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        # trainLoader = torch.utils.data.DataLoader(
        #     trainDs,
        #     sampler=ImbalancedDatasetSampler(trainDs),
        #     batch_size=batch_size)
    else:   
        sampler = None
    trainLoader = torch.utils.data.DataLoader(trainDs, batch_size, sampler=sampler)

    # Training loop
    for epoch in range(n_epochs):
          # Set model in training mode and run through each batch
        model.train()
        for batchIdx, (X_batch, y_batch) in enumerate(trainLoader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            # Forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            # Compute and store metrics
            acc = (torch.argmax(y_pred, 1) ==
                   torch.argmax(y_batch, 1)).float().mean()
            if batchIdx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}'.format(
                    epoch, n_epochs-1, batchIdx *
                    len(X_batch), len(trainLoader.dataset),
                    100. * batchIdx / len(trainLoader), loss.item(), acc*100))
                if use_tensorboard:
                    writer.add_scalar('training loss',
                                    loss.item(),
                                    epoch * len(trainLoader) + batchIdx)
        
        # Set model in evaluation mode and run through the val set
        X_val, y_val = X_val.to(device), y_val.to(device)
        model.eval()
        y_pred = model(X_val)
        ce = loss_fn(y_pred, y_val)
        acc = (torch.argmax(y_pred, 1) ==
               torch.argmax(y_val, 1)).float().mean()
        ce = float(ce)
        acc = float(acc)
        print(
            f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")
            
    return(model)

def test_mlp(model, X_test, y_test, device, ohe):

    X_test_dev, y_test = X_test.to(device), y_test.to(device)
    model.eval()
    y_pred = model(X_test_dev).cpu().detach().numpy()
    # y_pred_labels = ohe.inverse_transform(y_pred).flatten()
    
    return(y_pred)

def flatten(xss):
    return [x for xs in xss for x in xs]

def compute_score(index1, index2, cell_level, score_type, 
                  ohe, y_pred_labels_ls_ls, test, y_categories):
    # Remove potential None values
    y_pred_labels = ohe.inverse_transform(
        y_pred_labels_ls_ls[index1][index2]).flatten().tolist()
    nan_positions = [i for i in range(
        len(y_pred_labels)) if y_pred_labels[i] is None]
    y_pred_labels = np.delete(y_pred_labels, nan_positions, axis=0)
    y_test = test.obs[cell_level].values
    y_test = np.delete(y_test, nan_positions, axis=0)
    # Compute F1 score
    score = f1_score(
        y_test,
        y_pred_labels,
        labels=y_categories, average=score_type)
    return(score)

def compute_accuracy(index1, index2, cell_level,
                  ohe, y_pred_labels_ls_ls, test, y_categories):
    # Remove potential None values
    y_pred_labels = ohe.inverse_transform(
        y_pred_labels_ls_ls[index1][index2]).flatten().tolist()
    nan_positions = [i for i in range(
        len(y_pred_labels)) if y_pred_labels[i] is None]
    y_pred_labels = np.delete(y_pred_labels, nan_positions, axis=0)
    y_test = test.obs[cell_level].values
    y_test = np.delete(y_test, nan_positions, axis=0)
    # Compute accuracy score
    acc = accuracy_score(
        y_test,
        y_pred_labels)
    return (acc)

def compute_auc(index1, index2, y_test_transformed, y_pred_labels_ls_ls):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    output_dim = y_pred_labels_ls_ls[index1][index2].shape[1]
    for i in range(output_dim):
        fpr[i], tpr[i], thresholds = roc_curve(y_test_transformed[:, i],
                                               y_pred_labels_ls_ls[index1][index2][:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_test_transformed.ravel(), y_pred_labels_ls_ls[index1][index2].ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return (roc_auc)

def compute_representation_separately(adata, representation, var_name, n_comps=50):
    """
    Compute either PCA or scVI representation separately for each unique value of a given variable in the AnnData object.

    Parameters:
        adata (AnnData): Annotated data matrix with observations (cells) in rows and variables (features) in columns.
        representation (str): The type of representation to compute. Can be either "pca" or "scVI".
        var_name (str): Name of the variable in `adata.obs` that contains the values to separate the data.
        n_comps (int): Number of components to compute for PCA or scVI.

    Returns:
        np.ndarray: Concatenated PCA or scVI results for all separated datasets.

    """
    variable = adata.obs[var_name].unique().tolist()
    adata_ls = [adata[adata.obs[var_name] == value, :].copy() for value in variable]
    for adata in adata_ls:
        if representation == "pca":
            sc.tl.pca(adata, n_comps=n_comps)
        elif representation == "scVI":
            scvi.settings.seed = 0
            adata.layers["counts"] = adata.X.copy()
            scvi.model.SCVI.setup_anndata(adata, layer = "counts")
            vae = scvi.model.SCVI(adata, n_latent=n_comps)
            vae.train()
            adata.obsm['X_scvi'] = vae.get_latent_representation()
            # adata.uns["scVI_latent_dist"] = vae.get_latent_distribution(return_dist=True)
    rep_ls = [adata.obsm['X_pca'].copy() if representation == "pca" else adata.obsm['X_scvi'].copy() for adata in adata_ls]
    rep = np.concatenate(rep_ls, axis=0)
    return rep

def get_training_ind(test_ind, train_on):
    s = list(range(0, 4))
    tmp = list(itertools.combinations(s, train_on))
    tmp = [list(aa) for aa in tmp]
    return tmp
    

def create_sim_setting(n_studies):
    
    train_ind_ls = []
    for train_on in range(1, n_studies):
        for test_ind in list(range(0, n_studies)):
            train_ind_ls.extend(get_training_ind(test_ind, train_on))

    train_on = np.repeat(range(1, n_studies), n_studies, axis=0)
    test_ind_ls = []
    for ind_rep in [4, 6, 4]:
        test_ind_ls.extend(np.repeat(range(0, n_studies), ind_rep, axis=0))
        
        
    sim_setting_df = pd.DataFrame(zip(test_ind_ls, train_ind_ls),
                                columns=['test_ind', 'train_ind'], dtype='object')
    sim_setting_df.insert(0, "train_on_n_studies", [len(a) for a in train_ind_ls])
    ## Set dtype of test_ind and train_on_n_studies to int
    sim_setting_df = sim_setting_df.astype({'test_ind': 'int', 'train_on_n_studies': 'int'})
    ## Remove lines where test_ind is in train_ind
    sim_setting_df = sim_setting_df[sim_setting_df.apply(lambda x: x.test_ind not in x.train_ind, axis=1)]
    
    return sim_setting_df

def load_dataset(organ, use_union_hvgs=True, filtering=False, label=None, renorm=False, new_order=False):
    if organ not in ["heart", "lung", "breast"]:
        raise ValueError("The organ must be either 'heart', 'lung', or 'breast'.")
    
    data_dir = f'/rwthfs/rz/cluster/work/zd775156/{organ}/'
    
    if organ == "heart":
        
        if use_union_hvgs:
            
            adata = sc.read_h5ad(data_dir+organ+f"_clean_subset_4_union_hvgs.h5ad")
            
            if label is None:
                raise ValueError("label must be specified when use_union_hvgs is True and organ is heart.")
            elif label == "cell_subtype":
                ## remove cells for which label is NaN
                adata = adata[~adata.obs[label].isna()].copy()
            
            if filtering:
                with open(os.path.join(data_dir, 'cell_types_to_rm_union_hvgs.json'), 'rb') as f:
                    cts_to_rm_union_hvgs = json.load(f)
                key = "cts_hf" if label == "cell_type" else "csts_hf"
                adata = adata[~adata.obs[label].isin(cts_to_rm_union_hvgs[key])]

        elif not use_union_hvgs:
            adata = sc.read_h5ad(os.path.join(
                data_dir, organ+'_clean_subset_4.h5ad'))

            if filtering:
                # Hard filtering boils down to removing Epicardium
                adata = adata[~adata.obs["cell_type"].isin(["Epicardium"])]
                
    elif organ == "lung":
        
        if use_union_hvgs:
            adata = sc.read_h5ad(
                data_dir+organ+"_clean_subset_4_union_hvgs.h5ad")
            
            if filtering:
                with open(os.path.join(data_dir, 'cell_types_to_rm_union_hvgs.json'), 'rb') as f:
                    cts_to_rm_union_hvgs = json.load(f)
                key = "cts_hf" if label == "cell_type" else "csts_hf"
                adata = adata[~adata.obs[label].isin(
                    cts_to_rm_union_hvgs[key])]
                
        elif not use_union_hvgs:
            adata = sc.read_h5ad(os.path.join(data_dir, 'lung_clean_subset_4.h5ad'))

            if filtering:
                with open(os.path.join(data_dir, 'cell_types_to_rm.json'), 'rb') as f:
                    cell_types_to_rm = json.load(f)
                adata = adata[~adata.obs["cell_type"].isin(cell_types_to_rm["cts_hf"])]
        
    elif organ == "breast":
        
        if use_union_hvgs:
            
            if new_order:
                # adata = sc.read_h5ad(data_dir+organ+"_clean_subset_4_union_hvgs_train_kevin_normal_vars.h5ad")
                adata = sc.read_h5ad(data_dir+"final_order/"+organ+"_clean_subset_4_union_hvgs_renorm.h5ad")
            else:
                if renorm:
                    adata = sc.read_h5ad(
                        data_dir+organ+"_clean_subset_4_union_hvgs_renorm.h5ad")
                else:
                    adata = sc.read_h5ad(
                        data_dir+organ+"_clean_subset_4_union_hvgs.h5ad")
            
            if filtering:
                with open(os.path.join(data_dir, 'cell_types_to_rm_union_hvgs.json'), 'rb') as f:
                    cts_to_rm_union_hvgs = json.load(f)
                key = "cts_hf" if label == "cell_type" else "csts_hf"
                adata = adata[~adata.obs[label].isin(
                    cts_to_rm_union_hvgs[key])]

        elif not use_union_hvgs:
            
            adata = sc.read_h5ad(os.path.join(data_dir, f'{organ}_clean_subset_4.h5ad'))
            
            if filtering:
                with open(os.path.join(data_dir, 'cell_types_to_rm.json'), 'rb') as f:
                    cell_types_to_rm = json.load(f)
                adata = adata[~adata.obs["cell_type"].isin(cell_types_to_rm["cts_hf"])]
    
    # Studies
    studies = sorted(adata.obs['study'].unique())
    
    # Result_dir
    if use_union_hvgs:
        result_dir = "/home/zd775156/sclabel/results/"+organ+"/subset_4/local_union_hvgs/"
    else:
        result_dir = "/home/zd775156/sclabel/results/"+organ+"/subset_4/local/"
        
    return result_dir, data_dir, adata, studies

## Define a function returning the lowest common ancestor of a list of nodes, given a nx.DiGraph graph
def get_lowest_common_ancestor(graph, nodes, id_to_name, name_to_id):
    import networkx as nx
    ## Get the list of nodes
    nodes_id = [name_to_id[x] for x in nodes]
    ## Get the lowest common ancestor of the nodes
    lca = nx.lowest_common_ancestor(graph, nodes_id[0], nodes_id[1])
    for node in nodes_id[2:]:
        lca = nx.lowest_common_ancestor(graph, lca, node)
    return id_to_name[lca]


def is_outlier(adata, metric: str, nmads: int):
        M = adata.obs[metric]
        outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
            np.median(M) + nmads * median_abs_deviation(M) < M)
        return outlier
    

def rare_cell_types(adata, label, filtering_type):
    if filtering_type not in ["soft", "hard"]:
        raise ValueError("filtering_type must be either 'soft' or 'hard'")
    if label not in ["cell_type", "cell_subtype"]:
        raise ValueError("label must be either 'cell_type' or 'cell_subtype'")

    if filtering_type == "hard":

        # Remove cell types with < 10 cells in any dataset:
        tmp = adata.obs[[label, 'study']].value_counts()
        multiindex = pd.MultiIndex.from_product(
            [adata.obs[label].unique().tolist(),
             adata.obs.study.unique().tolist()],
            names=[label, 'study'])
        tmp12 = tmp.reindex(multiindex, fill_value=0).reset_index(name="count")

        cts_to_rm = list(set(
            tmp12[tmp12["count"] < 10].reset_index()[label].tolist()))

        print(str(adata.obs[label].nunique() - len(cts_to_rm)) + " " +
              label.replace("_", " ") +
              ' remain after hard filtering out of ' +
              str(adata.obs[label].nunique()))

        return cts_to_rm

    if filtering_type == "soft":

        # Remove cell types with < 10 cells in a majority of datasets:

        tmp2 = adata.obs[[label, 'study']].value_counts()
        multiindex = pd.MultiIndex.from_product(
            [adata.obs[label].unique().tolist(),
             adata.obs.study.unique().tolist()],
            names=[label, 'study'])
        tmp22 = tmp2.reindex(
            multiindex, fill_value=0).reset_index(name="count")
        assert tmp22.shape[0] == adata.obs[label].nunique() * \
            adata.obs.study.nunique()
        tmp22['is_rare'] = tmp22['count'] < 10
        del tmp22['count']
        tmp22.set_index("is_rare").value_counts()
        tmp3 = tmp22.groupby(label)['is_rare'].sum()

        cts_to_rm = tmp3[tmp3 >= math.floor(
            (adata.obs['study'].nunique()+1)/2)]
        cts_to_rm = cts_to_rm.index.tolist()
        print(str(adata.obs[label].nunique() - len(cts_to_rm)) + " " +
              label.replace("_", " ") +
              " remain after soft filtering out of " +
              str(adata.obs[label].nunique()))

        return cts_to_rm


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
    ## Initiate dst to white
    dst.paste((255, 255, 255), [0, 0, dst.width, dst.height])
    # If width are not equal, offset the image to the bottom by the difference
    if im1.height < im2.height:
        dst.paste(im1, (0, (im2.height-im1.height) // 2))
        dst.paste(im2, (im1.width, 0))
    elif im1.height > im2.height:
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, (im1.height-im2.height) // 2))
    else:
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_h_reccursive(im_list):
    if len(im_list) == 1:
        return im_list[0]
    im1 = im_list[0]
    im2 = get_concat_h_reccursive(im_list[1:])
    return get_concat_h(im1, im2)

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height))
    # Initiate dst to white
    dst.paste((255, 255, 255), [0, 0, dst.width, dst.height])
    ## If width are not equal, offset the image to the right by the difference
    if im1.width < im2.width:
        dst.paste(im1, ((im2.width-im1.width) // 2, 0))
        dst.paste(im2, (0, im1.height))
    elif im1.width > im2.width:
        dst.paste(im1, (0, 0))
        dst.paste(im2, ((im1.width-im2.width) // 2, im1.height))
    else:
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
    return dst

def get_concat_v_reccursive(im_list):
    if len(im_list) == 1:
        return im_list[0]
    im1 = im_list[0]
    im2 = get_concat_v_reccursive(im_list[1:])
    return get_concat_v(im1, im2)

## Define function get_concat_v_reccursive, which does not center the pictures horizontally
## but rather aligns them to the left
def get_concat_v_reccursive_left(im_list):
    if len(im_list) == 1:
        return im_list[0]
    im1 = im_list[0]
    im2 = get_concat_v_reccursive_left(im_list[1:])
    return get_concat_v_left(im1, im2)

def get_concat_v_left(im1, im2):
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height))
    # Initiate dst to white
    dst.paste((255, 255, 255), [0, 0, dst.width, dst.height])
    # Paste im1 to the top-left corner
    dst.paste(im1, (0, 0))
    # Paste im2 to the left, below im1
    dst.paste(im2, (0, im1.height))
    return dst
