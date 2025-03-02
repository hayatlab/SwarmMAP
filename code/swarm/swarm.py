## Load packages
# General utility
import os
import glob
import json
import datetime
# Computation 
import pandas as pd
import numpy as np
import scanpy as sc
# Graphics
import matplotlib.pyplot as plt
import seaborn as sns
# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, auc, classification_report
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
# Swarm Learning
#from swarmlearning.pyt import SwarmCallback

#from importlib import util
#import sys
#spec = util.spec_from_file_location("swarmlearning", "/path/to/file.py")
#foo = util.module_from_spec(spec)
#sys.modules["module.name"] = foo
#spec.loader.exec_module(foo)
#foo.MyClass()


## MLP model
class Multiclass(nn.Module):
    """
    A simple multilayer perceptron (MLP) for multiclass classification.

    Attributes:
        - input_dim (int): The dimensionality of the input features.
        - output_dim (int): The number of classes for classification.

    Layers:
        - Fully connected layer (fc1) with input_dim features and 32 neurons.
        - Tanh activation function.
        - Fully connected layer (fc2) with 32 neurons.
        - Tanh activation function.
        - Fully connected layer (fc3) with output_dim neurons.

    Methods:
        - forward(x): Forward pass of the neural network.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        return out


## Data Import and Preparation
def loadData(dataDir, experiment):
    """
    load data from dataDir, preprocess and return train and test data in torch tensors

    Parameters
    ----------
    dataDir : str
        Directory where data is stored
    experiment : int
        Experiment: 0, 1, 2, 3

    Returns
    -------
    X_train_tensor : torch tensor
    y_train_tensor : torch tensor
    X_test_tensor : torch tensor
    y_test_tensor : torch tensor
    input_dim : int
    output_dim : int
    ohe : sklearn.preprocessing.OneHotEncoder fitted
    """
    
    # Load data
    data_path = glob.glob(os.path.join(dataDir,'heart_atlas'))
    data_path = data_path[0]
    print(f'Loading data from {data_path}')
    atlas = sc.read_h5ad(data_path+'/subatlas.h5ad')
    print("Data loaded successfully")
    
    # Read the JSON file for experiment information
    with open(data_path+'/exp_data.json', 'r') as json_file:
        data = json.load(json_file)
    #Specify the experiment ID you're interested in
    target_exp_id = int(experiment)
    #Find the experiment with the specified ID
    target_exp = next((exp for exp in data if exp['exp'] == target_exp_id), None)
    #Check if the experiment is found
    if target_exp:
        exp_id = target_exp['exp']
        training_data_id = target_exp['data']['training_data']
        test_data_id = target_exp['data']['test_data']

        # Print selected experiment with Datasets
        print(f"Experiment {exp_id}: Training Data = {training_data_id}, Test Data = {test_data_id}")
    else:
        print(f"Experiment {target_exp_id} not found.")
    
    # Define train and test sets
    studies = atlas.obs['Study'].value_counts().index.values
    training_study = studies[training_data_id] # ind 0, 1, 2, 3
    test_study = studies[test_data_id] # ind 0, 1, 2, 3
    
    # Train and test split
    train = atlas[atlas.obs['Study'] == training_study].copy()
    test = atlas[atlas.obs['Study'] == test_study].copy()
    test_study = str.replace(test_study, ' ', '_')
    del atlas # to save RAM

    # Test that the train/test split has no overlap
    print('Train:')
    print(train.obs['Study'].value_counts())
    print('Test:')
    print(test.obs['Study'].value_counts())

    # Select the `n_genes` most variable genes
    n_genes = 1000
    top_var_genes = train.var['highly_variable_rank'].sort_values().head(n_genes).index.values # Has no effect here because train.n_var = 1000

    # Define X and y
    X_train = train[:, top_var_genes].layers['counts']
    X_test = test[:, top_var_genes].layers['counts']
    y_train = train.obs['Annotation_1']
    y_test = test.obs['Annotation_1']

    # Define categories for categories encoder
    y_categories = ['Adipocytes',
    'Cardiomyocytes',
    'Endocardial',
    'Endothelial',
    'Epicardium',
    'Fibroblast',
    'Ischemic cells (MI)',
    'Lymphatic EC',
    'Lymphocytes',
    'Mast cells',
    'Monocytes',
    'Neuronal',
    'Pericytes',
    'VSMC']

    # Create categories encoder
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder_data = pd.DataFrame(y_categories).values.reshape(-1, 1)
    ohe.fit(encoder_data)

    # Transform y labels from categorial in numeric NumPy array
    y_train_transformed = ohe.transform(y_train.values.reshape(-1, 1))
    y_test_transformed = ohe.transform(y_test.values.reshape(-1, 1))

    # Converte sparse matrix to NumPy array
    X_train_array = X_train.toarray()
    X_test_array = X_test.toarray()

    # Convert data from NumPy array to PyTorch Tensor
    X_train_tensor = torch.tensor(X_train_array, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_array, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_transformed, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_transformed, dtype=torch.float32)

    # Define input and output dimension
    input_dim = X_train_tensor.size()[1]
    output_dim = y_train_tensor.size()[1]
    
    print("Data preprocessed successfully")
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_dim, output_dim, ohe


## Statistics
def stats(model, X_test, y_test, output_dim, ohe, device, scratchDir, experiment_name):
    # Define file name
    file_name_prefix = experiment_name.replace(" ", "_")

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
    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
    report = pd.DataFrame(report).transpose()

    print(report.to_string)
    report.to_csv(os.path.join(scratchDir, f"{file_name_prefix}_classification_report.csv"))

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
    colors = plt.cm.tab20c(np.linspace(0,1, output_dim))
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
    plt.title(f"{experiment_name}")
    plt.legend(loc="lower left", bbox_to_anchor=(1,0))

    plt.savefig(os.path.join(scratchDir, f"{file_name_prefix}_roc.png"), bbox_inches = 'tight')
    plt.clf()

    # F1 Score
    catagories = np.transpose(ohe.categories_).flatten()

    score = f1_score(y_test_labels, y_pred_labels, labels=catagories, average=None)
    score_df = pd.DataFrame({'cell_type': catagories, 'score': score})

    plt.figure(figsize=(8, 8))
    sns.barplot(score_df, x='cell_type', y='score')
    plt.ylabel("F1 score")
    plt.xlabel("Cell type")
    plt.xticks(rotation=90)
    plt.title(f"{experiment_name}")

    plt.savefig(os.path.join(scratchDir, f"{file_name_prefix}_f1.png"), bbox_inches = 'tight')
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
    plt.title(f"{experiment_name}")

    plt.savefig(os.path.join(scratchDir, f"{file_name_prefix}_confusion_matrix.png"), bbox_inches = 'tight')
    plt.clf()

    # Save Rersult Arrays
    np.save(os.path.join(scratchDir, f"{file_name_prefix}_y_pred.npy"), y_pred_labels)
    np.save(os.path.join(scratchDir, f"{file_name_prefix}_y_test.npy"), y_test_labels)


## Create and define folder structure
def directory(experiment):
    # Get directory information from swarm learning platform
    dataDir = os.getenv('DATA_DIR', '/platform/data')
    scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
    
    # Get current time
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d_%H-%M")

    # Define folder name
    folder_name_prefix = experiment.replace(" ", "_")
    folder = folder_name_prefix + "_" + date_string
    
    # Create new folder
    new_dir = os.path.join(scratchDir, folder)
    os.makedirs(new_dir)

    if os.path.exists(new_dir):
        print("Directory created successfully:", new_dir)
    else:
        print("Failed to create directory:", new_dir)
    
    scratchDir = new_dir
    
    return dataDir, scratchDir

## Main    
def main():
    # Set parameters and directories
    batchSz = 500
    default_max_epochs = 5
    default_min_peers = 2
    default_syncFrequency = 100
    default_experiment_name = "Swarm Learning"
    max_epochs = int(os.getenv('MAX_EPOCHS', str(default_max_epochs)))
    min_peers = int(os.getenv('MIN_PEERS', str(default_min_peers)))
    syncFrequency = int(os.getenv('SYNC_FREQUENCY', str(default_syncFrequency)))
    experiment = os.getenv('EXPERIMENT')
    experiment_name = os.getenv('EXPERIMENT_NAME', default_experiment_name)
    dataDir, scratchDir = directory(experiment_name)
    
    # Check if CUDA is available
    usecuda = torch.cuda.is_available()
    if usecuda:
        print('CUDA is accessable')
    else:
        print('CUDA  is not accesable')
    device = torch.device("cuda" if usecuda else "cpu")
    
    # Load data
    X_train, y_train, X_test, y_test, input_dim, output_dim, ohe = loadData(dataDir, experiment)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
    
    # Define model, loss function, optimizer and number of batches per epoch
    model = Multiclass(input_dim=input_dim , output_dim=output_dim).to(device)
    model_name = 'multiclass'
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    trainDs = torch.utils.data.TensorDataset(X_train, y_train)
    valDs = torch.utils.data.TensorDataset(X_val, y_val)
    
    trainLoader = torch.utils.data.DataLoader(trainDs, batch_size=batchSz)
    valLoader = torch.utils.data.DataLoader(valDs, batch_size=batchSz)
    
    # Create Swarm callback
    swarmCallback = None
    swarmCallback = SwarmCallback(syncFrequency=syncFrequency,
                                  minPeers=min_peers,
                                  useAdaptiveSync=False,
                                  adsValData=valLoader,
                                  adsValBatchSize=batchSz,
                                  model=model)
    
    # initalize swarmCallback and do first sync 
    swarmCallback.on_train_begin()
    
    # Train model
    for epoch in range(max_epochs):
        # Set model in training mode and run through each batch
        model.train()
        for batchIdx, (X_batch, y_batch) in enumerate(trainLoader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # compute and store metrics
            acc = (torch.argmax(y_pred, 1) == torch.argmax(y_batch, 1)).float().mean()
            if batchIdx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}'.format(
                    epoch, max_epochs-1, batchIdx * len(X_batch), len(trainLoader.dataset),
                    100. * batchIdx / len(trainLoader), loss.item(), acc*100))
            if swarmCallback is not None:
                    swarmCallback.on_batch_end()     
                
        # Set model in evaluation mode and run through the test set
        model.eval()
        X_val, y_val = X_val.to(device), y_val.to(device)
        y_pred = model(X_val)
        ce = loss_fn(y_pred, y_val)
        acc = (torch.argmax(y_pred, 1) == torch.argmax(y_val, 1)).float().mean()
        ce = float(ce)
        acc = float(acc)
        print(f"Epoch {epoch} validation: Cross-entropy={ce:.2f}, Accuracy={acc*100:.1f}%")
    
    # Handles what to do when training ends        
    swarmCallback.on_train_end()

    # Statistics
    stats(model, X_test, y_test, output_dim, ohe, device, scratchDir, experiment_name)
    
    # Save model and weights
    saved_model_path = os.path.join(scratchDir, model_name, 'saved_model.pt')
    os.makedirs(scratchDir, exist_ok=True)
    os.makedirs(os.path.join(scratchDir, model_name), exist_ok=True)
    torch.save(model, saved_model_path)
    print('Saved the trained model!')

if __name__ == '__main__':
  main()
