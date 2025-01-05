# General utility
import os
import json
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

# Define project directory
if os.getcwd().startswith('/rwthfs'):
    proj_dir = '/rwthfs/rz/cluster/home/zd775156/sclabel'
else:
    proj_dir = '/home/swarm/Desktop/single_cell_new'

atlas = sc.read_h5ad(proj_dir+'/subatlas.h5ad')

with open('code/host_1.json', 'r') as json_file: #ToDo change file name 
    data = json.load(json_file)
#Specify the experiment ID you're interested in
target_exp_id = '3' #ToDo add id from yaml in swarm learning setup
target_exp_id = int(target_exp_id)
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


studies = atlas.obs['Study'].value_counts().index.values
training_study = studies[training_data_id] # ind 0, 1, 2, 3
test_study = studies[test_data_id] # ind 0, 1, 2, 3

# Cell types present
exp_id = 0
for exp_id in range(0, 3):
    study = studies[exp_id]
    subset = atlas[atlas.obs['Study']==study]
    aa = subset.obs['Annotation_1'].value_counts().to_frame()
    aa = aa.rename_axis('cell_type').reset_index()

    plt.figure(figsize=(8, 8))
    sns.barplot(aa, x='cell_type', y='count')
    plt.ylabel("Count")
    plt.xlabel("Cell type")
    plt.xticks(rotation=90)
    plt.savefig(proj_dir+'/results_v1/cell_types/study_'+str(exp_id)+'.png')



