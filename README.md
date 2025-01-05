<<<<<<< HEAD
# SwarmMAP
Swarm learning single-cell transcriptomics

This repository contains the code to reproduce the paper:
Saldanha, O. L., Goepp, V., Pfeiffer, K., Kim, H., Kramann, R., Hayat, S., & Kather, J. N. (2025). SwarmMAP: Swarm Learning for Decentralized Cell Type Annotation in Single Cell Sequencing Data. bioRxiv, 2025-01. [https://doi.org/10.1101/2025.01.13.632775]
=======
# SwarmMAP: Swarm learning for cell type annotation

## Description
This repo reproduces the results in the paper \cite[]

## Content of this repository
code/
├── mlp/
│   └── local/
│       ├── local.py                  # Main script for managing and training MLP (Multi-Layer Perceptron) models.
│       ├── local_effect_of_dropout.py # Evaluates the impact of dropout rate on MLP model performance.
│       ├── local_effect_of_n_hvgs.py  # Analyzes how the number of highly variable genes (HVGs) affects MLP models.
│       ├── local_effect_of_repres.py  # Investigates the effect of different representations on MLP model outputs.
│       ├── local_plot_conf_mat.py     # Generates and visualizes confusion matrix plots for model evaluation.
│       ├── local_plot_effect_of_n_hvgs.py # Plots the impact of the number of HVGs on MLP model performance.
│       ├── local_plot_effect_of_repres.py # Visualizes the impact of different representations on model results.
│       ├── local_plot_f1.py           # Plots F1-score for evaluating classification model performance.
│       ├── local_plot_f1_xgboost.py   # Plots F1-score for XGBoost classification model.
│       ├── local_plot_indep_representation.py # Visualizes independent representations of data for analysis.
│       ├── local_plot_umaps.py        # Plots UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction visualization.
│       ├── local_xgboost.py           # Implements XGBoost model training and evaluation.
│
├── prepare_data/
│   ├── preproc_breast_union_hvgs.py  # Preprocessing script for breast cancer data, focusing on union of HVGs.
│   ├── preproc_heart_union_hvgs.py   # Preprocessing script for heart disease data, focusing on HVG selection.
│   └── preproc_lung_union_hvgs.py    # Preprocessing script for lung cancer data with focus on HVG filtering.
│
├── swarm/
│   ├── host_1.json                  # Configuration file for swarm node 1.
│   ├── host_2.json                  # Configuration file for swarm node 2.
│   ├── host_3.json                  # Configuration file for swarm node 3.
│   ├── hosts.ipynb                  # Jupyter Notebook for managing swarm hosts.
│   ├── hosts.py                     # Python script for managing and interacting with swarm hosts.
│   └── swarm.py                     # Main script for initializing and managing the swarm.
│
├── utils/
│   ├── get_heart_markers.R          # R script for identifying heart-related markers from data.
│   ├── mlp.py                       # Helper functions for MLP model training and evaluation.
│   ├── sclabel.py                   # Utility functions for label-related tasks (likely related to scRNA-seq).
│   └── write_cell_info.py           # Writes cell-specific information to files for downstream analysis.


>>>>>>> 147bcbd (describe scripts)
