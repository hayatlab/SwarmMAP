from math import exp, log
from statistics import mean, median
from scipy.stats import gmean
from scipy import sparse
import numpy as np
import pandas as pd
import sys
import os

# from symbol import return_stmt

def get_garnett_scale(counts):
    # with converting to dense matrix
    counts = counts.todense()
    nn_zeros = np.apply_along_axis(np.count_nonzero, 1, counts)
    median_nb_exprsd_genes = median(nn_zeros.tolist())
    colsums = counts.sum(axis=0).tolist()
    del X
    dataset_scale = gmean(colsums) / median_nb_exprsd_genes
    return(dataset_scale)    

def garnett_normalize(counts):
    colwise_gmean = np.apply_along_axis(geom_mean, 1, counts)
    return(counts / colwise_gmean[None, :])

def set_garnett_scale(X, scaling_factor):
    # ---- with converting to dense matrix
    size_sparse = sys.getsizeof(X)
    X = X.todense()
    size_dense = sys.getsizeof(X)
    nn_zeros = np.apply_along_axis(np.count_nonzero, 1, X)
    median_nb_exprsd_genes = median(nn_zeros.tolist())
    tmp = np.squeeze(np.asarray(X.sum(axis=0)))
    column_scale = tmp / median_nb_exprsd_genes / scaling_factor
    X = X / column_scale[None, :]
    if size_sparse < size_dense:
        X = sparse.csr_matrix(X)
    return(X)

### new
def get_size_factor(counts):
    counts = np.array(counts.todense())
    cell_totals = counts.sum(axis=0, keepdims=False)
    size_factor = cell_totals / np.exp(np.mean(np.log(cell_totals)))
    return(size_factor)

def get_train_cell_totals(train_counts):
    train_counts = np.array(train_counts.todense())
    train_cell_total = train_counts.sum(axis=0, keepdims=False)
    num_exprsd_genes = np.apply_along_axis(np.count_nonzero, 0, train_counts)
    train_cell_totals = np.exp(np.mean(np.log(train_cell_total))) / median(num_exprsd_genes)
    return(train_cell_totals)

def normalize_sf(counts, size_factor):
    # counts = counts.todense()
    norm_counts = counts / size_factor[None, :]
    # norm_counts = sparse.csr_array(norm_counts)
    return(norm_counts.tocsr())

def get_test_size_factor(test_counts, train_cell_totals):
    test_counts = np.array(test_counts.todense())
    test_cell_totals = test_counts.sum(axis=0, keepdims=False)
    test_num_exprsd_genes = np.apply_along_axis(np.count_nonzero, 0, test_counts)
    size_factor = test_cell_totals / (train_cell_totals * median(test_num_exprsd_genes))
    return(size_factor)
## end new

def harmonize_ct(labels, dataset, celltype_ref):
    corresp = pd.Series(celltype_ref.standardization.values,
                       index=celltype_ref[dataset].values).to_dict()
    harmonized = pd.Series(labels, dtype = 'category').map(corresp)
    harmonized = np.array(harmonized.tolist())
    return(harmonized)

from scipy.stats import gmean
def geom_mean(a):
    b = a[a != 0]
    return(gmean(b, axis=1))

def read(a):
	col1= list(map(lambda x:x.split('\n')[0].strip(), open(a, "r")))
	col2= list(map(lambda x:x.split('\t'), col1))
	return (col2)

def make_outdir(outdir, key):
	# outdir : top dir to save outputs
	# key : folder name to save outputs
	new_outdir = outdir+key
	if not os.path.exists(new_outdir):
		os.mkdir(new_outdir)
	return (new_outdir)