import os
import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
import harmony as hp
from sympy import intersection
import pyarrow.feather as f

# Define project directory
if os.getcwd().startswith('/rwthfs'):
    proj_dir = '/rwthfs/rz/cluster/home/zd775156/sclabel'
else: 
    proj_dir = '/home/vivien/work/aachen/sclabel'

broad = sc.read_h5ad(proj_dir+'/data/broad/broad_vst_common.h5ad')
mi = sc.read_h5ad(proj_dir+'/data/mi/mi_vst_common.h5ad')

broad.obs.to_csv("../project/broad_cell_info.csv")
mi.obs.to_csv("../project/mi_cell_info.csv")
# f.write_feather(broad.obs,
                # dest="../project/broad_cell_info.feather",
                # version = 1)
# f.write_feather(mi.obs, 
                # dest="../project/mi_cell_info.feather",
                # version = 1)
