import os
os.getcwd()
import numpy as np
import sys
import multiprocessing  
print(sys.argv)
code_path = sys.argv[1] # e.g., '/home/amb2022/clusterCCA_revision1/clusterCCA/'
sys.path.append(code_path)
sys.path.append(code_path+'/experiments/icml-2022/')
sys.path.append(code_path+'/utils/subspace-clustering-master')
# os.chdir(code_path+'/utils/subspace-clustering-master')
# os.chdir('/home/amb2022/clusterCCA_revision1/clusterCCA/utils/subspace-clustering-master')
# import cluster
# os.chdir('/home/amb2022/clusterCCA_revision1/clusterCCA')

import mosek
import cvxpy as cp
import time
from itertools import combinations
from admm_utils import prox as cprox
from pcmf import pcmf_full, pcmf_approx_uV
from sklearn.preprocessing import StandardScaler

import argparse

import sys
from datetime import datetime as dt
import pandas as pd
from shutil import copyfile

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import multivariate_normal
from collections import OrderedDict
np.random.seed(1234)

from aistats_genomics_experiments import *
# Parallel function

# Pool
if __name__ == '__main__':   
    pool = multiprocessing.Pool(processes=6)
    res = pool.map(smap, [f_NCI_a, f_NCI_b, f_NCI_c, f_NCI_d, f_NCI_e, f_NCI_f])

# NCI
# def f_NCI_a():
#     run_NCIgenomics_Full(gauss_coef=gauss_coef, neighbors=25, penalty_list = penalty_list)

# def f_NCI_b():
#     run_NCIgenomics_Approx_uV(gauss_coef=gauss_coef, neighbors=25, penalty_list = penalty_list)

# def f_NCI_c(): 
#     run_NCIgenomics_Full(gauss_coef=gauss_coef, neighbors=None, penalty_list = penalty_list)

# def f_NCI_d():
#     run_NCIgenomics_Approx_uV(gauss_coef=gauss_coef, neighbors=None, penalty_list = penalty_list)

# def f_NCI_e(): 
#     run_NCIgenomics_Full(gauss_coef=gauss_coef, neighbors=15, penalty_list = penalty_list)

# def f_NCI_f():
#     run_NCIgenomics_Approx_uV(gauss_coef=gauss_coef, neighbors=15, penalty_list = penalty_list)

