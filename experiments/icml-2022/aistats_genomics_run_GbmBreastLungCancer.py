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

if __name__ == '__main__':   
    pool = multiprocessing.Pool(processes=9)
    res = pool.map(smap, [f_GbmBreastLungCancer_a, f_GbmBreastLungCancer_b, f_GbmBreastLungCancer_c, f_GbmBreastLungCancer_d, f_GbmBreastLungCancer_e, f_GbmBreastLungCancer_f, f_GbmBreastLungCancer_g, f_GbmBreastLungCancer_h, f_GbmBreastLungCancer_i])
