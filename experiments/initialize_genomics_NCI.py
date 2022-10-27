import os
import numpy as np
import sys
import multiprocessing  
print(sys.argv)
code_path = sys.argv[1] # e.g., '/home/amb2022/clusterCCA_revision1/clusterCCA/'
os.chdir(code_path)
os.getcwd()
# dir_path = sys.argv[2] # saving directory
sys.path.append(code_path)
sys.path.append(code_path+'/experiments/')
sys.path.append(code_path+'/utils/subspace-clustering-master')
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

from genomics_experiments import *
# Parallel function

# Pool
if __name__ == '__main__':   
    pool = multiprocessing.Pool(processes=6)
    res = pool.map(smap, [f_NCI_a, f_NCI_b, f_NCI_c, f_NCI_d, f_NCI_e, f_NCI_f])
