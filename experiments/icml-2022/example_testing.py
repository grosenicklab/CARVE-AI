srun --pty -n 1 -c 30 --mem=200G -p scu-cpu,sackler-cpu /bin/bash
conda activate jupyter-lab

import numpy as np
import mosek
import cvxpy as cp
import matplotlib.pyplot as plt
import time
from itertools import combinations
import os
import sys
os.chdir('/home/amb2022/clusterCCA_revision1/clusterCCA/')
from admm_utils import prox as cprox
from pcmf import pcmf_full, pcmf_approx_uV, pcmf_full_consensus, save_multi_image, two_cluster_data, generate_cluster_PMD_data, chol_D, get_weights, prox_numba_arr

m = 2500
# ms=[2500,2500,2500,2500]
means=[-1.0, 1.0, -0.4, 0.4]
num_vars=20000
density=0.5
sigma=0.075
r=1
rho=1.0
problem_rank=1
weights='Gaussian'
gauss_coef=20.0
neighbors=25
split_size=100

n_clusters=4    
print('run: '+str(r))
seeds=[r,r+100]
ms = [m,m,m,m]
# Get clustered CCA data
X_clusters, u_true, v_true, _=generate_cluster_PMD_data(ms, num_vars, sigma, density, n_clusters, means=means) 
X_c=np.vstack(X_clusters)
true_clusters=np.repeat([0,1,2,3],m)
np.random.seed(seed=1234)
idx_perm=np.random.permutation(X_c.shape[0])
X_all=X_c[idx_perm,:]
true_clusters_all=true_clusters[idx_perm]

penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
parallel=False
print_progress=True
i=0
    
penalty_list = [np.inf, 1000, 10, 0.01]
tic = time.time()
A_list, U_list, s_list, V_list = pcmf_full_consensus(X_all, penalty_list, problem_rank=1, rho=1.0, admm_iters = 1,weights=weights, neighbors=neighbors, gauss_coef=gauss_coef, split_size=split_size, print_progress=True, parallel=False)
toc = time.time() - tic
print("Running length PCMF full:", toc)

 A_list, U_list, s_list, V_list




