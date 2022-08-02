# Initialize consensus comparison experiments for ICML 2022
import sys
import os
os.getcwd()
import numpy as np
import sys
import multiprocessing  

print('starting python initialize consensus experiments experiments for ICML 2022')

print(sys.argv)

# print sys.argv[0] # prints initialize_genomics.py
code_path = sys.argv[1] # e.g., '/home/amb2022/clusterCCA_revision1/clusterCCA/'
gauss_coef = float(sys.argv[2]) #
neighbors = int(sys.argv[3]) # 
split_size = int(sys.argv[4]) # 
m = int(sys.argv[5]) # 
density = float(sys.argv[6]) # 
num_vars = int(sys.argv[7]) # 
r = int(sys.argv[8]) # 
admm_iters = int(sys.argv[9]) # 
rho = float(sys.argv[10]) # 


sys.path.append(code_path)
sys.path.append(code_path+'/experiments/icml-2022/')
sys.path.append(code_path+'/utils/subspace-clustering-master')
os.chdir(code_path+'/utils/subspace-clustering-master')
import cluster
# from cluster.selfrepresentation import ElasticNetSubspaceClustering

os.chdir(code_path)
from pcmf import pcmf_full_consensus
os.chdir(code_path+'/experiments/icml-2022/')
from full_vs_consensus_experiments_icml import run_numerical_experiments_icml, run_numerical_experiments_icml_consensusOnly_8
os.chdir(code_path)

penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,50))[::-1]),axis=0)

run_numerical_experiments_icml(split_size=split_size, m=m, gauss_coef=gauss_coef, neighbors=neighbors, sigma=0.075, density=density, num_vars=num_vars, r=r, admm_iters=admm_iters, rho=rho, parallel=True, penalty_list=penalty_list)
