# Initialize genomics experiments for ICML 2022
import sys
import os
os.getcwd()
import numpy as np
import sys
import multiprocessing  

print('starting python initialize genomics experiments for ICML 2022')

print(sys.argv)
# print sys.argv[0] # prints initialize_genomics.py
code_path = sys.argv[1] # e.g., '/home/amb2022/clusterCCA_revision1/clusterCCA/'
gauss_coef = float(sys.argv[2]) #
neighbors = eval(sys.argv[3]) # 
# neighbors = int(sys.argv[3]) # 
split_size = int(sys.argv[4]) # 
dataset = sys.argv[5] # 
admm_iters = int(sys.argv[6]) # 
rho = float(sys.argv[7]) # 
penalty_type = float(sys.argv[8]) # 

sys.path.append(code_path)
sys.path.append(code_path+'/experiments/icml-2022/')
sys.path.append(code_path+'/utils/subspace-clustering-master')
os.chdir(code_path+'/utils/subspace-clustering-master')
import cluster
# from cluster.selfrepresentation import ElasticNetSubspaceClustering

# print(os.getcwd())
os.chdir(code_path)
from pcmf import pcmf_full_consensus
os.chdir(code_path+'/experiments/icml-2022/')
from genomics_experiments_icml import run_GbmBreastLungCancergenomics_Full_consensus, run_MouseOrgansgenomics_Full_consensus, run_GbmBreastLungCancergenomics_Full, run_MouseOrgansgenomics_Full_consensus_smallP
os.chdir(code_path)

if penalty_type == 0:
	penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,50))[::-1]),axis=0)
elif penalty_type == 1:
	penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
elif penalty_type == 2:
	penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(0,10,50))[::-1]),axis=0)
elif penalty_type == 3:
	penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-5,10,50))[::-1]),axis=0)
elif penalty_type == 4:
	penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-5,10,300))[::-1]),axis=0)
elif penalty_type == 5:
	penalty_list = np.concatenate((np.repeat(np.inf,2),np.exp(np.linspace(-10,10,20))[::-1]),axis=0)
elif penalty_type == 6:
	penalty_list = np.concatenate((np.repeat(np.inf,2),np.exp(np.linspace(-100,10,20))[::-1]),axis=0)
elif penalty_type == 7:
	penalty_list = np.concatenate((np.repeat(np.inf,2),np.exp(np.linspace(0,5,50))[::-1]),axis=0)
elif penalty_type == 8:
	penalty_list = np.concatenate((np.repeat(np.inf,5),np.exp(np.linspace(-50,10,100))[::-1]),axis=0)
elif penalty_type == 9:
	penalty_list = np.concatenate((np.repeat(np.inf,2),np.exp(np.linspace(0,5,150))[::-1]),axis=0)
else:
	penalty_list = np.concatenate((np.repeat(np.inf,1),np.exp(np.linspace(-10,10,2))[::-1]),axis=0)

# penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
# penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(0,10,150))[::-1]),axis=0)

if dataset == 'GbmBreastLungCancergenomics_skip3':
	run_GbmBreastLungCancergenomics_Full_consensus(rows_skip=3, penalty_type=penalty_type, gauss_coef=gauss_coef, neighbors=neighbors, split_size=split_size, penalty_list = penalty_list, admm_iters=admm_iters, rho = rho)
elif dataset == 'GbmBreastLungCancergenomics_skip0':
	run_GbmBreastLungCancergenomics_Full_consensus(rows_skip=0, penalty_type=penalty_type, gauss_coef=gauss_coef, neighbors=neighbors, split_size=split_size, penalty_list = penalty_list, admm_iters=admm_iters, rho = rho)
elif dataset == 'run_MouseOrgansgenomics_skip50':
	rows_skip=50
	run_MouseOrgansgenomics_Full_consensus(rows_skip=rows_skip, penalty_type=penalty_type, gauss_coef=gauss_coef, neighbors=neighbors, split_size=split_size, penalty_list = penalty_list, admm_iters=admm_iters, rho = rho)
elif dataset == 'run_MouseOrgansgenomics_skip50smallP':
	rows_skip=50
	run_MouseOrgansgenomics_Full_consensus_smallP(rows_skip=rows_skip, penalty_type=penalty_type, gauss_coef=gauss_coef, neighbors=neighbors, split_size=split_size, penalty_list = penalty_list, admm_iters=admm_iters, rho = rho)
elif dataset == 'run_MouseOrgansgenomics_skip25':
	rows_skip=25
	run_MouseOrgansgenomics_Full_consensus(rows_skip=rows_skip, penalty_type=penalty_type, gauss_coef=gauss_coef, neighbors=neighbors, split_size=split_size, penalty_list = penalty_list, admm_iters=admm_iters, rho = rho)
elif dataset == 'run_MouseOrgansgenomics_skip25smallP':
	rows_skip=25
	run_MouseOrgansgenomics_Full_consensus_smallP(rows_skip=rows_skip, penalty_type=penalty_type, gauss_coef=gauss_coef, neighbors=neighbors, split_size=split_size, penalty_list = penalty_list, admm_iters=admm_iters, rho = rho)
elif dataset == 'run_MouseOrgansgenomics_skip10':
	rows_skip=10
	run_MouseOrgansgenomics_Full_consensus(rows_skip=rows_skip, penalty_type=penalty_type, gauss_coef=gauss_coef, neighbors=neighbors, split_size=split_size, penalty_list = penalty_list, admm_iters=admm_iters, rho = rho)
elif dataset == 'run_MouseOrgansgenomics_skip10smallP':
	rows_skip=10
	run_MouseOrgansgenomics_Full_consensus_smallP(rows_skip=rows_skip, penalty_type=penalty_type, gauss_coef=gauss_coef, neighbors=neighbors, split_size=split_size, penalty_list = penalty_list, admm_iters=admm_iters, rho = rho)
elif dataset == 'run_MouseOrgansgenomics_skip5':
	rows_skip=5
	run_MouseOrgansgenomics_Full_consensus(rows_skip=rows_skip, penalty_type=penalty_type, gauss_coef=gauss_coef, neighbors=neighbors, split_size=split_size, penalty_list = penalty_list, admm_iters=admm_iters, rho = rho)
elif dataset == 'run_MouseOrgansgenomics_skip5smallP':
	rows_skip=5
	run_MouseOrgansgenomics_Full_consensus_smallP(rows_skip=rows_skip, penalty_type=penalty_type, gauss_coef=gauss_coef, neighbors=neighbors, split_size=split_size, penalty_list = penalty_list, admm_iters=admm_iters, rho = rho)
elif dataset == 'run_MouseOrgansgenomics_skip2':
	rows_skip=2
	run_MouseOrgansgenomics_Full_consensus(rows_skip=rows_skip, penalty_type=penalty_type, gauss_coef=gauss_coef, neighbors=neighbors, split_size=split_size, penalty_list = penalty_list, admm_iters=admm_iters, rho = rho)
elif dataset == 'run_MouseOrgansgenomics_skip2smallP':
	rows_skip=2
	run_MouseOrgansgenomics_Full_consensus_smallP(rows_skip=rows_skip, penalty_type=penalty_type, gauss_coef=gauss_coef, neighbors=neighbors, split_size=split_size, penalty_list = penalty_list, admm_iters=admm_iters, rho = rho)
elif dataset == 'run_MouseOrgansgenomics_skip1':
	rows_skip=1
	run_MouseOrgansgenomics_Full_consensus(rows_skip=rows_skip, penalty_type=penalty_type, gauss_coef=gauss_coef, neighbors=neighbors, split_size=split_size, penalty_list = penalty_list, admm_iters=admm_iters, rho = rho)
elif dataset == 'run_MouseOrgansgenomics_skip1smallP':
	rows_skip=1
	run_MouseOrgansgenomics_Full_consensus_smallP(rows_skip=rows_skip, penalty_type=penalty_type, gauss_coef=gauss_coef, neighbors=neighbors, split_size=split_size, penalty_list = penalty_list, admm_iters=admm_iters, rho = rho)
elif dataset == 'run_MouseOrgansgenomics_skip0':
	rows_skip=0
	run_MouseOrgansgenomics_Full_consensus(rows_skip=rows_skip, penalty_type=penalty_type, gauss_coef=gauss_coef, neighbors=neighbors, split_size=split_size, penalty_list = penalty_list, admm_iters=admm_iters, rho = rho)
elif dataset == 'run_MouseOrgansgenomics_skip0smallP':
	rows_skip=0
	run_MouseOrgansgenomics_Full_consensus_smallP(rows_skip=rows_skip, penalty_type=penalty_type, gauss_coef=gauss_coef, neighbors=neighbors, split_size=split_size, penalty_list = penalty_list, admm_iters=admm_iters, rho = rho)
elif dataset == 'GbmBreastLungCancergenomics_skip3_FULL':
		run_GbmBreastLungCancergenomics_Full(rows_skip=3, penalty_type=penalty_type, gauss_coef=gauss_coef, neighbors=neighbors, penalty_list = penalty_list, admm_iters=admm_iters, rho = rho)
else:
	print('Dataset',dataset, 'not found.')





