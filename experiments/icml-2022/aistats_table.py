import os
os.getcwd()
import numpy as np
import sys
import multiprocessing  
# os.chdir('/home/amb2022/clusterCCA_revision1/clusterCCA/utils/subspace-clustering-master')
# import cluster
os.chdir('/home/amb2022/clusterCCA_revision1/clusterCCA')
from pcmf import pcmf_full, pcmf_approx_uV, pcmf_approx_V, two_cluster_data

def smap(f):
    return f()

def run_numerical_experiments_aistats(dir_path='../results/',data_type='test', pcmf_type='test',intercept = True, cluster_sizes = [50,50], sigma = 0.075, density = 0.5, num_vars = 5, cluster_means = [-0.2,0.2], r = 1, rho = 1.0, weights='Gaussian', gauss_coef=2.0, neighbors = 25, admm_iters = 5, penalty_list = np.concatenate((np.repeat(np.inf,10), np.exp(np.linspace(-50,0,150))[::-1]), axis=0), parallel=False):
    import numpy as np
    import mosek
    import cvxpy as cp
    import matplotlib.pyplot as plt
    import time
    from itertools import combinations
    from admm_utils import prox as cprox
    
    print('run: '+str(r))
    seeds = [r,r+100]
    X_c, true_clusters = two_cluster_data(m=cluster_sizes, means=cluster_means, n_X=num_vars, sigma=sigma, density=density, 
                                       gen_seeds=False, seeds=seeds, plot=False, scale_data=True, intercept=intercept)
    tic = time.time()
    pcmf_type = 'AISTATS_pcmf_full'
    save_path = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)
    print(save_path)
    A, U, S, V = pcmf_full(X_c, penalty_list, numba=True, rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    toc = time.time() - tic
    np.savez(save_path+'.npz', true_clusters=true_clusters, means=cluster_means, n_X=num_vars, sigma=sigma, density=density, seeds=seeds, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=np.asarray(penalty_list), rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=intercept) 

    tic = time.time()
    pcmf_type = 'AISTATS_pcmf_approx_uV'
    save_path = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)
    print(save_path)
    V, U, S = pcmf_approx_uV(X_c, penalty_list, numba=True, rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    toc = time.time() - tic
    np.savez(save_path+'.npz', true_clusters=true_clusters, means=cluster_means, n_X=num_vars, sigma=sigma, density=density, seeds=seeds, V=V, U=U, S=S, runtime=toc, penalty_list=np.asarray(penalty_list), rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=intercept) 

# Parallel function
def run_numerical_experiments_parallel_20_agglom_samemeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_samemeans'
    num_vars = 20
    density = 0.2
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_20_agglom_samemeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_samemeans'
    num_vars = 20
    density = 0.5
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_20_agglom_diffmeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_diffmeans'
    num_vars = 20
    density = 0.2
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_20_agglom_diffmeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_diffmeans'
    num_vars = 20
    density = 0.5
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)


def run_numerical_experiments_parallel_20_agglom_unbalanced_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_unbalanced'
    num_vars = 20
    density = 0.2
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_20_agglom_unbalanced_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_unbalanced'
    num_vars = 20
    density = 0.5
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

### 200
def run_numerical_experiments_parallel_200_agglom_samemeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_samemeans'
    num_vars = 200
    density = 0.2
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_200_agglom_samemeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_samemeans'
    num_vars = 200
    density = 0.5
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_200_agglom_diffmeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_diffmeans'
    num_vars = 200
    density = 0.2
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_200_agglom_diffmeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_diffmeans'
    num_vars = 200
    density = 0.5
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_200_agglom_unbalanced_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_unbalanced'
    num_vars = 200
    density = 0.2
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_200_agglom_unbalanced_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_unbalanced'
    num_vars = 200
    density = 0.5
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

## 2000

def run_numerical_experiments_parallel_2000_agglom_samemeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_samemeans'
    num_vars = 2000
    density = 0.2
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_2000_agglom_samemeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_samemeans'
    num_vars = 2000
    density = 0.5
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_2000_agglom_diffmeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_diffmeans'
    num_vars = 2000
    density = 0.2
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_2000_agglom_diffmeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_diffmeans'
    num_vars = 2000
    density = 0.5
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_2000_agglom_unbalanced_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_unbalanced'
    num_vars = 2000
    density = 0.2
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_2000_agglom_unbalanced_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'agglom_unbalanced'
    num_vars = 2000
    density = 0.5
    penalty_list = np.concatenate((np.exp(np.linspace(-10,10,150)), np.repeat(np.inf,10)),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

##########
# 20

def run_numerical_experiments_parallel_20_div_samemeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_samemeans'
    num_vars = 20
    density = 0.2
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_20_div_samemeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_samemeans'
    num_vars = 20
    density = 0.5
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_20_div_diffmeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_diffmeans'
    num_vars = 20
    density = 0.2
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_20_div_diffmeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_diffmeans'
    num_vars = 20
    density = 0.5
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_20_div_unbalanced_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_unbalanced'
    num_vars = 20
    density = 0.2
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_20_div_unbalanced_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_unbalanced'
    num_vars = 20
    density = 0.5
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

# 200
def run_numerical_experiments_parallel_200_div_samemeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_samemeans'
    num_vars = 200
    density = 0.2
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_200_div_samemeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_samemeans'
    num_vars = 200
    density = 0.5
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_200_div_diffmeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_diffmeans'
    num_vars = 200
    density = 0.2
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_200_div_diffmeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_diffmeans'
    num_vars = 200
    density = 0.5
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)


def run_numerical_experiments_parallel_200_div_unbalanced_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_unbalanced'
    num_vars = 200
    density = 0.2
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_200_div_unbalanced_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_unbalanced'
    num_vars = 200
    density = 0.5
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

# 2000
def run_numerical_experiments_parallel_2000_div_samemeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_samemeans'
    num_vars = 2000
    density = 0.2
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_2000_div_samemeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_samemeans'
    num_vars = 2000
    density = 0.5
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_2000_div_diffmeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_diffmeans'
    num_vars = 2000
    density = 0.2
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_2000_div_diffmeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_diffmeans'
    num_vars = 2000
    density = 0.5
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_2000_div_unbalanced_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_unbalanced'
    num_vars = 2000
    density = 0.2
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

def run_numerical_experiments_parallel_2000_div_unbalanced_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_unbalanced'
    num_vars = 2000
    density = 0.5
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, parallel=True)

# DIV no neighbors
# 20
def run_numerical_experiments_parallel_20noneighbors_div_samemeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_samemeans'
    num_vars = 20
    density = 0.2
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    neighbors=None
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)

def run_numerical_experiments_parallel_20noneighbors_div_samemeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_samemeans'
    num_vars = 20
    density = 0.5
    neighbors=None
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)

def run_numerical_experiments_parallel_20noneighbors_div_diffmeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_diffmeans'
    num_vars = 20
    density = 0.2
    neighbors=None
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)

def run_numerical_experiments_parallel_20noneighbors_div_diffmeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_diffmeans'
    num_vars = 20
    density = 0.5
    neighbors=None
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)


def run_numerical_experiments_parallel_20noneighbors_div_unbalanced_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_unbalanced'
    num_vars = 20
    density = 0.2
    neighbors=None
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)

def run_numerical_experiments_parallel_20noneighbors_div_unbalanced_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_unbalanced'
    num_vars = 20
    density = 0.5
    neighbors=None
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)

# 200
def run_numerical_experiments_parallel_200noneighbors_div_samemeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_samemeans'
    num_vars = 200
    density = 0.2
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    neighbors=None
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)

def run_numerical_experiments_parallel_200noneighbors_div_samemeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_samemeans'
    num_vars = 200
    density = 0.5
    neighbors=None
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)

def run_numerical_experiments_parallel_200noneighbors_div_diffmeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_diffmeans'
    num_vars = 200
    density = 0.2
    neighbors=None
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)

def run_numerical_experiments_parallel_200noneighbors_div_diffmeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_diffmeans'
    num_vars = 200
    density = 0.5
    neighbors=None
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)


def run_numerical_experiments_parallel_200noneighbors_div_unbalanced_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_unbalanced'
    num_vars = 200
    density = 0.2
    neighbors=None
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)

def run_numerical_experiments_parallel_200noneighbors_div_unbalanced_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_unbalanced'
    num_vars = 200
    density = 0.5
    neighbors=None
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)

# 2000
def run_numerical_experiments_parallel_2000noneighbors_div_samemeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_samemeans'
    num_vars = 2000
    density = 0.2
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    neighbors=None
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)

def run_numerical_experiments_parallel_2000noneighbors_div_samemeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_samemeans'
    num_vars = 2000
    density = 0.5
    neighbors=None
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-0.05,0.05], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)

def run_numerical_experiments_parallel_2000noneighbors_div_diffmeans_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_diffmeans'
    num_vars = 2000
    density = 0.2
    neighbors=None
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)

def run_numerical_experiments_parallel_2000noneighbors_div_diffmeans_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_diffmeans'
    num_vars = 2000
    density = 0.5
    neighbors=None
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [50,50], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)


def run_numerical_experiments_parallel_2000noneighbors_div_unbalanced_dpt2(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_unbalanced'
    num_vars = 2000
    density = 0.2
    neighbors=None
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)

def run_numerical_experiments_parallel_2000noneighbors_div_unbalanced_dpt5(r):
#     print('run '+str(r)+ ' ')
    data_type = 'div_unbalanced'
    num_vars = 2000
    density = 0.5
    neighbors=None
    penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
    run_numerical_experiments_aistats(dir_path='results/',r=r, data_type=data_type, density=density, num_vars=num_vars, cluster_sizes = [80,20], cluster_means = [-.2,.2], intercept = True, penalty_list = penalty_list, neighbors=neighbors, parallel=True)





# # Run it
# if __name__ == '__main__':
#     pool = multiprocessing.Pool(processes=5)
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_samemeans_dpt2, args=(r,)) for r in range(10)]
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_diffmeans_dpt2, args=(r,)) for r in range(10)]
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_unbalanced_dpt2, args=(r,)) for r in range(10)]
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_samemeans_dpt5, args=(r,)) for r in range(10)]
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_diffmeans_dpt5, args=(r,)) for r in range(10)]
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_unbalanced_dpt5, args=(r,)) for r in range(10)]

# if __name__ == '__main__':
#     pool = multiprocessing.Pool(processes=5)
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_samemeans_dpt2, args=(r,)) for r in range(10)] # 28642.pts-19.scu-login02
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_diffmeans_dpt2, args=(r,)) for r in range(10)] # 28711.pts-19.scu-login02
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_unbalanced_dpt2, args=(r,)) for r in range(10)] # 28957.pts-19.scu-login02
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_samemeans_dpt5, args=(r,)) for r in range(10)] # 29019.pts-19.scu-login02
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_diffmeans_dpt5, args=(r,)) for r in range(10)] # 29080.pts-19.scu-login02
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_unbalanced_dpt5, args=(r,)) for r in range(10)] # 29140.pts-19.scu-login02

# if __name__ == '__main__':
#     pool = multiprocessing.Pool(processes=10)
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_samemeans_dpt2, args=(r,)) for r in range(10)] # 
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_diffmeans_dpt2, args=(r,)) for r in range(10)] # 
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_unbalanced_dpt2, args=(r,)) for r in range(10)] # 
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_samemeans_dpt5, args=(r,)) for r in range(10)] #
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_diffmeans_dpt5, args=(r,)) for r in range(10)] # 
#     [pool.apply_async(run_numerical_experiments_parallel_agglom_unbalanced_dpt5, args=(r,)) for r in range(10)] # 21021.pts-28.scu-login02



