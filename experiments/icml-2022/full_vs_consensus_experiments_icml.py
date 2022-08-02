import numpy as np
import mosek
import cvxpy as cp
import matplotlib.pyplot as plt
import time
from itertools import combinations
from admm_utils import prox as cprox
from pcmf import pcmf_full, pcmf_approx_uV, pcmf_full_consensus, save_multi_image, two_cluster_data, generate_cluster_PMD_data
#
def run_numerical_experiments_icml(dir_path='results/',data_type='test', pcmf_type='test',intercept=False, split_size=50, m=50, sigma=0.075, density=0.5, num_vars=5, means=[-1.0, 1.0, -0.4, 0.4], r=1, rho=1.0, weights='Gaussian', gauss_coef=5.0, neighbors=25, admm_iters=5, penalty_list=np.concatenate((np.repeat(np.inf,10), np.exp(np.linspace(-50,0,150))[::-1]), axis=0), parallel=False):
    n_clusters=4    
    print('run: '+str(r))
    seeds=r
    ms = [m,m,m,m]
    # Get clustered CCA data
    X_clusters, u_true, v_true, _=generate_cluster_PMD_data(ms, num_vars, sigma, density, n_clusters, means=means) 
    X_c=np.vstack(X_clusters)
    true_clusters=np.repeat([0,1,2,3],m)
    np.random.seed(seed=1234)
    idx_perm=np.random.permutation(X_c.shape[0])
    X_all=X_c[idx_perm,:]
    true_clusters_all=true_clusters[idx_perm]
#
    data_type='div_fourcluster_ConsensusComparison'
#
    tic=time.time()
    pcmf_type='pcmf_full_consensus'
    save_path=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)
    # output_file=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.txt'
    A, U, S, V=pcmf_full_consensus(X_all, penalty_list, split_size=split_size, rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+".txt")
    toc=time.time() - tic
    np.savez(save_path+".npz", ms=ms, n_clusters=n_clusters, true_clusters=true_clusters_all, split_size=split_size, means=means, n_X=num_vars, sigma=sigma, density=density, seeds=seeds, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=np.asarray(penalty_list), rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=intercept) 
    # plot_save_name = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.pdf'
    generate_plots_synthetic(A, U, V, S, split_size, penalty_list, true_clusters_all, save_path+".pdf")
    del A, U, S, V, toc, save_path, pcmf_type

    tic=time.time()
    pcmf_type='pcmf_full'
    save_path=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)
    # output_file=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.txt'
    print(save_path)
    A, U, S, V=pcmf_full(X_all, penalty_list, numba=True, rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+".txt")
    # A, U, S, V=pcmf_full(X_c, penalty_list, numba=True, rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+".txt")
    toc=time.time() - tic
    np.savez(save_path+".npz", ms=ms, n_clusters=n_clusters, true_clusters=true_clusters_all, means=means, n_X=num_vars, sigma=sigma, density=density, seeds=seeds, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=np.asarray(penalty_list), rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=intercept) 
    # plot_save_name = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.pdf'
    generate_plots_synthetic(A, U, V, S, X_c.shape[0], penalty_list, true_clusters_all, save_path+".pdf")
#

def run_numerical_experiments_icml_4UNBALANCED(dir_path='results/',data_type='test', pcmf_type='test',intercept=False, split_size=50, m=50, sigma=0.075, density=0.5, num_vars=5, means=[-1.0, 1.0, -0.4, 0.4], r=1, rho=1.0, weights='Gaussian', gauss_coef=5.0, neighbors=25, admm_iters=5, penalty_list=np.concatenate((np.repeat(np.inf,10), np.exp(np.linspace(-50,0,150))[::-1]), axis=0), parallel=False):
    n_clusters=4    
    print('run: '+str(r))
    seeds=r
    if m == 100:
        ms = [20,55,15,10]
    elif m == 1000:
        ms = [200,550,150,100]
    elif m == 10000:
        ms = [2000,5500,1500,1000]
    elif m == 100000:
        ms = [20000,55000,15000,10000]
    else:
        ms = [20,55,15,10]

    # Get clustered CCA data
    X_clusters, u_true, v_true, _=generate_cluster_PMD_data(ms, num_vars, sigma, density, n_clusters, means=means) 
    X_c=np.vstack(X_clusters)
    true_clusters=np.repeat([0,1,2,3],m)
    np.random.seed(seed=1234)
    idx_perm=np.random.permutation(X_c.shape[0])
    X_all=X_c[idx_perm,:]
    true_clusters_all=true_clusters[idx_perm]
#
    data_type='div_4UNBALANCED'
#
    tic=time.time()
    pcmf_type='pcmf_full_consensus'
    save_path=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)
    # output_file=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.txt'
    A, U, S, V=pcmf_full_consensus(X_all, penalty_list, split_size=split_size, rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+".txt")
    toc=time.time() - tic
    np.savez(save_path+".npz", X_all=X_all, ms=ms, n_clusters=n_clusters, true_clusters=true_clusters_all, split_size=split_size, means=means, n_X=num_vars, sigma=sigma, density=density, seeds=seeds, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=np.asarray(penalty_list), rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=intercept) 
    # plot_save_name = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.pdf'
    generate_plots_synthetic(A, U, V, S, split_size, penalty_list, true_clusters_all, save_path+".pdf")
    del A, U, S, V, toc, save_path, pcmf_type

    if m == 100:
        tic=time.time()
        pcmf_type='pcmf_full'
        save_path=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)
        # output_file=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.txt'
        print(save_path)
        A, U, S, V=pcmf_full(X_all, penalty_list, numba=True, rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+".txt")
        # A, U, S, V=pcmf_full(X_c, penalty_list, numba=True, rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+".txt")
        toc=time.time() - tic
        np.savez(save_path+".npz", X_all=X_all, ms=ms, n_clusters=n_clusters, true_clusters=true_clusters_all, means=means, n_X=num_vars, sigma=sigma, density=density, seeds=seeds, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=np.asarray(penalty_list), rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=intercept) 
        # plot_save_name = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.pdf'
        generate_plots_synthetic(A, U, V, S, X_all.shape[0], penalty_list, true_clusters_all, save_path+".pdf")
#


def run_numerical_experiments_icml_consensusOnly(dir_path='results/',data_type='test', pcmf_type='test',intercept=False, split_size=50, m=50, sigma=0.075, density=0.5, num_vars=5, means=[-1.0, 1.0, -0.4, 0.4], r=1, rho=1.0, weights='Gaussian', gauss_coef=5.0, neighbors=25, admm_iters=5, penalty_list=np.concatenate((np.repeat(np.inf,10), np.exp(np.linspace(-50,0,150))[::-1]), axis=0), parallel=False):
    n_clusters=4    
    print('run: '+str(r))
    # seeds=[r,r+100]
    seeds = r
    ms = [m,m,m,m]
    # Get clustered CCA data
    X_clusters, u_true, v_true, _=generate_cluster_PMD_data(ms, num_vars, sigma, density, n_clusters, means=means) 
    X_c=np.vstack(X_clusters)
    true_clusters=np.repeat([0,1,2,3],m)
    np.random.seed(seed=1234)
    idx_perm=np.random.permutation(X_c.shape[0])
    X_all=X_c[idx_perm,:]
    true_clusters_all=true_clusters[idx_perm]
#
    data_type='div_fourcluster_ConsensusONLY'
#
    tic=time.time()
    pcmf_type='pcmf_full_consensus'
    save_path=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.npz'
    output_file=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.txt'
    A, U, S, V=pcmf_full_consensus(X_all, penalty_list, split_size=split_size, rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=output_file)
    toc=time.time() - tic
    np.savez(save_path, true_clusters=true_clusters_all, split_size=split_size, means=means, n_X=num_vars, sigma=sigma, density=density, seeds=seeds, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=np.asarray(penalty_list), rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=intercept) 
    plot_save_name = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.pdf'
    generate_plots_synthetic(A, U, V, S, split_size, penalty_list, true_clusters_all, plot_save_name)


def run_numerical_experiments_icml_consensusOnly_8(dir_path='results/',data_type='test', pcmf_type='test',intercept=False, split_size=50, m=50, sigma=0.075, density=0.5, num_vars=5, means=[-2.1, 2.1, -1.5, 1.5, -1.0, 1.0, -0.4, 0.4], r=1, rho=1.0, weights='Gaussian', gauss_coef=5.0, neighbors=25, admm_iters=5, penalty_list=np.concatenate((np.repeat(np.inf,10), np.exp(np.linspace(-50,0,150))[::-1]), axis=0), parallel=False):
    n_clusters=8  
    print('run: '+str(r))
    # seeds=[r,r+100]
    seeds = r
    ms = [m,m,m,m,m,m,m,m]
    # Get clustered CCA data
    X_clusters, u_true, v_true, _=generate_cluster_PMD_data(ms, num_vars, sigma, density, n_clusters, means=means) 
    X_c=np.vstack(X_clusters)
    true_clusters=np.repeat([0,1,2,3,4,5,6,7],m)
    np.random.seed(seed=1234)
    idx_perm=np.random.permutation(X_c.shape[0])
    X_all=X_c[idx_perm,:]
    true_clusters_all=true_clusters[idx_perm]
#
    data_type='div_eightcluster_ConsensusONLY'
#
    tic=time.time()
    pcmf_type='pcmf_full_consensus'
    save_path=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)
    # output_file=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.txt'
    A, U, S, V=pcmf_full_consensus(X_all, penalty_list, split_size=split_size, rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    toc=time.time() - tic
    np.savez(save_path+'.npz', true_clusters=true_clusters_all, split_size=split_size, means=means, n_X=num_vars, sigma=sigma, density=density, seeds=seeds, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=np.asarray(penalty_list), rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=intercept) 
    # plot_save_name = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.pdf'
    generate_plots_synthetic(A, U, V, S, split_size, penalty_list, true_clusters_all, save_path+".pdf")



def run_numerical_experiments_icml_20classes(dir_path='results/',data_type='test', pcmf_type='test',intercept=False, split_size=50, m=50, sigma=0.075, density=0.5, num_vars=5, means=[-5.2, 5.2, -4.8, 4.8, -4.1, 4.1, -3.7, 3.7, -3.1, 3.1,-2.8, 2.8, -2.1, 2.1, -1.5, 1.5, -1.0, 1.0, -0.4, 0.4], r=1, rho=1.0, weights='Gaussian', gauss_coef=5.0, neighbors=25, admm_iters=5, penalty_list=np.concatenate((np.repeat(np.inf,10), np.exp(np.linspace(-50,0,150))[::-1]), axis=0), parallel=False):
    n_clusters=20  
    print('run: '+str(r))
    # seeds=[r,r+100]
    seeds = r
    ms = [m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m]
    # Get clustered CCA data
    X_clusters, u_true, v_true, _=generate_cluster_PMD_data(ms, num_vars, sigma, density, n_clusters, means=means) 
    X_c=np.vstack(X_clusters)
    true_clusters=np.repeat([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],m)
    np.random.seed(seed=1234)
    idx_perm=np.random.permutation(X_c.shape[0])
    X_all=X_c[idx_perm,:]
    true_clusters_all=true_clusters[idx_perm]
#
    data_type='div_consensusComparison_20Classes'

    tic=time.time()
    pcmf_type='pcmf_full_consensus'
    save_path=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)
    # output_file=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.txt'
    A, U, S, V=pcmf_full_consensus(X_all, penalty_list, split_size=split_size, rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    toc=time.time() - tic
    np.savez(save_path+'.npz', true_clusters=true_clusters_all, split_size=split_size, means=means, n_X=num_vars, sigma=sigma, density=density, seeds=seeds, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=np.asarray(penalty_list), rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=intercept) 
    # plot_save_name = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.pdf'
    generate_plots_synthetic(A, U, V, S, split_size, penalty_list, true_clusters_all, save_path+".pdf")
    del A, U, S, V, toc, save_path, pcmf_type

    tic=time.time()
    pcmf_type='pcmf_full'
    save_path=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)
    # output_file=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.txt'
    print(save_path)
    # THIS WAS WRONG - it was X_c instead of X_all; explaining why clustering was weird...
    A, U, S, V=pcmf_full(X_all, penalty_list, numba=True, rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    # A, U, S, V=pcmf_full(X_c, penalty_list, numba=True, rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    toc=time.time() - tic
    np.savez(save_path+'.npz', true_clusters=true_clusters_all, means=means, n_X=num_vars, sigma=sigma, density=density, seeds=seeds, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=np.asarray(penalty_list), rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=intercept) 
    # plot_save_name = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.pdf'
    generate_plots_synthetic(A, U, V, S, X_c.shape[0], penalty_list, true_clusters_all, save_path+".pdf")


def run_numerical_experiments_icml_20classes_PALS(dir_path='results/',data_type='test', pcmf_type='test',intercept=False, m=50, sigma=0.075, density=0.5, num_vars=5, means=[-5.2, 5.2, -4.8, 4.8, -4.1, 4.1, -3.7, 3.7, -3.1, 3.1,-2.8, 2.8, -2.1, 2.1, -1.5, 1.5, -1.0, 1.0, -0.4, 0.4], r=1, rho=1.0, weights='Gaussian', gauss_coef=5.0, neighbors=25, admm_iters=5, penalty_list=np.concatenate((np.repeat(np.inf,10), np.exp(np.linspace(-50,0,150))[::-1]), axis=0), parallel=False):
    n_clusters=20  
    print('run: '+str(r))
    # seeds=[r,r+100]
    seeds = r
    ms = [m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m]
    # Get clustered CCA data
    X_clusters, u_true, v_true, _=generate_cluster_PMD_data(ms, num_vars, sigma, density, n_clusters, means=means) 
    X_c=np.vstack(X_clusters)
    true_clusters=np.repeat([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],m)
    np.random.seed(seed=1234)
    idx_perm=np.random.permutation(X_c.shape[0])
    X_all=X_c[idx_perm,:]
    true_clusters_all=true_clusters[idx_perm]
#
    data_type='div_PALS_20Classes'

    tic=time.time()
    pcmf_type='pcmf_approx_uV'
    save_path=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)
    # output_file=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.txt'
    print(save_path)
    V, U, S=pcmf_approx_uV(X_all, penalty_list, numba=True, rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    toc=time.time() - tic
    A = []
    for p in range(len(penalty_list)):
        A.append((U[p]*S[p])@V[p])
    
    np.savez(save_path+'.npz', X=X_all, ms=ms, u_true=u_true, v_true=v_true, true_clusters=true_clusters_all, means=means, n_X=num_vars, sigma=sigma, density=density, seeds=seeds, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=np.asarray(penalty_list), rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=intercept) 
    # plot_save_name = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.pdf'
    generate_plots_synthetic(A, U, V, S, X_all.shape[0], penalty_list, true_clusters_all, save_path+".pdf")


def run_numerical_experiments_icml_20classes_onlyconsensus(dir_path='results/',data_type='test', pcmf_type='test',intercept=False, split_size=50, m=50, sigma=0.075, density=0.5, num_vars=5, means=[-5.2, 5.2, -4.8, 4.8, -4.1, 4.1, -3.7, 3.7, -3.1, 3.1,-2.8, 2.8, -2.1, 2.1, -1.5, 1.5, -1.0, 1.0, -0.4, 0.4], r=1, rho=1.0, weights='Gaussian', gauss_coef=5.0, neighbors=25, admm_iters=5, penalty_list=np.concatenate((np.repeat(np.inf,10), np.exp(np.linspace(-50,0,150))[::-1]), axis=0), parallel=False):
    n_clusters=20  
    print('run: '+str(r))
    # seeds=[r,r+100]
    seeds = r
    ms = [m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m]
    # Get clustered CCA data
    X_clusters, u_true, v_true, _=generate_cluster_PMD_data(ms, num_vars, sigma, density, n_clusters, means=means) 
    X_c=np.vstack(X_clusters)
    true_clusters=np.repeat([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],m)
    np.random.seed(seed=1234)
    idx_perm=np.random.permutation(X_c.shape[0])
    X_all=X_c[idx_perm,:]
    true_clusters_all=true_clusters[idx_perm]
#
    data_type='div_consensusComparison_20Classes'

    tic=time.time()
    pcmf_type='pcmf_full_consensus'
    save_path=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)
    # output_file=dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.txt'
    A, U, S, V=pcmf_full_consensus(X_all, penalty_list, split_size=split_size, rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    toc=time.time() - tic
    np.savez(save_path+'.npz', true_clusters=true_clusters_all, split_size=split_size, means=means, n_X=num_vars, sigma=sigma, density=density, seeds=seeds, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=np.asarray(penalty_list), rho=rho, admm_iters=admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=intercept) 
    # plot_save_name = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_n'+str(X_all.shape[0])+'_p'+str(num_vars)+'_density'+str(density)+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'.pdf'
    generate_plots_synthetic(A, U, V, S, split_size, penalty_list, true_clusters_all, save_path+".pdf")
    del A, U, S, V, toc, save_path, pcmf_type
#


from matplotlib import cm
def path_plot(coefficient_arr, penalty_list,plot_range=[0,-1], cut_vars=False, first_vars_only=False, var_sel=1):
    # Crop x axis (e.g, to remove 'burn-in' period at beginning)
    coefficient_arr = coefficient_arr[plot_range[0]:plot_range[1],:,:]
    penalty_list = penalty_list[plot_range[0]:plot_range[1]]
    if cut_vars is True:
        coefficient_arr = coefficient_arr[:,:,[1,2,coefficient_arr.shape[2]-1]]
        
    if first_vars_only is True:
        coefficient_arr = coefficient_arr[:,:,[var_sel]]

    # Colormap
    cmap = cm.get_cmap('viridis', coefficient_arr.shape[2])
    colors = cmap(np.linspace(0.0,1.0,coefficient_arr.shape[2]))

    # Define x-axis range
    penalty_range = range(len(penalty_list))

    fig, ax = plt.subplots(1,1, figsize=(20,10))
    for i in range(coefficient_arr.shape[2]):
        x = np.round(np.array(penalty_list),8)[penalty_range]
        y = coefficient_arr[penalty_range,:,i]
        ax.plot(np.arange(x.shape[0]), y, color=colors[i], alpha=0.15)
        ax.set_xticks(range(x.shape[0]), minor=False);
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.set_xticklabels(x,fontsize=24)
        evens = np.arange(0,len(ax.xaxis.get_ticklabels())+1,2)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    ax.tick_params(axis='y', labelsize=24)
    plt.xlabel(r'$\lambda$',fontsize=24)
    plt.ylabel('Coefficients',fontsize=24)
    return plt

def generate_plots_synthetic(A_list, U_list, V_list, s_list, split_size, penalty_list, true_clusters_all, plot_save_name):
    plt = path_plot(np.asarray(A_list)[:,:,0:3], penalty_list, plot_range=[0,-1])
    plt.xlabel("A")
    plt.ylabel("Estimates")
    #
    plt = path_plot(np.asarray(A_list)[:,true_clusters_all==0,0:3], penalty_list, plot_range=[0,-1])
    plt.xlabel("A (1)")
    plt.ylabel("Estimates")
#
    plt = path_plot(np.asarray(A_list)[:,true_clusters_all==1,0:3], penalty_list, plot_range=[0,-1])
    plt.xlabel("A (2)")
    plt.ylabel("Estimates")
#
    plt = path_plot(np.asarray(A_list)[:,true_clusters_all==2,0:3], penalty_list, plot_range=[0,-1])
    plt.xlabel("A (3)")
    plt.ylabel("Estimates")
#
    plt = path_plot(np.asarray(A_list)[:,true_clusters_all==3,0:3], penalty_list, plot_range=[0,-1])
    plt.xlabel("A (4)")
    plt.ylabel("Estimates")
#
    try:                                                     
        plt = path_plot(np.asarray(A_list)[:,0:split_size,0:3], penalty_list, plot_range=[0,-1])
        plt.xlabel("A Batch 1")
        plt.ylabel("Estimates")
#
        plt = path_plot(np.asarray(A_list)[:,split_size:split_size*2,0:3], penalty_list, plot_range=[0,-1])
        plt.xlabel("A Batch 2")
        plt.ylabel("Estimates")
    except:
        print('Could not make A batches')
#
    plt = path_plot(np.asarray(U_list)[:,:,0:3], penalty_list, plot_range=[0,-1])
    plt.xlabel("U")
    plt.ylabel("Estimates")
#
    plt = path_plot(np.asarray(U_list)[:,true_clusters_all==0,0:3], penalty_list, plot_range=[0,-1])
    plt.xlabel("U (1)")
    plt.ylabel("Estimates")
#
    plt = path_plot(np.asarray(U_list)[:,true_clusters_all==1,0:3], penalty_list, plot_range=[0,-1])
    plt.xlabel("U (2)")
    plt.ylabel("Estimates")
#
    plt = path_plot(np.asarray(U_list)[:,true_clusters_all==2,0:3], penalty_list, plot_range=[0,-1])
    plt.xlabel("U (3)")
    plt.ylabel("Estimates")
#
    plt = path_plot(np.asarray(U_list)[:,true_clusters_all==3,0:3], penalty_list, plot_range=[0,-1])
    plt.xlabel("U (4)")
    plt.ylabel("Estimates")
#
    plt = path_plot(np.asarray(V_list)[:,:,0:3], penalty_list, plot_range=[0,-1])
    plt.xlabel("V")
    plt.ylabel("Estimates")
#
    plt = path_plot(np.atleast_3d(np.asarray(s_list)), penalty_list, plot_range=[0,-1])
    plt.xlabel("s")
    plt.ylabel("Estimates")
#
    save_multi_image(plot_save_name)
    del plt


# # !! THIS IS IT !!
# p=58
# Vv=np.tile(results['V'][p],(X.shape[0],1))*np.sign(results['U'][p])
# # Vv=np.tile(results['V'][p],(X.shape[0],1))

# U_pcmf_cluster=np.zeros((X.shape[0],1))
# V_pcmf_cluster=np.zeros((20,X.shape[1]))
# S_pcmf_cluster=np.zeros((20,1))
# U_SVD_cluster=np.zeros((X.shape[0],1))
# V_SVD_cluster=np.zeros((20,X.shape[1]))
# S_SVD_cluster=np.zeros((20,1))
# for m in range(20):
#     inds = true_clusters==m
#     V_pcmf_cluster[m,:] = np.mean(Vv[inds,:])
# #     Uu = (results['U'][p][inds])
#     Uu = np.abs(results['U'][p][inds])
#     U_pcmf_cluster[inds] = np.sqrt(1./np.sum(Uu**2))*Uu # Unit vector
#     S_pcmf_cluster[m] = U_pcmf_cluster[inds].T@X[inds,:]@V_pcmf_cluster[m,:].T
#     U, S, V = randomized_svd(X[inds,:], n_components=1,random_state=1234)
#     U_SVD_cluster[inds] = U
#     S_SVD_cluster[m] = S
#     V_SVD_cluster[m,:] = V
# print(np.vstack((S_pcmf_cluster.T,S_SVD_cluster.T)))

# # print(np.hstack((U_pcmf_cluster[inds],U_SVD_cluster[inds])))

# print(np.hstack((V_pcmf_cluster[m,0:3],V_SVD_cluster[m,0:3])))