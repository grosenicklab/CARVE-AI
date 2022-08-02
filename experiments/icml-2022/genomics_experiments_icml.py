	################## Genomics Experiments ###################
import mosek
import cvxpy as cp
import time
from itertools import combinations
from admm_utils import prox as cprox
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
import os
import sys

from matplotlib import pyplot as plt

from math import floor

#
from pcmf import pcmf_full, pcmf_approx_uV, pcmf_full_consensus, save_multi_image, pcmf_full_consensus_OLD
#
np.random.seed(1234)
#

# scale data
def run_GbmBreastLungCancergenomics_Full_consensus(dir_path='results/',data_path='data/', penalty_type=None,rows_skip=3, split_size=50, admm_iters = 5, gauss_coef=2.0, neighbors=None, rho = 1, weights='Gaussian', penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0), parallel=True, output_file='NaN'):
    # Load GbmBreastLungCancer dataset
    gbm = pd.read_csv(os.path.join(data_path, 'gbm_z.txt'), sep='\t', header=0)
    lung = pd.read_csv(os.path.join(data_path, 'lung_z.txt'), sep='\t', header=0)
    breast = pd.read_csv(os.path.join(data_path, 'breast_z.txt'), sep='\t', header=0)
    labels = np.hstack((np.repeat('gbm',gbm.shape[0]),np.repeat('lung',lung.shape[0]),np.repeat('breast',breast.shape[0])))
    gbm_T = gbm.T
    gbm_T['variables'] = gbm.columns
    lung_T = lung.T
    lung_T['variables'] = lung.columns
    breast_T = breast.T
    breast_T['variables'] = breast.columns
    data_T = pd.merge(gbm_T,pd.merge(lung_T,breast_T,on='variables'),on='variables')
    data = data_T.drop(columns='variables').T
    data.columns = data_T['variables']
    # 
    intercept=True
    # intercept=False
    if rows_skip>0:
        data_subset = data[::rows_skip]
        labels_subset = labels[::rows_skip]
        X_c = data_subset.to_numpy()
        if intercept==True:
        	X_c = np.hstack((X_c,np.ones((X_c.shape[0],1))))
        true_clusters = labels_subset
        print("X_c is:",X_c.shape)
        np.random.seed(seed=1234)
        idx_perm = np.random.permutation(X_c.shape[0])
        X_c = X_c[idx_perm,:]
        true_clusters = true_clusters[idx_perm]
        rounded = floor(X_c.shape[0]/split_size)*split_size
        X_all = X_c[0:rounded,:]
        print("After rounding down X_c is:",X_c.shape)
        true_clusters_all=true_clusters[0:rounded]
    else:
        X_c = data.to_numpy()
        true_clusters = labels
        if intercept==True:
        	X_c = np.hstack((X_c,np.ones((X_c.shape[0],1))))
        np.random.seed(seed=1234)
        idx_perm = np.random.permutation(X_c.shape[0])
        X_c = X_c[idx_perm,:]
        true_clusters = true_clusters[idx_perm]
        rounded = floor(X_c.shape[0]/split_size)*split_size
        X_all = X_c[0:rounded,:]
        true_clusters_all=true_clusters[0:rounded]
    #
    # np.random.seed(seed=1234)
    # idx_perm = np.random.permutation(X_c.shape[0])
    # X_all = X_c[idx_perm,:]
    # true_clusters_all = true_clusters[idx_perm]
    print('Size of X_all is:',X_all.shape)
    # Run PCMF
    tic = time.time()
    pcmf_type = 'pcmf_full_consensus'
    # pcmf_type = 'pcmf_full_consensus_OLD'

    save_path = dir_path+pcmf_type+'_GbmBreastLungCancer_genomics_run'+'_N_'+str(X_all.shape[0])+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)+'_addmIters'+str(admm_iters)+'_penalty_type'+str(penalty_type)+'_intercept'+str(intercept)
    # output_file = dir_path+pcmf_type+'_GbmBreastLungCancer_genomics_run'+'_N_'+str(X_all.shape[0])+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)+'_addmIters'+str(admm_iters)+'.txt'
    print(save_path)
    # A, U, S, V = pcmf_full_consensus_OLD(X_all, penalty_list, split_size=split_size, rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    A, U, S, V = pcmf_full_consensus(X_all, penalty_list, split_size=split_size, rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    toc = time.time() - tic
    np.savez(save_path+'.npz', pcmf_type=pcmf_type, X_c=X_all, true_clusters=true_clusters_all, A=A, V=V, U=U, S=S, split_size=split_size, runtime=toc, penalty_list=penalty_list, rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=intercept) 
    # plot_save_name = dir_path+pcmf_type+'_GbmBreastLungCancer_genomics_run'+'_N_'+str(X_all.shape[0])+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)+'_addmIters'+str(admm_iters)+'.pdf'
    generate_plots(A, U, V, S, split_size, penalty_list, true_clusters_all, save_path+'.pdf', 'gbmbreastlung')
#
def load_mouse_organs(data_path,quantile_prob=0.2):
    # 'quantile_prob is Percentile for data filtering.'
    data = pd.read_csv(os.path.join(data_path, 'mouse_organs', 'data', 'data_merged_tpm.csv.bz2'), index_col=0, compression='bz2')
    labels = pd.read_csv(os.path.join(data_path, 'mouse_organs', 'data', 'labels_merged.csv'), header=None).values.squeeze()
    ind_train = pd.read_csv(os.path.join(data_path, 'mouse_organs', 'data', 'indices_train.csv'), header=None).values.squeeze()
    ind_test = pd.read_csv(os.path.join(data_path, 'mouse_organs', 'data', 'indices_test.csv'), header=None).values.squeeze()
    prior = pd.read_csv(os.path.join(data_path, 'mouse_organs', 'data', 'labels_prior_genes.csv'), dtype=np.float32)
    #
    training_data = data.values[ind_train]
    training_labels = labels[ind_train].squeeze()
    test_data = data.values[ind_test]
    test_labels = labels[ind_test].squeeze()
    training_prior = prior.values[ind_train]
    # filter data for points without prior knowledge
    training_prior_sum = np.sum(training_prior, axis=1)
    ind_train_prior = np.where(training_prior_sum == 1)[0]
    training_data = training_data[ind_train_prior]
    training_labels = training_labels[ind_train_prior]
    training_prior = training_prior[ind_train_prior]
    training_labels_unique = np.unique(training_labels)
    test_labels_unique = np.unique(test_labels)
    print('scale')
    # scale data between -1 and 1
    # scaler = MinMaxScaler(feature_range=(-1,1))
    # scale data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(training_data)
    training_data_scaled = scaler.transform(training_data)
    test_data_scaled = scaler.transform(test_data)
    print('vor filter')
    # filter vor genes with low variance
    sum_gene_count_total = np.sum(training_data, axis=0)
    var_gene = np.var(training_data, axis=0)
    ind_filter_count = np.where(sum_gene_count_total > np.quantile(sum_gene_count_total, quantile_prob))[0]
    ind_filter_var = np.where(var_gene > np.quantile(var_gene, quantile_prob))[0]
    ind_gene_filter = np.unique([ind_filter_count, ind_filter_var])
    training_data_scaled = training_data_scaled[:, ind_gene_filter]
    test_data_scaled = test_data_scaled[:, ind_gene_filter]
    return training_data_scaled, test_data_scaled, training_labels, test_labels, ind_gene_filter
#
def run_MouseOrgansgenomics_Full_consensus(dir_path='results/',data_path='data/', penalty_type=None,split_size=50, rows_skip = 10, admm_iters = 5, gauss_coef=2.0, neighbors=None, rho = 1, weights='Gaussian', penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-5,500,150))[::-1]),axis=0), parallel=True, output_file='NaN'):
    # Load MouseOrgans dataset
    training_data_scaled, test_data_scaled, training_labels, test_labels, ind_gene_filter = load_mouse_organs(data_path)
    print("training_data_scaled",training_data_scaled.shape,"test_data_scaled",test_data_scaled.shape)
    intercept=True
    # intercept=False
    if rows_skip>0:
        training_data_scaled_subset = training_data_scaled[::rows_skip]
        training_labels_subset = training_labels[::rows_skip]
        training_labels_unique = np.unique(training_labels_subset)
        X_c = training_data_scaled_subset
        if intercept==True:
        	X_c = np.hstack((X_c,np.ones((X_c.shape[0],1))))
        true_clusters = training_labels_subset
        np.random.seed(seed=1234)
        idx_perm = np.random.permutation(X_c.shape[0])
        X_c = X_c[idx_perm,:]
        true_clusters = true_clusters[idx_perm]
        rounded = floor(X_c.shape[0]/split_size)*split_size
        X_all = X_c[0:rounded,:]
        true_clusters_all=true_clusters[0:rounded]
    else:
        X_c = np.vstack((training_data_scaled, test_data_scaled))
        if intercept==True:
        	X_c = np.hstack((X_c,np.ones((X_c.shape[0],1))))
        true_clusters = np.hstack((training_labels, test_labels))
        np.random.seed(seed=1234)
        idx_perm = np.random.permutation(X_c.shape[0])
        X_c = X_c[idx_perm,:]
        true_clusters = true_clusters[idx_perm]
        rounded = floor(X_c.shape[0]/split_size)*split_size
        X_all = X_c[0:rounded,:]
        true_clusters_all=true_clusters[0:rounded]
        # X_c = np.vstack((training_data_scaled, test_data_scaled))
        # true_clusters = np.vstack((training_labels, test_labels))
    #
    # np.random.seed(seed=1234)
    # idx_perm = np.random.permutation(X_c.shape[0])
    # X_all = X_c[idx_perm,:]
    # true_clusters_all = true_clusters[idx_perm]
    print('Size of X_all is:',X_all.shape)
    #
    # Run PCMF
    tic = time.time()
    pcmf_type = 'pcmf_full_consensus'
    save_path = dir_path+pcmf_type+'_MouseOrgans_genomics_run'+'_N_'+str(X_all.shape[0])+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)+'_addmIters'+str(admm_iters)+'_penalty_type'+str(penalty_type)+'_intercept'+str(intercept)
    # output_file = dir_path+pcmf_type+'_MouseOrgans_genomics_run'+'_N_'+str(X_all.shape[0])+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)+'_addmIters'+str(admm_iters)+'.txt'
    print(save_path)
    A, U, S, V = pcmf_full_consensus(X_all, penalty_list, split_size=split_size, rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    toc = time.time() - tic
    np.savez(save_path+'.npz', pcmf_type=pcmf_type, X_c=X_all, true_clusters=true_clusters_all, A=A, V=V, U=U, S=S, split_size=split_size, runtime=toc, penalty_list=penalty_list, rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=intercept) 
    # plot_save_name = dir_path+pcmf_type+'_MouseOrgans_genomics_run'+'_N_'+str(X_all.shape[0])+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)+'_addmIters'+str(admm_iters)+'.pdf'
    generate_plots(A, U, V, S, split_size, penalty_list, true_clusters_all, save_path+'.pdf', 'mouse')
#

def run_MouseOrgansgenomics_Full_consensus_smallP(dir_path='results/',data_path='data/', penalty_type=None,split_size=50, rows_skip = 10, admm_iters = 5, gauss_coef=2.0, neighbors=None, rho = 1, weights='Gaussian', penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-5,500,150))[::-1]),axis=0), parallel=True, output_file='NaN'):
    # Load MouseOrgans dataset
    training_data_scaled, test_data_scaled, training_labels, test_labels, ind_gene_filter = load_mouse_organs(data_path)
    print("training_data_scaled",training_data_scaled.shape,"test_data_scaled",test_data_scaled.shape)
    intercept=True
    # intercept=False
    if rows_skip>0:
        training_data_scaled_subset = training_data_scaled[::rows_skip]
        training_labels_subset = training_labels[::rows_skip]
        training_labels_unique = np.unique(training_labels_subset)
        X_c = training_data_scaled_subset
        if intercept==True:
        	X_c = np.hstack((X_c,np.ones((X_c.shape[0],1))))
        true_clusters = training_labels_subset
        np.random.seed(seed=1234)
        idx_perm = np.random.permutation(X_c.shape[0])
        X_c = X_c[idx_perm,:]
        true_clusters = true_clusters[idx_perm]
        rounded = floor(X_c.shape[0]/split_size)*split_size
        X_all = X_c[0:rounded,:]
        true_clusters_all=true_clusters[0:rounded]
    else:
        X_c = np.vstack((training_data_scaled, test_data_scaled))
        if intercept==True:
        	X_c = np.hstack((X_c,np.ones((X_c.shape[0],1))))
        true_clusters = np.hstack((training_labels, test_labels))
        np.random.seed(seed=1234)
        idx_perm = np.random.permutation(X_c.shape[0])
        X_c = X_c[idx_perm,:]
        true_clusters = true_clusters[idx_perm]
        rounded = floor(X_c.shape[0]/split_size)*split_size
        X_all = X_c[0:rounded,:]
        true_clusters_all=true_clusters[0:rounded]
        # X_c = np.vstack((training_data_scaled, test_data_scaled))
        # true_clusters = np.vstack((training_labels, test_labels))

    X_c = X_c[:,0:1000]
    #
    # np.random.seed(seed=1234)
    # idx_perm = np.random.permutation(X_c.shape[0])
    # X_all = X_c[idx_perm,:]
    # true_clusters_all = true_clusters[idx_perm]
    print('Size of X_all is:',X_all.shape)
    #
    # Run PCMF
    tic = time.time()
    pcmf_type = 'pcmf_full_consensus_smallP'
    save_path = dir_path+pcmf_type+'_MouseOrgans_genomics_run'+'_N_'+str(X_all.shape[0])+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)+'_addmIters'+str(admm_iters)+'_penalty_type'+str(penalty_type)+'_intercept'+str(intercept)
    # output_file = dir_path+pcmf_type+'_MouseOrgans_genomics_run'+'_N_'+str(X_all.shape[0])+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)+'_addmIters'+str(admm_iters)+'.txt'
    print(save_path)
    A, U, S, V = pcmf_full_consensus(X_all, penalty_list, split_size=split_size, rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    toc = time.time() - tic
    np.savez(save_path+'.npz', pcmf_type=pcmf_type, X_c=X_all, true_clusters=true_clusters_all, A=A, V=V, U=U, S=S, split_size=split_size, runtime=toc, penalty_list=penalty_list, rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=intercept) 
    # plot_save_name = dir_path+pcmf_type+'_MouseOrgans_genomics_run'+'_N_'+str(X_all.shape[0])+'_split_size_'+str(split_size)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)+'_addmIters'+str(admm_iters)+'.pdf'
    generate_plots(A, U, V, S, split_size, penalty_list, true_clusters_all, save_path+'.pdf', 'mouse')
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
#
def generate_plots(A_list, U_list, V_list, s_list, split_size, penalty_list, true_clusters_all, plot_save_name, data_type):
    plt = path_plot(np.asarray(A_list)[:,:,0:4], penalty_list, plot_range=[0,-1])
    plt.xlabel("A")
    plt.ylabel("Estimates")
#
    try:                                                     
        plt = path_plot(np.asarray(A_list)[:,0:split_size,0:4], penalty_list, plot_range=[0,-1])
        plt.xlabel("A Batch 1")
        plt.ylabel("Estimates")
#
#
        plt = path_plot(np.asarray(A_list)[:,split_size:split_size*2,0:4], penalty_list, plot_range=[0,-1])
        plt.xlabel("A Batch 2")
        plt.ylabel("Estimates")
#
    except:
        print('Could not make A batches')
#
    plt = path_plot(np.asarray(U_list)[:,:,0:4], penalty_list, plot_range=[0,-1])
    plt.xlabel("U")
    plt.ylabel("Estimates")
#
    plt = path_plot(np.asarray(V_list)[:,:,0:4], penalty_list, plot_range=[0,-1])
    plt.xlabel("V")
    plt.ylabel("Estimates")
#
    plt = path_plot(np.atleast_3d(np.asarray(s_list)), penalty_list, plot_range=[0,-1])
    plt.xlabel("s")
    plt.ylabel("Estimates")
#
#     try:           
#         usV_list = []
#         for p in range(len(penalty_list)):
#             usV_list.append(U_list[p]*s_list[p]@V_list[p])
# #
#         path_plot(np.asarray(usV_list)[:,:,0:4], penalty_list, plot_range=[0,-1])
#         plt.xlabel("u*s*V")
#         plt.ylabel("Estimates")
#
    if data_type=='mouse':
        plt = path_plot(np.asarray(A_list)[:,true_clusters_all=='Heart',0:4], penalty_list, plot_range=[0,-1])
        plt.xlabel("A (Heart)")
        plt.ylabel("Estimates")
#
#
        plt = path_plot(np.asarray(A_list)[:,true_clusters_all=='Kidney',0:4], penalty_list, plot_range=[0,-1])
        plt.xlabel("A (Kidney)")
        plt.ylabel("Estimates")
#
#
        plt = path_plot(np.asarray(A_list)[:,true_clusters_all=='Large_Intestine',0:4], penalty_list, plot_range=[0,-1])
        plt.xlabel("A (Large_Intestine)")
        plt.ylabel("Estimates")
#
#
        plt = path_plot(np.asarray(A_list)[:,true_clusters_all=='Liver',0:4], penalty_list, plot_range=[0,-1])
        plt.xlabel("A (Liver)")
        plt.ylabel("Estimates")
#
#
        plt = path_plot(np.asarray(A_list)[:,true_clusters_all=='Lung',0:4], penalty_list, plot_range=[0,-1])
        plt.xlabel("A (Lung)")
        plt.ylabel("Estimates")
#
#
        plt = path_plot(np.asarray(A_list)[:,true_clusters_all=='Spleen',0:4], penalty_list, plot_range=[0,-1])
        plt.xlabel("A (Spleen)")
        plt.ylabel("Estimates")
#
#
    elif data_type=='gbmbreastlung':
        plt = path_plot(np.asarray(A_list)[:,true_clusters_all=='breast',0:4], penalty_list, plot_range=[0,-1])
        plt.xlabel("A (Breast)")
        plt.ylabel("Estimates")
#
#
        plt = path_plot(np.asarray(A_list)[:,true_clusters_all=='gbm',0:4], penalty_list, plot_range=[0,-1])
        plt.xlabel("A (gbm)")
        plt.ylabel("Estimates")
#
#
        plt = path_plot(np.asarray(A_list)[:,true_clusters_all=='lung',0:4], penalty_list, plot_range=[0,-1])
        plt.xlabel("A (lung)")
        plt.ylabel("Estimates")
#
    save_multi_image(plot_save_name)
#

#
# scale data
def run_GbmBreastLungCancergenomics_Full(dir_path='results/',data_path='data/', penalty_type=None,rows_skip=3, admm_iters = 5, gauss_coef=2.0, neighbors=None, rho = 1, weights='Gaussian', penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0), parallel=True, output_file='NaN'):
    # Load GbmBreastLungCancer dataset
    gbm = pd.read_csv(os.path.join(data_path, 'gbm_z.txt'), sep='\t', header=0)
    lung = pd.read_csv(os.path.join(data_path, 'lung_z.txt'), sep='\t', header=0)
    breast = pd.read_csv(os.path.join(data_path, 'breast_z.txt'), sep='\t', header=0)
    labels = np.hstack((np.repeat('gbm',gbm.shape[0]),np.repeat('lung',lung.shape[0]),np.repeat('breast',breast.shape[0])))
    gbm_T = gbm.T
    gbm_T['variables'] = gbm.columns
    lung_T = lung.T
    lung_T['variables'] = lung.columns
    breast_T = breast.T
    breast_T['variables'] = breast.columns
    data_T = pd.merge(gbm_T,pd.merge(lung_T,breast_T,on='variables'),on='variables')
    data = data_T.drop(columns='variables').T
    data.columns = data_T['variables']
    # 
    if rows_skip>0:
        data_subset = data[::rows_skip]
        labels_subset = labels[::rows_skip]
        X_c = data_subset.to_numpy()
        true_clusters = labels_subset
        print("X_c is:",X_c.shape)
    else:
        X_c = data.to_numpy()
        true_clusters = labels
    #
    np.random.seed(seed=1234)
    idx_perm = np.random.permutation(X_c.shape[0])
    X_all = X_c[idx_perm,:]
    true_clusters_all = true_clusters[idx_perm]
    print('Size of X_all is:',X_all.shape)
    # Run PCMF
    tic = time.time()
    pcmf_type = 'pcmf_full'
    save_path = dir_path+pcmf_type+'_GbmBreastLungCancer_genomics_run'+'_N_'+str(X_all.shape[0])+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)+'_addmIters'+str(admm_iters)+'_penalty_type'+str(penalty_type)

    # save_path = dir_path+pcmf_type+'_GbmBreastLungCancer_genomics_run'+'_N_'+str(X_all.shape[0])+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)+'_addmIters'+str(admm_iters)+'.npz'
    # output_file = dir_path+pcmf_type+'_GbmBreastLungCancer_genomics_run'+'_N_'+str(X_all.shape[0])+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)+'_addmIters'+str(admm_iters)+'.txt'
    print(save_path)
    A, U, S, V = pcmf_full(X_all, penalty_list, numba=True, rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    toc = time.time() - tic
    np.savez(save_path+'.npz', pcmf_type=pcmf_type, X_c=X_all, true_clusters=true_clusters_all, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=penalty_list, rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=False) 
    # plot_save_name = dir_path+pcmf_type+'_GbmBreastLungCancer_genomics_run'+'_N_'+str(X_all.shape[0])+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)+'_addmIters'+str(admm_iters)+'.pdf'
    generate_plots(A, U, V, S, X_all.shape[0], penalty_list, true_clusters_all, save_path+'.pdf', 'gbmbreastlung')


#