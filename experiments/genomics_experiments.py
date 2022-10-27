####
import os
os.getcwd()
import numpy as np
import sys
import multiprocessing  
# os.chdir('/home/amb2022/clusterCCA_revision1/clusterCCA/utils/subspace-clustering-master')
# import cluster
os.chdir('/athena/listonlab/store/amb2022/clusterCCA_revision1/PCMF')

saving_dir_fullpath='/athena/listonlab/scratch/amb2022/PCMF/results/'

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


def run_14Cancergenomics_Full_problem_rank(dir_path='results/',data_path='data/', problem_rank=3, admm_iters = 5, gauss_coef=2.0, neighbors=None, rho = 1, weights='Gaussian', penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-5,500,150))[::-1]),axis=0), parallel=True, output_file='NaN'):
    # Load 14 Cancer dataset
    train_14c = np.loadtxt(data_path+'14cancer.xtrain')
    labels_14c = np.loadtxt(data_path+'14cancer.ytrain')
    test_14c = np.loadtxt(data_path+'14cancer.xtest')
    testlabels_14c = np.loadtxt(data_path+'14cancer.ytest')
    X_c = np.hstack((train_14c,test_14c)).T
    labels = np.hstack((labels_14c,testlabels_14c)).T
    true_clusters = labels
    labels_names = ['breast','prostate','lung','collerectal','lymphoma','bladder','melanoma','uterus','leukemia','renal','pancreas','ovary','meso','cns']
    scaler = StandardScaler()
    scaler.fit(X_c)
    X_c = scaler.transform(X_c)
    X_c = np.hstack((X_c,np.ones((X_c.shape[0],1))))
    # Run PCMF
    tic = time.time()
    pcmf_type = 'AISTATS_pcmf_full'
    save_path = dir_path+pcmf_type+'_14Cancer_genomics_run'+'_rank'+str(problem_rank)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)
    print(save_path)
    A, U, S, V = pcmf_full(X_c, penalty_list, rho=rho, problem_rank=problem_rank, numba=True, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    toc = time.time() - tic
    np.savez(save_path+'.npz', problem_rank=problem_rank, pcmf_type=pcmf_type, X_c=X_c, true_clusters=true_clusters, labels_names=labels_names, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=penalty_list, rho=rho, numba=True, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=True) 

def run_NCIgenomics_Full_problem_rank(dir_path='results/',data_path='data/', problem_rank=3, admm_iters = 5, gauss_coef=2.0, neighbors=None, rho = 1, weights='Gaussian', penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-5,500,150))[::-1]),axis=0), parallel=True, output_file='NaN'):
    # Load NCI dataset
    X_c = np.genfromtxt(data_path+'nci.data.csv',delimiter=',')
    X_c = X_c[1:,:]
    X_c = X_c[:,1:].T
    raw_labels = np.loadtxt(data_path+'nci.label.txt',dtype=str)
    classes, class_counts = np.unique(raw_labels,return_counts=True)
    labels = np.array([np.where(classes == l)[0][0] for l in raw_labels])
    true_clusters = labels
    scaler = StandardScaler()
    scaler.fit(X_c)
    X_c = scaler.transform(X_c)
    X_c = np.hstack((X_c,np.ones((X_c.shape[0],1))))
    # Run PCMF
    tic = time.time()
    pcmf_type = 'AISTATS_pcmf_full'
    save_path = dir_path+pcmf_type+'_NCI_genomics_run_run'+'_rank'+str(problem_rank)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)
    print(save_path)
    A, U, S, V = pcmf_full(X_c, penalty_list, rho=rho, numba=True, problem_rank=problem_rank, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    toc = time.time() - tic
    np.savez(save_path+'.npz', problem_rank=problem_rank, pcmf_type=pcmf_type, X_c=X_c, true_clusters=true_clusters, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=penalty_list, rho=rho, numba=True, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=True) 

def run_SRBCTgenomics_Full_problem_rank(dir_path='results/',data_path='data/', problem_rank=3, admm_iters = 5, gauss_coef=2.0, neighbors=None, rho = 1, weights='Gaussian', penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-5,500,150))[::-1]),axis=0), parallel=True, output_file='NaN'):
    # Load SRBCT dataset
    train_khan = np.loadtxt(data_path+'khan.xtrain')
    labels_khan = np.loadtxt(data_path+'khan.ytrain')
    test_khan = np.loadtxt(data_path+'khan.xtest')
    testlabels_khan = np.loadtxt(data_path+'khan.ytest')
    X_c = np.hstack((train_khan,test_khan)).T
    labels = np.hstack((labels_khan,testlabels_khan)).T
    true_clusters = labels
    idx_keep = ~np.isnan(true_clusters)
    X_c = X_c[idx_keep,:]
    true_clusters = labels[idx_keep]
    labels_names = ['EWS','RMS','NB','BL']
    scaler = StandardScaler()
    scaler.fit(X_c)
    X_c = scaler.transform(X_c)
    X_c = np.hstack((X_c,np.ones((X_c.shape[0],1))))
    # Run PCMF
    tic = time.time()
    pcmf_type = 'AISTATS_pcmf_full'
    save_path = dir_path+pcmf_type+'_SRBCT_genomics_run_run'+'_rank'+str(problem_rank)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)
    print(save_path)
    A, U, S, V = pcmf_full(X_c, penalty_list, rho=rho, numba=True, problem_rank=problem_rank, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    toc = time.time() - tic
    np.savez(save_path+'.npz', problem_rank=problem_rank, pcmf_type=pcmf_type, X_c=X_c, true_clusters=true_clusters, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=penalty_list, rho=rho, numba=True, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=True) 

def load_mouse_organs(data_path,quantile_prob=0.2):
    'quantile_prob is Percentile for data filtering.'
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

def run_MouseOrgansgenomics_Full_problem_rank(dir_path='results/',data_path='data/', problem_rank=3, admm_iters = 5, gauss_coef=2.0, neighbors=None, rho = 1, weights='Gaussian', penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-5,500,150))[::-1]),axis=0), parallel=True, output_file='NaN'):
    # Load MouseOrgans dataset
    training_data_scaled, test_data_scaled, training_labels, test_labels, ind_gene_filter = load_mouse_organs(data_path)
    rows_skip = 50
    training_data_scaled_subset = training_data_scaled[::rows_skip]
    training_labels_subset = training_labels[::rows_skip]
    training_labels_unique = np.unique(training_labels_subset)
    X_c = training_data_scaled_subset
    X_c = np.hstack((X_c,np.ones((X_c.shape[0],1))))
    true_clusters = training_labels_subset
    # Run PCMF
    tic = time.time()
    pcmf_type = 'AISTATS_pcmf_full'
    save_path = dir_path+pcmf_type+'_MouseOrgans_genomics_run_run'+'_rank'+str(problem_rank)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)
    print(save_path)
    A, U, S, V = pcmf_full(X_c, penalty_list, rho=rho, numba=True, problem_rank=problem_rank, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    toc = time.time() - tic
    np.savez(save_path+'.npz', problem_rank=problem_rank, pcmf_type=pcmf_type, X_c=X_c, true_clusters=true_clusters, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=penalty_list, rho=rho, numba=True, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=True) 

def run_GbmBreastLungCancergenomics_Full_problem_rank(dir_path='results/',data_path='data/', problem_rank=3, admm_iters = 5, gauss_coef=2.0, neighbors=None, rho = 1, weights='Gaussian', penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-5,500,150))[::-1]),axis=0), parallel=True, output_file='NaN'):
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
    rows_skip = 3
    data_subset = data[::rows_skip]
    labels_subset = labels[::rows_skip]
    X_c = data_subset.to_numpy()
    true_clusters = labels_subset
    # Run PCMF
    tic = time.time()
    pcmf_type = 'AISTATS_pcmf_full'
    save_path = dir_path+pcmf_type+'_GbmBreastLungCancer_genomics_run_run'+'_rank'+str(problem_rank)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'_rho'+str(rho)
    print(save_path)
    A, U, S, V = pcmf_full(X_c, penalty_list, rho=rho, numba=True, problem_rank=problem_rank, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=save_path+'.txt')
    toc = time.time() - tic
    np.savez(save_path+'.npz', problem_rank, pcmf_type=pcmf_type, X_c=X_c, true_clusters=true_clusters, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=penalty_list, rho=rho, numba=True, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, intercept=True) 

def smap(f):
    return f()

penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
# penalty_list = np.repeat(np.inf,2)
gauss_coef=2.0
# 14 cancer
def f_14c_a():
    run_14Cancergenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=3, neighbors=15, penalty_list = penalty_list)

def f_14c_b():
    run_14Cancergenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=15, neighbors=15, penalty_list = penalty_list)

def f_14c_c(): 
    run_14Cancergenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=20, neighbors=15, penalty_list = penalty_list)

def f_14c_d():
    run_14Cancergenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=3, neighbors=25, penalty_list = penalty_list)

def f_14c_e(): 
    run_14Cancergenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=15, neighbors=25, penalty_list = penalty_list)

def f_14c_f():
    run_14Cancergenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=20, neighbors=25, penalty_list = penalty_list)

# NCI
def f_NCI_a():
    run_NCIgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=3, neighbors=15, penalty_list = penalty_list)

def f_NCI_b():
    run_NCIgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=14, neighbors=15, penalty_list = penalty_list)

def f_NCI_c(): 
    run_NCIgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=20, neighbors=15, penalty_list = penalty_list)

def f_NCI_d():
    run_NCIgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=3, neighbors=25, penalty_list = penalty_list)

def f_NCI_e(): 
    run_NCIgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=14, neighbors=25, penalty_list = penalty_list)

def f_NCI_f():
    run_NCIgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=20, neighbors=25, penalty_list = penalty_list)

# SRBCT
def f_SRBCT_a():
    run_SRBCTgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=3, neighbors=15, penalty_list = penalty_list)

def f_SRBCT_b():
    run_SRBCTgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=5, neighbors=15, penalty_list = penalty_list)

def f_SRBCT_c(): 
    run_SRBCTgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=10, neighbors=15, penalty_list = penalty_list)

def f_SRBCT_d():
    run_SRBCTgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=3, neighbors=25, penalty_list = penalty_list)

def f_SRBCT_e(): 
    run_SRBCTgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=5, neighbors=25, penalty_list = penalty_list)

def f_SRBCT_f():
    run_SRBCTgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=10, neighbors=25, penalty_list = penalty_list)

# GbmBreastLungCancer
def f_GbmBreastLungCancer_a():
    run_GbmBreastLungCancergenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=3, neighbors=15, penalty_list = penalty_list)

def f_GbmBreastLungCancer_b():
    run_GbmBreastLungCancergenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=4, neighbors=15, penalty_list = penalty_list)

def f_GbmBreastLungCancer_c(): 
    run_GbmBreastLungCancergenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=10, neighbors=15, penalty_list = penalty_list)

def f_GbmBreastLungCancer_d():
    run_GbmBreastLungCancergenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=3, neighbors=25, penalty_list = penalty_list)

def f_GbmBreastLungCancer_e(): 
    run_GbmBreastLungCancergenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=4, neighbors=25, penalty_list = penalty_list)

def f_GbmBreastLungCancer_f():
    run_GbmBreastLungCancergenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=10, neighbors=25, penalty_list = penalty_list)

# MouseOrgans
def f_MouseOrgans_a():
    run_MouseOrgansgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=3, neighbors=15, penalty_list = penalty_list)

def f_MouseOrgans_b():
    run_MouseOrgansgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=8, neighbors=15, penalty_list = penalty_list)

def f_MouseOrgans_c(): 
    run_MouseOrgansgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=20, neighbors=15, penalty_list = penalty_list)

def f_MouseOrgans_d():
    run_MouseOrgansgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=3, neighbors=25, penalty_list = penalty_list)

def f_MouseOrgans_e(): 
    run_MouseOrgansgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=8, neighbors=25, penalty_list = penalty_list)

def f_MouseOrgans_f():
    run_MouseOrgansgenomics_Full_problem_rank(dir_path=saving_dir_fullpath, gauss_coef=gauss_coef, problem_rank=20, neighbors=25, penalty_list = penalty_list)


















