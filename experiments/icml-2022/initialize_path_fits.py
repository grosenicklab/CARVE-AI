
# srun --pty -n1 --mem=32G -p scu-cpu /bin/bash

os.chdir('/home/amb2022/clusterCCA_revision1/clusterCCA')

# RUN CLUSTERING ALGORITHMS AND SAVE OUTPUTS
def run_cluster_fits(data_path, pcmf_type, data_type='NA', r=0, save_path='results/cluster_fits/', pcmf_c_method='spectral', PCMFclusterpath_n_replicates=1):
    # LOAD DATA
    X, Xhat_list, true_clusters, results = load_experiments(data_path, pcmf_type, data_type, r=r)
    n_clusters = len(np.unique(true_clusters))
    #
    # INITIALIZE AND RUN FITS
    save_path_fits = save_path+'FIT_Table_'+os.path.splitext(data_path)[0]+'.npz' # data_path
    labels_list = []
    fits_table_list = []
    fits_table_list.append(['Cluster Method','Accuracy','ARI','NMI','Time Elapsed','Data Path']) 
    #
    try:
        cluster_method = 'PCA + K-Means'
        tic = time.time()
        labels, ari, nmi, acc = fit_pca_kmeans(X, true_clusters, n_clusters)
        toc = time.time() - tic
        fits_table_list.append([cluster_method,acc,ari,nmi,toc,data_path])
        labels_list.append([labels,true_clusters,n_clusters,cluster_method,data_path])
    except:
        fits_table_list.append([cluster_method,np.nan,np.nan,np.nan,np.nan,data_path])
        labels_list.append([np.nan,true_clusters,n_clusters,cluster_method,data_path])
    #
    try:
        cluster_method = 'Ward'
        tic = time.time()
        labels, ari, nmi, acc = fit_ward(X, true_clusters, n_clusters)
        toc = time.time() - tic
        fits_table_list.append([cluster_method,acc,ari,nmi,toc,data_path])
        labels_list.append([labels,true_clusters,n_clusters,cluster_method,data_path])
    except:
        fits_table_list.append([cluster_method,np.nan,np.nan,np.nan,np.nan,data_path])
        labels_list.append([np.nan,true_clusters,n_clusters,cluster_method,data_path])
    #
    try:
        cluster_method = 'Spectral'
        tic = time.time()
        labels, ari, nmi, acc = fit_spectral(X, true_clusters, n_clusters)
        toc = time.time() - tic
        fits_table_list.append([cluster_method,acc,ari,nmi,toc,data_path])
        labels_list.append([labels,true_clusters,n_clusters,cluster_method,data_path])
    except:
        fits_table_list.append([cluster_method,np.nan,np.nan,np.nan,np.nan,data_path])
        labels_list.append([np.nan,true_clusters,n_clusters,cluster_method,data_path])
    #
    try:
        cluster_method = 'DP-GMM'
        tic = time.time()
        labels, ari, nmi, acc = fit_dpgmm(X, true_clusters, n_clusters)
        toc = time.time() - tic
        fits_table_list.append([cluster_method,acc,ari,nmi,toc,data_path])
        labels_list.append([labels,true_clusters,n_clusters,cluster_method,data_path])
    except:
        fits_table_list.append([cluster_method,np.nan,np.nan,np.nan,np.nan,data_path])
        labels_list.append([np.nan,true_clusters,n_clusters,cluster_method,data_path])
    #
    try:
        cluster_method = 'Elastic Subspace'
        tic = time.time()
        labels, ari, nmi, acc = fit_elasticnetsubspace(X, true_clusters, n_clusters)
        toc = time.time() - tic
        fits_table_list.append([cluster_method,acc,ari,nmi,toc,data_path])
        labels_list.append([labels,true_clusters,n_clusters,cluster_method,data_path])
    except:
        fits_table_list.append([cluster_method,np.nan,np.nan,np.nan,np.nan,data_path])
        labels_list.append([np.nan,true_clusters,n_clusters,cluster_method,data_path])
    #
    try:
        cluster_method = 'gMADD'
        tic = time.time()
        labels, ari, nmi, acc = fit_gMADD(X, true_clusters, n_clusters)
        toc = time.time() - tic
        fits_table_list.append([cluster_method,acc,ari,nmi,toc,data_path])
        labels_list.append([labels,true_clusters,n_clusters,cluster_method,data_path])
    except:
        fits_table_list.append([cluster_method,np.nan,np.nan,np.nan,np.nan,data_path])
        labels_list.append([np.nan,true_clusters,n_clusters,cluster_method,data_path])
    #
    try:
        cluster_method = 'HDDC'
        tic = time.time()
        labels, ari, nmi, acc = fit_hddc(X, true_clusters, n_clusters)
        toc = time.time() - tic
        fits_table_list.append([cluster_method,acc,ari,nmi,toc,data_path])
        labels_list.append([labels,true_clusters,n_clusters,cluster_method,data_path])
    except:
        fits_table_list.append([cluster_method,np.nan,np.nan,np.nan,np.nan,data_path])
        labels_list.append([np.nan,true_clusters,n_clusters,cluster_method,data_path])
    #
    try:
        cluster_method = 'GMCM'
        tic = time.time()
        labels, ari, nmi, acc = fit_GMCM(X, true_clusters, n_clusters)
        toc = time.time() - tic
        fits_table_list.append([cluster_method,acc,ari,nmi,toc,data_path])
        labels_list.append([labels,true_clusters,n_clusters,cluster_method,data_path])
    except:
        fits_table_list.append([cluster_method,np.nan,np.nan,np.nan,np.nan,data_path])
        labels_list.append([np.nan,true_clusters,n_clusters,cluster_method,data_path])
    #
    try:
        cluster_method = 'MixGlasso'
        tic = time.time()
        labels, ari, nmi, acc = fit_mixglasso(X, true_clusters, n_clusters)
        toc = time.time() - tic
        fits_table_list.append([cluster_method,acc,ari,nmi,toc,data_path])
        labels_list.append([labels,true_clusters,n_clusters,cluster_method,data_path])
    except:
        fits_table_list.append([cluster_method,np.nan,np.nan,np.nan,np.nan,data_path])
        labels_list.append([np.nan,true_clusters,n_clusters,cluster_method,data_path])
    #
    try:
        cluster_method = 'VarSel'
        tic = time.time()
        labels, ari, nmi, acc = fit_VarSel(X, true_clusters, n_clusters)
        toc = time.time() - tic
        fits_table_list.append([cluster_method,acc,ari,nmi,toc,data_path])
        labels_list.append([labels,true_clusters,n_clusters,cluster_method,data_path])
    except:
        fits_table_list.append([cluster_method,np.nan,np.nan,np.nan,np.nan,data_path])
        labels_list.append([np.nan,true_clusters,n_clusters,cluster_method,data_path])
    #
    try:
        cluster_method = 'CARP'
        tic = time.time()
        labels, ari, nmi, acc = fit_carp(X, true_clusters, n_clusters)
        toc = time.time() - tic
        fits_table_list.append([cluster_method,acc,ari,nmi,toc,data_path])
        labels_list.append([labels,true_clusters,n_clusters,cluster_method,data_path])
    except:
        fits_table_list.append([cluster_method,np.nan,np.nan,np.nan,np.nan,data_path])
        labels_list.append([np.nan,true_clusters,n_clusters,cluster_method,data_path])
    #
    try:
        cluster_method = 'CBASS'
        tic = time.time()
        labels, ari, nmi, acc = fit_cbass(X, true_clusters, n_clusters)
        toc = time.time() - tic
        fits_table_list.append([cluster_method,acc,ari,nmi,toc,data_path])
        labels_list.append([labels,true_clusters,n_clusters,cluster_method,data_path])
    except:
        fits_table_list.append([cluster_method,np.nan,np.nan,np.nan,np.nan,data_path])
        labels_list.append([np.nan,true_clusters,n_clusters,cluster_method,data_path])
    #
    try:
        cluster_method = 'PCMF Cluster Path'
        tic = time.time()
        labels, ari, nmi, acc, best_idx, n_clusts_list,ics_list,centroids_list,best_idx_list,n_clusts_median,ics_median = fit_clusterpath(X, Xhat_list, true_clusters, n_clusters, 
                                                                   gauss_coef=gauss_coef, n_replicates=PCMFclusterpath_n_replicates, 
                                                                   c_method=pcmf_c_method, verbose=False)
        toc = time.time() - tic
        fits_table_list.append([cluster_method,acc,ari,nmi,toc,data_path])
        labels_list.append([labels,true_clusters,n_clusters,cluster_method,data_path])
        save_path = save_path+'FIT_'+pcmf_c_method+'_'+os.path.splitext(data_path)[0]+'.npz' # data_path
        np.savez(save_path, data_path=data_path, pcmf_type=pcmf_type, pcmf_c_method=pcmf_c_method, PCMFclusterpath_n_replicates=PCMFclusterpath_n_replicates, 
                 labels=labels, ari=ari, nmi=nmi, acc=acc, best_idx=best_idx, 
                 n_clusts_list=n_clusts_list, ics_list=ics_list, centroids_list=centroids_list, best_idx_list=best_idx_list, 
                 n_clusts_median=n_clusts_median, ics_median=ics_median) 
    except:
        fits_table_list.append([cluster_method,np.nan,np.nan,np.nan,np.nan,data_path])
        labels_list.append([np.nan,true_clusters,n_clusters,cluster_method,data_path])
    #
    # SAVE FITS
    np.savez(save_path_fits, data_path=data_path, labels_list=labels_list, fits_table_list=fits_table_list)



# run_cluster_fits(data_path, pcmf_type, data_type, r=0, pcmf_c_method='spectral', PCMFclusterpath_n_replicates=1):
# def fit_1():
#     data_path = '/results/pcmf_full_consensus_MouseOrgans_genomics_run_N_100_split_size_50_gausscoef5.0_neighbors25_rho1.0_addmIters10_penalty_type0.0_interceptTrue.npz'
#     pcmf_type = 'pcmf_approxUV'
#     run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)

# def fit_1():
#     data_path = '/results/pcmf_full_consensus_MouseOrgans_genomics_run_N_100_split_size_50_gausscoef5.0_neighbors25_rho1.0_addmIters10_penalty_type0.0_interceptTrue.npz'
#     pcmf_type = 'pcmf_full'
#     run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)

# def fit_1():
#     data_path = '/results/pcmf_full_consensus_MouseOrgans_genomics_run_N_100_split_size_50_gausscoef5.0_neighbors25_rho1.0_addmIters10_penalty_type0.0_interceptTrue.npz'
#     pcmf_type = 'pcmf_full_consensus'
#     run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)

## AISTATS GENOMICS
def fit_AISTATS_1():
    # 14 Cancer
    pcmf_type = 'pcmf_full'
    data_path = 'results/AISTATS_pcmf_full_14Cancer_genomics_run_gausscoef2.0_neighbors15_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'results/AISTATS_pcmf_full_14Cancer_genomics_run_gausscoef2.0_neighbors25_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'results/AISTATS_pcmf_full_14Cancer_genomics_run_gausscoef2.0_neighborsNone_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)

def fit_AISTATS_2():
    # 14 Cancer
    pcmf_type = 'pcmf_approx_uV'
    data_path = 'AISTATS_pcmf_approx_uV_14Cancer_genomics_run_gausscoef2.0_neighbors15_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'AISTATS_pcmf_approx_uV_14Cancer_genomics_run_gausscoef2.0_neighbors25_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'AISTATS_pcmf_approx_uV_14Cancer_genomics_run_gausscoef2.0_neighborsNone_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)

def fit_AISTATS_3():
    # GbmBreastLungCancer
    pcmf_type = 'pcmf_full'
    data_path = 'AISTATS_pcmf_full_GbmBreastLungCancer_genomics_run_gausscoef2.0_neighbors15_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'AISTATS_pcmf_full_GbmBreastLungCancer_genomics_run_gausscoef2.0_neighbors25_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'AISTATS_pcmf_full_GbmBreastLungCancer_genomics_run_gausscoef2.0_neighborsNone_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)

def fit_AISTATS_4():
    # GbmBreastLungCancer
    pcmf_type = 'pcmf_approx_uV'
    data_path = 'AISTATS_pcmf_approx_uV_GbmBreastLungCancer_genomics_run_gausscoef2.0_neighbors15_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'AISTATS_pcmf_approx_uV_GbmBreastLungCancer_genomics_run_gausscoef2.0_neighbors25_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'AISTATS_pcmf_approx_uV_GbmBreastLungCancer_genomics_run_gausscoef2.0_neighborsNone_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    pcmf_type = 'pcmf_approx_uV'
    data_path = 'AISTATS_pcmf_approx_uV_GbmBreastLungCancer_genomics_run_gausscoef2.0_neighbors15_rho1.5.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'AISTATS_pcmf_approx_uV_GbmBreastLungCancer_genomics_run_gausscoef2.0_neighbors15_rho2.0.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'AISTATS_pcmf_approx_uV_GbmBreastLungCancer_genomics_run_gausscoef2.0_neighbors15_rho2.5.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)

def fit_AISTATS_5():
    # NCI
    pcmf_type = 'pcmf_full'
    data_path = 'AISTATS_pcmf_full_NCI_genomics_run_gausscoef2.0_neighbors15_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'AISTATS_pcmf_full_NCI_genomics_run_gausscoef2.0_neighbors25_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'AISTATS_pcmf_full_NCI_genomics_run_gausscoef2.0_neighborsNone_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)

def fit_AISTATS_6():
    # NCI
    pcmf_type = 'pcmf_approx_uV'
    data_path = 'AISTATS_pcmf_approx_uV_NCI_genomics_run_gausscoef2.0_neighbors15_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'AISTATS_pcmf_approx_uV_NCI_genomics_run_gausscoef2.0_neighbors25_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'AISTATS_pcmf_approx_uV_NCI_genomics_run_gausscoef2.0_neighborsNone_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)

def fit_AISTATS_7():
    # SRBCT
    pcmf_type = 'pcmf_full'
    data_path = 'results/AISTATS_pcmf_full_SRBCT_genomics_run_gausscoef2.0_neighbors15_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'results/AISTATS_pcmf_full_SRBCT_genomics_run_gausscoef2.0_neighbors25_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'results/AISTATS_pcmf_full_SRBCT_genomics_run_gausscoef2.0_neighborsNone_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)

def fit_AISTATS_8():
    # SRBCT
    pcmf_type = 'pcmf_approx_uV'
    data_path = 'results/AISTATS_pcmf_approx_uV_SRBCT_genomics_run_gausscoef2.0_neighbors15_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'results/AISTATS_pcmf_approx_uV_SRBCT_genomics_run_gausscoef2.0_neighbors25_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)
    data_path = 'results/AISTATS_pcmf_approx_uV_SRBCT_genomics_run_gausscoef2.0_neighborsNone_rho1.npz'
    run_cluster_fits(data_path, pcmf_type, data_type='genomics', PCMFclusterpath_n_replicates=10)

## AISTATS TABLE 1
# # 
# def f1_a(run, diffmeans_p20_ari_nmi_toc_types=[]):
#     ari_nmi_toc_type = run_numerical_experiments_clusterfits_parallel_aistats(r=run, data_type = 'diffmeans', num_vars = 20)
#     diffmeans_p20_ari_nmi_toc_types.append(ari_nmi_toc_type)
#     return diffmeans_p20_ari_nmi_toc_types

# def f1_b(run, diffmeans_p200_ari_nmi_toc_types=[]):
#     ari_nmi_toc_type = run_numerical_experiments_clusterfits_parallel_aistats(r=run, data_type = 'diffmeans', num_vars = 200)
#     diffmeans_p200_ari_nmi_toc_types.append(ari_nmi_toc_type)
#     return diffmeans_p200_ari_nmi_toc_types

# def f1_c(run, diffmeans_p2000_ari_nmi_toc_types=[]):
#     ari_nmi_toc_type = run_numerical_experiments_clusterfits_parallel_aistats(r=run, data_type = 'diffmeans', num_vars = 2000)
#     diffmeans_p2000_ari_nmi_toc_types.append(ari_nmi_toc_type)
#     return diffmeans_p2000_ari_nmi_toc_types

# def f2_a(run, samemeans_p20_ari_nmi_toc_types=[]):
#     ari_nmi_toc_type = run_numerical_experiments_clusterfits_parallel_aistats(r=run, data_type = 'samemeans', num_vars = 20)
#     samemeans_p20_ari_nmi_toc_types.append(ari_nmi_toc_type)
#     return samemeans_p20_ari_nmi_toc_types

# def f2_b(run, samemeans_p200_ari_nmi_toc_types=[]):
#     ari_nmi_toc_type = run_numerical_experiments_clusterfits_parallel_aistats(r=run, data_type = 'samemeans', num_vars = 200)
#     samemeans_p200_ari_nmi_toc_types.append(ari_nmi_toc_type)
#     return samemeans_p200_ari_nmi_toc_types

# def f2_c(run, samemeans_p2000_ari_nmi_toc_types=[]):
#     ari_nmi_toc_type = run_numerical_experiments_clusterfits_parallel_aistats(r=run, data_type = 'samemeans', num_vars = 2000)
#     samemeans_p2000_ari_nmi_toc_types.append(ari_nmi_toc_type)
#     return samemeans_p2000_ari_nmi_toc_types

# def f3_a(run, diffmeans_p20_ari_nmi_toc_types=[]):
#     ari_nmi_toc_type = run_numerical_experiments_clusterfits_parallel_aistats(r=run, data_type = 'diffmeans', num_vars = 20)
#     diffmeans_p20_ari_nmi_toc_types.append(ari_nmi_toc_type)
#     return diffmeans_p20_ari_nmi_toc_types

# def f3_b(run, diffmeans_p200_ari_nmi_toc_types=[]):
#     ari_nmi_toc_type = run_numerical_experiments_clusterfits_parallel_aistats(r=run, data_type = 'diffmeans', num_vars = 200)
#     diffmeans_p200_ari_nmi_toc_types.append(ari_nmi_toc_type)
#     return diffmeans_p200_ari_nmi_toc_types

# def f3_c(run, diffmeans_p2000_ari_nmi_toc_types=[]):
#     ari_nmi_toc_type = run_numerical_experiments_clusterfits_parallel_aistats(r=run, data_type = 'diffmeans', num_vars = 2000)
#     diffmeans_p2000_ari_nmi_toc_types.append(ari_nmi_toc_type)
#     return diffmeans_p2000_ari_nmi_toc_types

## PCMF consensus fits

# Genomics

# Synthetic








# if __name__ == '__main__':   
#     pool = multiprocessing.Pool(processes=10)
#     res = pool.map(smap, [run_div_neighbors25(0), run_div_neighbors25(1), run_div_neighbors25(2), 
#                         run_div_neighbors25(3), run_div_neighbors25(4), run_div_neighbors25(5),
#                         run_div_neighbors25(6), run_div_neighbors25(7), run_div_neighbors25(8), run_div_neighbors25(10)])



