import os
import sys
import multiprocessing  
import time
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cross_decomposition import CCA
from sklearn.cluster import SpectralClustering
from sklearn import mixture
from sklearn.cluster import KMeans
# from sklearn.metrics.cluster import normalized_mutual_info_score as NMI_score, adjusted_rand_score as ARI_score
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI_score, adjusted_rand_score as ARI_score, rand_score as rand_score
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score, mean_squared_error # cohen_kappa_score, hinge_loss, coverage_error, consensus_score
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linprog
from scipy.optimize import linear_sum_assignment as linear_assignment

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    mixture.BayesianGaussianMixture()

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning,ignore::ConvergenceWarning,ignore::RuntimeWarning')

# # os.environ['R_HOME'] = "/Library/Frameworks/R.framework/Resources/"
# import rpy2.robjects as robjects
# import rpy2.rinterface as rinterface
# from rpy2.robjects.packages import importr
# base = importr('base')
# conflicted = importr('conflicted')
# VarSelLCM = importr('VarSelLCM')

os.chdir('/home/amb2022/clusterCCA_revision1/clusterCCA')


# from cluster.selfrepresentation import ElasticNetSubspaceClustering

def smap(f):
    return f()

# def smap(f_args):
#     f, *args = f_args
#     return f(*args)

def calculate_scores_nonpath(pred_clusters, true_clusters):
    """
    Calculate cluster accuracy scores and requires true clusters
    Calls cluster_along_path on each index i along penalties in path
    Params:
        true_clusters
        path_length
    Returns:
        path_clusters = list of clusterings at each penalty on path
        nmi_score = np-array of normalized mutual information score for each penalty
        adj_rand_score = np-array of adjusted rand score for each penalty
        rand_score = np-array of rand score for each penalty
        mse_score = np-array of mean squared error score for each penalty
    """
    nmi_score = NMI_score(true_clusters, pred_clusters, average_method='arithmetic')
    adj_rand_score = ARI_score(true_clusters, pred_clusters)
    ri_score = rand_score(true_clusters, pred_clusters)
    mse_score = mean_squared_error(true_clusters, pred_clusters)
     
    return nmi_score, adj_rand_score, ri_score, mse_score


def confusion_matrix_ordered(pred, true):
    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)
    conf_mat = confusion_matrix(pred,true)
    indexes = linear_assignment(_make_cost_m(conf_mat))
    js = [e for e in sorted(indexes, key=lambda x: x[0])[1]]
    conf_mat_ord = conf_mat[:, js]
    return conf_mat_ord

def fit_pca_kmeans(X, true_clusters, n_clusters):
    '''CCA + Kmeans clustering'''
    from sklearn.cluster import KMeans
    # Fit PCA-kmeans
    u,d,vh = np.linalg.svd(X, full_matrices=False)
    Xhat = u[:,0].reshape((X.shape[0], 1))*vh[0,:].reshape((1, X.shape[1]))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Xhat)
    labels = kmeans.labels_
    #
    # Calculate scores
    nmi, ari, ri, mse = calculate_scores_nonpath(labels, true_clusters)
    # Calculate accuracy
    conf_mat_ord = confusion_matrix_ordered(labels,true_clusters)
    acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
    #
    return labels, ari, nmi, acc

def fit_ward(X, true_clusters, n_clusters):
    '''Ward clustering'''
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph
    #
    data_in = X
    #
    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(data_in, n_neighbors=10, include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    #
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', connectivity=connectivity).fit(data_in)
    labels = clustering.labels_    
    #
    # Calculate scores
    nmi, ari, ri, mse = calculate_scores_nonpath(labels, true_clusters)
    # Calculate accuracy
    conf_mat_ord = confusion_matrix_ordered(labels,true_clusters)
    acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
    #
    return labels, ari, nmi, acc

def fit_spectral(X, true_clusters, n_clusters):
    '''Spectral clustering'''
    from sklearn.cluster import SpectralClustering
    #
    data_in = X
    #
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, random_state=20, affinity="nearest_neighbors").fit(data_in)
    labels = spectral_clustering.labels_
    #
    # Calculate scores
    nmi, ari, ri, mse = calculate_scores_nonpath(labels, true_clusters)
    # Calculate accuracy
    conf_mat_ord = confusion_matrix_ordered(labels,true_clusters)
    acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
    #
    return labels, ari, nmi, acc

def fit_dpgmm(X, true_clusters, n_clusters):
    '''DPGMM clustering'''
    from sklearn import mixture
    #
    data_in = X
    #
    dpgmm = mixture.BayesianGaussianMixture(n_components=n_clusters, covariance_type='full', weight_concentration_prior_type='dirichlet_process',max_iter=1000)
    dpgmm_mod = dpgmm.fit(data_in)
    labels = dpgmm.predict(data_in)
    #
    # Calculate scores
    nmi, ari, ri, mse = calculate_scores_nonpath(labels, true_clusters)
    # Calculate accuracy
    conf_mat_ord = confusion_matrix_ordered(labels,true_clusters)
    acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
    #
    return labels, ari, nmi, acc

def fit_elasticnetsubspace(X, true_clusters, n_clusters):
    '''Elastic net clustering'''
    from cluster.selfrepresentation import ElasticNetSubspaceClustering
    #
    data_in = X
    #
    essc = ElasticNetSubspaceClustering(n_clusters=n_clusters, algorithm='spams', gamma=5, active_support=False, tau=0.1).fit(data_in)
    labels = essc.labels_
    #
    # Calculate scores
    nmi, ari, ri, mse = calculate_scores_nonpath(labels, true_clusters)
    # Calculate accuracy
    conf_mat_ord = confusion_matrix_ordered(labels,true_clusters)
    acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
    #
    return labels, ari, nmi, acc

def fit_gMADD(X, true_clusters, n_clusters):
    '''gMADD clustering'''    #                                                                                                          
    from rpy2.robjects.packages import importr
    base = importr('base')
    utils = importr('utils')
    conflicted = importr('conflicted')
    HDLSSkST = importr('HDLSSkST')
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    rpy2.robjects.r('''                                                                                                                       
            # create a function `f`                                                                                                      
            f = function(X,true_clusters,n_clusters) {                                                                                                                                                                                                              
                N = dim(X)[1]                                                                                                            
                labels = gMADD(1,1,n_clusters,1,X)                                                                                             
                return(labels)                                                                                                           
            }                                                                                                                            
            # call the function `f` with argument value 3                                                                                
            ''')
    r_f = rpy2.robjects.globalenv['f']
    labels = r_f(X,true_clusters,n_clusters)
    #
    # Calculate scores
    nmi, ari, ri, mse = calculate_scores_nonpath(labels, true_clusters)
    # Calculate accuracy
    conf_mat_ord = confusion_matrix_ordered(labels,true_clusters)
    acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
    #
    return labels, ari, nmi, acc

def fit_hddc(X, true_clusters, n_clusters):
    '''HDDC clustering'''    #
    from rpy2.robjects.packages import importr
    base = importr('base')
    utils = importr('utils')
    conflicted = importr('conflicted')
    HDclassif = importr('HDclassif')
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    rpy2.robjects.r('''
            # create a function `f`
            f = function(X,true_clusters,n_clusters) {
                prms1 = hddc(X, K=n_clusters, model="ALL", algo="EM")
                labels = prms1$class
                return(labels)
            }
            # call the function `f` with argument value 3
            ''')
    r_f = rpy2.robjects.globalenv['f']
    labels = r_f(X,true_clusters,n_clusters)
    #
    # Calculate scores
    nmi, ari, ri, mse = calculate_scores_nonpath(labels, true_clusters)
    # Calculate accuracy
    conf_mat_ord = confusion_matrix_ordered(labels,true_clusters)
    acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
    #
    return labels, ari, nmi, acc

def fit_GMCM(X, true_clusters, n_clusters):
    '''GMCM clustering'''    #
    from rpy2.robjects.packages import importr
    base = importr('base')
    utils = importr('utils')
    conflicted = importr('conflicted')
    GMCM = importr('GMCM')
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    robjects.r('''
            # create a function `f`
            f = function(X,true_clusters,n_clusters) {
                uhat = Uhat(X)
                init_theta = choose.theta(uhat, m = n_clusters) # m is number of components/clusters
                est_theta = fit.full.GMCM(u = uhat,  # Ranking function is applied automatically
                                           theta = init_theta,
                                           method = "NM",
                                           max.ite = 5000,
                                           reltol = 1e-4,
                                           verbose = FALSE)
                labels = classify(uhat, est_theta)
                return(labels)
            }
            # call the function `f` with argument value 3
            ''')
    r_f = robjects.globalenv['f']
    labels = r_f(X,true_clusters,n_clusters)
    # Calculate scores
    nmi, ari, ri, mse = calculate_scores_nonpath(labels, true_clusters)
    # Calculate accuracy
    conf_mat_ord = confusion_matrix_ordered(labels,true_clusters)
    acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
    #
    return labels, ari, nmi, acc

def fit_mixglasso(X, true_clusters, n_clusters):
    '''mixGlasso clustering'''    #
    from rpy2.robjects.packages import importr
    base = importr('base')
    utils = importr('utils')
    conflicted = importr('conflicted')
    nethet = importr('nethet')
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    robjects.r('''
            # create a function `f`
            f = function(X,true_clusters,n_clusters) {
                mixglasso.result = mixglasso(X], n.comp=n_clusters)
                labels = mixglasso.result$comp
                return(labels)
            }
            # call the function `f` with argument value 3
            ''')
    r_f = robjects.globalenv['f']
    labels = r_f(X,true_clusters,n_clusters).flatten()
    # Calculate scores
    nmi, ari, ri, mse = calculate_scores_nonpath(labels, true_clusters)
    # Calculate accuracy
    conf_mat_ord = confusion_matrix_ordered(labels,true_clusters)
    acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
    #
    return labels, ari, nmi, acc

def fit_VarSel(X, true_clusters, n_clusters, num_cores):
    '''VarSel with variable selection clustering'''    #
    from rpy2.robjects.packages import importr
    base = importr('base')
    utils = importr('utils')
    conflicted = importr('conflicted')
    VarSelLCM = importr('VarSelLCM')
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    robjects.r('''
            # create a function `f`
            f = function(X,true_clusters,n_clusters,num_cores) {
                res_with = VarSelCluster(X, gvals = n_clusters, nbcores = num_cores, crit.varsel = "BIC")
                labels = fitted(res_with)
                return(labels)
            }
            # call the function `f` with argument value 3
            ''')
    r_f = robjects.globalenv['f']
    labels = r_f(X,true_clusters,n_clusters)
    # Calculate scores
    nmi, ari, ri, mse = calculate_scores_nonpath(labels, true_clusters)
    # Calculate accuracy
    conf_mat_ord = confusion_matrix_ordered(labels,true_clusters)
    acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
    #
    return labels, ari, nmi, acc

def fit_carp(X, true_clusters, n_clusters):
    '''CARP clustering'''    #
    from rpy2.robjects.packages import importr
    base = importr('base')
    utils = importr('utils')
    conflicted = importr('conflicted')
    clustRviz = importr('clustRviz')
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    robjects.r('''
            # create a function `f`
            f = function(X,true_clusters,n_clusters) {
                carp_fit = CARP(X)
                labels = get_cluster_labels(carp_fit, k = n_clusters)
                return(labels)
            }
            # call the function `f` with argument value 3
            ''')
    r_f = robjects.globalenv['f']
    labels = r_f(X,true_clusters,n_clusters)
    # Calculate scores
    nmi, ari, ri, mse = calculate_scores_nonpath(labels, true_clusters)
    # Calculate accuracy
    conf_mat_ord = confusion_matrix_ordered(labels,true_clusters)
    acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
    #
    return labels, ari, nmi, acc

def fit_cbass(X, true_clusters, n_clusters):
    '''CBASS clustering'''    #
    from rpy2.robjects.packages import importr
    base = importr('base')
    utils = importr('utils')
    conflicted = importr('conflicted')
    clustRviz = importr('clustRviz')
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    robjects.r('''
            # create a function `f`
            f = function(X,true_clusters,n_clusters) {
                cbass_fit = CBASS(X)
                labels = get_cluster_labels(cbass_fit,k.row=n_clusters)
                return(labels)
            }
            # call the function `f` with argument value 3
            ''')
    r_f = robjects.globalenv['f']
    labels = r_f(X,true_clusters,n_clusters)
    # Calculate scores
    nmi, ari, ri, mse = calculate_scores_nonpath(labels, true_clusters)
    # Calculate accuracy
    conf_mat_ord = confusion_matrix_ordered(labels,true_clusters)
    acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
    #
    return labels, ari, nmi, acc

# ########### Model Selection functions ############
# from spectral_clustering_mod import SpectralClustering_mod

from pcmf import sparse_D, get_weights
def cluster_fit(Xhat, num_clusters=2, gamma=2.0,  method='spectral'):
    # Cluster on the rows of Xhat with k=n_clust_previous
    if method == 'spectral':
        # clustering = SpectralClustering(n_clusters = num_clusters, gamma=gamma, \
                                        # assign_labels='discretize',affinity='rbf')
        # clustering = SpectralClustering_mod(n_clusters=num_clusters, random_state=0, gamma=gamma).fit(Xhat)
        clustering = SpectralClustering_mod(n_clusters = num_clusters, random_state=0, gamma=gamma, \
                                        assign_labels='discretize',affinity='rbf').fit(Xhat)
    elif method == 'kmeans':
        clustering = KMeans(n_clusters=num_clusters, random_state=0)
    elif method == 'gmm':
        clustering = GaussianMixture(n_components=num_clusters, warm_start=True)
    else:
        return ValueError('Selected clustering method not yet implemented.')
    clustering.fit(Xhat)
    #
    return clustering.labels_

def select_fit(n_clusts, ics, num_clusters):
    start_num = np.where(np.array(n_clusts)==num_clusters)[0]
    if len(start_num) > 1:
        start_num = start_num[0]
    vals = np.array(ics)[np.where(np.array(n_clusts)==num_clusters)]
    idxs = np.where(vals == np.min(vals))[0]
    if len(idxs) > 1:
        idxs = idxs[0]
    return start_num + idxs

def cluster_path(X_c, Xhat_list, penalty_list, gauss_coef, neighbors, r, n_clust_true=20,c_method='spectral', verbose=False, early_stopping=False):
    '''
    Estimate number of clusters and fit quality at each value of the penalty.
    '''
    ics = []
    n_clusts = []
    centroids = []
    n_clust = 1
    # !! Could be problematic for consensus to use D
    D = sparse_D(X_c.shape[0],X_c.shape[1])
    weights = get_weights(X_c, gauss_coef=gauss_coef, neighbors=neighbors)
    # !! 
    #
    notinf_idx = np.where(np.asarray(penalty_list)<np.inf)[0]
    penalty_list = penalty_list[notinf_idx]
    Xhat_list = [Xhat_list[i] for i in range(len(Xhat_list)) if i in notinf_idx]
    for i,Xhat in enumerate(Xhat_list):
        if early_stopping == True:
            if n_clust > n_clust_true + 1:
                break
        #
        penalty = penalty_list[i]
        if verbose:
            print("Run",r,", Penalty IDX:",i+1,"/",len(Xhat_list))
        #
        if n_clust < Xhat.shape[0]:
            out, n_clust, labels, ic = cluster_forwardstep(Xhat, X_c, D, n_clust, penalty, weights, r,
                                                            method=c_method, gamma=gauss_coef, selection='lik', verbose=verbose)
        n_clusts.append(n_clust)
        ics.append(ic)
        centroids.append(out)
    return n_clusts, ics, centroids

def centroid_matrix(X,labels):
    '''
    Given a matrix X and labels clustering each of it's rows, return a matrix 
    where each row has been replaced by it's nearest centroid.
    '''
    centroid_dict = dict()
    X_out = np.zeros_like(X)
    keys, counts = np.unique(labels, return_counts=True)
    for k in keys:
        centroid_dict[k] = list() # np.zeros(X.shape[1])
    for i,l in enumerate(labels):
        centroid_dict[l].append(X[i,:])
    for j,k in enumerate(keys):
        centroid_dict[k] = np.mean(np.asarray(centroid_dict[k]), axis=0)
    for i,l in enumerate(labels):
        X_out[i,:] = centroid_dict[l]
    return X_out

def cluster_forwardstep(Xhat, X, D, n_clust_previous, penalty, weights, r, method='spectral', gamma=1.0, selection='bic', verbose=False):
    epsilon = penalty
    #
    # Cluster on the rows of Xhat with k=n_clust_previous
    if method == 'spectral':
        clustering = SpectralClustering(n_clusters = n_clust_previous, gamma=gamma, \
                                        assign_labels='discretize',affinity='rbf',random_state=r)
    elif method == 'kmeans':
        clustering = KMeans(n_clusters=n_clust_previous, random_state=0)
    elif method == 'gmm':
        clustering = GaussianMixture(n_components=n_clust_previous, warm_start=True)
    else:
        return ValueError('Selected clustering method not yet implemented.')
    clustering.fit(Xhat)
    #
    labels1 = clustering.labels_
    Xhat1 = centroid_matrix(Xhat, labels1)
    df1 = n_clust_previous + 2
    #df1 = convex_clust_df(Xhat1, penalty, epsilon = epsilon)  
    if np.isnan(df1):
        df1 = 1
    #print('DF1:',df1, n_clust_previous)
    #
    # Cluster on the rows of Xhat with k=n_clust_previous+1 
    if method == 'spectral':
        clustering = SpectralClustering(n_clusters = n_clust_previous+1, gamma=gamma, \
                                        assign_labels='discretize',affinity='rbf',random_state=r)
        # clustering = SpectralClustering(n_clusters = n_clust_previous+1, gamma=gamma, \
        #                                 assign_labels='discretize',affinity='rbf') 
                        #affinity='nearest_neighbors', neighbors=neighbors, n_jobs=-1 # set to # cores
        
    elif method == 'kmeans':
        clustering = KMeans(n_clusters=n_clust_previous+1, random_state=0)
    elif method == 'gmm':
        clustering = GaussianMixture(n_components=n_clust_previous+1, warm_start=True)
    else:
        return ValueError('Selected clustering method not yet implemented.')
    clustering.fit(Xhat)
    #
    labels2 = clustering.labels_
    Xhat2 = centroid_matrix(Xhat, labels2)
    df2 = n_clust_previous + 1 + 2
    #df2 = convex_clust_df(Xhat2, penalty, epsilon = epsilon)      
    if np.isnan(df2):
        df2 = 2
    #print('DF2:',df2,n_clust_previous+1)
    #
    # Check loss for both clusterings
    # !! Could be problematic for consensus to use D
    loglik1 = np.linalg.norm(X - Xhat1, 2)**2 + penalty*np.sum(weights*np.sum(np.abs(D*Xhat1),axis=1))
    loglik2 = np.linalg.norm(X - Xhat2, 2)**2 + penalty*np.sum(weights*np.sum(np.abs(D*Xhat2),axis=1))
    # !!
    #
    if selection == 'aic':
        ic1 = loglik1 + 2*df1 
        ic2 = loglik2 + 2*df2 
    elif selection == 'bic':
        ic1 = loglik1 + df1*np.log(Xhat.shape[0])
        ic2 = loglik2 + df2*np.log(Xhat.shape[0])
    else:
        ic1 = loglik1
        ic2 = loglik2
    #
    if verbose:
        print("IC1, IC2, Penalty, Penalty Term:",ic1,ic2,penalty,np.sum(weights*np.sum(np.abs(D*Xhat2),axis=1)))
    if np.isnan(ic1):
        ic1 = np.inf
    #
    # Return the better clustering 
    if ic1 >= ic2:
        if verbose:
            print('Num clusters:', n_clust_previous+1)
        return centroid_matrix(Xhat,labels2), n_clust_previous+1, labels2, ic2
    else:
        if verbose:
            print('Num clusters:', n_clust_previous)
        return centroid_matrix(Xhat,labels1), n_clust_previous, labels1, ic1

####
def fit_clusterpath(X, Xhat_list, true_clusters, n_clusters_true, gauss_coef=2.0, n_replicates=1, c_method='spectral', early_stopping=False, verbose=False):
    '''Clusterpath clustering'''    #
    print('Fitting clusterpath')
    n_clusts_list = []
    ics_list = []
    centroids_list = []
    best_idx_list = []
    n_clusts_list_len = []
    for r in range(n_replicates):
        print('Replicate:',r)
        n_clusts, ics, centroids = cluster_path(X, Xhat_list, np.asarray(penalty_list), gauss_coef, neighbors, r, n_clust_true=n_clusters_true, c_method=c_method, early_stopping=early_stopping, verbose=verbose)
        n_clusts_list.append(n_clusts)
        ics_list.append(ics)
        centroids_list.append(centroids)
        try:
            best_idx = int(select_fit(n_clusts, ics, n_clusters_true))
        except:
            best_idx = np.nan
            print('Did not reach # true clusters.')
        best_idx_list.append(best_idx)
        n_clusts_list_len.append(len(n_clusts_list[r]))
    #
    if n_replicates > 1:
        if early_stopping==True:
            if len(np.unique(n_clusts_list_len)) > 1:
                max_len = np.max(n_clusts_list_len)
                for r in range(n_replicates):
                    print('r',r)
                    if n_clusts_list_len[r]<max_len:
                        end_val1 = n_clusts_list[r][n_clusts_list_len[r]-1]
                        end_val2 = ics_list[r][n_clusts_list_len[r]-1]
                        for i in range(max_len-n_clusts_list_len[r]):
                            print('i',i)
                            n_clusts_list[r].append(end_val1)
                            ics_list[r].append(end_val2)
        #
        try:
            n_clusts_median = np.round(np.median(np.asarray(n_clusts_list),axis=0))
            ics_median = np.median(np.asarray(ics_list),axis=0)
            best_idx = int(select_fit(n_clusts_median,ics_median,n_clusters_true))
        except:
            n_clusts_median = np.nan
            ics_median = np.nan
            best_idx = np.nan
            print('Failed to get ics and n_clusts median.')
        #
    else:
        n_clusts_median = np.nan
        ics_median = np.nan
        best_idx = best_idx_list[0]
    #
    try:
        labels = cluster_fit(Xhat_list[best_idx], num_clusters=n_clusters_true, gamma=gauss_coef, method=c_method)
        # Calculate scores
        nmi, ari, ri, mse = calculate_scores_nonpath(labels, true_clusters)
        #
        # Calculate accuracy
        conf_mat_ord = confusion_matrix_ordered(labels,true_clusters)
        acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
    except:
        labels = nmi =ari = ri = mse = acc= np.nan
        print('Failed to cluster.')
    #
    return labels, ari, nmi, acc, best_idx, n_clusts_list,ics_list,centroids_list,best_idx_list,n_clusts_median,ics_median

def load_experiments(data_path, pcmf_type, data_type, r=0):
    from pcmf import two_cluster_data
    results = np.load(data_path, allow_pickle=True)
    gauss_coef = float(results['gauss_coef'])
    neighbors = int(results['neighbors'])
    penalty_list = list(results['penalty_list'])
    if data_type == 'numerical':
        cluster_means = results['means']
        sigma = results['sigma']
        seeds = results['seeds']
        scale_data = results['scale_data'] # True
        intercept = results['intercept']
        cluster_sizes = results['cluster_sizes'] # True
        cluster_means = results['cluster_means']
        num_vars = results['num_vars']
        density = results['density']
        X, true_clusters = two_cluster_data(m=cluster_sizes, means=cluster_means, n_X=num_vars, sigma=sigma, density=density, gen_seeds=False, seeds=seeds, plot=False, scale_data=scale_data, intercept=intercept)
    else:
        X = results['X_c']
        true_clusters = results['true_clusters']
     #
    # if dataset=='_SRBCT_genomics':
    #     idx_keep = ~np.isnan(true_clusters)
    #     true_clusters = true_clusters[idx_keep]
    if pcmf_type == 'pcmf_full':
        A = results['A']
        Xhat_list = A
    elif pcmf_type == 'pcmf_approx_uV':
        U = results['U']
        V = results['V']
        Xhat_list = []
        #
        for i in range(len(penalty_list)):
            Xhat_list.append(U[i] * V[i])
    elif pcmf_type == 'pcmf_full_consensus':
        A = results['A']
        Xhat_list = A
    else:
        print('Incorrect pcmf_type')
    #
    return X, Xhat_list, true_clusters, results



# #### Fixed to not have infinite loop on Github; also lowering max SVD restarts to 5#
# def discretize(#
#     vectors, *, copy=True, max_svd_restarts=5, n_iter_max=20, random_state=None#
# ):#
#     """Search for a partition matrix which is closest to the eigenvector embedding.#
#     This implementation was proposed in [1]_.#
#     Parameters#
#     ----------#
#     vectors : array-like of shape (n_samples, n_clusters)#
#         The embedding space of the samples.#
#     copy : bool, default=True#
#         Whether to copy vectors, or perform in-place normalization.#
#     max_svd_restarts : int, default=30#
#         Maximum number of attempts to restart SVD if convergence fails#
#     n_iter_max : int, default=30#
#         Maximum number of iterations to attempt in rotation and partition#
#         matrix search if machine precision convergence is not reached#
#     random_state : int, RandomState instance, default=None#
#         Determines random number generation for rotation matrix initialization.#
#         Use an int to make the randomness deterministic.#
#         See :term:`Glossary <random_state>`.#
#     Returns#
#     -------#
#     labels : array of integers, shape: n_samples#
#         The labels of the clusters.#
#     References#
#     ----------#
#     .. [1] `Multiclass spectral clustering, 2003#
#            Stella X. Yu, Jianbo Shi#
#            <https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf>`_#
#     Notes#
#     -----#
#     The eigenvector embedding is used to iteratively search for the#
#     closest discrete partition.  First, the eigenvector embedding is#
#     normalized to the space of partition matrices. An optimal discrete#
#     partition matrix closest to this normalized embedding multiplied by#
#     an initial rotation is calculated.  Fixing this discrete partition#
#     matrix, an optimal rotation matrix is calculated.  These two#
#     calculations are performed until convergence.  The discrete partition#
#     matrix is returned as the clustering solution.  Used in spectral#
#     clustering, this method tends to be faster and more robust to random#
#     initialization than k-means.#
#     """#
# #
#     from scipy.sparse import csc_matrix#
#     from scipy.linalg import LinAlgError#
# #
#     random_state = check_random_state(random_state)#
# #
#     vectors = as_float_array(vectors, copy=copy)#
# #
#     eps = np.finfo(float).eps#
#     n_samples, n_components = vectors.shape#
# #
#     # Normalize the eigenvectors to an equal length of a vector of ones.#
#     # Reorient the eigenvectors to point in the negative direction with respect#
#     # to the first element.  This may have to do with constraining the#
#     # eigenvectors to lie in a specific quadrant to make the discretization#
#     # search easier.#
#     norm_ones = np.sqrt(n_samples)#
#     for i in range(vectors.shape[1]):#
#         vectors[:, i] = (vectors[:, i] / np.linalg.norm(vectors[:, i])) * norm_ones#
#         if vectors[0, i] != 0:#
#             vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])#
# #
#     # Normalize the rows of the eigenvectors.  Samples should lie on the unit#
#     # hypersphere centered at the origin.  This transforms the samples in the#
#     # embedding space to the space of partition matrices.#
#     vectors = vectors / np.sqrt((vectors ** 2).sum(axis=1))[:, np.newaxis]#
# #
#     svd_restarts = 0#
#     has_converged = False#
# #
#     # If there is an exception we try to randomize and rerun SVD again#
#     # do this max_svd_restarts times.#
#     while (svd_restarts < max_svd_restarts) and not has_converged:#
# #
#         # Initialize first column of rotation matrix with a row of the#
#         # eigenvectors#
#         rotation = np.zeros((n_components, n_components))#
#         rotation[:, 0] = vectors[random_state.randint(n_samples), :].T#
# #
#         # To initialize the rest of the rotation matrix, find the rows#
#         # of the eigenvectors that are as orthogonal to each other as#
#         # possible#
#         c = np.zeros(n_samples)#
#         for j in range(1, n_components):#
#             # Accumulate c to ensure row is as orthogonal as possible to#
#             # previous picks as well as current one#
#             c += np.abs(np.dot(vectors, rotation[:, j - 1]))#
#             rotation[:, j] = vectors[c.argmin(), :].T#
# #
#         last_objective_value = 0.0#
#         n_iter = 0#
# #
#         while not has_converged:#
#             n_iter += 1#
# #
#             t_discrete = np.dot(vectors, rotation)#
# #
#             labels = t_discrete.argmax(axis=1)#
#             vectors_discrete = csc_matrix(#
#                 (np.ones(len(labels)), (np.arange(0, n_samples), labels)),#
#                 shape=(n_samples, n_components),#
#             )#
# #
#             t_svd = vectors_discrete.T * vectors#
# #
#             try:#
#                 U, S, Vh = np.linalg.svd(t_svd)#
#             except LinAlgError:#
#                 svd_restarts += 1#
#                 print("SVD did not converge, randomizing and trying again")#
#                 break#
# #
#             ncut_value = 2.0 * (n_samples - S.sum())#
#             if (abs(ncut_value - last_objective_value) < eps) or (n_iter > n_iter_max):#
#                 has_converged = True#
#             else:#
#                 # otherwise calculate rotation and continue#
#                 last_objective_value = ncut_value#
#                 rotation = np.dot(Vh.T, U.T)#
# #
#     if not has_converged:#
#         raise LinAlgError("SVD did not converge")#
#     return labels