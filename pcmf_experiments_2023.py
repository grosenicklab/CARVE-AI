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

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI_score, adjusted_rand_score as ARI_score, rand_score as rand_score
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score, mean_squared_error # cohen_kappa_score, hinge_loss, coverage_error, consensus_score


from pcmf_dataloaders import load_humanATAC, load_mouseLGNrnaseq, load_macaqueLGNrnaseq, load_mouseOrganRNAseq
from pcmf_dataloaders import load_NCI, load_SRBCT, load_mouseorgans, load_gbmBreastLung
from pcmf_dataloaders import load_COVID, load_penguins, load_NCI_MV, load_SRBCT_MV, load_mouseorgans_MV, load_gbmBreastLung_MV



# from rpy2.robjects.packages import importr
# import rpy2.robjects as robjects
# import rpy2.rinterface as rinterface
# import rpy2.robjects.numpy2ri
# base = importr('base')
# conflicted = importr('conflicted')
# VarSelLCM = importr('VarSelLCM')
# HDclassif = importr('HDclassif')
# nethet = importr('nethet')
# clustRviz = importr('clustRviz')
# HDLSSkST = importr('HDLSSkST')

# Functions
def run_fitPCMF_experiment_consensusICML2023(savedir='/Users/amandabuch/Documents/clusterCCA/PCMF/results_ICML2023/',dataset='MNIST', problem_rank=100, rho=2.0, gc=1.0, neighbors=40, admm_iters=2, penalty_list = np.concatenate((np.repeat(np.inf,5),np.exp(np.linspace(-50,10,10))[::-1]),axis=0), skip=1, labels_keep=[0,1,2,3,4,5], data_path='/Users/amandabuch/Documents/clusterCCA/revision1/clusterCCA/data/',randomize=False):
    from pcmf import pcmf_full_consensus_2023 as pcmf_consensus
    import time
    from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
    # LOAD DATA
    batch_size = 50
    if dataset == 'MNIST':
        save_path=savedir+'MNIST_'+'skip'+str(skip)+'_nclusters'+str(len(labels_keep))+'_batchSize_'+str(batch_size)+'_pathLength'+str(len(penalty_list))+'problem_rank'+str(problem_rank)+'_gausscoef'+str(gc)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'_randomize'+str(randomize)
        X_in, true_clusters_in, X_in_test, true_clusters_in_test, true_clusters_labels, num_clusters = load_MNIST(labels_keep=labels_keep, plot=False, skip=skip, batch_size=batch_size, randomize=randomize)
        u_true = []
        v_true = []
    elif dataset == 'FashionMNIST':
        save_path=savedir+'FashionMNIST_'+'skip'+str(skip)+'_nclusters'+str(len(labels_keep))+'_batchSize_'+str(batch_size)+'_pathLength'+str(len(penalty_list))+'problem_rank'+str(problem_rank)+'_gausscoef'+str(gc)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'_randomize'+str(randomize)
        X_in, true_clusters_in, X_in_test, true_clusters_in_test, true_clusters_labels, num_clusters = load_FashionMNIST(labels_keep=labels_keep, plot=False, skip=skip, batch_size=batch_size, randomize=randomize)
        u_true = []
        v_true = []
    elif dataset == 'GBMBreastLung':
        save_path=savedir+'GBMBreastLung_'+'skip'+str(skip)+'_batchSize_'+str(batch_size)+'_pathLength'+str(len(penalty_list))+'problem_rank'+str(problem_rank)+'_gausscoef'+str(gc)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'_randomize'+str(randomize)
        X_in, true_clusters_in, X_in_test, true_clusters_in_test, true_clusters_labels, num_clusters, data  = load_GBMBreastLung(data_path=data_path, plot=False, skip=skip, batch_size=batch_size, randomize=randomize)
        u_true = []
        v_true = []
    elif dataset == 'macaqueLGNrnaseq':
        save_path=savedir+'macaqueLGNrnaseq'+'skip'+str(skip)+'_batchSize_'+str(batch_size)+'_pathLength'+str(len(penalty_list))+'problem_rank'+str(problem_rank)+'_gausscoef'+str(gc)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'_randomize'+str(randomize)
        X_in, true_clusters_in, X_in_test, true_clusters_in_test, true_clusters_labels, num_clusters, data  = load_macaqueLGNrnaseq(data_path=data_path, plot=False, skip=skip, batch_size=batch_size, randomize=randomize)
        u_true = []
        v_true = []
    elif dataset == 'mouseLGNrnaseq':
        save_path=savedir+'mouseLGNrnaseq'+'skip'+str(skip)+'_batchSize_'+str(batch_size)+'_pathLength'+str(len(penalty_list))+'problem_rank'+str(problem_rank)+'_gausscoef'+str(gc)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'_randomize'+str(randomize)
        X_in, true_clusters_in, X_in_test, true_clusters_in_test, true_clusters_labels, num_clusters, data  = load_mouseLGNrnaseq(data_path=data_path, plot=False, skip=skip, batch_size=batch_size, randomize=randomize)
        u_true = []
        v_true = []
    elif dataset == 'humanATAC':
        save_path=savedir+'humanATAC'+'skip'+str(skip)+'_batchSize_'+str(batch_size)+'_pathLength'+str(len(penalty_list))+'problem_rank'+str(problem_rank)+'_gausscoef'+str(gc)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'_randomize'+str(randomize)
        X_in, true_clusters_in, X_in_test, true_clusters_in_test, true_clusters_labels, num_clusters, data  = load_humanATAC(data_path=data_path, plot=False, skip=skip, batch_size=batch_size, randomize=randomize)
        u_true = []
        v_true = []  
    elif dataset == 'mouseOrganRNAseq':
        save_path=savedir+'mouseOrganRNAseq'+'skip'+str(skip)+'_batchSize_'+str(batch_size)+'_pathLength'+str(len(penalty_list))+'problem_rank'+str(problem_rank)+'_gausscoef'+str(gc)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'_randomize'+str(randomize)
        X_in, true_clusters_in, X_in_test, true_clusters_in_test, true_clusters_labels, num_clusters, data  = load_mouseOrganRNAseq(data_path=data_path, plot=False, skip=skip, batch_size=batch_size, randomize=randomize)
        u_true = []
        v_true = []  
    elif dataset == 'Synthetic':
        save_path=savedir+'Synthetic100000_'+'_batchSize_'+str(batch_size)+'_pathLength'+str(len(penalty_list))+'problem_rank'+str(problem_rank)+'_gausscoef'+str(gc)+'_neighbors'+str(neighbors)+'_admm_iters'+str(admm_iters)+'_rho'+str(rho)+'_randomize'+str(randomize)
        X_in, true_clusters_in, X_in_test, true_clusters_in_test, num_clusters, u_true, v_true = load_syntheticDataConsensus(n=100000, n_test=10000, p=1000, m=25000, m_test=2500, plot=False, randomize=randomize)
    else:
        print('Dataset not implemented:',dataset)
        return
    
    print('Problem rank',problem_rank)
    print('Save path is:',save_path+'.npz')

    # STANDARDIZE DATA
    scaler = StandardScaler(with_mean=True,with_std=False).fit(X_in)
    X_in = scaler.transform(X_in)
    normalizer = Normalizer().fit(X_in) # normalize columns
    X_in = normalizer.transform(X_in)
    scaler2 = StandardScaler(with_mean=False,with_std=True).fit(X_in)
    X_in = scaler2.transform(X_in)

    X_in_test = scaler.transform(X_in_test)
    X_in_test = normalizer.transform(X_in_test)
    X_in_test = scaler2.transform(X_in_test)

    # Fit PCMF
    tic=time.time()
    A_list, U_list, s_list, V_list = pcmf_consensus(X_in, penalty_list, split_size=batch_size, problem_rank=np.min((problem_rank,X_in.shape[1])), rho=rho, gauss_coef=gc, weights='Gaussian', admm_iters = admm_iters, neighbors=neighbors)
    toc=time.time() - tic
    
    A = A_list
    U = U_list
    S = s_list
    V = V_list

    X_compA = []
    X_compB = []
    UV = []
    USV = []
    for p in range(len(penalty_list)):
        x_pcmf = U[p]*S[p][:,0] # should the 0 be here??
        x_pcmf2 = (X_in[:,:] @ V[p].T) 
        X_compA.append(np.array(x_pcmf))
        X_compB.append(np.array(x_pcmf2))
        UV.append((np.array(U[p])[:,:]@np.array(V[p])[:,:]))
        USV.append((np.array(U[p])[:,:]*np.array(S[p]).flatten())@np.array(V[p])[:,:])

    X_compA=np.asarray(X_compA)
    X_compB=np.asarray(X_compB)
    
    # SAVE DATASET    
    np.savez(save_path+".npz",  A=A, U=U, S=S, V=V, UV=UV, USV=USV, X_compA=X_compA, X_compB=X_compB, X_in=X_in, true_clusters_in=true_clusters_in, X_in_test=X_in_test, true_clusters_in_test=true_clusters_in_test, true_clusters_labels=true_clusters_labels, num_clusters=num_clusters, u_true=u_true, v_true=v_true, split_size=batch_size, runtime=toc, penalty_list=np.asarray(penalty_list), rho=rho, admm_iters=admm_iters, gc=gc, neighbors=neighbors) 
#     generate_plots_synthetic(A, U, V, S, split_size, penalty_list, true_clusters_all, save_path+".pdf")
    del A, U, S, V, UV, USV, X_compA, X_compB, X_in, true_clusters_in, toc
    
    return save_path+".npz"


def get_clusters(true_clusters_in, X_in, neighbors, A, UV, USV, U, V, S):
    num_clusters = len(np.unique(true_clusters_in))
    n_X = X_in.shape[1]
    n_Y = X_in.shape[0]
    # NN = min(300,int(len(true_clusters_in[0:n_Y])/num_clusters))
    NN = neighbors
    for P in range(len(penalty_list)):
    #     kmeans = KMeans(n_clusters=3, random_state=0).fit(np.array(UV[P]))
        if n_Y < 4000:
    #         spectral_clustering = SpectralClustering(affinity='nearest_neighbors', n_neighbors=NN, n_clusters=num_clusters).fit(np.array(UV[P])) #U[P][:,0:1])) #UV[P]))
    #         spectral_clustering = SpectralClustering(affinity='nearest_neighbors', n_neighbors=int(len(true_clusters_in)/num_clusters), n_clusters=num_clusters).fit(np.array(USV[P]))
    #         spectral_clustering = SpectralClustering(affinity='nearest_neighbors', n_neighbors=int(len(true_clusters_in)/num_clusters), n_clusters=num_clusters).fit(np.array(USV[P]))
    #         spectral_clustering = SpectralClustering(affinity='nearest_neighbors', n_neighbors=int(len(true_clusters_in)/num_clusters), n_clusters=num_clusters).fit((np.array(U[P])[:,:]*np.array(S[P]).flatten()@np.array(V[P])[:,:]))
    #         spectral_clustering = SpectralClustering(affinity='nearest_neighbors', n_neighbors=int(len(true_clusters_in)/num_clusters), n_clusters=num_clusters).fit((np.array(U[P])[:,:]@np.array(V[P])[:,:]))
            spectral_clustering = SpectralClustering(affinity='nearest_neighbors', n_neighbors=NN, n_clusters=num_clusters).fit(np.array(A[P])[:,:])
    #         spectral_clustering = SpectralClustering(affinity='nearest_neighbors', n_neighbors=int(len(true_clusters_in)/num_clusters), n_clusters=num_clusters).fit(np.array(X_compB[P])[:,:])
            cs = spectral_clustering.labels_
        else:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(np.array(UV[P]))
            cs = kmeans.labels_
        import sklearn

        conf_mat_ord = confusion_matrix_ordered(cs,true_clusters_in[0:n_Y])
        acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
        print('Penalty idx',P,'ARI',sklearn.metrics.adjusted_rand_score(cs,true_clusters_in[0:n_Y]), 'ACC',acc)

    if n_Y < 2000:
        # Fit PCMF
        tic=time.time()
        spectral_clustering = SpectralClustering(affinity='nearest_neighbors', n_neighbors=NN, n_clusters=num_clusters).fit(X_in[:,0:n_X])
        toc=time.time() - tic
        cs = spectral_clustering.labels_
        conf_mat_ord = confusion_matrix_ordered(cs,true_clusters_in[0:n_Y])
        acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
        print('X_c spectral','ARI',sklearn.metrics.adjusted_rand_score(cs,true_clusters_in), 'ACC',acc, 'TOC',toc)

    from sklearn.decomposition import PCA
    tic=time.time()
    pca = PCA(n_components=U[0].shape[1])
    pca_in = pca.fit_transform(X_in[:,0:n_X])
    if n_Y < 2:
        spectral_clustering = SpectralClustering(affinity='nearest_neighbors', n_neighbors=NN, n_clusters=num_clusters).fit(pca_in)
        toc=time.time() - tic
        cs = spectral_clustering.labels_
        conf_mat_ord = confusion_matrix_ordered(cs,true_clusters_in[0:n_Y])
        acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
        print('X_c PCA1 spectral','ARI',sklearn.metrics.adjusted_rand_score(cs,true_clusters_in), acc, toc)
    else:
        kmeans = KMeans(n_clusters=3, random_state=0).fit(pca_in)
        toc=time.time() - tic
        cs = kmeans.labels_
        conf_mat_ord = confusion_matrix_ordered(cs,true_clusters_in[0:n_Y])
        acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
        print('X_c PCA1 kmeans','ARI',sklearn.metrics.adjusted_rand_score(cs,true_clusters_in), 'ACC',acc, 'TOC',toc)

    k = problem_rank
    tic=time.time()
    u,d,vh = np.linalg.svd(X_in[:,0:n_X], full_matrices=False)
    pca_in = ((u[:,0:k]*d[0:k]) @ vh[0:k,:])
    pca_in = (u[:,0:k] @ vh[0:k,:])
    if n_Y < 2:
        spectral_clustering = SpectralClustering(affinity='nearest_neighbors', n_neighbors=NN, n_clusters=num_clusters).fit(pca_in)
        toc=time.time() - tic
        cs = spectral_clustering.labels_
        conf_mat_ord = confusion_matrix_ordered(cs,true_clusters_in[0:n_Y])
        acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
        print('X_c PCA2 spectral','ARI',sklearn.metrics.adjusted_rand_score(cs,true_clusters_in), 'ACC',acc, 'TOC', toc)
    else:
        kmeans = KMeans(n_clusters=3, random_state=0).fit(pca_in)
        toc=time.time() - tic
        cs = kmeans.labels_
        conf_mat_ord = confusion_matrix_ordered(cs,true_clusters_in[0:n_Y])
        acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
        print('X_c PCA2 kmeans','ARI',sklearn.metrics.adjusted_rand_score(cs,true_clusters_in), 'ACC',acc, 'TOC',toc)

    kmeans = KMeans(n_clusters=3, random_state=0).fit(X_in[:,0:n_X])
    cs = kmeans.labels_
    conf_mat_ord = confusion_matrix_ordered(cs,true_clusters_in[0:n_Y])
    acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
    print('X_c kmeans','ARI',sklearn.metrics.adjusted_rand_score(cs,true_clusters_in), 'ACC',acc, 'TOC',toc)

    if n_Y < 2000:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import kneighbors_graph
        tic=time.time()
        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(X_in[:,0:n_X], n_neighbors=NN, include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)
        clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward', connectivity=connectivity).fit(X_in[:,0:n_X])
        toc=time.time() - tic
        cs = clustering.labels_
        acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
        conf_mat_ord = confusion_matrix_ordered(cs,true_clusters_in[0:n_Y])
        print('X_c Ward','ARI',sklearn.metrics.adjusted_rand_score(cs,true_clusters_in),'ACC',acc, 'TOC',toc)



from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, KMeans, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linprog, linear_sum_assignment as linear_assignment


def calculate_scores(pred_clusters, true_clusters):
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
    #
    return nmi_score, adj_rand_score, ri_score, mse_score


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

def fit_cca_kmeans(X, Y, true_clusters, n_clusters):
    '''CCA + Kmeans clustering'''
    from sklearn.cross_decomposition import CCA
    from sklearn.cluster import KMeans
    # Fit CCA-kmeans
    cca = CCA(n_components=1)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(np.hstack((X_c,Y_c)))
    labels = kmeans.labels_
    #
    # Calculate scores
    nmi, ari, ri, mse = calculate_scores_nonpath(labels,true_clusters)
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

# def fit_spectral(X, true_clusters, n_clusters):
#     '''Spectral clustering'''
#     from sklearn.cluster import SpectralClustering
#     #
#     data_in = X
#     #
#     spectral_clustering = SpectralClustering(n_clusters=n_clusters, random_state=20, affinity="nearest_neighbors").fit(data_in)
#     labels = spectral_clustering.labels_
#     #
#     # Calculate scores
#     nmi, ari, ri, mse = calculate_scores(labels, true_clusters)
#     # Calculate accuracy
#     conf_mat_ord = confusion_matrix_ordered(labels,true_clusters)
#     acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
#     #
#     return labels, ari, nmi, acc

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

def fit_leiden_or_louvain(X_in, true_clusters_in, ctype='Leiden', neighbors=5):
    ''' neighbors=5,10,15,20,25 '''
    import pandas as pd
    import scanpy as sc
    from anndata import AnnData
    dataset_x = X_in
    dataset_y = true_clusters_in
    adata = AnnData(dataset_x)

    if ctype=='Leiden':
        # Leiden algorithm
        sc.pp.pca(adata)
        sc.pp.neighbors(adata,n_neighbors=neighbors)
        sc.tl.leiden(adata)
        y_pred = pd.factorize(adata.obs['leiden'].tolist())[0]
        # Calculate accuracy
        conf_mat_ord = confusion_matrix_ordered(y_pred, dataset_y)
        acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
        print('NN 5 Leiden ACC',acc)

    elif ctype=='Louvain':
        sc.tl.louvain(adata)
        y_pred = pd.factorize(adata.obs['louvain'].tolist())[0]
        # Calculate accuracy
        conf_mat_ord = confusion_matrix_ordered(y_pred, dataset_y)
        acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
        print('NN 5 Louvain ACC',acc)

    else:
        print('ctype error '+str(ctype))
        return

    labels = y_pred
    true_clusters = dataset_y

    nmi, ari, ri, mse = calculate_scores_nonpath(labels, true_clusters)


    return labels, ari, nmi, acc

def fit_gMADD(X, true_clusters, n_clusters):
    '''gMADD clustering'''    #
    import rpy2.robjects as robjects                                                                                       
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

def fit_elasticsubspace(X, true_clusters, n_clusters,elasticsubpace_path='/Users/amandabuch/Documents/clusterCCA/revision1/clusterCCA/utils/subspace-clustering-master'):
    import os
    os.setwd(elasticsubpace_path)
    import progressbar2
    from cluster.selfrepresentation import ElasticNetSubspaceClustering
    cluster_method = 'Elastic Subspace'
    tic = time.time()
    labels, ari, nmi, acc = fit_elasticnetsubspace(X_in, true_clusters_in, num_clusters)
    toc = time.time() - tic
    print(cluster_method,acc,toc)

    return labels, ari, nmi, acc


def fit_hddc(X, true_clusters, n_clusters):
    '''HDDC clustering'''    #
    import rpy2.robjects as robjects
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
    import rpy2.robjects as robjects
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
    import rpy2.robjects as robjects
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
    import rpy2.robjects as robjects
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


def fit_pca_carp(X, true_clusters, n_clusters):
    '''CCA + Kmeans clustering'''
    from sklearn.cluster import KMeans
    # Fit PCA-kmeans
    u,d,vh = np.linalg.svd(X, full_matrices=False)
    Xhat = u[:,0].reshape((X.shape[0], 1))*vh[0,:].reshape((1, X.shape[1]))

    labels, ari, nmi, acc = fit_carp(Xhat, true_clusters, n_clusters)

    return labels, ari, nmi, acc

def fit_cca_carp(X, Y, true_clusters, n_clusters, n_components=1):
    '''CCA + Kmeans clustering'''
    from sklearn.cross_decomposition import CCA
    from sklearn.cluster import KMeans
    # Fit CCA-kmeans
    cca = CCA(n_components=n_components)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)

    labels, ari, nmi, acc = fit_carp(np.hstack((X_c,Y_c)), true_clusters, n_clusters)

    return labels, ari, nmi, acc


def fit_carp(X, true_clusters, n_clusters):
    '''CARP clustering'''    #
    import rpy2.robjects as robjects
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
    import rpy2.robjects as robjects
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


from matplotlib import pyplot as plt
def path_plot2(coefficient_arr, penalty_list, plot_range=[0,-1], cut_vars=False, 
              first_vars_only=False, var_sel=1, true_clusters=None,true_clusters_index=None, figsize=(20,10), xticks=None):
    import numpy as np
    import matplotlib
    from matplotlib import pyplot as plt
    import seaborn as sns; sns.set()

    # Crop x axis (e.g, to remove 'burn-in' period at beginning)                                                         
    coefficient_arr = coefficient_arr[plot_range[0]:plot_range[1],:,:]
    penalty_list = penalty_list[plot_range[0]:plot_range[1]]
    if cut_vars is True:
        coefficient_arr = coefficient_arr[:,:,[1,2,coefficient_arr.shape[2]-1]]

    if first_vars_only is True:
        coefficient_arr = coefficient_arr[:,:,[var_sel]]

    # Colormap                                                                                                           
    cmap = matplotlib.cm.get_cmap('tab20b')

    #cm.get_cmap('viridis', coefficient_arr.shape[2])
    print(true_clusters)
    if true_clusters is not None:
        colors = cmap(np.linspace(0.0,1.0,len(np.unique(true_clusters)) ))
        colors = ['darkblue','darkorange','red','green','purple','gray','pink']
        print(len(colors))
    else:
        colors = cmap(np.linspace(0.0,1.0,coefficient_arr.shape[1]))
        print(len(colors))
    
    # Define x-axis range                                                                                                
    penalty_range = range(len(penalty_list))

    # Make figure
    fig, ax = plt.subplots(1,1, figsize=figsize)

    for i in range(coefficient_arr.shape[2]):
        x = np.round(np.array(penalty_list),8)[penalty_range]
        y = coefficient_arr[penalty_range,:,i]
        if true_clusters is not None:
            # Make different line types for different clusters
            linetypes = ['dotted']*len(true_clusters)
            color_list = [colors[i]]*len(true_clusters)
#             for j, tc in enumerate(zip(true_clusters,pd.factorize(np.sort(true_clusters))[0])):
            for j, tc in enumerate(zip(true_clusters,true_clusters)):
                tcc = tc[1]
                tc = tc[0]
#                 print(tcc,tc, colors[tcc],true_clusters_index[tcc])
                if tc == 0:
                    linetypes[j] = 'dashed'
                    color_list[j] = colors[tcc]
                elif tc == 1:
                    linetypes[j] = 'solid'
                    color_list[j] = colors[tcc]
                elif tc == 2:
                    linetypes[j] = 'solid'
                    color_list[j] = colors[tcc]
                else:
                    linetypes[j] = 'dashed'
                    color_list[j] = colors[tcc]
#                 print(color_list)
            # Plot the lines and set linestyle
            ax.plot(np.arange(x.shape[0]), y, color=colors[i], alpha=0.5)
            for l, line in enumerate(ax.get_lines()):
                line.set_linestyle(linetypes[l])
                line.set_color(color_list[l])
        else:
            ax.plot(np.arange(x.shape[0]), y, color=colors[i], alpha=0.5)
        # Set plot ticks and labels
        ax.set_facecolor('white')
        ax.set_xticks(range(x.shape[0]), minor=False);
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        x = [str(xx)[0:9] for xx in x]
        ax.set_xticklabels(x) #,fontsize=24)
        evens = np.arange(0,len(ax.xaxis.get_ticklabels())+1,2)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False) 
    ax.tick_params(axis='y') #, labelsize=24)
#     print(color_list)
#     print(np.where(true_clusters==0)[0][0], ax.get_lines()[np.where(true_clusters==0)[0][0]].get_color(),
#           np.where(true_clusters==1)[0][0], ax.get_lines()[np.where(true_clusters==1)[0][0]].get_color(),
#           np.where(true_clusters==2)[0][0], ax.get_lines()[np.where(true_clusters==2)[0][0]].get_color() )
    
#     ax.legend([ax.get_lines()[np.where(true_clusters==0)[0][0]], ax.get_lines()[np.where(true_clusters==1)[0][0]], ax.get_lines()[np.where(true_clusters==2)[0][0]]], true_clusters_index)

    
    plt.xlabel(r'$\lambda$',fontsize=16)
    plt.ylabel('Coefficients',fontsize=16)
    sns.despine()
    
from matplotlib import pyplot as plt
def path_plot3(coefficient_arr, penalty_list, plot_range=[0,-1], cut_vars=False, colors = ["green","blue","pink"],
              first_vars_only=False, var_sel=1, true_clusters=None, true_clusters_index=None, figsize=(20,10), xticks=None):
    import numpy as np
    import matplotlib
    from matplotlib import pyplot as plt
    import seaborn as sns; sns.set()
    # Crop x axis (e.g, to remove 'burn-in' period at beginning)                                                         
    coefficient_arr = coefficient_arr[plot_range[0]:plot_range[1],:,:]
    penalty_list = penalty_list[plot_range[0]:plot_range[1]]
    if cut_vars is True:
        coefficient_arr = coefficient_arr[:,:,[1,2,coefficient_arr.shape[2]-1]]

    if first_vars_only is True:
        coefficient_arr = coefficient_arr[:,:,[var_sel]]

    # Colormap                                                                                                           
    cmap = matplotlib.cm.get_cmap('tab20b')

    #cm.get_cmap('viridis', coefficient_arr.shape[2])
    if true_clusters is not None:
#         colors = cmap(np.linspace(0.0,1.0,len(np.unique(true_clusters)) ))
        colors = ['darkblue','darkorange','red','green','purple','gray','pink']

    else:
        colors = cmap(np.linspace(0.0,1.0,coefficient_arr.shape[1]))
    
    # Define x-axis range                                                                                                
    penalty_range = range(len(penalty_list))
    with plt.style.context('default'):
        # Make figure
        fig, ax = plt.subplots(1,1, figsize=figsize)

        for i in range(coefficient_arr.shape[2]):
            x = np.round(np.array(penalty_list),5)[penalty_range]
            y = coefficient_arr[penalty_range,:,i]
            if true_clusters is not None:
                # Make different line types for different clusters
                linetypes = ['dotted']*len(true_clusters)
                color_list = [colors[i]]*len(true_clusters)
                print(len(colors),len(np.unique(true_clusters)))
                for j, tc in enumerate(true_clusters):
                    if tc == 1:
                        linetypes[j] = 'solid'
                        color_list[j] = colors[tc]
                    elif tc == 2:
                        linetypes[j] = 'dashed'
                        color_list[j] = colors[tc]
                    elif tc == 0:
                        linetypes[j] = 'dotted'
                        color_list[j] = colors[tc]
                    else:
                        linetypes[j] = 'dashdot'
                        color_list[j] = colors[tc]

                # Plot the lines and set linestyle
                ax.plot(np.arange(x.shape[0]), y, color=colors[i], alpha=0.5)
                for l, line in enumerate(ax.get_lines()):
                    line.set_linestyle(linetypes[l])
                    line.set_color(color_list[l])
            else:
                ax.plot(np.arange(x.shape[0]), y, color=colors[i], alpha=0.5)
            # Set plot ticks and labels
            ax.set_facecolor('white')
            ax.set_xticks(range(x.shape[0]), minor=False);
            plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
#             x = [str(xx)[0:5] for xx in x]
            x = [str('{:0.3e}'.format(xx)) for xx in x]
            ax.set_xticklabels(x) #,fontsize=24)
            evens = np.arange(0,len(ax.xaxis.get_ticklabels())+1,2)
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False) 
        ax.tick_params(axis='y') #, labelsize=24)
        plt.xlabel(r'$\lambda$',fontsize=16)
        plt.ylabel('Coefficients',fontsize=16)
        sns.despine()


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

def centroid_matrix_mixture(X,proba,means):
    '''
    Given a matrix X and labels clustering each of it's rows, return a matrix 
    where each row has been replaced by it's nearest centroid.
    '''
    centroid_dict = dict()
    X_out = np.zeros_like(X)
    for i in range(X_out.shape[0]):
        X_out[i,:] = np.dot(proba[i],means)
    return X_out

def cluster_match(pred, true):
    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)
    conf_mat = confusion_matrix(pred,true)
    indexes = linear_assignment(_make_cost_m(conf_mat))
    js = [e for e in sorted(indexes, key=lambda x: x[0])[1]]
    conf_mat_ord = conf_mat[:, js]
    pred_update = []
    for i in range(pred.shape[0]):
        pred_update.append(js[pred[i]])
        
    return np.array(pred_update)


from numba import jit, prange
from sklearn.neighbors import NearestNeighbors
from scipy import stats

@jit(nopython=False, parallel=True, fastmath=True)
def predict_NN(neighbors, X_in_test, X_in, true_clusters_in, true_clusters_in_test, true_clusters_test_predict):
    true_clusters_test_predict2 = []
    print(X_in_test.shape, true_clusters_in_test.shape, true_clusters_test_predict.shape, X_in.shape, true_clusters_in.shape)
    for i in prange(X_in_test.shape[0]):
        nbrs = NearestNeighbors(n_neighbors=int(neighbors), algorithm='ball_tree').fit(np.vstack((X_in_test[i,:],X_in)))
        _, indices = nbrs.kneighbors(np.vstack((X_in_test[i,:],X_in)))
        NN_train_inds = np.array(indices)[0,:][1:neighbors]-1
        mode, count = stats.mode(true_clusters_in[NN_train_inds])
        cl = int(mode)
        true_clusters_test_predict2.append(cl)
        print(i, cl, true_clusters_test_predict[i], true_clusters_in_test[i])
        
    return np.array(true_clusters_test_predict2)


from scipy.sparse import csr_matrix
from itertools import combinations
import networkx as nx
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def sparse_D(n,p):
    '''                                                                                                                  
    Construct a sparse matrix, that when applied to a vector containing concatenated vectors                             
    of coefficients b = [b_1 b_2 ... b_n] where each b_i is p=num_var long and there are                                 
    n = num_vec of them. Differences are taken between conformal elements (e.g. b_11 and b_21)                           
    across all unique pairwise combinations of vectors.                                                                  
    '''
    comb_list = list(combinations(range(n),2))
    combs_arr = np.array(comb_list)
    num_combs = combs_arr.shape[0]
    data = np.ones_like(combs_arr)
    data[:,1] *= -1
    row = np.repeat(range(num_combs),2)
    col = combs_arr.flatten()
    return csr_matrix((data.flatten(), (row, col)), shape=(num_combs, n)), comb_list

def diff_graph_cluster(Xhat, D, comb_list, num_clusters, thresh_sd=6, pca_clean=True, num_fits=1, verbose=False):
    '''
    Given a PCMF data approximation 'Xhat' for a fixed lambda and a differencing matrix 'D', calculate the 
    difference variable graph as suggested in Chi and Lange JCGC (2015), clustering on the graph adjacency 
    matrix (or a PCA embedding of it if pca_clean=True).
    
    Args:
        Xhat - PCMF data approximation at a fixed penalty parameter.
        D - a sparse differencing matrix given by 'sparse_D'.
        comb_list - the combination indices returned by 'sparse_D'.
        num_clusters - the number of clusters.
        thresh_sd - a threshold standard deviation cuttoff for thresholding the difference graph.
        pca_clean - boolean; should the PCA of the adjacency matrix be used for clustering. 
        num_fits - number of spectral clusterings to take the median of for output. 
        verbose - Print threshold adjustment; plot histogram graph edges and show GMM fit used to choose threshold.

    '''
    # Get graph edges from distances, and estimate graph threshold from edge mode centered around zero
    edges = np.sum(D*Xhat,axis=1)
    #edges = np.median(D*Xhat,axis=1)

    gmm = GaussianMixture(n_components = 6, max_iter=200, n_init=10).fit(edges.reshape(-1, 1))
    zero_mode_idx = np.where(np.abs(gmm.means_)==np.min(np.abs(gmm.means_)))[0]
    thresh = thresh_sd*np.sqrt(gmm.covariances_[zero_mode_idx])
    
    # Make adjacency from sum of differences (and adjust threshold if necessary) 
    flag = True
    while flag==True:
        # Generate graph
        G = nx.Graph()
        for i,e in enumerate(edges):
            if np.abs(e) < thresh:
                G.add_edge(comb_list[i][0], comb_list[i][1])
        A = nx.adjacency_matrix(G).toarray()
        if verbose:
            print('threshold:',thresh)
        if A.shape[0] < Xhat.shape[0]:
            thresh *= 1.1
        else:
            flag = False
            
    # Apply spectral clustering, taking median of 'num_fits' tryes to get output labels
    out_labels = []
    for f in range(num_fits):
        # Use PCA of A if 'pca_clean' flag set
        if pca_clean:
            uA, sA, vhA = np.linalg.svd(A)
            spectral_clustering = SpectralClustering(n_clusters=num_clusters, random_state=20, affinity="nearest_neighbors").fit(uA[:,0:num_clusters])
            out_labels.append(spectral_clustering.labels_)
        else:
            spectral_clustering = SpectralClustering(n_clusters=num_clusters, random_state=20, affinity="nearest_neighbors").fit(A)
            out_labels.append(spectral_clustering.labels_)
            
    # Plot thresholding histogram if 'plot_thresh_hist' is true
    if verbose:
        plt.figure(figsize=(10,5))
        _ = plt.hist(edges,bins=100,density=True)

        f_axis = edges.copy().ravel()
        f_axis.sort()
        a = []
        for weight, mean, covar in zip(gmm.weights_, gmm.means_, gmm.covariances_):
            a.append(weight*norm.pdf(f_axis, mean, np.sqrt(covar)).ravel())
            plt.plot(f_axis, a[-1])
        plt.plot(f_axis, np.array(a).sum(axis=0), 'k-')
        plt.xlabel('Variable')
        plt.ylabel('PDF')
        plt.tight_layout()
        plt.show()
        
    return np.median(np.array(out_labels),axis=0)



# tic = time.time()
# labels, ari, nmi, acc = fit_cca_kmeans(X, Y, true_clusters, len(np.unique(true_clusters)))
# toc = time.time() - tic
# print('CCA+Kmeans ACC',acc,'Time elapsed:',toc)

# tic = time.time()
# labels, ari, nmi, acc = fit_ward(X, true_clusters, len(np.unique(true_clusters)))
# toc = time.time() - tic
# print('Ward ACC',acc,'Time elapsed:',toc)

# tic = time.time()
# labels, ari, nmi, acc = fit_spectral(X, true_clusters, len(np.unique(true_clusters)))
# toc = time.time() - tic
# print('Spectral ACC',acc,'Time elapsed:',toc)

# tic = time.time()
# labels, ari, nmi, acc = fit_elasticnetsubspace(X, true_clusters, len(np.unique(true_clusters)))
# toc = time.time() - tic
# print('Elastic Subspace ACC',acc,'Time elapsed:',toc)

# tic = time.time()
# labels, ari, nmi, acc = fit_dpgmm(X, true_clusters, len(np.unique(true_clusters)))
# toc = time.time() - tic
# print('DPGMM ACC',acc,'Time elapsed:',toc)

# tic = time.time()
# labels, ari, nmi, acc = fit_gMADD(X, true_clusters, len(np.unique(true_clusters)))
# toc = time.time() - tic
# print('gMADD ACC',acc,'Time elapsed:',toc)

# tic = time.time()
# labels, ari, nmi, acc = fit_hddc(X, true_clusters, len(np.unique(true_clusters)))
# toc = time.time() - tic
# print('HDDC ACC',acc,'Time elapsed:',toc)

# tic = time.time()
# labels, ari, nmi, acc = fit_carp(X, true_clusters, len(np.unique(true_clusters)))
# toc = time.time() - tic
# print('CARP ACC',acc,'Time elapsed:',toc)


def fit_cluster_comparisons(X, Y=None, dtype='PCA', run_nondeep=True, run_deep=False, dataset_in=None, data_dir='/athena/listonlab/store/amb2022/PCMF'):
    ''' dtype='PCA' or 'CCA' '''
    import time
    smallenough_ward = X.shape[0] < 1000 # < N=XXX
    smallenough_spectral = X.shape[0] < 1000 # < N=XXX
    smallenough_elastic = X.shape[0] < 400 # < N=XXX
    smallenough_gmadd = X.shape[0] < 1000 # < N=XXX
    smallenough_hdcc = X.shape[0] < 400 # < N=XXX
    smallenough_dpgmm = X.shape[0] < 1000 # < N=XXX
    smallenough_carp = X.shape[0] < 400 # < N=XXX

    clustering_names = []
    labels_all = []
    ari_all = []
    nmi_all = []
    acc_all = []
    toc_all = []

    if run_nondeep is True:
        if dtype=='PCA':
            try:
                tic = time.time()
                labels, ari, nmi, acc = fit_pca_kmeans(X, true_clusters, n_clusters) # append labels, ari, nmi, acc and string of name to dataframe...
                toc = time.time() - tic
                clustering_names.append('PCA + K-means')
                labels_all.append(labels)
                ari_all.append(ari)
                nmi_all.append(nmi)
                acc_all.append(acc)
                toc_all.append(toc)
            except:
                print('Failed on PCA + K-means')
        if dtype=='CCA':
            try:
                tic = time.time()
                labels, ari, nmi, acc = fit_cca_kmeans(X, Y, true_clusters, n_clusters)
                toc = time.time() - tic
                clustering_names.append('CCA + K-means')
                labels_all.append(labels)
                ari_all.append(ari)
                nmi_all.append(nmi)
                acc_all.append(acc)
                toc_all.append(toc)
            except:
                print('Failed on CCA + K-means')
        ## If dataset is small enough
        if dataset is smallenough_ward:
            try:
                tic = time.time()
                labels, ari, nmi, acc = fit_ward(X, true_clusters, n_clusters)
                toc = time.time() - tic
                clustering_names.append('Ward')
                labels_all.append(labels)
                ari_all.append(ari)
                nmi_all.append(nmi)
                acc_all.append(acc)
                toc_all.append(toc)
            except:
                print('Failed on Ward')
        ## If dataset is small enough
        if dataset is smallenough_spectral:
            try:
                tic = time.time()
                labels, ari, nmi, acc = fit_spectral(X, true_clusters, n_clusters)
                toc = time.time() - tic
                clustering_names.append('Spectral')
                labels_all.append(labels)
                ari_all.append(ari)
                nmi_all.append(nmi)
                acc_all.append(acc)
                toc_all.append(toc)
            except:
                print('Failed on Spectral')
        ## If dataset is small enough
        if dataset is smallenough_elastic:
            try:
                tic = time.time()
                labels, ari, nmi, acc = fit_elasticnetsubspace(X, true_clusters, n_clusters) / fit_elasticsubspace(X, true_clusters, n_clusters)
                toc = time.time() - tic
                clustering_names.append('Elastic Subspace')
                labels_all.append(labels)
                ari_all.append(ari)
                nmi_all.append(nmi)
                acc_all.append(acc)
                toc_all.append(toc)
            except:
                print('Failed on Elastic Subspace')

        ## If dataset is small enough
        if dataset is smallenough_gmadd:
            try:
                tic = time.time()
                labels, ari, nmi, acc = fit_gMADD(X, true_clusters, n_clusters)
                toc = time.time() - tic
                clustering_names.append('gMADD')
                labels_all.append(labels)
                ari_all.append(ari)
                nmi_all.append(nmi)
                acc_all.append(acc)
                toc_all.append(toc)
            except:
                print('Failed on gMADD')

        ## If dataset is small enough
        if dataset is smallenough_hdcc:
            try:
                tic = time.time()
                labels, ari, nmi, acc = fit_hddc(X, true_clusters, n_clusters)
                toc = time.time() - tic
                clustering_names.append('HDDC')
                labels_all.append(labels)
                ari_all.append(ari)
                nmi_all.append(nmi)
                acc_all.append(acc)
                toc_all.append(toc)
            except:
                print('Failed on HDDC')

        for neighbors in np.arange(5,26,5):
            try:
                tic = time.time()
                labels, ari, nmi, acc = fit_leiden_or_louvain(X, true_clusters, ctype='Leiden', neighbors=neighbors)
                toc = time.time() - tic
                clustering_names.append('Leiden')
                labels_all.append(labels)
                ari_all.append(ari)
                nmi_all.append(nmi)
                acc_all.append(acc)
                toc_all.append(toc)
            except:
                print('Failed on Leiden')

            try:
                tic = time.time()
                labels, ari, nmi, acc = fit_leiden_or_louvain(X, true_clusters, ctype='Louvain', neighbors=neighbors)
                toc = time.time() - tic
                clustering_names.append('Louvain')
                labels_all.append(labels)
                ari_all.append(ari)
                nmi_all.append(nmi)
                acc_all.append(acc)
                toc_all.append(toc)
            except:
                print('Failed on Louvain')

        ## If dataset is small enough
        if dataset is smallenough_dpgmm:
            try:
                tic = time.time()
                labels, ari, nmi, acc = fit_dpgmm(X, true_clusters, n_clusters)
                toc = time.time() - tic
                clustering_names.append('DP-GMM')
                labels_all.append(labels)
                ari_all.append(ari)
                nmi_all.append(nmi)
                acc_all.append(acc)
                toc_all.append(toc)
            except:
                print('Failed on DP-GMM')
        ## If dataset is small enough
        if dataset is smallenough_carp:
            try:
                tic = time.time()
                labels, ari, nmi, acc = fit_carp(X, true_clusters, n_clusters)
                toc = time.time() - tic
                clustering_names.append('hCARP')
                labels_all.append(labels)
                ari_all.append(ari)
                nmi_all.append(nmi)
                acc_all.append(acc)
                toc_all.append(toc)
            except:
                print('Failed on hCARP')

            if dtype=='PCA':
                try:
                    tic = time.time()
                    labels, ari, nmi, acc = fit_pca_carp(X, true_clusters, n_clusters)
                    toc = time.time() - tic
                    clustering_names.append('PCA + hCARP')
                    labels_all.append(labels)
                    ari_all.append(ari)
                    nmi_all.append(nmi)
                    acc_all.append(acc)
                    toc_all.append(toc)
                except:
                    print('Failed on PCA + hCARP')

            elif dtype=='CCA':
                try:
                    tic = time.time()
                    labels, ari, nmi, acc = fit_cca_carp(X, Y, true_clusters, n_clusters)
                    toc = time.time() - tic
                    clustering_names.append('CCA + hCARP')
                    labels_all.append(labels)
                    ari_all.append(ari)
                    nmi_all.append(nmi)
                    acc_all.append(acc)
                    toc_all.append(toc)
                except:
                    print('Failed on CCA + hCARP')

    # if run_deep is True:
    #     if dtype=='PCA':
    #         X_in = X
    #     elif dtype=='CCA':
    #         X_in = np.hstack((X,Y))

    #     accuracies, toc, _ = fit_dec(X_in, true_clusters) # append best accuracy and string of name to dataframe...
    #     accuracies, toc, _ = fit_IDEC_deep_cluster(dataset_in=dataset_in, data_dir=data_dir) # append best accuracy and string of name to dataframe...
    #     accuracies, toc, _ = fit_cardec(X_in, true_clusters) # append best accuracy and string of name to dataframe...
    return [clustering_names, acc_all, ari_all, nmi_all, toc_all], labels_all



def plot_umap(X_in, true_clusters_in=None, n_neighbors=15):
    import pandas as pd
    import scanpy as sc
    from anndata import AnnData 
    dataset_x = X_in
    dataset_y = true_clusters_in
    adata = AnnData(dataset_x)
    adata.obs["labels"] = true_clusters_in
    # Add the fit umap...
    sc.pp.pca(adata)
    # sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors = n_neighbors)
    sc.tl.umap(adata)
    #
    if true_clusters_in is not None:
        sc.pl.umap(adata, color="labels")
    else:
        sc.pl.umap(adata)




