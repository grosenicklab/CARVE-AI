"""
File with pcmf - cca functions
To use: from pcmf_cca_functions import *
"""

###------- IMPORT FUNCTIONS -------###

import numpy as np
#import mosek
import cvxpy as cp
import matplotlib.pyplot as plt
import time

# %matplotlib inline

from itertools import combinations 

from sklearn import datasets, mixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI_score, adjusted_rand_score as ARI_score, rand_score as rand_score
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score, mean_squared_error # cohen_kappa_score, hinge_loss, coverage_error, consensus_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# %load_ext line_profiler

# to use first run 'python setup.py build_ext --inplace'
from admm_utils import prox as cprox

import scipy.spatial as sp
from scipy.spatial import procrustes
from scipy.optimize import linear_sum_assignment
from scipy.stats.stats import pearsonr

from kneed import DataGenerator, KneeLocator
import collections

# from cluster.selfrepresentation import ElasticNetSubspaceClustering

from numba import jit, prange
# from pcmf import prox_numba_arr



# from collections import defaultdict, OrderedDict

###------- GENERATE DATA FUNCTIONS -------###
def generate_cluster_cca_data(m=100, n_X=20, n_Y=10, sigma=5, density=0.2, coeff_factor=1, n_clusters=2, cluster_means=None, diff_means=False, means_factor=1,gen_seeds=True,seeds_clusters='NaN'):
    '''
    Generates synthetic data for CCA (of clusters of the same size currently)
    Params:
        m = number of observations in single cluster
        n_X = number of variables in X
        n_Y = number of variables in Y
        sigma = standard deviation of observation distributions (~noise level)
        density = Fraction of coefficients to set to 0 in CCA space
        coeff_factor = factor to shrink u_star and v_star
        n_clusters = number of clusters to generate
        diff_means = If true, each cluster has X and Y drawn from distributions with different means
        cluster_means = If input, means of length of num_clusters
        means_factor = Multiplying factor that increases spread between cluster means
    Returns:
        X_out = list of X data of length n_clusters
        Y_out = list of Y data of length n_clusters
        u_stars = list of true u coefficients of length n_clusters
        v_stars = list of true v coefficients of length n_clusters
        seeds = list of seeds used to generate cluster data
    '''
    X_out = []
    Y_out = []
    u_stars = []
    v_stars = []
    seeds_all=[]
    for nc in range(n_clusters):
        
        if diff_means == True:
            if cluster_means is not None:
                mean_X = np.random.randn(1)*means_factor
                mean_Y = np.random.randn(1)*means_factor
            else:
                mean_X = cluster_means[nc]
                mean_Y = cluster_means[nc]
        else:
            mean_X = 0
            mean_Y = 0
        if isinstance(m, int):
            cluster_sz = m
        else:
            cluster_sz = m[nc]

        X, Y, u_star, v_star,seeds = generate_cca_data(cluster_sz, n_X, n_Y, sigma, density, coeff_factor, mean_X=mean_X, mean_Y=mean_Y, gen_seeds=gen_seeds,seeds=seeds_clusters[nc])
        X_out.append(X)
        Y_out.append(Y)
        u_stars.append(u_star)
        v_stars.append(v_star)
        seeds_all.append(seeds)
    return X_out, Y_out, u_stars, v_stars, seeds_all
def generate_cca_data(m=100, n_X=20, n_Y=10, sigma=1, density=0.5, coeff_factor=1, mean_X=0, mean_Y=0, gen_seeds=True,seeds='NaN'):
    ''' Generates data matrix X and covariates Y.
    Params:
        m = number of observations in single cluster
        n_X = number of variables in X
        n_Y = number of variables in Y
        sigma = standard deviation of observation distributions (~noise level)
        density = Fraction of coefficients to set to 0 in CCA space
        coeff_factor = factor to shrink u_star and v_star
        mean_X = mean X distribution drawn from, default is 0 means
        mean_Y = mean Y distribution drawn from, default is 0 means
        seed = random seed
    Returns:
        X = np-array of shape (m,n_X)
        Y = np-array of shape (m,n_Y)
        y = true u coefficients of shape (n_X)
        v = true v coefficients of shape (n_Y)
    '''
    if gen_seeds is True:
        seeds = [np.random.randint(99999),np.random.randint(99999)]
        print('generating seeds')
    else:
        print('NOT generating seeds')

    np.random.seed(seeds[0])
    z = np.random.randn(m)
    # Construct X
    u_star = np.random.randn(n_X) / coeff_factor
    X = np.random.normal(mean_X,sigma,size=((m,n_X)))
    X_idxs = np.random.choice(range(n_X), int(density*n_X), replace=False)
    for idx in X_idxs:
        X[:,idx] += u_star[idx]*z
#
    np.random.seed(seeds[1])
    # Construct Y
    v_star = np.random.randn(n_Y) / coeff_factor
    Y = np.random.normal(mean_Y,sigma,size=((m,n_Y)))
    Y_idxs = np.random.choice(range(n_Y), int(density*n_Y), replace=False)
    for idx in Y_idxs:
        Y[:,idx] += v_star[idx]*z
#
    return X, Y, u_star, v_star, seeds

###------- PATHWISE ADMM CCA -------###
from scipy.sparse import csr_matrix
from sksparse.cholmod import cholesky, cholesky_AAt
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
    return csr_matrix((data.flatten(), (row, col)), shape=(num_combs, n))

def chol_D(num_obs, num_vars, rho):
    D = sparse_D(num_obs, num_vars)
    return D, cholesky_AAt(np.sqrt(rho) * D.T, beta=1.0)

def group_soft_threshold(vec,alpha):
    '''
    Soft group thresholding for ADMM
    Params:
        vec
        alpha
    Returns:
        np.asarray(V_prox)
    '''
    if np.linalg.norm(vec) > alpha:
        return vec - alpha*vec/np.linalg.norm(vec) 
    else:
        return np.zeros_like(vec)

def prox(V, lamb, rho, w=None):
    if w is None:
        w = np.ones(V.shape[0])
    V_prox = []
    for i in range(V.shape[0]):
        alpha = w[i]*lamb/rho
        V_prox.append(group_soft_threshold(V[i,:],alpha))
    return np.asarray(V_prox)  


@jit(nopython=True, parallel=True, fastmath=True)
def prox_numba_arr(V_prox, V, lamb, rho, w=None):
    '''
    Numba formulation of Proximal operator for group sparse thresholding.
    '''
#     print(V_prox)
#     print(w)
    if w is None:
        w = np.ones(V.shape[0])
    for i in prange(V.shape[0]):
        if np.isinf(lamb):
            alpha = lamb/rho
        else:
            alpha = w[i]*lamb/rho
#         print(alpha)
        # group_soft_threshold
        vec_norm = np.linalg.norm(V[i,:])
#         print(alpha)
        if vec_norm > alpha:
#             print(vec_norm > alpha)
            V_prox[i, :] = V[i,:] - alpha*V[i,:]/vec_norm
        else:
            V_prox[i, :] = np.zeros_like(V[i,:])
#     print(V_prox)
    return V_prox

def l2_ball_proj(X):
    '''
    Normalize rows of X to satisfy $\|X_{i,\cdot}\|_2^2 \leq 1$.
    '''
    def row_norm(vec): 
        return np.sqrt(1./np.sum(vec**2)) * vec
    return np.apply_along_axis(row_norm,1,X)

def get_weights(X, gauss_coef=0.5, neighbors=None):
    '''
    Weights for ADMM
    Params:
        X
        gauss_coef
        neighbors
    Returns:
        w = weights of shape ()
    '''
    from scipy.spatial.distance import pdist, squareform
    dist_vec = pdist(X)
    w = np.exp(-1*gauss_coef*dist_vec**2)
    if neighbors is not None:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm='ball_tree').fit(X)
        _, indices = nbrs.kneighbors(X)
        comb_list = list(combinations(range(X.shape[0]),2))
        neighbors_indicator = []
        for comb in comb_list:
            nn_i = indices[comb[0],:]
            if comb[1] in nn_i:
                neighbors_indicator.append(1.0)
            else:
                neighbors_indicator.append(0.0)
        w *= np.array(neighbors_indicator)
    return w

def admm_CCA_update(mat_tilde, chol_factor, D, W, Z, rho, penalty, data, weights=False, neighbors=None,gauss_coef=0.5, numba=True):
    '''
    Updates V, W, and Z for ADMM
    Params:
        mat_tilde
        chol_factor
        D
        W
        Z
        rho
        penalty
        data
        weights
        neighbors
    Returns:
        V
        W
        Z
    '''
    if weights is False:
        weights = get_weights(data,gauss_coef=0.0)
    else:
        weights = get_weights(data,gauss_coef=gauss_coef, neighbors=neighbors)
    V = chol_factor(mat_tilde + rho*D.T*(W - Z))
#     W = prox(D*V+Z, penalty, rho, weights)
    if numba==True:
        W = prox_numba_arr(np.zeros_like(W), D*V+Z, penalty, rho, weights)
    else:
        W = cprox(D*V+Z, penalty, rho, weights)
    Z = Z + D*V - W
    V = l2_ball_proj(V)

    return V, W, Z

def admm_CCA_new(X, Y, penalty_list, rho=1.0, admm_iters = 2, cca_iters = 3, 
                 weights=False, neighbors=None, gauss_coef=0.5, verb=True, non_negative=False, parallel=False, output_file='', numba=True):
    '''
    Main function to run ADMM-version of PCMF-CCA, runs along penalty path penalty_list
    Params:
        X
        Y
        penalty_list
        rho
        admm_iters = number of admm updates for each u and each v
        cca_iters = number of u / v updates
        weights
        neighbors
        verb = whether to print output of iteratino and penalty at each outer iter
    Returns:
    '''
    # Initialize U as X and V as Y
    Dx, chol_factor_X = chol_D(X.shape[0], X.shape[1], rho)
    Wx = Zx = Dx*X
    Dy, chol_factor_Y = chol_D(Y.shape[0], Y.shape[1], rho)
    Wy = Zy = Dy*Y
    
    U_list = []
    V_list = []
    penalty = penalty_list[0]
    
    # Initial U update
    if penalty_list[0] > penalty_list[-1]:
        V_initial = np.tile(np.mean(Y,axis=0),Y.shape[0]).reshape(Y.shape)
    else:
        V_initial = Y.copy()
    Xu_tildes = []
    for i in range(X.shape[0]):
        Xu_tildes.append(np.dot(np.outer(X[i,:].T,Y[i,:]),V_initial[i,:]))
    Xu = np.asarray(Xu_tildes)
    U, Wx, Zx = admm_CCA_update(Xu, chol_factor_X, Dx, Wx, Zx, rho, penalty, X, weights=weights, neighbors=neighbors, numba=numba, gauss_coef=gauss_coef)
    # U_list.append(U)

    # Initial V update
    if penalty_list[0] > penalty_list[-1]:
        U_initial = np.tile(np.mean(X,axis=0),X.shape[0]).reshape(X.shape)
    else:
        U_initial = X.copy()
    Yv_tildes = []
    for i in range(Y.shape[0]):
        Yv_tildes.append(np.dot(np.outer(Y[i,:].T,X[i,:]),U_initial[i,:]))
    Yv = np.asarray(Yv_tildes)
    V, Wy, Zy = admm_CCA_update(Yv, chol_factor_Y, Dy, Wy, Zy, rho, penalty, Y, weights=weights, neighbors=neighbors, numba=numba, gauss_coef=gauss_coef)
    # V_list.append(V)

#     U = U_initial
#     V = V_initial
    try:
        for i in range(len(penalty_list)):
            penalty = penalty_list[i]
            print_progress = verb
            if print_progress == True:
                # # print("[",i+1,"]",round(penalty,5), end="")
                # print("[",i+1,"]","{:.25e}".format(penalty), end="")
                # print("...", end="")
                if parallel == True:
                    file1 = open(output_file, "a")
                    file1.write(str([i+1])+" {:.5e}".format(penalty))
                    file1.write("...")
                    file1.close()
                else:
                    print("[",i+1,"]","{:.5e}".format(penalty), end="")
                    print("...", end="")

            for it in range(cca_iters):
                # U update
                for it in range(admm_iters):
                    Xu_tildes = []
                    for i in range(X.shape[0]):
                        
                        Xu_tildes.append(np.dot(np.outer(X[i,:].T,Y[i,:]),V[i,:]))
                    Xu = np.asarray(Xu_tildes)

                    U, Wx, Zx = admm_CCA_update(Xu, chol_factor_X, Dx, Wx, Zx, rho, penalty, X, weights=weights, neighbors=neighbors, gauss_coef=gauss_coef)
                    if non_negative:
                        U[U<0] = 0

                # V update
                for it in range(admm_iters):
                    Yv_tildes = []
                    for i in range(X.shape[0]):
                        Yv_tildes.append(np.dot(np.outer(Y[i,:].T,X[i,:]),U[i,:]))
                    Yv = np.asarray(Yv_tildes)

                    V, Wy, Zy = admm_CCA_update(Yv, chol_factor_Y, Dy, Wy, Zy, rho, penalty, Y, weights=weights, neighbors=neighbors, gauss_coef=gauss_coef)
                    if non_negative:
                        V[V<0] = 0

            U_list.append(U)
            V_list.append(V)

    except KeyboardInterrupt:
        print("KeyboardInterrupt has been caught.")

    return U_list, V_list

###------- CLUSTER METRICS -------###
"""
    To do:
        Test split half
        Test predict X / Y
        Update clustering path from kmeans to GMM method
        Add CCA + cluster
        Add other clustering methods
        Add table output
        Check that ARI, etc. as expected
        Add option to pool means for coefficients by cluster in CCA cluster solution

"""

class cluster_metrics:
    """
    Compute cluster accuracy metrics along penalty path
    
        To do:
        Add elbow for calculate_heuristics
        Add random for plotting calculate_heuristics
        Add canonical correlation / singular values estimate
    """
    
    def __init__(self, coefficient_list, penalty_list, true_clusters, c_type='spectral'):
        """
        Initializes cluster_metrics object
        Params:
            coefficient_list = U_list or V_list, the list of coefficients over all penalties, length should be same as penalty_list
            penalty_list = list of penalties, length should be same as coefficient_list
            true_clusters = list of true clusters, (*should be optional)
        Properties:
            penalty_list
            coefficients_path = np-array of coefficient_list
            true_clusters
            num_clusters = number of clusters
            path_length = number of penalties in path
            
        """
        self.penalty_list = penalty_list
        self.coefficients_path = np.array(coefficient_list)
        self.true_clusters = np.array(true_clusters)
        self.num_clusters = len(np.unique(true_clusters))
        self.path_length = len(penalty_list)
        self.c_type=c_type
#         self.num_obs = shape(coefficient_list[0],2)
    
    def __repr__(self): 
        """"
        Gives output of number of clusters and number of penalties when you print the cluster_metrics object
        """
        return "num_clusters: % s path_length: % s" % (self.num_clusters, self.path_length) 
    
    def cluster_along_path(self,i, c_type='kmeans'):
        """"
        Kmeans clustering on coefficients at one penalty
        Params:
            self.coefficients_path = np-array of coefficients-path of shape (path_length,num_obs,num_variables)
            i = index of penalty in self.coefficients_path
        Returns: Clustering solution at one penalty
        """
        # ! consider adding sign flip / Procrustes option when true coefficients are known

        if c_type=="dpgmm":
            coefficient_arr = self.coefficients_path
            try:
                dpgmm = mixture.BayesianGaussianMixture(n_components=self.num_clusters, covariance_type='full', weight_concentration_prior_type='dirichlet_process',max_iter=1000)
                dpgmm_mod = dpgmm.fit(coefficient_arr[i,:,:])
            except:
                print('DPGMM failed so running with reg_covar=0.1')
                dpgmm = mixture.BayesianGaussianMixture(n_components=self.num_clusters, covariance_type='full', weight_concentration_prior_type='dirichlet_process',max_iter=1000,reg_covar=0.1)
                dpgmm_mod = dpgmm.fit(coefficient_arr[i,:,:])

            cs = dpgmm.predict(coefficient_arr[i,:,:])
        elif c_type=="kmeans":
            try:
                # print('kmeans')
                kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(self.coefficients_path[i,:,:])
                cs = kmeans.labels_
            except:
                print('kmeans fail')
                cs = 'NaN'
        elif c_type=="spectral":
            try:
                spectral = SpectralClustering(n_clusters=self.num_clusters, random_state=0, gamma=2.0).fit(self.coefficients_path[i,:,:])
                cs = spectral.labels_
            except:
                print('spectral fail')
                cs = 'NaN'
        elif c_type=="ElasticSubspace":
            try:
                essc = ElasticNetSubspaceClustering(n_clusters=self.num_clusters,algorithm='spams',gamma=5,active_support=False,tau=0.1).fit(self.coefficients_path[i,:,:])
                cs = essc.labels_
            except:
                print('ElasticSubspace failed')
                cs = np.zeros_like(self.true_clusters)
        return cs
    
    def gap_statistic(self, coeffs, nrefs=3):
        """
        Calculates gap statistic at one penalty
        Params:
            coeffs = ndarry of shape (n_samples, n_features)
            num_clusters = number of clusters
            nrefs = number of sample reference datasets to create
        Returns: 
            gap = gap statistic at one penalty
        """
        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=coeffs.shape)
            
            # Fit to it
            km = KMeans(self.num_clusters)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(self.num_clusters)
        km.fit(coeffs)
        
        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        return gap

    def calculate_scores(self):
        """
        Calculate cluster accuracy scores and requires true clusters
        Calls self.cluster_along_path on each index i along penalties in path
        Params:
            self.true_clusters
            self.path_length
        Returns:
            self.path_clusters = list of clusterings at each penalty on path
            self.nmi_score = np-array of normalized mutual information score for each penalty
            self.adj_rand_score = np-array of adjusted rand score for each penalty
            self.rand_score = np-array of rand score for each penalty
            self.mse_score = np-array of mean squared error score for each penalty
        """
        nmi = []
        ari = []
        ri = []
        mse = []
        path_clusters = []
        for i in range(self.path_length):
            pred_clusters = self.cluster_along_path(i, c_type=self.c_type)
            path_clusters.append(pred_clusters)
            nmi.append(NMI_score(self.true_clusters, pred_clusters, average_method='arithmetic'))
            ari.append(ARI_score(self.true_clusters, pred_clusters))
            ri.append(rand_score(self.true_clusters, pred_clusters))
            mse.append(mean_squared_error(self.true_clusters, pred_clusters))
            
        self.path_clusters = path_clusters
        self.nmi_score = np.array(nmi)
        self.adj_rand_score = np.array(ari)
        self.rand_score = np.array(ri)
        self.mse_score = np.array(mse)
        
        return self.path_clusters, self.nmi_score, self.adj_rand_score, self.rand_score, self.mse_score
    
    def calculate_heuristics(self):
        """""
        Calculates cluster heuristics when truth is not known along lambda path
        Parameters:
            self.coefficients_path
        Returns:
            self.davies_bouldin = DB values along penalty path * generally higher for convex clusters
        """""
        db = []
        si = []
        ch = []
        gs = []
        path_clusters = []
        for i in range(self.path_length):
            coeffs = self.coefficients_path[i,:,:]
            
            if hasattr(self,'nmi_score'):
                path_clusters = self.path_clusters
                pred_clusters = path_clusters[i]
            else:
                pred_clusters = self.cluster_along_path(i, c_type=self.c_type)
                path_clusters.append(pred_clusters)
                print('path clusters')
            
            try:
                db.append(davies_bouldin_score(coeffs, pred_clusters))
                si.append(silhouette_score(coeffs, pred_clusters, metric='euclidean', sample_size=None, random_state=None))
                ch.append(calinski_harabasz_score(coeffs, pred_clusters))
                gs.append(self.gap_statistic(coeffs, nrefs=3))
            except:
                db.append(np.nan)
                si.append(np.nan)
                ch.append(np.nan)
                gs.append(np.nan)
                
        self.path_clusters = path_clusters
        self.db_score = np.array(db)
        self.sil_score = np.array(si)
        self.ch_score = np.array(ch)
        self.gap_score = np.array(gs)

        return self.db_score, self.sil_score, self.ch_score, self.gap_score
    
    def calculate_scores_random(self, num_reps):
        """"
        Calculates random cluster statistic scores for num_reps random shuffles of true clusters
        * Can add functionality to input pred_clusters as true_clusters
        * Need to add ch, db, sil, gap
        Params:
            object
            num_reps = number of times to generate random sampling of clusters
        Returns:
            self.nmi_score_rand = array of nmi scores for each rep and penalty, shape (num_reps, path_length)
            self.adj_rand_score_rand = array of adj rand index scores for each rep and penalty, shape (num_reps, path_length)
            self.rand_score_rand = array of rand index scores for each rep and penalty, shape (num_reps, path_length)
        """
        #! make cluster along path output to self
        nmi_rand = ari_rand = ri_rand = np.zeros((num_reps, self.path_length))
        seeds = []
        for rep in range(num_reps):
            seed = np.random.randint(99999)
            clusters_rand = shuffle(self.true_clusters)
            seeds.append(seed)
            for i in range(self.path_length):
                pred_clusters = self.cluster_along_path(i,c_type=self.c_type)
                nmi_rand[rep,i] = NMI_score(clusters_rand, pred_clusters, average_method='arithmetic')
                ari_rand[rep,i] = ARI_score(clusters_rand, pred_clusters)
                ri_rand[rep,i] = rand_score(clusters_rand, pred_clusters)
        self.nmi_score_rand = nmi_rand
        self.adj_rand_score_rand = ari_rand
        self.rand_score_rand = nmi_rand
        return self.nmi_score_rand, self.adj_rand_score_rand, self.rand_score_rand, seeds

    def get_optimal_heuristic(self):
        """
        """
        kneedle_ch = KneeLocator(range(len(self.penalty_list)), self.ch_score, online=True, curve="concave", direction="increasing")
        kneedle_sil = KneeLocator(range(len(self.penalty_list)), self.sil_score, online=True, curve="concave", direction="increasing")
        kneedle_gap = KneeLocator(range(len(self.penalty_list)), self.gap_score, online=True, curve="convex", direction="decreasing")
        kneedle_db = KneeLocator(range(len(self.penalty_list)), self.db_score, online=True, curve="convex", direction="decreasing")

        self.ch_optimalpts = list(kneedle_ch.all_knees)
        self.sil_optimalpts = list(kneedle_sil.all_knees)
        self.gap_optimalpts = list(kneedle_gap.all_knees)
        self.db_optimalpts = list(kneedle_db.all_knees)
        self.heuristics_optimalpts = self.ch_optimalpts+self.sil_optimalpts+self.gap_optimalpts+self.db_optimalpts
        
        optimal_dict = dict(collections.Counter(self.ch_optimalpts + self.sil_optimalpts + self.db_optimalpts + self.gap_optimalpts))
        optimal_dict_freqs = np.array(list(optimal_dict.values()))
        optimal_dict_keys = np.array(list(optimal_dict.keys()))
        freq_max = np.max(optimal_dict_freqs)
        if freq_max > 1:
            print("using pool")
            opt_idx = np.where(optimal_dict_freqs > 1 )
            opt_val = [optimal_dict_keys[i] for i in opt_idx]
            # print(optimal_dict_keys)
            # print(opt_idx)
        else:
            print("using gap")
            # opt_val = np.max(np.array(self.ch_optimalpts))
            opt_val = np.max(np.array(self.gap_optimalpts))

        print(opt_val)
        optimal_penalty_idx = np.max(opt_val)

        return optimal_penalty_idx


    def print_best_heuristics(self):
        """
        Print best cluster accuracy scores
        Params:
            self.sil_score
            self.db_score
            self.ch_score
            self.gap_score
        Returns:
            text output
        """
        return "Calinski harbasz % s : % s, Silhouette % s : % s, Davies bouldin % s : % s, Gap statistic % s : % s" % (np.nanargmax(self.ch_score), np.nanmax(self.ch_score), np.nanargmax(self.sil_score), np.nanmax(self.db_score), np.argmin(self.gap_score), np.nanmin(self.gap_score), np.argmin(self.db_score), np.nanmin(self.db_score))
    

    def print_best_scores(self):
        """
        Print best cluster accuracy scores
        Params:
            self.nmi_score
            self.adj_rand_score
            self.rand_score
        Returns:
            text output
        """
        return "NMI % s : % s, ARI % s : % s, RI % s : % s" % (np.nanargmax(self.nmi_score), np.nanmax(self.nmi_score),np.nanargmax(self.adj_rand_score), np.nanmax(self.adj_rand_score),np.nanargmax(self.rand_score), np.nanmax(self.rand_score)) 
    
    def flip_sign(self, obs_coeff_true, obs_coeff):
        """
        Calculates Pearson correlation between true and predict coefficents and flips sign if negative correlation.
        Can also input predicted coefficients as truth to preserve original signs.
        Params:
            obs_coeff_true = True coefficients at one penalty and one observation, shape (num_variables, 1)
            obs_coeff = Predicted coefficients at one penalty and one observation, shape (num_variables, 1)
        Returns:
            obs_coeff with corrected signs
        """
        r,_ = pearsonr(obs_coeff_true, obs_coeff)
        return(np.sign(r)*obs_coeff)
    
    def compare_coefficients(self, coeff_true):
        """
        Calculates cosine similary between true coefficients and predicted coefficients, with Procrustes rotation
        Params:
            coeff_true = list of true coefficients, shape = (num_clusters, num_variables)
            self.true_clusters
            self.coefficients_path
            self.path_length
        Returns:
            self.path_coeff_fit = predicted coefficients after Procrustes rotation
            self.path_coeff_diff = disparity between predicted coefficients before and after Procrustes
            self.path_cluster_sim = cosine similarity between predicted coefficients (after Procrustes) and true coefficients
            self.path_cluster_sim_orig = cosine similarity between predicted coefficients (before Procrustes) and true coefficients
        """
        coeff_true_all = []
        for i in range(len(self.true_clusters)):
            cluster_idx = self.true_clusters[i]
            coeff_true_all.append(coeff_true[cluster_idx]) 
        coeff_true_all = np.array(coeff_true_all)
        path_coeff_fit = []
        path_coeff_diff = []
        path_cluster_sim = []
        path_cluster_sim_orig = []
        for i in range(self.path_length):
            coeff_pred = self.coefficients_path[i,:,:]
            for obs in range(len(self.true_clusters)):
                coeff_pred[obs,:] = self.flip_sign(coeff_true_all[obs,:], coeff_pred[obs,:])
            _, coeff_pred_fit, disparity = procrustes(coeff_true_all, coeff_pred)
            path_coeff_fit.append(coeff_pred_fit)
            path_coeff_diff.append(disparity)
            path_cluster_sim_orig.append(np.diag(1 - sp.distance.cdist(coeff_true_all, coeff_pred, 'cosine')))
            path_cluster_sim.append(np.diag(1 - sp.distance.cdist(coeff_true_all, coeff_pred_fit, 'cosine')))
        self.path_coeff_fit = path_coeff_fit
        self.path_coeff_diff = path_coeff_diff
        self.path_cluster_sim = path_cluster_sim
        self.path_cluster_sim_orig = path_cluster_sim_orig
        return self.path_coeff_diff, self.path_coeff_fit, self.path_cluster_sim, self.path_cluster_sim_orig

    def calculate_variate(self, mat, coeff_true=None, transform_coeffs=False):
        """
        Calculate canonical variate along penalty path
        Calls self.flip_sign
        Params:
            coeff_true: 
            self.true_clusters
            self.path_length
            mat = data matrix (X or Y) of shape (num_obs, num_vars)
            transform_coeffs = True if coeffs should first be sign-corrected and Procrustes rotation / scaling "corrected" * might be messed up currently
        Returns: 
            self.path_variates = list of canonical variate at each penalty
        """
        num_obs = mat.shape[0]
        
        if transform_coeffs == True:
            coeff_true_all = []
            for i in range(len(self.true_clusters)):
                cluster_idx = self.true_clusters[i]
                coeff_true_all.append(coeff_true[cluster_idx]) 
            coeff_true_all = np.array(coeff_true_all)

        path_variates = []
        for i in range(self.path_length):
            coeff_pred = self.coefficients_path[i,:,:]
            variate = np.zeros(num_obs)
            
            if transform_coeffs == True:
                for obs in range(num_obs):
                    coeff_pred[obs,:] = self.flip_sign(coeff_true_all[obs,:], coeff_pred[obs,:])
                _, coeff_pred_fit, disparity = procrustes(coeff_true_all, coeff_pred)
            else: 
                coeff_pred_fit = coeff_pred
                
            for obs in range(num_obs):
                variate[obs] = mat[obs,:] @ coeff_pred_fit[obs,:]
                
            path_variates.append(variate)
            
        self.path_variates = path_variates
        return self.path_variates
    
    def plot_coefficients(self, idx=None):
        """
        Takes cosine similarity between predicted coefficients and truth and plots it.
        Params:
            object
            idx = Index of penalty to add text indicator to - if none, it chooses the largest coefficient similarity 
        Returns:
            Plot
        """
        x = np.round(np.array(self.penalty_list),8)
        y = np.zeros(len(self.penalty_list))
        y_name = "Cosine similarity"
        for i in range(len(self.penalty_list)):
            y[i] = np.round(np.mean(self.path_cluster_sim[i]),5)
            
        fig, ax = plt.subplots(1,1, figsize=(20,10))
        ax.plot(np.arange(x.shape[0]), y);
#         if plot_rand == True:
#             ax.boxplot(y_rand, notch ='True', patch_artist = True)
        ax.set_xticks(range(x.shape[0]), minor=False);
        plt.setp(ax.get_xticklabels(), rotation=70, horizontalalignment='right');
        ax.set_xticklabels(x,fontsize=24);
        evens = np.arange(0,len(ax.xaxis.get_ticklabels())+1,2);
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        ax.tick_params(axis='y', labelsize=24)
        plt.xlabel(r'$\lambda$',fontsize=24)
        plt.ylabel(y_name,fontsize=24)
        if idx == None:
            plt.text(np.nanargmax(y), np.nanmax(y), [np.nanargmax(y),x[np.nanargmax(y)]], fontsize=24)
        else:
            plt.text(idx, y[idx], [idx,x[idx]], fontsize=24)
        
    def plot_score(self, score_name, plot_rand=False, num_reps=2, highlight_idx=None):
        """
        Takes input score_name and outputs corresponding plot. Plot_rand adds a boxplot of 
        random distribution of that statistic sampled num_reps times.
        Params:
            self = cluster_metric object
            score_name = name of score to plot (ARI, RI, NMI, ch, gap, db, sil)
            plot_rand = If True, runs self.calculate_scores_random(num_reps) * currently only works for ARI, RI, nmi
            num_reps = number of times plot_rand samples clusters
        Returns:
            Plot
        """
        x = np.round(np.array(self.penalty_list),8)
        
        if plot_rand == True:
            if hasattr(self,'nmi_score_rand'):
                nmi_rand = self.nmi_score_rand
                ari_rand = self.adj_rand_score_rand
                ri_rand = self.rand_score_rand
            else:
                nmi_rand, ari_rand, ri_rand, _ = self.calculate_scores_random(num_reps)
        else:
            nmi_rand = ari_rand = ri_rand = None
            
        if score_name=="ARI":
            y = self.adj_rand_score
            # y_rand = ari_rand
            y_name = "Adjusted Rand Index"
        elif score_name=="RI":
            y = self.rand_score
            # y_rand = ri_rand
            y_name = "Rand Index"
        elif score_name=="NMI":
            y = self.nmi_score
            # y_rand = nmi_rand
            y_name = "Normalized Mutual Information"
        elif score_name=="ch":
            y = self.ch_score
#             y_rand = ch_rand
            y_name = "Calinski Harabasz"
        elif score_name=="gap":
            y = self.gap_score
#             y_rand = gs_rand
            y_name = "Gap statistic"
        elif score_name=="db":
            y = self.db_score
#             y_rand = db_rand
            y_name = "Davies bouldin"
        elif score_name=="sil":
            y = self.sil_score
#             y_rand = sil_rand
            y_name = "Silhouette score"

        fig, ax = plt.subplots(1,1, figsize=(20,10))
        ax.plot(np.arange(x.shape[0]), y);
        if plot_rand == True:
            ax.boxplot(y_rand, notch ='True', patch_artist = True)
        ax.set_xticks(range(x.shape[0]), minor=False);
        plt.setp(ax.get_xticklabels(), rotation=70, horizontalalignment='right');
        ax.set_xticklabels(x,fontsize=24);
        evens = np.arange(0,len(ax.xaxis.get_ticklabels())+1,2);
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        ax.tick_params(axis='y', labelsize=24)
        plt.xlabel(r'$\lambda$',fontsize=24)
        plt.ylabel(y_name,fontsize=24)
        if highlight_idx!=None:
            plt.text(highlight_idx, y[highlight_idx], [highlight_idx,x[highlight_idx]], fontsize=24)
        elif score_name == "gap" or score_name == "db":
            plt.text(np.argmin(y), np.nanmin(y), [np.argmin(y),x[np.argmin(y)]], fontsize=24)
        else:
            plt.text(np.nanargmax(y), np.nanmax(y), [np.nanargmax(y),x[np.nanargmax(y)]], fontsize=24)


# def cross_data_predict(X, Y, pred_clusters_list, penalty_list, rho=1, admm_iters=2, cca_iters=3, verb=False, test_size=0.33, cutoff=1e-15, seed=42):
#     """
#     Cross-Dataset Prediction with CCA
#     Test sets are "novel" data
#     U is a in paper
#     V is b in paper
#     Params: 
#         X = np-array shape (num_obs, num_var_X)
#         Y = np-array shape (num_obs, num_var_Y)
#         penalty_list = list of length path_length (number of penalties)
#         cutoff = for pinv (pseudoinverse)
#         seed * change to random and save seeds
#     Returns: 
#     """
#     accuracy_X_list_cluster = []
#     accuracy_Y_list_cluster = []
#     num_clusters = len(np.unique(pred_clusters_list[0]))
#     for k in range(num_clusters):
#         accuracy_X_list=[]
#         accuracy_Y_list=[]
#         seed = np.random.randint(99999)
#         for i in range(len(penalty_list)):
#             pred_clusters = pred_clusters_list[i]
#             # Select data in cluster
#             X_cluster = X[pred_clusters==k,:]
#             Y_cluster = Y[pred_clusters==k,:]

#             # Subsample data
#             X_train, X_test, Y_train, Y_test = train_test_split(X_cluster, Y_cluster, test_size=test_size, random_state=seed)
#             num_obs_test = X_test.shape[0]

#             # Calculate U_list and V_list in training data along penalty path
#             print("Calculating training data coefficients for penalty index ",i)
#             U_list_train, V_list_train = admm_CCA_new(X_train, Y_train, penalty_list, rho=rho, admm_iters=admm_iters, cca_iters=cca_iters, verb=verb)

#             # Calculate mean coefficients U and V in cluster
#             U = np.mean(U_list_train[i], axis=0)
#             V = np.mean(V_list_train[i], axis=0)
            
#             ## Calculate accuracy of predicting Y
#             # Test samples from X projected onto the canonical space via the canonical coefficients U
#             Xu_test = np.array([X_test @ U])
#             # Invert the canonical coefficient matrix V (pseudoinverse)
#             V_inv = np.linalg.pinv([V], rcond = cutoff)
#             # Dot product of the pseudoinverse of the training canonical coefficients V and test canonical variates Xu
#             Y_pred = np.dot(V_inv, Xu_test).T
#             # Calculate accuracy between predicted Y samples with the actual Y samples
#             accuracy_Y=[]
#             for i in range(Y_test.shape[0]):
#                 r,_ = pearsonr(Y_test[i,:],Y_pred[i,:])
#                 accuracy_Y.append(r)
#             accuracy_Y = np.array(accuracy_Y)

#             ## Accuracy of predict X
#             # Test samples from Y projected onto the canonical space via the canonical coefficients V
#             Yv_test = np.array([Y_test @ V])

#             # Invert the canonical coefficient matrix U (pseudoinverse)
#             U_inv = np.linalg.pinv([U], rcond = cutoff)

#             # Dot product of the pseudoinverse of the training canonical coefficients U and test canonical variates Yv
#             X_pred = np.dot(U_inv, Yv_test).T

#             # Calculate accuracy between predicted X samples with the actual X samples
#             accuracy_X=[]
#             print("X test shape",X_test.shape[0])
#             for i in range(X_test.shape[0]):
#                 r,_ = pearsonr(X_test[i,:],X_pred[i,:])
#                 accuracy_X.append(r)
#             print(accuracy_X)
#             # accuracy_X = np.array(accuracy_X)

#             accuracy_X_list.append(accuracy_X)
#             accuracy_Y_list.append(accuracy_Y)

#         accuracy_X_list_cluster.append(accuracy_X_list)
#         accuracy_Y_list_cluster.append(accuracy_Y_list)

#     return accuracy_X_list_cluster, accuracy_Y_list_cluster

def cross_data_predict(X, Y, pred_clusters_list, penalty_list, rho=1, admm_iters=2, cca_iters=3, verb=False, test_size=0.33, cutoff=1e-15, seed=42):
    """
    Cross-Dataset Prediction with CCA
    Test sets are "novel" data
    U is a in paper
    V is b in paper

    Change it to use true clusters
    Params: 
        X = np-array shape (num_obs, num_var_X)
        Y = np-array shape (num_obs, num_var_Y)
        penalty_list = list of length path_length (number of penalties)
        cutoff = for pinv (pseudoinverse)
        seed * change to random and save seeds
    Returns: 
    """
    accuracy_X_list_cluster = []
    accuracy_Y_list_cluster = []
    num_clusters = len(np.unique(pred_clusters_list[0]))
    for k in range(num_clusters):
        accuracy_X_list=[]
        accuracy_Y_list=[]
        seed = np.random.randint(99999)
        
        pred_clusters = pred_clusters_list[i]
        # Select data in cluster
        X_cluster = X[pred_clusters==k,:]
        Y_cluster = Y[pred_clusters==k,:]

        # Subsample data
        X_train, X_test, Y_train, Y_test = train_test_split(X_cluster, Y_cluster, test_size=test_size, random_state=seed)
        num_obs_test = X_test.shape[0]

        # Calculate U_list and V_list in training data along penalty path
        print("Calculating training data coefficients for penalty index ",i)
        U_list_train, V_list_train = admm_CCA_new(X_train, Y_train, penalty_list, rho=rho, admm_iters=admm_iters, cca_iters=cca_iters, verb=verb)

        for i in range(len(penalty_list)):
            # Calculate mean coefficients U and V in cluster
            U = np.mean(U_list_train[i], axis=0)
            V = np.mean(V_list_train[i], axis=0)
            
            ## Calculate accuracy of predicting Y
            # Test samples from X projected onto the canonical space via the canonical coefficients U
            Xu_test = np.array([X_test @ U])
            # Invert the canonical coefficient matrix V (pseudoinverse)
            V_inv = np.linalg.pinv([V], rcond = cutoff)
            # Dot product of the pseudoinverse of the training canonical coefficients V and test canonical variates Xu
            Y_pred = np.dot(V_inv, Xu_test).T
            # Calculate accuracy between predicted Y samples with the actual Y samples
            accuracy_Y=[]
            for i in range(Y_test.shape[0]):
                r,_ = pearsonr(Y_test[i,:],Y_pred[i,:])
                accuracy_Y.append(r)
            accuracy_Y = np.array(accuracy_Y)

            ## Accuracy of predict X
            # Test samples from Y projected onto the canonical space via the canonical coefficients V
            Yv_test = np.array([Y_test @ V])

            # Invert the canonical coefficient matrix U (pseudoinverse)
            U_inv = np.linalg.pinv([U], rcond = cutoff)

            # Dot product of the pseudoinverse of the training canonical coefficients U and test canonical variates Yv
            X_pred = np.dot(U_inv, Yv_test).T

            # Calculate accuracy between predicted X samples with the actual X samples
            accuracy_X=[]
            print("X test shape",X_test.shape[0])
            for i in range(X_test.shape[0]):
                r,_ = pearsonr(X_test[i,:],X_pred[i,:])
                accuracy_X.append(r)
            print(accuracy_X)
            # accuracy_X = np.array(accuracy_X)

            accuracy_X_list.append(accuracy_X)
            accuracy_Y_list.append(accuracy_Y)

        accuracy_X_list_cluster.append(accuracy_X_list)
        accuracy_Y_list_cluster.append(accuracy_Y_list)

    return accuracy_X_list_cluster, accuracy_Y_list_cluster


def split_half(X, Y, penalty_list, num_splits=2, true_clusters=None, use_true_clusters=True,  rho=0.5, admm_iters=1, cca_iters=5, verb=False):
    """
    Split half analysis to compare coefficient stability between random cluster splits
    Params:
        num_splits = number of times to iterate splitting the data in half and then calculating statistics
        true_clusters = np.array of length num_obs
        use_true_clusters = indicator of whether to use true_clusters to split each cluser in half
        seed * set to random and save seeds
    Return:
        u_sims = u cos similarity between splits
        v_sims
        U_list_trains
        U_list_tests
        V_list_trains
        V_list_tests
        consider as dictionary instead of list of lists
    """
    seeds = []
    U_list_trains = []
    U_list_tests = []
    V_list_trains = []
    V_list_tests = []
    u_path_sims = []
    v_path_sims = []
    
    print(num_splits)
    for i in range(num_splits):
        print("Iteration ",i)

        num_clusters = len(np.unique(np.array(true_clusters)))
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        # Split each cluster's samples in half
        for k in range(num_clusters):
            X_c = X[true_clusters==k,:]
            Y_c = Y[true_clusters==k,:]
            X_tr, X_te, Y_tr, Y_te = train_test_split(X_c, Y_c, test_size=0.5)
            X_train.append(X_tr)
            Y_train.append(Y_tr)
            X_test.append(X_te)
            Y_test.append(Y_te)

        X_train=np.vstack(X_train)
        Y_train=np.vstack(Y_train)
        X_test=np.vstack(X_test)
        Y_test=np.vstack(Y_test)
        true_clusters_train = np.concatenate([np.zeros(int(X_train.shape[0]/2)),np.ones(int(X_train.shape[0]/2))])
        true_clusters_test = np.concatenate([np.zeros(int(X_test.shape[0]/2)),np.ones(int(X_test.shape[0]/2))])

        # if use_true_clusters==True:
        #     num_clusters = np.unique(np.array(true_clusters))
        #     X_train = Y_train = X_test = Y_test = []
        #     # Split each cluster's samples in half
        #     for cluster in range(num_clusters):
        #         X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, 0.5)
        #         X_train.append(X_tr)
        #         Y_train.append(Y_tr)
        #         X_test.append(X_te)
        #         Y_test.append(Y_te) 
        # else:
        #     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 0.5)

        # Calculate U_list and V_list in training data along penalty path
        # print("Calculating training data coefficients")
        U_list_train, V_list_train = admm_CCA_new(X_train, Y_train, penalty_list, rho=rho, admm_iters=admm_iters, cca_iters=cca_iters, verb=verb)
    
        # Calculate U_list and V_list in test data along penalty path
        # print("Calculating test data coefficients")
        U_list_test, V_list_test = admm_CCA_new(X_test, Y_test, penalty_list, rho=rho, admm_iters=admm_iters, cca_iters=cca_iters, verb=verb)
    
        # U_list_trains.append(U_list_train)
        # U_list_tests.append(U_list_test)
        # V_list_trains.append(V_list_train)
        # V_list_tests.append(V_list_test)
        
        u_path_sim = []
        v_path_sim = []
        # COSINE SIMILARITY like in compare_coefficients
        for i in range(len(penalty_list)):
            u_sim = []
            v_sim = []
            for k in range(num_clusters):
                u_coeff_train = np.mean(U_list_train[i][true_clusters_train==k,:],axis=0)
                u_coeff_test = np.mean(U_list_test[i][true_clusters_test==k,:],axis=0)
                v_coeff_train = np.mean(V_list_train[i][true_clusters_train==k,:],axis=0)
                v_coeff_test = np.mean(V_list_test[i][true_clusters_test==k,:],axis=0)
                # set as absolute value of cosine similarity
                u_sim.append(float(np.diag(1 - sp.distance.cdist([u_coeff_train], [u_coeff_test], 'cosine'))))
                v_sim.append(float(np.diag(1 - sp.distance.cdist([v_coeff_train], [v_coeff_test], 'cosine'))))
            u_path_sim.append(u_sim)
            v_path_sim.append(v_sim)
        u_path_sims.append(np.array(u_path_sim))
        v_path_sims.append(np.array(v_path_sim))

    return u_path_sims, v_path_sims



def calculate_variate_true(mat, coeff_true, true_clusters):
    """
    Returns true variates when given data matrix mat and true coefficients
    Params:
        mat: matrix (X or Y) of shape (num_obs,num_vars)
        coeff_true: List of coefficients for each cluster, length num_clusters
        true_clusters: True clusters, used to expand coeff_true
    """
    # Expand coefficients
    coeff_true_all = []
    for i in range(len(true_clusters)):
        cluster_idx = true_clusters[i]
        coeff_true_all.append(coeff_true[cluster_idx]) 
    coeff_true_all = np.array(coeff_true_all)
    # Project mat onto coeff_true
    num_obs = mat.shape[0]
    true_variate = np.zeros(num_obs)
    for obs in range(num_obs):
        true_variate[obs] = mat[obs,:] @ coeff_true_all[obs,:]

    return true_variate


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


###------- PLOTTING FUNCTIONS -------###
from matplotlib import cm
def path_plot(coefficient_arr, penalty_list,plot_range=[0,-1],cut_vars=False, cmap='tab20b'):
    # Crop x axis (e.g, to remove 'burn-in' period at beginning)
    coefficient_arr = coefficient_arr[plot_range[0]:plot_range[1],:,:]
    penalty_list = penalty_list[plot_range[0]:plot_range[1]]
    if cut_vars is True:
        coefficient_arr = coefficient_arr[:,:,[1,2,coefficient_arr.shape[2]-1]]

    # Colormap
    cmap = cm.get_cmap(cmap, coefficient_arr.shape[2])
    colors = cmap(np.linspace(0.0,1.0,coefficient_arr.shape[2]))

    # Define x-axis range
    penalty_range = range(len(penalty_list))

    fig, ax = plt.subplots(1,1, figsize=(20,10))
    for i in range(coefficient_arr.shape[2]):
        x = np.round(np.array(penalty_list),8)[penalty_range]
        y = coefficient_arr[penalty_range,:,i]
        ax.plot(np.arange(x.shape[0]), y, color=colors[i], alpha=0.15)
        ax.set_xticks(range(x.shape[0]), minor=False);
        plt.setp(ax.get_xticklabels(), rotation=70, horizontalalignment='right')
        ax.set_xticklabels(x,fontsize=24)
        evens = np.arange(0,len(ax.xaxis.get_ticklabels())+1,2)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    ax.tick_params(axis='y', labelsize=24)
    plt.xlabel(r'$\lambda$',fontsize=24)
    plt.ylabel('Coefficients',fontsize=24)

from sklearn.cluster import KMeans, SpectralClustering
def plot_cluster_assignments(X, true_clusters, coefficient_arr, num_clusters, skip=5, plot_idx=None, var_idx=[0,1]):
    if plot_idx is None:
        for i in range(coefficient_arr.shape[0]):
            if np.mod(i,skip) == 0:
                # kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(coefficient_arr[i,:,:])
                # cs = kmeans.labels_

                dpgmm = mixture.BayesianGaussianMixture(n_components=num_clusters, covariance_type='full', weight_concentration_prior_type='dirichlet_process',max_iter=1000,reg_covar=0.1)
                dpgmm_mod = dpgmm.fit(coefficient_arr[i,:,:])
                cs = dpgmm.predict(coefficient_arr[i,:,:])

                plt.figure(figsize=(12,6))
                plt.subplot(1,2,1)
                plt.scatter(X[:,var_idx[0]], X[:,var_idx[1]],c=true_clusters)
                plt.xlabel('Variable '+str(var_idx[0]))
                plt.ylabel('Variable '+str(var_idx[1]))

                plt.subplot(1,2,2)
                plt.scatter(X[:,var_idx[0]], X[:,var_idx[1]],c=cs)
                plt.xlabel('Variable '+str(var_idx[0]))
                plt.ylabel('Variable '+str(var_idx[1]))
    else:
        # kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(coefficient_arr[plot_idx,:,:])
        # cs = kmeans.labels_
        dpgmm = mixture.BayesianGaussianMixture(n_components=num_clusters, covariance_type='full', weight_concentration_prior_type='dirichlet_process',max_iter=1000,reg_covar=0.1)
        dpgmm_mod = dpgmm.fit(coefficient_arr[i,:,:])
        cs = dpgmm.predict(coefficient_arr[i,:,:])


        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.scatter(X[:,0], X[:,1],c=true_clusters)
        plt.xlabel('Variable '+str(var_idx[0]))
        plt.ylabel('Variable '+str(var_idx[1]))

        plt.subplot(1,2,2)
        plt.scatter(X[:,0], X[:,1],c=cs)  
        plt.xlabel('Variable '+str(var_idx[0]))
        plt.ylabel('Variable '+str(var_idx[1]))

def plot_ordercolor(X, true_clusters, coefficient_arr, split_variable, skip=5, plot_idx=None, var_idx=[0,1]):
    cmap = cm.get_cmap('viridis')
    if plot_idx is None:
        # Plot full path range (skiping every 'skip' variables)
        for i in range(coefficient_arr.shape[0]):
            if np.mod(i,skip) == 0:
                plt.figure(figsize=(12,6))
                plt.subplot(1,2,1)
                plt.scatter(X[:,var_idx[0]], X[:,var_idx[1]],c=true_clusters)
                plt.xlabel('Variable '+str(var_idx[0]))
                plt.ylabel('Variable '+str(var_idx[1]))

                plt.subplot(1,2,2)
                vec = coefficient_arr[i,:,split_variable]
                colors = cmap( (vec-np.min(vec))/np.max(vec), len(vec))
                plt.scatter(X[:,var_idx[1]], X[:,var_idx[1]], c=colors)
                plt.xlabel('Variable '+str(var_idx[0]))
                plt.ylabel('Variable '+str(var_idx[1]))
    else: 
        # Plot specific value along path
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.scatter(X[:,var_idx[0]], X[:,var_idx[1]],c=true_clusters)
        plt.xlabel('Variable '+str(var_idx[0]))
        plt.ylabel('Variable '+str(var_idx[1]))
                
        plt.subplot(1,2,2)
        vec = coefficient_arr[plot_idx,:,split_variable]
        colors = cmap( (vec-np.min(vec))/np.max(vec), len(vec))
        plt.scatter(X[:,var_idx[0]], X[:,var_idx[1]], c=colors)
        plt.xlabel('Variable '+str(var_idx[0]))
        plt.ylabel('Variable '+str(var_idx[1]))
        
def ari_subsample(v_metrics, test_size=0.1, n_splits=5, rho=1, admm_iters=2, cca_iters=2, verb=False):
    from sklearn.metrics.cluster import normalized_mutual_info_score as NMI_score, adjusted_rand_score as ARI_score, rand_score as RI_score
    from sklearn.model_selection import ShuffleSplit
    rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
    it = 1
    nmi_score = []
    adj_rand_score = []
    mse_score = []
    for train_index, test_index in rs.split(range(X.shape[0])):
        # Subsample data
        true_clusters_train = v_metrics.true_clusters[train_index]
        X_train = X[train_index,:]
        Y_train = Y[train_index,:]
        # Calculate coefficients
        print("Calculating training data coefficients for split ",it)
        U_list_train, V_list_train = admm_CCA_new(X_train, Y_train, v_metrics.penalty_list, rho=rho, admm_iters=admm_iters, cca_iters=cca_iters, verb=verb)
        it = it+1
        # Calculate clusters and scores
        coefficients_path = np.array(V_list_train)
        nmi = []
        ari = []
        ri = []
        mse = []
        path_clusters = []
        for i in range(v_metrics.path_length):
    #       Cluster coefficients at penalty
    #       ! change to dbgmm
            kmeans = KMeans(n_clusters=v_metrics.num_clusters, random_state=0).fit(coefficients_path[i,:,:])
            pred_clusters = kmeans.labels_
            path_clusters.append(pred_clusters)
            # Calculate cluster scores
            nmi.append(NMI_score(true_clusters_train, pred_clusters, average_method='arithmetic'))
            ari.append(ARI_score(true_clusters_train, pred_clusters))
            mse.append(mean_squared_error(true_clusters_train, pred_clusters))

        nmi_score.append(np.array(nmi))
        adj_rand_score.append(np.array(ari))
        mse_score.append(np.array(mse))

    # for it in range(5):
    #     print(np.nanargmax(adj_rand_score[it]))

    return adj_rand_score, nmi_score, mse_score

def convex_clust_df(uV, penalty, epsilon = 0.75):
    D = sparse_D(uV.shape[0], uV.shape[1])
    zero_idxs = np.where(np.sum(np.abs(D*uV),axis=1) <= epsilon)[0]

    D0 = D[zero_idxs,:]
    D1 = D[[i for i in range(D.shape[0]) if i not in zero_idxs],:]

    chol_D0 = cholesky_AAt(D0, beta=1.0)
    I = np.eye(D.shape[1])
    P = I - D0.T * chol_D0(D0)

    LHS = (1.0/np.linalg.norm(D1*uV)) * (D1.T * D1)
    RHS = (1.0/np.linalg.norm(D1*uV)**3) * ((D1.T*D1)*np.dot(uV, uV.T)*(D1.T*D1))

    df_hat = np.trace(np.linalg.inv(I + penalty*P*(LHS - RHS)) * P)
    return df_hat

def eBIC(mat_tilde, coeff, penalty, gamma, proximity_param=0.9):
    n,p = mat_tilde.shape
    RSS = np.sum((mat_tilde.flatten()-coeff.flatten())**2)
    fit_term = n*p*np.log(RSS/(n*p))
    df = convex_clust_df(coeff, penalty, proximity_param)
    model_complexity = (1 + 2*gamma)*df*np.log(n*p)
    ebic = fit_term + model_complexity
    return model_complexity, fit_term, ebic

def min_eBIC(X, Y, U_list, V_list, penalty_list, gamma=1.0, proximity_param=0.9):  
    ebics = []
    dfs = []
    RSSs = []
    
    for i in range(len(penalty_list)):
        U = U_list[i]
        V = V_list[i]

        pen = penalty_list[i]
        df, RSS, ebic = eBIC(np.cov(X,Y), np.cov(U,V), pen, gamma, proximity_param=proximity_param)
        ebics.append(ebic)
        dfs.append(df)
        RSSs.append(RSS)
        
    start = np.where(RSSs < np.quantile(RSSs,0.9))[0][0] # mask initial burn in period
    start = np.max((start,10))
    min_ebic = np.nanmin(ebics[start:])
    best_idx = np.where(ebics==min_ebic)[0][0]
    
    return ebics, best_idx, dfs, RSSs

def min_eBIC_U(X, Y, U_list, V_list, penalty_list, gamma=1.0, proximity_param=0.9):  
    ebics = []
    dfs = []
    RSSs = []
    
    for i in range(len(penalty_list)):
        U = U_list[i]
        V = V_list[i]

        Xu_tildes = []
        for obs in range(X.shape[0]):
            Xu_tildes.append(np.dot(np.outer(X[obs,:].T,Y[obs,:]),V[obs,:]))
        Xu = np.asarray(Xu_tildes)

        pen = penalty_list[i]
        df, RSS, ebic = eBIC(Xu, U, pen, gamma, proximity_param=proximity_param)
        ebics.append(ebic)
        dfs.append(df)
        RSSs.append(RSS)
        
    start = np.where(RSSs < np.quantile(RSSs,0.9))[0][0] # mask initial burn in period
    start = np.max((start,10))
    min_ebic = np.nanmin(ebics[start:])
    best_idx = np.where(ebics==min_ebic)[0][0]
    
    return ebics, best_idx, dfs, RSSs

def min_eBIC_V(X, Y, U_list, V_list, penalty_list, gamma=1.0, proximity_param=0.9):  
    ebics = []
    dfs = []
    RSSs = []
    
    for i in range(len(penalty_list)):
        U = U_list[i]
        V = V_list[i]

        Yv_tildes = []
        for obs in range(Y.shape[0]):
            Yv_tildes.append(np.dot(np.outer(Y[obs,:].T,X[obs,:]),U_initial[obs,:]))
        Yv = np.asarray(Yv_tildes)

        pen = penalty_list[i]
        df, RSS, ebic = eBIC(Yv, V, pen, gamma, proximity_param=proximity_param)
        ebics.append(ebic)
        dfs.append(df)
        RSSs.append(RSS)
        
    start = np.where(RSSs < np.quantile(RSSs,0.9))[0][0] # mask initial burn in period
    start = np.max((start,10))
    min_ebic = np.nanmin(ebics[start:])
    best_idx = np.where(ebics==min_ebic)[0][0]
    
    return ebics, best_idx

