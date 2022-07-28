###------- NeurIPS 2022 submission PCMF code -------###
### Hierarchically Clustered PCA and CCA via a Convex Clustering Penalty ###
###------- IMPORT FUNCTIONS -------###
import numpy as np
from sklearn.utils.extmath import randomized_svd
from itertools import combinations 
import cvxpy as cp
import time
from numba import jit, prange
import matplotlib.pyplot as plt

# to use first run 'python setup.py build_ext --inplace'
# from admm_utils import prox as cprox

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI_score, adjusted_rand_score as ARI_score, rand_score as rand_score
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score, mean_squared_error # cohen_kappa_score, hinge_loss, coverage_error, consensus_score

from scipy.sparse import csr_matrix
from sksparse.cholmod import cholesky, cholesky_AAt
from scipy.spatial.distance import pdist, squareform

###------- ADMM subproblems for u -------###
def clusterpath_PMD_subproblem_u(X_list, num_var, verb=True):
    '''
    Solves one subproblem in the alternating PMD-like pathwise clustered algorithm 
    for one value of lambda. Does not directly cluster u's.
    Note: we may want to relax convergence criteria for subproblem (at least in early iterations).
    '''
    X = np.hstack(X_list) # XXX add check - vector num_var*num_obs long
    num_obs = int(len(X)/num_var)
    beta = cp.Variable(num_var*num_obs)
    lambd = cp.Parameter(nonneg=True)
    l2_reg = cp.Parameter(nonneg=True)
    constraints = [cp.norm2(beta)**2 <= 1] 
    problem = cp.Problem(cp.Minimize(objective_fn_u(X, beta)), constraints)

    mosek_params = {} #{'MSK_DPAR_MIO_TOL_ABS_GAP':1e-1}
    problem.solve(solver='MOSEK',verbose=verb, warm_start=True, mosek_params=mosek_params)
    #err = mse(X, beta)
    return beta.value #, err

def clusterpath_PCMF_subproblem_u(X_list, num_var, penalty, verb=True):
    '''
    Solves one subproblem in the alternating PCMF pathwise clustered algorithm                                    
    for one value of lambda.
    Note: we may want to relax convergence criteria for subproblem (at least in early iterations).                       
    '''
    X = np.hstack(X_list) # XXX add check - vector num_var*num_obs long
    num_obs = int(len(X)/num_var)
    beta = cp.Variable(num_var*num_obs)
    lambd = cp.Parameter(nonneg=True)
    l2_reg = cp.Parameter(nonneg=True)
    constraints = [cp.norm2(beta)**2 <= 1]
    problem = cp.Problem(cp.Minimize(objective_fn_u2(X, beta, penalty)), constraints)
    mosek_params = {} #{'MSK_DPAR_MIO_TOL_ABS_GAP':1e-1}                                  
    problem.solve(solver='MOSEK',verbose=verb, warm_start=True, mosek_params=mosek_params)
    #err = mse(X, beta)                                                                    
    return beta.value #, err

###------- ADMM helper functions -------###

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
        # group_soft_threshold
        vec_norm = np.linalg.norm(V[i,:])
        if vec_norm > alpha:
            V_prox[i, :] = V[i,:] - alpha*V[i,:]/vec_norm
        else:
            V_prox[i, :] = np.zeros_like(V[i,:])
    return V_prox
@jit(nopython=True, parallel=True, fastmath=True) 
def nb_subtract(M1, M2):
    return M1-M2

@jit(nopython=True, parallel=True, fastmath=True)
def nb_add(M1, M2):
    return M1+M2

def get_weights(X, gauss_coef=0.5, neighbors=None):
    '''
    '''
    from scipy.spatial.distance import pdist, squareform
    dist_vec = pdist(X) / X.shape[0]
    w = np.exp(-1*gauss_coef*(dist_vec)**2)
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

def l2_ball_proj(X):
    '''
    Normalize rows of X to satisfy $\|X_{i,\cdot}\|_2^2 \leq 1$.
    '''
    def row_norm(vec): 
        return np.sqrt(1./np.sum(vec**2)) * vec
    return np.apply_along_axis(row_norm,1,X)

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

def chol_D(num_obs, num_vars, rho, reg_scale=1.0):
    '''
    '''
    D = sparse_D(num_obs, num_vars)
    return D, cholesky_AAt(np.sqrt(rho) * D.T, beta=1.0*reg_scale)

def objective_fn_u(X, beta):
    mat = loss_fn(X, beta)
    return mat

def objective_fn_u2(X, beta, penalty):
    D = sparse_D(X.shape[0],1)
    mat = loss_fn(X, beta) + penalty*cp.norm2(D@beta)
    return mat

def loss_fn(X, beta):
    return -1*cp.sum(cp.matmul(X,beta))

def mse(X, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, beta).value

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols

def SVD(X,return_rank=2):
    U, s, Vh = np.linalg.svd(X,full_matrices=False)
    U = U[:,0:return_rank]
    Vh = Vh[0:return_rank,:]
    s = s[0:return_rank]
    return U,s,Vh

def NMF(X,return_rank=2):
    from sklearn.decomposition import NMF as sk_NMF
    model = sk_NMF(n_components=2, init='random', random_state=0)
    U = model.fit_transform(X)
    Vh = model.components_
    U = U[:,0:return_rank]
    Vh = Vh[0:return_rank,:]
    return U,Vh

###------- PCMF functions -------###

def pcmf_PALS(X, penalty_list, rho=1.0, admm_iters = 5, verb=False, weights=False, neighbors=None, gauss_coef=0.5,  print_progress=True, parallel=False, output_file='NaN', non_negative=False, numba=False):
    '''
    Relaxation of full PCMF problem to iterate between clustering on u and clustering on V. 
    '''
    print('weights: '+str(weights), 'neighbors: '+str(neighbors), 'gauss_coef: '+str(gauss_coef), 'rho: '+str(rho))
    # Initialize                                                                                                     
    if weights is False:
        weights = get_weights(X, gauss_coef=0.0)
    else:
        weights = get_weights(X, gauss_coef=gauss_coef, neighbors=neighbors)
    D, chol_factor = chol_D(X.shape[0], X.shape[1], rho)
    V = X.copy()
    for i in range(V.shape[0]):
        V[i,:] = np.mean(X, axis=0)
    W = Z = W2 = Z2 = D*X
    d = np.ones(X.shape[0])
    V_list = []
    u_list = []
    s_list = []
    # First run initial on v_init                                                                                    
    Xu_tildes = []
    for i in range(X.shape[0]):
        Xu_tildes.append(np.dot(X[i,:], V[i,:]))
    u = clusterpath_PMD_subproblem_u(Xu_tildes, 1, verb)
    u.shape = (len(u),1)
    #
    try:
        # Iterate over penalty grid fitting problem for each value
        for i in range(len(penalty_list)):
            penalty = penalty_list[i]
            num_obs = X.shape[0]
            num_var = X.shape[1]
            if print_progress == True:
                if parallel == True:
                    file1 = open(output_file, "a")
                    file1.write(str([i+1])+" {:.5e}".format(penalty))
                    file1.write("...")
                    file1.close()
                else:
                    print("[",i+1,"]","{:.5e}".format(penalty), end="")
                    print("...", end="")
            # Update V                                                                                                   
            Xv_tildes = []
            for i in range(X.shape[0]):
                Xv_tildes.append(u[i]*X[i,:])
            Xv = np.asarray(Xv_tildes)
            for it in range(admm_iters):
                if numba is False:
                    V = chol_factor(Xv + rho*D.T*(W - Z))
                    W = cprox(D*V+Z, penalty, rho, weights)
                else:
                    V = chol_factor(nb_add(Xv, rho*D.T*nb_subtract(W,Z)))
                    W = prox_numba_arr(np.zeros_like(W), D*V+Z, penalty, rho, weights)
                #
                Z = Z + D*V - W
            #
            V = l2_ball_proj(V)
            if non_negative:
                V[V<0] = 0
            V_list.append(V)
            # Update u                                                                                                   
            Xu_tildes = []
            for i in range(X.shape[0]):
                Xu_tildes.append(np.dot(X[i,:], V[i,:]))
            Xu = np.asarray(Xu_tildes)
            try:
                u = clusterpath_PCMF_subproblem_u(Xu_tildes, 1, penalty, verb)
            except:
                print("PCMF subproblem is not defined for single cluster u, using PMD subproblem.")
                u = clusterpath_PMD_subproblem_u(Xu_tildes, 1, verb)
            u.shape = (len(u),1)
            s = u*Xv
            u_list.append(u)
            s_list.append(d)
            #
    except KeyboardInterrupt:
        print("KeyboardInterrupt has been caught.")
        #
    return V_list, u_list, s_list


def pcmf_ADMM(X, penalty_list, problem_rank=1, rho=1.0, admm_iters = 5, verb=False, weights=False, neighbors=None, gauss_coef=2.0,print_progress=True, parallel=False, output_file='NaN',factor_type='SVD', numba=False):
    '''
    Fits fully constrained PCMF problem iterating between convex clustering and an SVD of rank 'problem_rank' 
    using ADMM updates.
    '''
    print('weights: '+str(weights), 'neighbors: '+str(neighbors), 'gauss_coef: '+str(gauss_coef), 'rho: '+str(rho))
    # Initialize                                                                                                     
    if weights is False:
        weights = get_weights(X,gauss_coef=0.0)
    else:
        weights = get_weights(X,gauss_coef=gauss_coef, neighbors=neighbors)
    D, chol_factor = chol_D(X.shape[0], X.shape[1], rho, reg_scale=1.0+rho)
    G = Z1 = D*X
    A = Z2 = X.copy()
    if factor_type == 'SVD':
        U, s, Vh = SVD(A, problem_rank)
    elif factor_type == 'NMF':
        U, Vh = NMF(A, problem_rank)
        s = np.ones(problem_rank)
    A_list = []
    U_list = []
    s_list = []
    V_list = []
    #
    try:
        # Iterate over penalty grid fitting problem for each value
        for i in range(len(penalty_list)):
            penalty = penalty_list[i]
            num_obs = X.shape[0]
            num_var = X.shape[1]
            if print_progress == True:
                if parallel == True:
                    file1 = open(output_file, "a")
                    file1.write(str([i+1])+" {:.5e}".format(penalty))
                    file1.write("...")
                    file1.close()
                else:
                    # print("[",i+1,"]",round(penalty,5), end="")                                                         
                    print("[",i+1,"]","{:.5e}".format(penalty), end="")
                    print("...", end="")

            for it in range(admm_iters):
                if numba is False:
                    A = chol_factor(X + rho*D.T*(G - Z1) + rho*(np.dot(U*s,Vh) - Z2))
                    G = cprox(D*A+Z1, penalty, rho, weights)
                else:
                    A = chol_factor(X + rho*D.T*(G - Z1) + rho*(np.dot(U*s,Vh) - Z2))
                    G = prox_numba_arr(np.zeros_like(G), D*A+Z1, penalty, rho, weights)
                #
                if factor_type == 'SVD':
                    U, s, Vh = SVD(A + Z2, problem_rank)
                elif factor_type == 'NMF':
                    A[A<0] = 0
                    A_Z2 = A + Z2
                    A_Z2[A_Z2<0] = 0
                    U, Vh = NMF(A_Z2, problem_rank)
                    s = np.ones(problem_rank)
                #
                Z1 = Z1 + rho*(D*A - G)
                Z2 = Z2 + rho*(A - np.dot(U*s,Vh))
            #
            A_list.append(A)
            V_list.append(Vh)
            s_list.append(s)
            U_list.append(U)
            #
    except KeyboardInterrupt:
        print("KeyboardInterrupt has been caught.")
    return A_list, U_list, s_list, V_list

from sklearn.utils.extmath import randomized_svd
def pcmf_ADMM_consensus(X_all, penalty_list, problem_rank=1, rho=1.0, admm_iters = 5, verb=False, weights=False, neighbors=None, gauss_coef=2.0,print_progress=True, parallel=False, output_file='NaN',factor_type='SVD', split_size=10):
    '''
    Fits fully constrained PCMF problem iterating between convex clustering and an SVD of rank 'problem_rank' 
    using ADMM updates.
    '''
    print('weights: '+str(weights), 'neighbors: '+str(neighbors), 'gauss_coef: '+str(gauss_coef), 'rho: '+str(rho))
    #
    rho1 = rho2 = rho
    print("rho1:",rho1,"rho2",rho2)
    # Data split
    X_list = np.array_split(X_all, int(X_all.shape[0]/split_size), axis=0) # get batches
    X_list_inds = np.array_split(np.arange(X_all.shape[0]), int(X_all.shape[0]/split_size), axis=0) # get batches
    X = X_list[0]
    num_obs = X.shape[0]
    num_var = X.shape[1]
    D, chol_factor = chol_D(num_obs, num_var, rho1, reg_scale=1.0+rho1) # note num_var is not used in chol_D
    print("Number of batches:",len(X_list))
    #
    # Initialize
    G = []
    A = []
    Z1 = []
    Z2 = []
    U = []
    Vh = []
    s = []
    X_mean = []
    weights_list = []
    for X, idx in zip(X_list, np.arange(len(X_list))):
        print("Initialize IDX:",idx)
        X_mean.append(X.mean(0))
        X = X-np.mean(X,axis=0)
        X_list[idx] = X
        if weights is False:
            weights = get_weights(X,gauss_coef=0.0)
        else:
            weights = get_weights(X,gauss_coef=gauss_coef, neighbors=neighbors)
        weights_list.append(weights)
        #
        Ginit = Z1init = D*X
        Ainit = Z2init = X.copy()
        G.append(Ginit)
        A.append(Ainit)
        Z1.append(Z1init)
        Z2.append(Z2init)
    #
    M = []
    for X, idx in zip(X_list, np.arange(len(X_list))):
        M.append(np.tile(X_mean[idx],(X_list[0].shape[0],1)))
    M = np.vstack(M)
    print("Means matrix has shape:", M.shape)
    U, s, Vh = randomized_svd(X_all, n_components=problem_rank,random_state=1234)
    for X, idx in zip(X_list, np.arange(len(X_list))):
        X_inds = X_list_inds[idx]
        Z2[idx] = X.copy() #np.dot(U[X_inds]*s,Vh)
    #
    A_list = []
    U_list = []
    s_list = []
    V_list = []
    #
    try:
        # Iterate over penalty grid fitting problem for each value
        for i in range(len(penalty_list)):
            penalty = penalty_list[i]
            num_obs = X.shape[0]
            num_var = X.shape[1]
            if print_progress == True:
                if parallel == True:
                    file1 = open(output_file, "a")
                    file1.write(str([i+1])+" {:.5e}".format(penalty))
                    file1.write("...")
                    file1.close()
                else:                                                     
                    print("[",i+1,"]","{:.5e}".format(penalty), end="")
                    print("...", end="")
            for it in range(admm_iters):
                for X, idx in zip(X_list, np.arange(len(X_list))):
                    X_inds = X_list_inds[idx]
                    A[idx] = chol_factor(X + rho*D.T*(G[idx] - Z1[idx]) + rho*(np.dot(U[X_inds]*s,Vh) - Z2[idx]))
                    G[idx] = prox_numba_arr(np.zeros_like(G[idx]), D*A[idx]+Z1[idx], penalty, rho1, weights_list[idx])
                # consensus
                As = np.vstack(A)
                Z2s = np.vstack(Z2)
                B = As + Z2s
                U_b, s_b, Vh_b = randomized_svd(B, n_components=problem_rank,random_state=1234)
                U_a__s_a = np.dot(B + M, Vh_b.T)
                Vh = Vh_b
                s = s_b
                U = U_a__s_a / s
                U = np.sqrt(1./np.sum(U**2)) * U
                #
                Z2_update = rho2*(As - np.dot(U*s,Vh))
                #
                for X, idx in zip(X_list, np.arange(len(X_list))):
                    X_inds = X_list_inds[idx]
                    Z1[idx] = Z1[idx] + rho1*(D*A[idx] - G[idx])
                    Z2[idx] = Z2[idx] + Z2_update[X_inds,:]
            #
            A_list.append(np.vstack(A))
            V_list.append(np.vstack(Vh))
            s_list.append(np.vstack(s))
            U_list.append(np.vstack(U))
            #
    except KeyboardInterrupt:
        print("KeyboardInterrupt has been caught.")
        #
    return A_list, U_list, s_list, V_list


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
    #
    if numba==True:
        W = prox_numba_arr(np.zeros_like(W), D*V+Z, penalty, rho, weights)
    else:
        W = cprox(D*V+Z, penalty, rho, weights)
    Z = Z + D*V - W
    V = l2_ball_proj(V)
    #
    return V, W, Z

def pcmf_CCA(X, Y, penalty_list, rho=1.0, admm_iters = 2, cca_iters = 3, 
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
    #
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
    #
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
    #
    try:
        for i in range(len(penalty_list)):
            penalty = penalty_list[i]
            print_progress = verb
            if print_progress == True:
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
                    #
                    Xu = np.asarray(Xu_tildes)
                    U, Wx, Zx = admm_CCA_update(Xu, chol_factor_X, Dx, Wx, Zx, rho, penalty, X, weights=weights, neighbors=neighbors, gauss_coef=gauss_coef)
                    if non_negative:
                        U[U<0] = 0
                    #
                # V update
                for it in range(admm_iters):
                    Yv_tildes = []
                    for i in range(X.shape[0]):
                        Yv_tildes.append(np.dot(np.outer(Y[i,:].T,X[i,:]),U[i,:]))
                    Yv = np.asarray(Yv_tildes)

                    V, Wy, Zy = admm_CCA_update(Yv, chol_factor_Y, Dy, Wy, Zy, rho, penalty, Y, weights=weights, neighbors=neighbors, gauss_coef=gauss_coef)
                    if non_negative:
                        V[V<0] = 0
                    #
            U_list.append(U)
            V_list.append(V)
            #
    except KeyboardInterrupt:
        print("KeyboardInterrupt has been caught.")
        #
    return U_list, V_list

###------- Clustering functions -------###
def generate_PMD_data(m=100, n_X=10, sigma=1, density=0.5, mean=0, seed=1):
    "Generates data matrix."  
    np.random.seed(seed)      
    u_star = np.random.randn(m) / 3.0
    v_star = np.random.randn(n_X) / 3.0
    X = np.random.normal(mean,sigma,size=((m,n_X))) / 3.0
    X_idxs = np.random.choice(range(n_X), int(density*n_X), replace=False)
    for idx in X_idxs:
        X[:,idx] += v_star[idx]*u_star
        
    return X, u_star, v_star

def generate_cluster_PMD_data(m=[100,100], n_X=20, sigma=0.01, density=0.2, n_clusters=2, means=[0,0], gen_seeds=True, seeds=[], verbose=False):
    X_out = []
    u_stars = []
    v_stars = []
    
    for nc in range(n_clusters):
        if gen_seeds is True:
            seed = np.random.randint(99999)
            seeds.append(seed)
            # print('generating seeds')
        # else:
            # print(seeds)
            # print('NOT generating seeds')

        if verbose == True:
            print(seeds)
        
        X, u_star, v_star = generate_PMD_data(m[nc], n_X, sigma, density, mean=means[nc], seed=seeds[nc])
        X_out.append(X)
        u_stars.append(u_star)
        v_stars.append(v_star)
    return X_out, u_stars, v_stars, seeds

def two_cluster_data(m=[50,50],means=[0,0],n_X=200,sigma=0.075,density=1.0, seed=1, plot=True, intercept=False, gen_seeds=True, seeds='NaN',scale_data=False):
    '''
    Generates two clusters in n_X dimensions with m[0],m[1] observations per class.  
    '''
    # Get clustered data
    X_clusters, u_true, v_true, _ = generate_cluster_PMD_data(m, n_X, sigma, density, 2, means=means, gen_seeds=gen_seeds, seeds=seeds) 
    X_c = np.vstack(X_clusters)
    true_clusters = np.repeat([0,1],m)
    #
    if scale_data:
        scaler = StandardScaler()
        scaler.fit(X_c)
        X_c = scaler.transform(X_c)
    #
    if plot:
        plt.figure(figsize=(6,6))
        plt.scatter(X_clusters[0][:,0],X_clusters[0][:,1], c='darkblue')
        plt.scatter(X_clusters[1][:,0],X_clusters[1][:,1], c='darkorange')
        plt.axis("off")
	#        
    if intercept:
        X_c = np.hstack((X_c,np.ones((X_c.shape[0],1))))
    #
    return X_c, true_clusters

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

from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.optimize import linprog
from scipy.optimize import linear_sum_assignment as linear_assignment
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
    nmi, ari, ri, mse = calculate_scores(labels, true_clusters)
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
    nmi, ari, ri, mse = calculate_scores(labels, true_clusters)
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
    nmi, ari, ri, mse = calculate_scores(labels, true_clusters)
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
    nmi, ari, ri, mse = calculate_scores(labels, true_clusters)
    # Calculate accuracy
    conf_mat_ord = confusion_matrix_ordered(labels,true_clusters)
    acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
    #
    return labels, ari, nmi, acc


###------- PLOTTING FUNCTIONS -------###
from matplotlib import cm
def path_plot(coefficient_arr, penalty_list,plot_range=[0,-1],cut_vars=False):
    # Crop x axis (e.g, to remove 'burn-in' period at beginning)
    coefficient_arr = coefficient_arr[plot_range[0]:plot_range[1],:,:]
    penalty_list = penalty_list[plot_range[0]:plot_range[1]]
    if cut_vars is True:
        coefficient_arr = coefficient_arr[:,:,[1,2,coefficient_arr.shape[2]-1]]

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
        