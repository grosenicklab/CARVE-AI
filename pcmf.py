import numpy as np
#import mosek
import cvxpy as cp
import matplotlib.pyplot as plt
import time

from itertools import combinations 

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering

# to use first run 'python setup.py build_ext --inplace'
from admm_utils import prox as cprox
# from p3ca import cluster_metrics

from numba import jit, prange

def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60) 
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))
    
def tic():
    global _start_time 
    _start_time = time.time()
## !! 

def loss_fn(X, beta):
    return -1*cp.sum(cp.matmul(X,beta))

def mse(X, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, beta).value

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols

def mse(X, beta):
    return (1.0 / X.shape[0]) * loss_fn(X, beta).value

def group_regularizer(beta, num_var, num_obs):
    y_index = vec_diff_type1(num_var, num_obs)
    diff = beta[y_index[:,0]]-beta[y_index[:,1]]
    diff_mat = cp.reshape(diff, (int((num_obs-1)*num_obs/2), num_var))   
    return 1.0/(num_var*(num_obs-1)*num_obs/2) * cp.mixed_norm(diff_mat,p=2,q=1)

# slightly faster, but less readable
def vec_diff_type1(p,n):
    comb_list = list(combinations(range(n),2))
    combs_arr = np.array(comb_list)
    x_index = np.array(list(range(p*len(comb_list)))).astype(int)
    y_index = np.zeros(((p*len(comb_list),2))).astype(int)
    #     A1 = np.array(list(range(p)) * len(comb_list))
    A1 = np.tile(np.array(range(p)),len(comb_list)).astype(int)
    A2 = np.repeat(combs_arr[:,0]*p,p).astype(int)
    A3 = np.repeat(combs_arr[:,1]*p,p).astype(int)
    y_index[:,0] = list(A1 + A2)
    y_index[:,1] = list(A1 + A3)
    return y_index

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

def chol_D(num_obs, num_vars, rho, reg_scale=1.0):
    D = sparse_D(num_obs, num_vars)
    return D, cholesky_AAt(np.sqrt(rho) * D.T, beta=1.0*reg_scale)

def group_soft_threshold(vec,alpha):
    if np.linalg.norm(vec) > alpha:
        return vec - alpha*vec/np.linalg.norm(vec) 
    else:
        return np.zeros_like(vec)
    
def prox(V, lamb, rho, w=None):
    '''
    Proximal operator for group sparse thresholding.
    '''
    if w is None:
        w = np.ones(V.shape[0])
    V_prox = []
    for i in range(V.shape[0]):
        alpha = w[i]*lamb/rho
        V_prox.append(group_soft_threshold(V[i,:],alpha))
    return np.asarray(V_prox)  

def l2_ball_proj(X):
    '''
    Normalize rows of X to satisfy $\|X_{i,\cdot}\|_2^2 \leq 1$.
    '''
    def row_norm(vec): 
        return np.sqrt(1./np.sum(vec**2)) * vec
    return np.apply_along_axis(row_norm,1,X)

def objective_fn(X, beta, lambd, l2_reg, num_var, num_obs, min_path_X=None, reg_type='group'):
    if lambd.value == 0:
        mat = loss_fn(X, beta) + l2_reg*cp.norm2(beta)**2
    else:
        if reg_type == 'group':
            if min_path_X is None:
                reg = group_regularizer(beta, num_var, num_obs)
            else:
                reg = group_regularizer_approx(beta, num_var, num_obs, min_path_X)
        elif reg_type == 'L1':
            reg = L1_regularizer(beta, num_var, num_obs)
        else:
            return ValueError('reg_type value error')
        mat = loss_fn(X, beta) + lambd*reg + l2_reg*cp.norm2(beta)**2
    return mat

def objective_fn_u(X, beta):
    mat = loss_fn(X, beta)
    return mat

def objective_fn_u2(X, beta, penalty):
    D = sparse_D(X.shape[0],1)
    mat = loss_fn(X, beta) + penalty*cp.norm2(D@beta)
    return mat

###################### ADMM subproblems for u #######################
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

    Note: we may want to relax convergence criteria for subproblem (at least in early iterations).                       '''
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

def clusterpath_genLasso_subproblem_u(X_list, num_var, verb=True):
    '''Generalized Lasso to solve for U'''    #
    from rpy2.robjects.packages import importr
    base = importr('base')
    utils = importr('utils')
    conflicted = importr('conflicted')
    HDclassif = importr('genlasso')
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    rpy2.robjects.r('''
            # create a function `f`
            f = function(Xu_tildes,X,D,penalty) {
                out = genlasso(Xu_tildes,X,D)
                betas = coef(out, lambda=penalty)
                return(betas)
            }
            # call the function `f` with argument value 3
            ''')
    r_f = rpy2.robjects.globalenv['f']
    u = r_f(Xu_tildes,X,D,penalty)
    #

#def clusterpath_PMD_subproblem(X_list, num_var, lambda_val, l2_val, min_path_X=None, verb=True):
#    '''
#    Solves one subproblem in the alternating PMD-like pathwise clustered algorithm 
#    for one value of lambda.
#    
#    Note: we may want to relax convergence criteria for subproblem (at least in early iterations).
#    '''
#    X = np.hstack(X_list) # XXX add check - vector num_var*num_obs long
#    num_obs = int(len(X)/num_var)
#    beta = cp.Variable(num_var*num_obs)
#    lambd = cp.Parameter(nonneg=True)
#    l2_reg = cp.Parameter(nonneg=True)
#    constraints = []
#    for i in range(num_obs):
#        constraints += [cp.norm2(beta[(i*num_var):((i+1)*num_var)])**2 <= 1] 
#    problem = cp.Problem(cp.Minimize(objective_fn(X, beta, lambd, l2_reg, num_var, num_obs, min_path_X)), constraints)
#
#    lambd.value = lambda_val
#    l2_reg.value = l2_val
#    mosek_params = {} #{'MSK_DPAR_MIO_TOL_ABS_GAP':1e-1}
#    problem.solve(solver='MOSEK',verbose=verb,warm_start=True,mosek_params=mosek_params)
#    err = mse(X, beta)
#    return beta.value, err

############## NUMBA functions ##################
# from numba import jit, prange

# @jit(nopython=True, parallel=True, fastmath=True)
# def prox_numba_arr(V_prox, V, lamb, rho, w=None):
#     '''
#     Numba formulation of Proximal operator for group sparse thresholding.
#     '''
#     if w is None:
#         w = np.ones(V.shape[0])
#     for i in prange(V.shape[0]):
#         alpha = w[i]*lamb/rho
#         # group_soft_threshold
#         vec_norm = np.linalg.norm(V[i,:])
#         if vec_norm > alpha:
#             V_prox[i, :] = V[i,:] - alpha*V[i,:]/vec_norm
#         else:
#             V_prox[i, :] = np.zeros_like(V[i,:])  
#     return V_prox

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
@jit(nopython=True, parallel=True, fastmath=True) 
def nb_subtract(M1, M2):
    return M1-M2

@jit(nopython=True, parallel=True, fastmath=True)
def nb_add(M1, M2):
    return M1+M2


########################### ADMM helper functions ##########################

def get_weights(X, gauss_coef=0.5, neighbors=None):
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

def get_subject_singular_vals(X, u_list, V_list):
    d_list = []
    for i in range(len(u_list)):
        ds = []
        for j in range(V_list[0].shape[0]):
            ds.append(np.dot(u_list[i].T,np.dot(X,V_list[i][j,:])))
        d_list.append(np.asarray(ds))
    return d_list

def admm_V_update(X, penalty, rho, V=None, W=None, Z=None, D=None, chol_factor=None, V_iters=10):
    
    if chol_factor is None:
        D, chol_factor = chol_D(X.shape[0], X.shape[1], rho)
    if V is None:
        V = X.copy()
    if W is None:
        W = D*X
    if Z is None:
        Z = D*X

    for it in range(V_iters): 
        V = chol_factor(X + rho*D.T*(W - Z))
        W = prox(D*V+Z, penalty, rho, w)
        Z = Z + D*V - W
        V = l2_ball_proj(V)
        
    return V

############################ Main ADMM problem functions ###########################

def pcmf_approx_V(X, penalty_list, rho=1.0, admm_iters = 5, verb=False, weights=False, neighbors=None, gauss_coef=0.5,  print_progress=True, parallel=False, output_file='NaN'):
    '''
    Runs the ADMM PCMF problem with convex clustering on the right singular vectors only.
    '''
    print('weights: '+str(weights), 'neighbors: '+str(neighbors), 'gauss_coef: '+str(gauss_coef), 'rho: '+str(rho))
    # Initialize
    if weights is False:
        weights = get_weights(X,gauss_coef=0.0)
    else:
        weights = get_weights(X,gauss_coef=gauss_coef, neighbors=neighbors)
    D, chol_factor = chol_D(X.shape[0], X.shape[1], rho)
    V = X.copy()
    W = Z = D*X
    V_list = []
    u_list = []
    s_list = []
    
    # First run initial on v_init
    Xu_tildes = []
    for i in range(X.shape[0]):
        Xu_tildes.append(np.dot(X[i,:],V[i,:]))  
    u = clusterpath_PMD_subproblem_u(Xu_tildes, 1, verb) 
    u.shape = (len(u),1)

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
            V = chol_factor(Xv + rho*D.T*(W - Z))
            W = cprox(D*V+Z, penalty, rho, weights)
            Z = Z + D*V - W
        V = l2_ball_proj(V)
        V_list.append(V)
        
        # Update u
        Xu_tildes = []
        for i in range(X.shape[0]):
            Xu_tildes.append(np.dot(X[i,:],V[i,:]))  
        u = clusterpath_PMD_subproblem_u(Xu_tildes, 1, verb) 
        u.shape = (len(u),1)
        u_list.append(u)

        # TODO: add s output

    return V_list, u_list, s_list

def pcmf_approx_uV(X, penalty_list, rho=1.0, admm_iters = 5, verb=False, weights=False, neighbors=None, gauss_coef=0.5,  print_progress=True, parallel=False, output_file='NaN', non_negative=False, numba=False):
    '''
    Relaxation of full PCMF problem to iterate between clustering on u and clustering on V. 
    '''
    print('weights: '+str(weights), 'neighbors: '+str(neighbors), 'gauss_coef: '+str(gauss_coef), 'rho: '+str(rho))
    # Initialize                                                                                                     
    if weights is False:
        weights = get_weights(X,gauss_coef=0.0)
    else:
        weights = get_weights(X,gauss_coef=gauss_coef, neighbors=neighbors)
    D, chol_factor = chol_D(X.shape[0], X.shape[1], rho)
    V = X.copy()
    for i in range(V.shape[0]):
        V[i,:] = np.mean(X,axis=0)
    W = Z = W2 = Z2 = D*X
    d = np.ones(X.shape[0])
    V_list = []
    u_list = []
    s_list = []
#    accel_gamma = 0.9 # TODO: add and test acceleration

    # First run initial on v_init                                                                                    
    Xu_tildes = []
    for i in range(X.shape[0]):
        Xu_tildes.append(np.dot(X[i,:],V[i,:]))
    u = clusterpath_PMD_subproblem_u(Xu_tildes, 1, verb)
    u.shape = (len(u),1)

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
        #            W_prev = W.copy()
                    W = cprox(D*V+Z, penalty, rho, weights)
                else:
                    V = chol_factor(nb_add(Xv, rho*D.T*nb_subtract(W,Z)))
                    W = prox_numba_arr(np.zeros_like(W), D*V+Z, penalty, rho, weights)

                Z = Z + D*V - W
    #            Z_prev = Z.copy()
    #            W = W + accel_gamma*(W - W_prev)
    #            Z = Z + accel_gamma*(Z - Z_prev)
    #        accel_gamma = accel_gamma/(i+3)
            V = l2_ball_proj(V)
            if non_negative:
                V[V<0] = 0
            V_list.append(V)

            # Update u                                                                                                   
            Xu_tildes = []
            for i in range(X.shape[0]):
                Xu_tildes.append(np.dot(X[i,:],V[i,:]))
            Xu = np.asarray(Xu_tildes)
            try:
                u = clusterpath_PCMF_subproblem_u(Xu_tildes, 1, penalty, verb)
            except:
                print("PCMF subproblem is not defined for single cluster u, using PMD subproblem.")
                u = clusterpath_PMD_subproblem_u(Xu_tildes, 1, verb)
            u.shape = (len(u),1)
            s = u*Xv
            #u = u/s
            u_list.append(u)
            s_list.append(d)

    except KeyboardInterrupt:
        print("KeyboardInterrupt has been caught.")

    return V_list, u_list, s_list


def pcmf_full(X, penalty_list, problem_rank=1, rho=1.0, admm_iters = 5, verb=False, weights=False, neighbors=None, gauss_coef=2.0,print_progress=True, parallel=False, output_file='NaN',factor_type='SVD', numba=False):
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

#    accel_gamma = 0.9 # TODO: add and test acceleration
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
    #            G_prev = G.copy()
    #            U_prev = U.copy()
    #            s_prev = s.copy()
    #            Vh_prev = Vh.copy()
    #            Z1_prev = Z1.copy()
    #            Z2_prev = Z2.copy()
                if numba is False:
                    A = chol_factor(X + rho*D.T*(G - Z1) + rho*(np.dot(U*s,Vh) - Z2))
                    G = cprox(D*A+Z1, penalty, rho, weights)
                else:
                    A = chol_factor(X + rho*D.T*(G - Z1) + rho*(np.dot(U*s,Vh) - Z2))
                    # A = chol_factor(nb_add(X, rho*D.T*nb_subtract(G,Z1) + rho*(np.dot(U*s,Vh) - Z2)))
                    G = prox_numba_arr(np.zeros_like(G), D*A+Z1, penalty, rho, weights)

                if factor_type == 'SVD':
                    U, s, Vh = SVD(A + Z2, problem_rank)
                elif factor_type == 'NMF':
                    A[A<0] = 0
                    A_Z2 = A + Z2
                    A_Z2[A_Z2<0] = 0
                    U, Vh = NMF(A_Z2, problem_rank)
                    s = np.ones(problem_rank)

                Z1 = Z1 + rho*(D*A - G)
                Z2 = Z2 + rho*(A - np.dot(U*s,Vh))
                
    #            G = G + accel_gamma*(G - G_prev)
    #            U = U + accel_gamma*(U - U_prev)
    #            s = s + accel_gamma*(s - s_prev)
    #            Vh = Vh + accel_gamma*(Vh - Vh_prev)
    #            Z1 = Z1 + accel_gamma*(Z1 - Z1_prev)
    #            Z2 = Z2 + accel_gamma*(Z2 - Z2_prev)
    #        accel_gamma = accel_gamma/(i+3)
            
            A_list.append(A)
            V_list.append(Vh)
            s_list.append(s)
            U_list.append(U)

    except KeyboardInterrupt:
        print("KeyboardInterrupt has been caught.")

    return A_list, U_list, s_list, V_list

from sklearn.utils.extmath import randomized_svd
def pcmf_full_consensus(X_all, penalty_list, problem_rank=1, rho=1.0, admm_iters = 5, verb=False, weights=False, neighbors=None, gauss_coef=2.0,print_progress=True, parallel=False, output_file='NaN',factor_type='SVD', split_size=10):
    '''
    Fits fully constrained PCMF problem iterating between convex clustering and an SVD of rank 'problem_rank' 
    using ADMM updates.
    '''
    print('weights: '+str(weights), 'neighbors: '+str(neighbors), 'gauss_coef: '+str(gauss_coef), 'rho: '+str(rho))
    #
    rho1 = rho2 = rho
#     rho1 = rho - 0.02
#     rho2 = rho

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
#     print(M)
    
#     U, s, Vh = SVD(X_all, problem_rank)
    U, s, Vh = randomized_svd(X_all, n_components=problem_rank,random_state=1234)
    for X, idx in zip(X_list, np.arange(len(X_list))):
        X_inds = X_list_inds[idx]
#         Z2[idx] = X_all[X_inds,:] #np.dot(U[X_inds]*s,Vh)
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
                #
                # consensus
                As = np.vstack(A)
                Z2s = np.vstack(Z2)
#                 U, s, Vh = SVD(As + Z2s, problem_rank)
#                 print("Vh.shape:",Vh.shape)
#                 print("Vh:",Vh[0,0:3])
                B = As + Z2s
#                 U_b, s_b, Vh_b = SVD(B, problem_rank)
                U_b, s_b, Vh_b = randomized_svd(B, n_components=problem_rank,random_state=1234)
                U_a__s_a = np.dot(B + M, Vh_b.T)
#                 print("Vh_b:",Vh_b[0,0:3])
#                 print("U before:",U[0:3,0].T)
                Vh = Vh_b
                s = s_b
                U = U_a__s_a / s
                U = np.sqrt(1./np.sum(U**2)) * U
#                 print("U_a__s_a:",U_a__s_a[0:3,0].T)
#                 print("U after:",U[0:3,0].T)
                #
                Z2_update = rho2*(As - np.dot(U*s,Vh))
                #
                for X, idx in zip(X_list, np.arange(len(X_list))):
                    X_inds = X_list_inds[idx]
                    Z1[idx] = Z1[idx] + rho1*(D*A[idx] - G[idx])
                    Z2[idx] = Z2[idx] + Z2_update[X_inds,:] # should this be Z2_update[X_inds,:]?
#                     Z2[idx] = Z2[idx] + Z2_update[X_inds] # should this be Z2_update[X_inds,:]?

            A_list.append(np.vstack(A))
            V_list.append(np.vstack(Vh))
            s_list.append(np.vstack(s))
            U_list.append(np.vstack(U))
            
    except KeyboardInterrupt:
        print("KeyboardInterrupt has been caught.")
        
    return A_list, U_list, s_list, V_list

def pcmf_full_consensus_OLD(X_all, penalty_list, problem_rank=1, rho=1.0, admm_iters = 5, verb=False, weights=False, neighbors=None, gauss_coef=2.0,print_progress=True, parallel=False, output_file='NaN',factor_type='SVD', split_size=10):
    '''
    Fits fully constrained PCMF problem iterating between convex clustering and an SVD of rank 'problem_rank' 
    using ADMM updates.
    '''
    print('Running old version of pcmf full consensus')
    print('split_size:',str(split_size),'weights: '+str(weights), 'neighbors: '+str(neighbors), 'gauss_coef: '+str(gauss_coef), 'rho: '+str(rho))
    #
    rho1 = rho2 = rho

    print("rho1:",rho1,"rho2:",rho2)
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
    
    U, s, Vh = SVD(X_all, problem_rank)
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
                #
                # consensus
                As = np.vstack(A)
                Z2s = np.vstack(Z2)
                B = As + Z2s
                U_b, s_b, Vh_b = SVD(B, problem_rank)
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

            A_list.append(np.vstack(A))
            V_list.append(np.vstack(Vh))
            s_list.append(np.vstack(s))
            U_list.append(np.vstack(U))
            
    except KeyboardInterrupt:
        print("KeyboardInterrupt has been caught.")
        
    return A_list, U_list, s_list, V_list

######### UPDATED CONSENUS ########
def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.
    Parameters
    ----------
    u : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
    v : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
        The input v should really be called vt to be consistent with scipy's
        output.
    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.
    Returns
    -------
    u_adjusted : ndarray
        Array u with adjusted columns and the same dimensions as u.
    v_adjusted : ndarray
        Array v with adjusted rows and the same dimensions as v.
    """
    if u_based_decision:
#         print("U shape",u.shape)
#         print("V shape",v.shape)
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
#         print("signs shape",signs.shape, "max", np.abs(u))
#         print(signs)
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
#         print("U shape",u.shape)
#         print("V shape",v.shape)
        max_abs_rows = np.argmax(np.abs(v), axis=1)
#         print("max_abs_rows shape",max_abs_rows.shape)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
#         print("signs shape",signs.shape)
#         signs = np.atleast_2d(np.sign(v[range(v.shape[0]), max_abs_rows])).T
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v

def SVD2(X,return_rank=2):
#     U, s, Vh = np.linalg.svd(X,full_matrices=False)
    U, s, Vh = randomized_svd(X, n_components=return_rank,random_state=1234)
    U = U[:,0:return_rank]
    Vh = Vh[0:return_rank,:]
    s = s[0:return_rank]
    
    U, Vh = svd_flip(U, Vh, u_based_decision=True)
    return U,s,Vh


from sklearn.utils.extmath import randomized_svd
def pcmf_full_consensus2(X_all, penalty_list, problem_rank=1, rho=1.0, admm_iters = 5, verb=False, weights=False, neighbors=None, gauss_coef=2.0,print_progress=True, parallel=False, output_file='NaN',factor_type='SVD', split_size=10):
    '''
    Fits fully constrained PCMF problem iterating between convex clustering and an SVD of rank 'problem_rank' 
    using ADMM updates.
    '''
    print('weights: '+str(weights), 'neighbors: '+str(neighbors), 'gauss_coef: '+str(gauss_coef), 'rho: '+str(rho))
    #
    rho1 = rho2 = rho
#     rho1 = rho - 0.02
#     rho2 = rho

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

    U, s, Vh = SVD2(X_all, problem_rank)
    Vhorig = Vh
#     U, s, Vh = randomized_svd(X_all, n_components=problem_rank,random_state=1234)
    
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
        
        Ginit = np.zeros_like(Ginit)
        Z1init = np.zeros_like(Z1init)
        
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
#     print(M)
    
    for X, idx in zip(X_list, np.arange(len(X_list))):
        X_inds = X_list_inds[idx]
#         Z2[idx] = X_all[X_inds,:] #np.dot(U[X_inds]*s,Vh)
#         Z2[idx] = X.copy() #np.dot(U[X_inds]*s,Vh)
        A[idx] = np.dot(U[X_inds]*s,Vh)
        Z2[idx] = np.dot(U[X_inds]*s,Vh)
    
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
                #
                # consensus
                As = np.vstack(A)
                Z2s = np.vstack(Z2)
#                 U, s, Vh = SVD(As + Z2s, problem_rank)
#                 print("Vh.shape:",Vh.shape)
#                 print("Vh:",Vh[0,0:3])
                B = As + Z2s
                U_b, s_b, Vh_b = SVD2(B, problem_rank)
#                 U_b, s_b, Vh_b = randomized_svd(B, n_components=problem_rank,random_state=1234)
                U_a__s_a = np.dot(B + M, Vh_b.T)
#                 print("Vh_b:",Vh_b[0,0:3])
#                 print("U before:",U[0:3,0].T)
                Vh = Vh_b
                s = s_b
                U = U_a__s_a / s
                U = np.sqrt(1./np.sum(U**2)) * U
#                 print("U_a__s_a:",U_a__s_a[0:3,0].T)
#                 print("U after:",U[0:3,0].T)
#                  for k in range(0,problem_rank):
#                     U[:,k] = U_a__s_a[:,k] / s[k]
#                     U[:,k] = np.sqrt(1./np.sum(U[:,k]**2)) * U[:,k]
#                 print("U_a__s_a:",U_a__s_a[0:3,0].T)
#                 print("U after:",U[0:3,0].T)
#
                if problem_rank > 0:
                    for k in range(0,problem_rank):
                        if np.round(s[k],1)<=0.2:
                            U[:,k] = 0
                            Vh[k,:] = 0
                            s[k] = 0
                        if penalty==np.inf:
                            U[:,k] = 0
                            Vh[k,:] = 0
                            s[k] = 0
    #                             U, Vh = V_flip(U, Vh, Vh_orig)
#                         if np.sign(Vh[k,0]) != np.sign(Vhorig[k,0]):
                        if np.sign(Vh[k,0]) == -1:
                            U[:,k] = U[:,k] * -1
                            Vh[k,:] = Vh[k,:] * -1
#                         for idx in np.arange(1,len(X_list)):
#                             if np.abs(np.mean(U[X_list_inds[0],k])) >= 0.01:
#                                 if np.sign(np.mean(U[X_list_inds[idx],k])) == np.sign(np.mean(U[X_list_inds[0],k])):
#                                     U[X_list_inds[idx],k] = U[X_list_inds[idx],k] * -1
                                
                Z2_update = rho2*(As - np.dot(U*s,Vh))
                #
                for X, idx in zip(X_list, np.arange(len(X_list))):
                    X_inds = X_list_inds[idx]
                    Z1[idx] = Z1[idx] + rho1*(D*A[idx] - G[idx])
                    Z2[idx] = Z2[idx] + Z2_update[X_inds,:] # should this be Z2_update[X_inds,:]?
#                     Z2[idx] = Z2[idx] + Z2_update[X_inds] # should this be Z2_update[X_inds,:]?

            A_list.append(np.vstack(A))
            V_list.append(np.vstack(Vh))
            s_list.append(np.vstack(s))
            U_list.append(np.vstack(U))
            
    except KeyboardInterrupt:
        print("KeyboardInterrupt has been caught.")
        
    return A_list, U_list, s_list, V_list





########################## New cluster fitting method using adjacency matrix #######################
def diff_graph_cluster(Xhat, D, comb_list, num_clusters, thresh_sd=6, pca_clean=True, num_fits=1, verbose=False, use_adjacency=False):
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
    n,p = Xhat.shape

    # Get graph edges from distances, and estimate graph threshold from edge mode centered around zero
    edges = np.sum(np.abs(D*Xhat),axis=1)
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

    if not use_adjacency:
        deg = np.diag(np.sum(A, axis=1))
        K = nx.modularity_matrix(G)
        A = n*np.sqrt(deg)@K@np.sqrt(deg) # From RMT4ML pg 265

    # Apply spectral clustering, taking median of 'num_fits' tryes to get output labels
    out_labels = []
    for f in range(num_fits):
        # Use PCA of A if 'pca_clean' flag set
        if pca_clean:
            uA, sA, vhA = np.linalg.svd(A)
            spectral_clustering = SpectralClustering(n_clusters=num_clusters, random_state=20, affinity="nearest_neighbors", assign_labels="cluster_qr").fit(uA[:,0:num_clusters])
            out_labels.append(spectral_clustering.labels_)
        else:
            spectral_clustering = SpectralClustering(n_clusters=num_clusters, random_state=20, affinity="nearest_neighbors", assign_labels="cluster_qr").fit(A)
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

########################## Predict and project on test data / new data #######################
def PCMF_predict_clusters(X_test, X_train, V, p, true_clusters_train, PCMFtype='Full', true_clusters_test=None):# Get cluster PCA component
    '''Function to take held out test data and project it to PCA component and predict clusters
       Inputs:  X_test: n_test x p (already centered using np.mean(X_train, axis=0))
                X_train: n_train x p  (already centered using np.mean(X_train, axis=0))
                V: #penalties x r x p; list output of PCMF_Full of rank r
                p: penalty index on path
                true_clusters_train: n_train x 1; true cluster labels for training set
                true_clusters_test: n_test x 1; Optional argument; if available, the true labels for test set to get accuracy
    '''
    from scipy.spatial.distance import cdist
    
    XV_c = []
    if PCMFtype is 'Full':
        for cluster in np.unique(true_clusters_train):
            XV_c.append(np.mean((X_train[true_clusters_train==cluster,:] @ np.array(V)[p,:,:].T),0))
        XV_test = (X_test @ np.array(V)[p,:,:].T)
        
        cluster_dist = cdist( XV_test, XV_c )
        
    elif PCMFtype is 'PALS':
        for cluster in np.unique(true_clusters_train):
            XV_c.append(np.mean((X_train[true_clusters_train==cluster,:] @ np.mean(np.array(V)[p,true_clusters_train==cluster,:], axis=0)[:, np.newaxis] ),0) )
        XV_test = (X_test @ np.mean(np.array(V)[p,true_clusters_train==cluster,:], axis=0)[:, np.newaxis])
        
        cluster_dist = []
        for cluster in np.unique(true_clusters_train):
            cluster_dist.append( cdist( XV_tests[cluster], XV_c[cluster:cluster+1] )[:,0] ) 
        cluster_dist = np.array(cluster_dist).T

    else:
        print("Wrong PCMFtype: ", PCMFtype)
        return [], [], [], []
            
    XV_c = np.array(XV_c)

    # Project new subjects and assign to clusters
    true_clusters_test_predict = np.argmin(cluster_dist, 1)

    if len(true_clusters_test)>1:
        # Check alignment to true clusters
        cluster_acc = np.sum(true_clusters_test == true_clusters_test_predict) / len(true_clusters_test)
        print('Test set cluster accuracy:', cluster_acc)
    else:
        cluster_acc = []

    return XV_c, true_clusters_test_predict, XV_test, cluster_acc

########################## Clustering helper functions #######################

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

########### Model Selection functions ############
def cluster_fit(Xhat, num_clusters=2, gamma=2.0,  method='spectral'):
    # Cluster on the rows of Xhat with k=n_clust_previous
    if method == 'spectral':
        # clustering = SpectralClustering(n_clusters = num_clusters, gamma=gamma, \
                                        # assign_labels='discretize',affinity='rbf')
        clustering = SpectralClustering(n_clusters=num_clusters, random_state=0, gamma=gamma).fit(Xhat)
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

def cluster_path(X_c, Xhat_list, penalty_list, gauss_coef, neighbors, verbose=False):
    '''
    Estimate number of clusters and fit quality at each value of the penalty.
    '''
    ics = []
    n_clusts = []
    centroids = []
    n_clust = 1
    D = sparse_D(X_c.shape[0],X_c.shape[1])
    weights = get_weights(X_c, gauss_coef=gauss_coef, neighbors=neighbors)
    #
    notinf_idx = np.where(np.asarray(penalty_list)<np.inf)[0]
    penalty_list = penalty_list[notinf_idx]
    Xhat_list = [Xhat_list[i] for i in range(len(Xhat_list)) if i in notinf_idx]
    for i,Xhat in enumerate(Xhat_list):
        penalty = penalty_list[i]
        if n_clust < Xhat.shape[0]:
            out, n_clust, labels, ic = cluster_forwardstep(Xhat, X_c, D, n_clust, penalty, weights, 
                                                            method='spectral', gamma=gauss_coef, selection='lik', verbose=verbose)
        n_clusts.append(n_clust)
        ics.append(ic)
        centroids.append(out)
    return n_clusts, ics, centroids

from p3ca import convex_clust_df
def cluster_forwardstep(Xhat, X, D, n_clust_previous, penalty, weights, method='spectral', gamma=1.0, selection='bic', verbose=False):
    epsilon = penalty
    #
    # Cluster on the rows of Xhat with k=n_clust_previous
    if method == 'spectral':
        clustering = SpectralClustering(n_clusters = n_clust_previous, gamma=gamma, \
                                        assign_labels='discretize',affinity='rbf')
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
    # Cluster on the rows of V with k=n_clust_previous+1 
    if method == 'spectral':
        clustering = SpectralClustering(n_clusters = n_clust_previous+1, gamma=gamma, \
                                        assign_labels='discretize',affinity='rbf')
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
    loglik1 = np.linalg.norm(X - Xhat1, 2)**2 + penalty*np.sum(weights*np.sum(np.abs(D*Xhat1),axis=1))
    loglik2 = np.linalg.norm(X - Xhat2, 2)**2 + penalty*np.sum(weights*np.sum(np.abs(D*Xhat2),axis=1))
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


############ Data generation functions #############

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

    if scale_data:
        scaler = StandardScaler()
        scaler.fit(X_c)
        X_c = scaler.transform(X_c)

    if plot:
        plt.figure(figsize=(6,6))
        plt.scatter(X_clusters[0][:,0],X_clusters[0][:,1], c='darkblue')
        plt.scatter(X_clusters[1][:,0],X_clusters[1][:,1], c='darkorange' )
        #plt.scatter(X_clusters[2][:,0],X_clusters[2][:,1])
        plt.axis("off")

        #X_c = np.hstack((X_c,np.ones((X_c.shape[0],1)))) # For rows 1,3

#         plt.figure(figsize=(6,6))
#         maxval = np.max(np.abs(X_c))
#         plt.imshow(X_c,aspect='auto',interpolation='nearest',cmap='twilight_shifted',vmin=-1*maxval, vmax=maxval)  
#         plt.axis('off')
        
    if intercept:
        X_c = np.hstack((X_c,np.ones((X_c.shape[0],1))))
    
    return X_c, true_clusters

def two_cluster_data_outputUV(m=[50,50],means=[0,0],n_X=200,sigma=0.075,density=1.0, seed=1, plot=True, intercept=False, gen_seeds=True, seeds='NaN',scale_data=False):
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
        plt.scatter(X_clusters[1][:,0],X_clusters[1][:,1], c='darkorange' )
        #plt.scatter(X_clusters[2][:,0],X_clusters[2][:,1])
        plt.axis("off")
    #
    if intercept:
        X_c = np.hstack((X_c,np.ones((X_c.shape[0],1))))
    #
    return X_c, true_clusters, u_true, v_true

################## Experiments ###################

def run_numerical_experiments(dir_path='../results/',data_type='test', intercept = True, cluster_sizes = [50,50], sigma = 0.075, density = 0.5, num_vars = 5, cluster_means = [-0.2,0.2], r = 1, rho = 1.0, weights='Gaussian', penalty_list = np.concatenate((np.repeat(np.inf,10), np.exp(np.linspace(-50,0,150))[::-1]), axis=0), parallel=False, output_file='NaN'):
    import numpy as np
    import mosek
    import cvxpy as cp
    import matplotlib.pyplot as plt
    import time
    from itertools import combinations
    from admm_utils import prox as cprox
    from pcmf import pcmf_admm, two_cluster_data
    
    save_path = dir_path+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'.npz'
    print(save_path)

    seeds = [r,r+100]
    X_c, true_clusters = two_cluster_data(m=cluster_sizes, means=cluster_means, n_X=num_vars, sigma=sigma, density=density, 
                                       gen_seeds=False, seeds=seeds, plot=False, scale_data=False, intercept=intercept)
    tic = time.time()
    V, u = pcmf_admm(X_c, penalty_list, rho=rho, weights=weights, print_progress=True, parallel=parallel, output_file=output_file)
    toc = time.time() - tic
    np.savez(save_path, true_clusters=true_clusters, means=cluster_means, n_X=num_vars, sigma=sigma, density=density, seeds=seeds, V=np.asarray(V), U=np.asarray(u), runtime=toc, penalty_list=np.asarray(penalty_list), rho=rho, weights=weights) 

def run_numerical_experiments_aistats(dir_path='../results/',data_type='test', pcmf_type='test',intercept = True, cluster_sizes = [50,50], sigma = 0.075, density = 0.5, num_vars = 5, cluster_means = [-0.2,0.2], r = 1, rho = 1.0, weights='Gaussian', gauss_coef=2.0, neighbors = 25, admm_iters = 5, penalty_list = np.concatenate((np.repeat(np.inf,10), np.exp(np.linspace(-50,0,150))[::-1]), axis=0), parallel=False):
    import numpy as np
    import mosek
    import cvxpy as cp
    import matplotlib.pyplot as plt
    import time
    from itertools import combinations
    from admm_utils import prox as cprox
    from pcmf import pcmf_full, pcmf_approx_uV, pcmf_approx_V, two_cluster_data
    
    print('run: '+str(r))
    seeds = [r,r+100]
    X_c, true_clusters = two_cluster_data(m=cluster_sizes, means=cluster_means, n_X=num_vars, sigma=sigma, density=density, 
                                       gen_seeds=False, seeds=seeds, plot=False, scale_data=True, intercept=intercept)
    tic = time.time()
    pcmf_type = 'pcmf_full'
    save_path = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'.npz'
    output_file = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'.txt'
    print(save_path)
    A, U, S, V = pcmf_full(X_c, penalty_list, rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=output_file)
    toc = time.time() - tic
    np.savez(save_path, true_clusters=true_clusters, means=cluster_means, n_X=num_vars, sigma=sigma, density=density, seeds=seeds, A=A, V=V, U=U, S=S, runtime=toc, penalty_list=np.asarray(penalty_list), rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, ntercept=intercept) 

    tic = time.time()
    pcmf_type = 'pcmf_approx_uV'
    save_path = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'.npz'
    output_file = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)+'.txt'
    print(save_path)
    V, U, S = pcmf_approx_uV(X_c, penalty_list, rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=parallel, output_file=output_file)
    toc = time.time() - tic
    np.savez(save_path, true_clusters=true_clusters, means=cluster_means, n_X=num_vars, sigma=sigma, density=density, seeds=seeds, V=V, U=U, S=S, runtime=toc, penalty_list=np.asarray(penalty_list), rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, scale_data=True, ntercept=intercept) 



################# Plotting Functions ################
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf', bbox_inches='tight')
    pp.close()


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

from sklearn.cluster import KMeans
def plot_cluster_assignments(X, true_clusters, coefficient_arr, num_clusters, skip=5, plot_idx=None, var_idx=[0,1]):
    if plot_idx is None:
        for i in range(coefficient_arr.shape[0]):
            if np.mod(i,skip) == 0:
                #kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(coefficient_arr[i,:,:])
                #cs = kmeans.labels_
                dpgmm = mixture.BayesianGaussianMixture(n_components=num_clusters, covariance_type='full', \
                                                        weight_concentration_prior_type='dirichlet_process')
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
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(coefficient_arr[plot_idx,:,:])
        cs = kmeans.labels_

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
            
if __name__ == '__main__':
    pass

######### Compare Xhat #########
def split_half_clusterpath(r, pcmf_type, data_type, X, num_vars, density, penalty_list, best_idx, num_splits=2, true_clusters=None, admm_iters=5, rho = 1.0, weights='Gaussian', gauss_coef=2.0, neighbors=25, verb=False,dir_path='results/'):
    """
    Split half analysis to compare coefficient stability between random cluster splits
    Params:
        num_splits = number of times to iterate splitting the data in half and then calculating statistics
        true_clusters = np.array of length num_obs
        use_true_clusters = indicator of whether to use true_clusters to split each cluser in half
        seed * set to random and save seeds
    Return:
        cluster_sim = cos similarity between splits
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.stats.stats import pearsonr
    from sklearn.model_selection import train_test_split
    from pcmf import pcmf_full, pcmf_approx_uV
    seeds = []
    U_list_trains = []
    U_list_tests = []
    V_list_trains = []
    V_list_tests = []
    u_path_sims = []
    v_path_sims = []
    num_clusters = 2
    for i in range(num_splits):
        print("PCMF split ",i)
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        # Split each cluster's samples in half
        for k in range(2):
            X_c = X[true_clusters==k,:]
            X_tr, X_te = train_test_split(X_c, test_size=0.5, random_state=1234+k)
            X_train.append(X_tr)
            X_test.append(X_te)
        #
        X_train = np.vstack(X_train)
        X_test = np.vstack(X_test)
        true_clusters_train = np.concatenate([np.zeros(int(X_train.shape[0]/2)),np.ones(int(X_train.shape[0]/2))])
        true_clusters_test = np.concatenate([np.zeros(int(X_test.shape[0]/2)),np.ones(int(X_test.shape[0]/2))])
        #
        penalty_list = np.concatenate((np.repeat(np.inf,10),np.exp(np.linspace(-10,10,150))[::-1]),axis=0)
        # Calculate U_list and V_list in training data along penalty path
        # print("Calculating training data coefficients")
        if pcmf_type == 'pcmf_full':
            tic = time.time()
            model_name = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)
            print(model_name)
            A_train, U, S, V = pcmf_full(X_train, penalty_list, rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=False, output_file='')
            A_test, U, S, V = pcmf_full(X_test, penalty_list, rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=False, output_file='')
            toc = time.time() - tic
            print('Time elapsed: '+str(toc))
            v_sim = []
            for k in range(num_clusters):
                v_coeff_train = np.mean(A_train[best_idx][true_clusters_train==k,:],axis=0)
                v_coeff_test = np.mean(A_test[best_idx][true_clusters_test==k,:],axis=0)
                # set as absolute value of cosine similarity
                v_sim.append(float(np.diag(1 - sp.distance.cdist([v_coeff_train], [v_coeff_test], 'cosine'))))
            # Xhat_list_best = Xhat_list[best_idx][:,:-1]
            cluster_sim = v_sim
        elif pcmf_type == 'pcmf_approx_uV':
            tic = time.time()
            model_name = dir_path+pcmf_type+'_'+'synthetic_run'+str(r)+'_'+data_type+'_p'+str(num_vars)+'_density'+str(density)+'_gausscoef'+str(gauss_coef)+'_neighbors'+str(neighbors)
            print(model_name)
            V_train, U_train, S = pcmf_approx_uV(X_train, penalty_list, rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=False, output_file='')
            V_test, U_train, S = pcmf_approx_uV(X_test, penalty_list, rho=rho, admm_iters = admm_iters, weights=weights, gauss_coef=gauss_coef, neighbors=neighbors, print_progress=True, parallel=False, output_file='')
            toc = time.time() - tic
            print('Time elapsed: '+str(toc))
            v_sim = []
            for k in range(num_clusters):
                v_coeff_train = np.mean(V_train[best_idx][true_clusters_train==k,:],axis=0)
                v_coeff_test = np.mean(V_test[best_idx][true_clusters_test==k,:],axis=0)
                # set as absolute value of cosine similarity
                v_sim.append(float(np.diag(1 - sp.distance.cdist([v_coeff_train], [v_coeff_test], 'cosine'))))
            #
            cluster_sim = v_sim
    #
    return np.mean(np.abs(cluster_sim)), cluster_sim

def flip_sign(obs_coeff_true, obs_coeff):
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

def compare_coefficients_clusterpath(coeff_pred, coeff_true, true_clusters):
    """
    Calculates cosine similary between true coefficients and predicted coefficients, with Procrustes rotation
    Params:
        coeff_true = list of true coefficients, shape = (num_clusters, num_variables)
        true_clusters
        coefficients_path
        path_length
    Returns:
        path_coeff_fit = predicted coefficients after Procrustes rotation
        path_coeff_diff = disparity between predicted coefficients before and after Procrustes
        path_cluster_sim = cosine similarity between predicted coefficients (after Procrustes) and true coefficients
        path_cluster_sim_orig = cosine similarity between predicted coefficients (before Procrustes) and true coefficients
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.stats.stats import pearsonr
    cluster_sim = compare_coefficients_clusterpath(Xhat, X_hat_true, true_clusters)
    return cluster_sim



    coeff_true_all = []
    for i in range(len(true_clusters)):
        cluster_idx = true_clusters[i]
        coeff_true_all.append(coeff_true[cluster_idx]) 
    #
    coeff_true_all = np.array(coeff_true_all)
    #
    cluster_sim = np.diag(1 - sp.distance.cdist(coeff_true_all, coeff_pred, 'cosine'))
    #
    return cluster_sim

def compare_coefficients_pca(X, X_hat_true, true_clusters, num_vars, intercept):
    from scipy.optimize import linear_sum_assignment
    from scipy.stats.stats import pearsonr
    # Fit PCA-kmeans
    u,d,vh = np.linalg.svd(X, full_matrices=False)
    Xhat = u[:,0].reshape((len(true_clusters), 1))*vh[0,:].reshape((1, X.shape[1]))
    if intercept==True:
        Xhat = Xhat[:,:-1]
    #
    cluster_sim = compare_coefficients_clusterpath(Xhat, X_hat_true, true_clusters)
    return cluster_sim


def split_half_pca(X, true_clusters, num_splits=2):
    """
    Split half analysis to compare coefficient stability between random cluster splits
    Params:
        num_splits = number of times to iterate splitting the data in half and then calculating statistics
        true_clusters = np.array of length num_obs
        use_true_clusters = indicator of whether to use true_clusters to split each cluser in half
        seed * set to random and save seeds
    Return:
        cluster_sim = cos similarity between splits
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.stats.stats import pearsonr
    from sklearn.model_selection import train_test_split
    from pcmf import pcmf_full, pcmf_approx_uV
    seeds = []
    num_clusters = 2
    for i in range(num_splits):
        print("PCA split ",i)
        X_train = []
        X_test = []
        # Split each cluster's samples in half
        for k in range(2):
            X_c = X[true_clusters==k,:]
            X_tr, X_te = train_test_split(X_c, test_size=0.5, random_state=1234+k)
            X_train.append(X_tr)
            X_test.append(X_te)
        #
        X_train=np.vstack(X_train)
        X_test=np.vstack(X_test)
        true_clusters_train = np.concatenate([np.zeros(int(X_train.shape[0]/2)),np.ones(int(X_train.shape[0]/2))])
        true_clusters_test = np.concatenate([np.zeros(int(X_test.shape[0]/2)),np.ones(int(X_test.shape[0]/2))])
        #
        # Calculate U_list and V_list in training data along penalty path
        # print("Calculating training data coefficients")
        u,d,vh = np.linalg.svd(X_train, full_matrices=False)
        Xhat_train = u[:,0].reshape((len(true_clusters_train), 1))*vh[0,:].reshape((1,X.shape[1]))
        u,d,vh = np.linalg.svd(X_test, full_matrices=False)
        Xhat_test = u[:,0].reshape((len(true_clusters_test), 1))*vh[0,:].reshape((1,X.shape[1]))
        v_sim = []
        for k in range(num_clusters):
            v_coeff_train = np.mean(Xhat_train,axis=0)
            v_coeff_test = np.mean(Xhat_test,axis=0)
            # set as absolute value of cosine similarity
            v_sim.append(float(np.diag(1 - sp.distance.cdist([v_coeff_train], [v_coeff_test], 'cosine'))))
        # 
        cluster_sim = v_sim
    #
    return np.mean(np.abs(cluster_sim)), cluster_sim
