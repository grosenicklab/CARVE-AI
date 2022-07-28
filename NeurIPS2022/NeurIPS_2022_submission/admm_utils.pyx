import numpy as np
cimport numpy as np
cimport cython

## Compile-time datatypes
DTYPE_float = np.float64
ctypedef np.float_t DTYPE_float_t

DTYPE_int = np.int32
ctypedef np.int_t DTYPE_int_t

@cython.boundscheck(False)
@cython.wraparound(False) 
def group_soft_threshold(np.ndarray[DTYPE_float_t, ndim=1] vec, alpha):
    cdef int n = vec.shape[0]
    cdef double vec_norm = 0.0

    for i in range(n):
        vec_norm += vec[i]*vec[i]
    vec_norm = np.sqrt(vec_norm)
    if vec_norm > alpha:
        for i in range(n):
            vec[i] = vec[i] - alpha * vec[i] / vec_norm
        return vec
    else:
        for i in range(n):
            vec[i] = 0.0
        return vec

@cython.boundscheck(False)
@cython.wraparound(False) 
def prox(np.ndarray[DTYPE_float_t, ndim=2] V, lamb, rho, np.ndarray[DTYPE_float_t, ndim=1] w):
    n = V.shape[0]
    for i in range(n):
        alpha = w[i]*lamb/rho
        V[i,:] = group_soft_threshold(V[i,:],alpha)
    return V

if __name__=="__main__":
   pass