import os
os.getcwd()
import numpy as np
import sys
import multiprocessing  
print(sys.argv)
code_path = sys.argv[1] # e.g., '/home/amb2022/clusterCCA_revision1/clusterCCA/'
sys.path.append(code_path)
sys.path.append(code_path+'/experiments/icml-2022/')
sys.path.append(code_path+'/utils/subspace-clustering-master')
# os.chdir(code_path+'/utils/subspace-clustering-master')
os.chdir('/home/amb2022/clusterCCA_revision1/clusterCCA')

from pcmf import pcmf_full, pcmf_approx_uV, pcmf_approx_V, two_cluster_data
from aistats_table import *
# Parallel function

######
def run_div_neighborsNone(r):
    run_numerical_experiments_parallel_20noneighbors_div_samemeans_dpt2(r)

    run_numerical_experiments_parallel_20noneighbors_div_samemeans_dpt5(r)

    run_numerical_experiments_parallel_20noneighbors_div_diffmeans_dpt2(r)

    run_numerical_experiments_parallel_20noneighbors_div_diffmeans_dpt5(r)

    run_numerical_experiments_parallel_20noneighbors_div_unbalanced_dpt2(r)

    run_numerical_experiments_parallel_20noneighbors_div_unbalanced_dpt5(r)

    run_numerical_experiments_parallel_200noneighbors_div_samemeans_dpt2(r)

    run_numerical_experiments_parallel_200noneighbors_div_samemeans_dpt5(r)

    run_numerical_experiments_parallel_200noneighbors_div_diffmeans_dpt2(r)

    run_numerical_experiments_parallel_200noneighbors_div_diffmeans_dpt5(r)

    run_numerical_experiments_parallel_200noneighbors_div_unbalanced_dpt2(r)

    run_numerical_experiments_parallel_200noneighbors_div_unbalanced_dpt5(r)

    run_numerical_experiments_parallel_2000noneighbors_div_samemeans_dpt2(r)

    run_numerical_experiments_parallel_2000noneighbors_div_samemeans_dpt5(r)

    run_numerical_experiments_parallel_2000noneighbors_div_diffmeans_dpt2(r)

    run_numerical_experiments_parallel_2000noneighbors_div_diffmeans_dpt5(r)

    run_numerical_experiments_parallel_2000noneighbors_div_unbalanced_dpt2(r)

    run_numerical_experiments_parallel_2000noneighbors_div_unbalanced_dpt5(r)

# # Run it
# if __name__ == '__main__':
#     print('Running main')
#     pool = multiprocessing.Pool(processes=10)
#     [pool.apply_async(run_div_neighborsNone, args=(r,)) for r in range(10)]

# Pool
if __name__ == '__main__':   
    pool = multiprocessing.Pool(processes=10)
    res = pool.map(smap, [run_div_neighborsNone(0), run_div_neighborsNone(1), run_div_neighborsNone(2), 
                        run_div_neighborsNone(3), run_div_neighborsNone(4), run_div_neighborsNone(5),
                        run_div_neighborsNone(6), run_div_neighborsNone(7), run_div_neighborsNone(8), run_div_neighborsNone(10)])






