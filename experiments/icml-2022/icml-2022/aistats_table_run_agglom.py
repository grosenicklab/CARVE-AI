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
def run_agglom(r):
    run_numerical_experiments_parallel_20_agglom_samemeans_dpt2(r)

    run_numerical_experiments_parallel_20_agglom_samemeans_dpt5(r)

    run_numerical_experiments_parallel_20_agglom_diffmeans_dpt2(r)

    run_numerical_experiments_parallel_20_agglom_diffmeans_dpt5(r)

    run_numerical_experiments_parallel_20_agglom_unbalanced_dpt2(r)

    run_numerical_experiments_parallel_20_agglom_unbalanced_dpt5(r)

    run_numerical_experiments_parallel_200_agglom_samemeans_dpt2(r)

    run_numerical_experiments_parallel_200_agglom_samemeans_dpt5(r)

    run_numerical_experiments_parallel_200_agglom_diffmeans_dpt2(r)

    run_numerical_experiments_parallel_200_agglom_diffmeans_dpt5(r)

    run_numerical_experiments_parallel_200_agglom_unbalanced_dpt2(r)

    run_numerical_experiments_parallel_200_agglom_unbalanced_dpt5(r)

    run_numerical_experiments_parallel_2000_agglom_samemeans_dpt2(r)

    run_numerical_experiments_parallel_2000_agglom_samemeans_dpt5(r)

    run_numerical_experiments_parallel_2000_agglom_diffmeans_dpt2(r)

    run_numerical_experiments_parallel_2000_agglom_diffmeans_dpt5(r)

    run_numerical_experiments_parallel_2000_agglom_unbalanced_dpt2(r)

    run_numerical_experiments_parallel_2000_agglom_unbalanced_dpt5(r)

# # Run it
# if __name__ == '__main__':
#     pool = multiprocessing.Pool(processes=10)
#     [pool.apply_async(run_agglom, args=(r,)) for r in range(10)]
if __name__ == '__main__':   
    pool = multiprocessing.Pool(processes=10)
    res = pool.map(smap, [run_agglom(0), run_agglom(1), run_agglom(2), 
                        run_agglom(3), run_agglom(4), run_agglom(5),
                        run_agglom(6), run_agglom(7), run_agglom(8), run_agglom(10)])



