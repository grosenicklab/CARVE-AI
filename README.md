# Welcome to Cluster-Aware Routines for Versatile Embedding (CARVE) 



# Installation

- Installing Conda environment using .yml file with Mamba:
mamba env create -f aistats_2024.yml

- Mosek licence: To run the code, you will need to get a free Mosek license from: and put it in /home/<user>/mosek/mosek.lic (follow directions at link). 

- If you encounter problems with 'admm_utils', you can rebuild the Cython files locally by first installing Cython:

conda install -c anaconda cython

...and then running:

python setup.py build_ext --inplace

- The jupyter notebook 'Example notebook for  aistats 2024.ipynb' contains runnable examples. 

