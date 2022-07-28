
The written Appendix is in 'NeurIPS_2022_Appendix.pdf'.

#--------------------------------------#
#          CODE  INSTALLATION          #
#--------------------------------------#

- Installing Conda environment using .yml file:
conda env create -f neurips_2022.yml

- Mosek licence: To run the code, you will need to get a free Mosek license from: and put it in /home/<user>/mosek/mosek.lic (follow directions at link). 

- If you encounter problems with 'admm_utils', you can rebuild the Cython files locally by first installing Cython:

conda install -c anaconda cython

...and then running:

python setup.py build_ext --inplace

- The jupyter notebook 'Example notebook for NeurIPS 2022.ipynb' contains runnable examples. 

