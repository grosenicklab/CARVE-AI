#!/bin/bash
#SBATCH --job-name=icml_genomics    # Job name
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --cpus-per-task=16                   # Run 16 tasks	
#SBATCH --mem=32gb                    # Job memory request
#SBATCH -p scu-cpu
pwd; hostname; date

echo 'Submitting python initialize genomics experiments for ICML 2022'

echo 'activating conda environment'
eval "$(conda shell.bash hook)"
conda activate jupyter-lab

echo 'Code path is : '${code_path}
cd $code_path
pwd

echo python ${code_path}experiments/icml-2022/initialize_genomics.py $code_path $gauss_coef $neighbors $split_size $dataset $admm_iters $rho $penalty_type

python ${code_path}experiments/icml-2022/initialize_genomics.py $code_path $gauss_coef $neighbors $split_size $dataset $admm_iters $rho $penalty_type
