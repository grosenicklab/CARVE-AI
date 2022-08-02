#!/bin/bash
#SBATCH --job-name=aistats_div25    # Job name
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --cpus-per-task=16                   # Run 16 tasks	
#SBATCH --mem=50gb                    # Job memory request
#SBATCH -p scu-cpu
pwd; hostname; date

echo 'Submitting python initialize rerun AISTATS (div neighbors 25 table) experiments for ICML 2022'

echo 'activating conda environment'
eval "$(conda shell.bash hook)"
conda activate jupyter-lab

echo 'Code path is : '${code_path}
cd $code_path
pwd

echo python ${code_path}experiments/icml-2022/aistats_table_run_div_neighbors25.py $code_path
python ${code_path}experiments/icml-2022/aistats_table_run_div_neighbors25.py $code_path
