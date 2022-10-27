#!/bin/bash
#SBATCH --job-name=aistats_genomics_SRBCT)    # Job name
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --cpus-per-task=10                   # Run 16 tasks	
#SBATCH --mem=200gb                    # Job memory request
#SBATCH -p scu-cpu
pwd; hostname; date

echo 'Submitting python initialize rerun AISTATS (genomics SRBCT) experiments for ICML 2022'

echo 'activating conda environment'
eval "$(conda shell.bash hook)"
conda activate jupyter-lab

echo 'Code path is : '${code_path}
cd $code_path
pwd

echo python ${code_path}experiments/initialize_genomics_SRBCT.py $code_path
python ${code_path}experiments/initialize_genomics_SRBCT.py $code_path
