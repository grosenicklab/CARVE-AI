#!/bin/bash
#SBATCH --job-name=aistats_genomics_14c)    # Job name
#SBATCH --nodes=1                    # Run all processes on a single node	
#SBATCH --cpus-per-task=10                   # Run 16 tasks	
#SBATCH --mem=200gb                    # Job memory request
#SBATCH -p scu-cpu
pwd; hostname; date

echo 'Submitting python initialize rerun AISTATS (genomics 14 cancer) experiments for ICML 2022'

echo 'activating conda environment'
eval "$(conda shell.bash hook)"
conda activate jupyter-lab

echo 'Code path is : '${code_path}
cd $code_path
pwd

echo python ${code_path}experiments/icml-2022/aistats_genomics_run_14c.py $code_path
python ${code_path}experiments/icml-2022/aistats_genomics_run_14c.py $code_path