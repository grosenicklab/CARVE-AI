#SRBCT
code_path='/athena/listonlab/store/amb2022/PCMF/'
echo sbatch --export=code_path=$code_path ${code_path}experiments/run_genomics_run_SRBCT.sh.sh
sbatch --export=code_path=$code_path ${code_path}experiments/run_genomics_run_SRBCT.sh.sh

