#SRBCT
code_path='/athena/listonlab/store/amb2022/PCMF/'
echo sbatch --export=code_path=$code_path ${code_path}experiments/run_genomics_run_SRBCT.sh
sbatch --export=code_path=$code_path ${code_path}experiments/run_genomics_run_SRBCT.sh

echo sbatch --export=code_path=$code_path ${code_path}experiments/run_genomics_run_NCI.sh
sbatch --export=code_path=$code_path ${code_path}experiments/run_genomics_run_NCI.sh

echo sbatch --export=code_path=$code_path ${code_path}experiments/run_genomics_run_GbmBreastLungCancer.sh
sbatch --export=code_path=$code_path ${code_path}experiments/run_genomics_run_GbmBreastLungCancer.sh

echo sbatch --export=code_path=$code_path ${code_path}experiments/run_genomics_run_14c.sh
sbatch --export=code_path=$code_path ${code_path}experiments/run_genomics_run_14c.sh

echo sbatch --export=code_path=$code_path ${code_path}experiments/run_genomics_run_MouseOrgans.sh
sbatch --export=code_path=$code_path ${code_path}experiments/run_genomics_run_MouseOrgans.sh

