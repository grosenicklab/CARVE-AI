# srun --pty -n 1 -c 1 --mem=10G -p scu-cpu,sackler-cpu /bin/bash

srun --pty -n 1 -c 30 --mem=10G -p scu-cpu,sackler-cpu /bin/bash

# conda activate jupyter-lab
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
addm_iters=20
rho=1.0
for gauss_coef in 20.0 5.0 #0.0
do
    for neighbors in 25 15 #None
    do
        for split_size in 50 100 #25
        do
            for dataset in 'GbmBreastLungCancergenomics' 'run_MouseOrgansgenomics'
            do
                echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho ${code_path}experiments/icml-2022/submit_genomics.sh
                # sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset ${code_path}experiments/icml-2022/submit_genomics.sh
            done
        done
    done
done

gauss_coef=5.0
num_vars=1000
density=0.5
for m in 50 100 250 500
do
    for neighbors in 25 #None
    do
        for split_size in 50 100
        do
            for r in 7 #0 2 .. 9
            do
                echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,m=$m,density=$density,num_vars=$num_vars,r=$r ${code_path}experiments/icml-2022/submit_consensus_comparison.sh 
                sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,m=$m,density=$density,num_vars=$num_vars,r=$r ${code_path}experiments/icml-2022/submit_consensus_comparison.sh 
            done
        done
    done
done


code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
admm_iters=20
rho=1.0
gauss_coef=20.0
neighbors=15
split_size=71
dataset='GbmBreastLungCancergenomics' 
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho ${code_path}experiments/icml-2022/submit_genomics.sh

neighbors=15
split_size=25
dataset='run_MouseOrgansgenomics'
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho ${code_path}experiments/icml-2022/submit_genomics.sh


neighbors=15
split_size=100
dataset='run_MouseOrgansgenomics'
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho ${code_path}experiments/icml-2022/submit_genomics.sh

# Note check how split size is working...
# modify so it will pad data to correct size
# add plotting function to PDF
# Should I actually be padding with ones??

ls -lth ../../results/*consensus*

admm_iters=20
rho=1.0
gauss_coef=5.0
neighbors=15
split_size=25
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=50
rho=1.0
gauss_coef=5.0
neighbors=15
split_size=25
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=50
rho=1.0
gauss_coef=5.0
neighbors=15
split_size=25
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='run_MouseOrgansgenomics_skip50'
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=20
rho=1.0
gauss_coef=5.0
neighbors=15
split_size=25
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='run_MouseOrgansgenomics_skip50'
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=20
rho=1.0
gauss_coef=5.0
neighbors=25
split_size=50
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='run_MouseOrgansgenomics_skip25'
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=20
rho=1.0
gauss_coef=5.0
neighbors=25
split_size=50
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='run_MouseOrgansgenomics_skip0'
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=200gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=20
rho=1.0
gauss_coef=5.0
neighbors=25
split_size=50
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=200gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=50
rho=1.0
gauss_coef=5.0
neighbors=25
split_size=50
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=200gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho ${code_path}experiments/icml-2022/submit_genomics.sh


admm_iters=5
rho=1.0
gauss_coef=2.0
neighbors=15
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=0
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=5
rho=1.0
gauss_coef=2.0
neighbors=15
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=0
split_size=71
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=5
rho=1.0
gauss_coef=2.0
neighbors=15
penalty_type=0
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3_FULL'
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=5
rho=2.0
gauss_coef=2.0
neighbors=15
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=5
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=5
rho=3.0
gauss_coef=2.0
neighbors=15
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=5
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=5
rho=0.5
gauss_coef=2.0
neighbors=15
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=5
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=5
rho=10
gauss_coef=2.0
neighbors=15
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=5
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=50
rho=1.0
gauss_coef=2.0
neighbors=15
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=5
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=500
rho=1.0
gauss_coef=2.0
neighbors=15
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=5
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


admm_iters=5
rho=1.0
gauss_coef=2.0
neighbors=5
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=0
split_size=20
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

ls -lth ../../../results/*pcmf_full*

cat ../../../results/pcmf_full_GbmBreastLungCancer_genomics_run_N_142_gausscoef2.0_neighbors15_rho1.0_addmIters5.txt
cat ../../../results/pcmf_full_GbmBreastLungCancer_genomics_run_N_142_gausscoef2.0_neighbors15_rho1.0_addmIters5_penalty_type10.0.txt

admm_iters=5
rho=1.0
gauss_coef=2.0
neighbors=10
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=0
split_size=20
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


admm_iters=5
rho=1.0
gauss_coef=2.0
neighbors=10
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
penalty_type=0
split_size=25
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh




admm_iters=5
rho=1.0
gauss_coef=2.0
neighbors=None
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=0
split_size=20
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


admm_iters=10
rho=1.0
gauss_coef=2.0
neighbors=5
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=0
split_size=20
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=20
rho=1.0
gauss_coef=2.0
neighbors=5
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=0
split_size=20
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=20
rho=1.0
gauss_coef=2.0
neighbors=15
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=0
split_size=20
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


admm_iters=20
rho=1.0
gauss_coef=2.0
neighbors=10
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=0
split_size=20
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=20
rho=1.0
gauss_coef=2.0
neighbors=None
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=0
split_size=20
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


admm_iters=5
rho=1.0
gauss_coef=2.0
neighbors=5
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
penalty_type=0
split_size=25
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


admm_iters=5
rho=1.0
gauss_coef=2.0
neighbors=15
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
penalty_type=0
split_size=25
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=5
rho=1.0
gauss_coef=2.0
neighbors=20
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
penalty_type=0
split_size=25
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


admm_iters=5
rho=1.5
gauss_coef=2.0
neighbors=10
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
penalty_type=0
split_size=25
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=5
rho=2.0
gauss_coef=2.0
neighbors=10
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
penalty_type=0
split_size=25
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


admm_iters=5
rho=2.0
gauss_coef=2.0
neighbors=10
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=0
split_size=25
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=20
rho=1.0
gauss_coef=2.0
neighbors=5
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=0
split_size=25
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


admm_iters=10
rho=0.5
gauss_coef=2.0
neighbors=5
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=0
split_size=25
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


admm_iters=10
rho=0.5
gauss_coef=2.0
neighbors=15
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
penalty_type=0
split_size=25
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=10
rho=0.5
gauss_coef=2.0
neighbors=15
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
penalty_type=1
split_size=25
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


admm_iters=10
rho=0.5
gauss_coef=2.0
neighbors=20
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
penalty_type=1
split_size=25
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=10
rho=1
gauss_coef=5.0
neighbors=25
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
penalty_type=0
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=10
rho=1
gauss_coef=5.0
neighbors=50
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
penalty_type=0
split_size=100
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=10
rho=1
gauss_coef=5.0
neighbors=70
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
penalty_type=0
split_size=100
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=20
rho=1.0
gauss_coef=2.0
neighbors=5
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=2
split_size=25
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch --mem=50gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=20
rho=1.0
gauss_coef=2.0
neighbors=5
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip3'
penalty_type=2
split_size=25
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


admm_iters=100
rho=1
gauss_coef=5.0
neighbors=5
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
penalty_type=0
split_size=10
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh



admm_iters=100
rho=1
gauss_coef=5.0
neighbors=5
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
penalty_type=4
split_size=10
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=100
rho=1
gauss_coef=5.0
neighbors=25
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
penalty_type=4
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh



admm_iters=50
rho=1
gauss_coef=5.0
neighbors=25
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
penalty_type=7
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=20
rho=1
gauss_coef=5.0
neighbors=25
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='GbmBreastLungCancergenomics_skip0'
penalty_type=7
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh



admm_iters=20
rho=1
gauss_coef=5.0
neighbors=25
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='run_MouseOrgansgenomics_skip50'
penalty_type=7
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


admm_iters=20
rho=1
gauss_coef=5.0
neighbors=25
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='run_MouseOrgansgenomics_skip25'
penalty_type=7
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


admm_iters=1
rho=1
gauss_coef=5.0
neighbors=25
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='run_MouseOrgansgenomics_skip50'
penalty_type=7
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=20
rho=1
gauss_coef=5.0
neighbors=25
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='run_MouseOrgansgenomics_skip10'
penalty_type=7
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=100
rho=1
gauss_coef=5.0
neighbors=25
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='run_MouseOrgansgenomics_skip5'
penalty_type=4
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


admm_iters=50
rho=1
gauss_coef=5.0
neighbors=50
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='run_MouseOrgansgenomics_skip5'
penalty_type=4
split_size=100
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh



admm_iters=50
rho=1
gauss_coef=5.0
neighbors=50
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='run_MouseOrgansgenomics_skip5'
penalty_type=6
split_size=100
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=20
rho=1
gauss_coef=5.0
neighbors=50
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='run_MouseOrgansgenomics_skip5'
penalty_type=0
split_size=100
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

admm_iters=20
rho=1
gauss_coef=5.0
neighbors=50
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='run_MouseOrgansgenomics_skip2'
penalty_type=0
split_size=100
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh




admm_iters=20
rho=1
gauss_coef=5.0
neighbors=25
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='run_MouseOrgansgenomics_skip50'
penalty_type=6
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh



admm_iters=50
rho=1
gauss_coef=5.0
neighbors=25
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='run_MouseOrgansgenomics_skip50'
penalty_type=6
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh



admm_iters=50
rho=1
gauss_coef=5.0
neighbors=25
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='run_MouseOrgansgenomics_skip25'
penalty_type=6
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


admm_iters=50
rho=1
gauss_coef=5.0
neighbors=25
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset='run_MouseOrgansgenomics_skip10'
penalty_type=6
split_size=50
echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh



sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=100,m=250,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type='eight_clusters' ${code_path}experiments/icml-2022/submit_consensus.sh

sbatch --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=100,m=250,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type='eight_clusters' ${code_path}experiments/icml-2022/submit_consensus.sh

mkdir ../../../results/icml_pdfs
mv ../../../results/*OLD*pdf ../../../results/icml_pdfs
ls -lth ../../../results/*OLD*pdf

ls -lth ../../../results/*pdf

# mv ../../../results/*pdf ../../../results/icml_pdfs

ls -lth ../../../results/*Mouse*txt


# admm_iters=20
# rho=1.0
# gauss_coef=5.0
# neighbors=15
# split_size=25
# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
# dataset='run_MouseOrgansgenomics_skip25'
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch --mem=50gb --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho ${code_path}experiments/icml-2022/submit_genomics.sh

# admm_iters=20
# rho=1.0
# gauss_coef=5.0
# neighbors=15
# split_size=50
# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
# dataset='run_MouseOrgansgenomics_skip0'
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch --mem=200gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho ${code_path}experiments/icml-2022/submit_genomics.sh


# sbatch --mem=40gb --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=250,density=0.5,num_vars=50,r=7,admm_iters=20,rho=1.0 ${code_path}experiments/icml-2022/submit_consensus.sh


# sbatch --mem=100gb --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=100,m=250,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0 ${code_path}experiments/icml-2022/submit_consensus.sh
# sbatch --mem=200gb --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=100,m=2500,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0 ${code_path}experiments/icml-2022/submit_consensus.sh
# sbatch --mem=200gb --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=100,m=25000,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0 ${code_path}experiments/icml-2022/submit_consensus.sh
# sbatch --mem=200gb --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=100,m=250000,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0 ${code_path}experiments/icml-2022/submit_consensus.sh

sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=100,m=25000,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0 ${code_path}experiments/icml-2022/submit_consensus.sh

sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=12500,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0 ${code_path}experiments/icml-2022/submit_consensus.sh

# sbatch --mem=200gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=100,m=250,density=0.5,num_vars=20000,r=7,admm_iters=20,rho=1.0 ${code_path}experiments/icml-2022/submit_consensus.sh
# sbatch --mem=200gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=100,m=2500,density=0.5,num_vars=20000,r=7,admm_iters=20,rho=1.0 ${code_path}experiments/icml-2022/submit_consensus.sh


# sbatch --mem=200gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=250,density=0.5,num_vars=20000,r=7,admm_iters=20,rho=1.0 ${code_path}experiments/icml-2022/submit_consensus.sh
# sbatch --mem=200gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=2500,density=0.5,num_vars=20000,r=7,admm_iters=20,rho=1.0 ${code_path}experiments/icml-2022/submit_consensus.sh
sbatch --mem=200gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=25000,density=0.5,num_vars=20000,r=7,admm_iters=20,rho=1.0 ${code_path}experiments/icml-2022/submit_consensus.sh






sbatch --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=100,m=25000,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0 ${code_path}experiments/icml-2022/submit_consensus_comparison.sh

sbatch --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=100,m=2500,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0 ${code_path}experiments/icml-2022/submit_consensus_comparison.sh

sbatch --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=100,m=2500,density=0.5,num_vars=50,r=0,admm_iters=20,rho=1.0 ${code_path}experiments/icml-2022/submit_consensus_comparison.sh



