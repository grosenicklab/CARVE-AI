###### 
# Consensus vs full
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
                echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,m=$m,density=$density,num_vars=$num_vars,r=$r,admm_iters=5,rho=1.0 ${code_path}experiments/icml-2022/submit_consensus_comparison.sh 
                sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,m=$m,density=$density,num_vars=$num_vars,r=$r,admm_iters=5,rho=1.0 ${code_path}experiments/icml-2022/submit_consensus_comparison.sh 
            done
        done
    done
done
######
# Consensus only (4 clusters):   $code_path $gauss_coef $neighbors $split_size $m $density $num_vars $r $admm_iters $rho $dataset_type
# 100, 50 neighbors
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset_type='four'
sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=25,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus_comparison.sh 
dataset_type='four'
sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=25,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus_comparison.sh 

# Consensus only (4 UNBALANCED):   $code_path $gauss_coef $neighbors $split_size $m $density $num_vars $r $admm_iters $rho $dataset_type
# 100, 50 neighbors
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
dataset_type='4UNBALANCED'	
sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=100,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh 
dataset_type='4UNBALANCED'
sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=100,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh 

dataset_type='4UNBALANCED'
sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=1000,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh 
dataset_type='4UNBALANCED'
sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=1000,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh 

dataset_type='4UNBALANCED'
sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=10000,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh 
dataset_type='4UNBALANCED'
sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=10000,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh 

dataset_type='4UNBALANCED'
sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=100,m=100000,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh 
dataset_type='4UNBALANCED'
sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=100,m=100000,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh 

######
# Consensus only (4 clusters):   $code_path $gauss_coef $neighbors $split_size $m $density $num_vars $r $admm_iters $rho $dataset_type
# 100,000, 50 neighbors
# dataset_type='four'
# sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=50,split_size=100,m=25000,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='four'
# sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=50,split_size=100,m=25000,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh

#100,000
# dataset_type='four'
# sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=100,m=25000,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='four'
# sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=100,m=25000,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='four'
# sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=100,split_size=200,m=25000,density=0.5,num_vars=10000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# # 50,000
# dataset_type='four'
# sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=12500,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='four'
# sbatch --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=12500,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh

# # 10,000
# dataset_type='four'
# sbatch --mem=100gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=2500,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='four'
# sbatch --mem=100gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=2500,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh

# # 1,000
# dataset_type='four'
# sbatch --mem=50gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=250,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='four'
# sbatch --mem=50gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=250,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh

######
# Consensus only (8 clusters):

# # 1,000
# dataset_type='eight_clusters'
# sbatch --mem=25gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=5.0,neighbors=50,split_size=100,m=250,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='eight_clusters'
# sbatch --mem=50gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=50,split_size=100,m=250,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh

# # 500
# dataset_type='eight_clusters'
# sbatch --mem=25gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=5.0,neighbors=50,split_size=100,m=125,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='eight_clusters'
# sbatch --mem=25gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=5.0,neighbors=50,split_size=100,m=125,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh

# 100
# dataset_type='eight_clusters'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=25,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='eight_clusters'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=25,split_size=50,m=25,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh

######
# Consensus comparison (20 clusters):
code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
# # 100
# dataset_type='20_clusters'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=10,split_size=50,m=5,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='20_clusters'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=10,split_size=50,m=5,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh

# # 200
# dataset_type='20_clusters'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=15,split_size=50,m=10,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='20_clusters'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=15,split_size=50,m=10,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh

# # 500
# dataset_type='20_clusters'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=30,split_size=50,m=25,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='20_clusters'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=30,split_size=50,m=25,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh

# # 500
# dataset_type='20_clusters_consensus_only'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=50,split_size=100,m=25,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='20_clusters_consensus_only'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=50,split_size=100,m=25,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh

# dataset_type='20_clusters_consensus_only'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=100,split_size=200,m=25,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='20_clusters_consensus_only'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=100,split_size=200,m=25,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh

# # 1000
# dataset_type='20_clusters_consensus_only'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=50,split_size=100,m=50,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='20_clusters_consensus_only'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=50,split_size=100,m=50,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh

# dataset_type='20_clusters_consensus_only'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=100,split_size=200,m=50,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='20_clusters_consensus_only'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=100,split_size=200,m=50,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh

# # 100
# dataset_type='20_clusters_PALS'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=10,split_size=50,m=5,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='20_clusters_PALS'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=10,split_size=50,m=5,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh

# # 200
# dataset_type='20_clusters_PALS'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=15,split_size=50,m=10,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='20_clusters_PALS'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=15,split_size=50,m=10,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh

# # 500
# dataset_type='20_clusters_PALS'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=50,split_size=100,m=25,density=0.5,num_vars=100,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh
# dataset_type='20_clusters_PALS'
# sbatch --mem=25gb --cpus-per-task=10 --export=code_path=$code_path,gauss_coef=5.0,neighbors=50,split_size=100,m=25,density=0.5,num_vars=1000,r=7,admm_iters=20,rho=1.0,dataset_type=$dataset_type ${code_path}experiments/icml-2022/submit_consensus.sh

######
# gbm lung breast consensus
	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=10
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=50
	# split_size=71
	# penalty_type=0
	# dataset='GbmBreastLungCancergenomics_skip3' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=20
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=50
	# split_size=71
	# penalty_type=0
	# dataset='GbmBreastLungCancergenomics_skip3' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=50
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=50
	# split_size=71
	# penalty_type=0
	# dataset='GbmBreastLungCancergenomics_skip3' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=10
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# penalty_type=0
	# dataset='GbmBreastLungCancergenomics_skip3' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=20
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# penalty_type=0
	# dataset='GbmBreastLungCancergenomics_skip3' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=50
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# penalty_type=0
	# dataset='GbmBreastLungCancergenomics_skip3' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=10
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=50
	# split_size=100
	# penalty_type=0
	# dataset='GbmBreastLungCancergenomics_skip0' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=20
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=50
	# split_size=100
	# penalty_type=0
	# dataset='GbmBreastLungCancergenomics_skip0' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=50
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=50
	# split_size=100
	# penalty_type=0
	# dataset='GbmBreastLungCancergenomics_skip0' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
#	 sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=10
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=50
	# split_size=100
	# penalty_type=8
	# dataset='GbmBreastLungCancergenomics_skip0' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=20
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=50
	# split_size=100
	# penalty_type=8
	# dataset='GbmBreastLungCancergenomics_skip0' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=50
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=50
	# split_size=100
	# penalty_type=8
	# dataset='GbmBreastLungCancergenomics_skip0' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=10
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=50
	# split_size=100
	# penalty_type=7
	# dataset='GbmBreastLungCancergenomics_skip0' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=20
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=50
	# split_size=100
	# penalty_type=7
	# dataset='GbmBreastLungCancergenomics_skip0' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=50
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=50
	# split_size=100
	# penalty_type=7
	# dataset='GbmBreastLungCancergenomics_skip0' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=10
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=50
	# split_size=100
	# penalty_type=9
	# dataset='GbmBreastLungCancergenomics_skip0' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=20
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=50
	# split_size=100
	# penalty_type=9
	# dataset='GbmBreastLungCancergenomics_skip0' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=50
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=50
	# split_size=100
	# penalty_type=9
	# dataset='GbmBreastLungCancergenomics_skip0' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=10
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# penalty_type=9
	# dataset='GbmBreastLungCancergenomics_skip0' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=20
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# penalty_type=9
	# dataset='GbmBreastLungCancergenomics_skip0' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# admm_iters=50
	# rho=1.0
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# penalty_type=9
	# dataset='GbmBreastLungCancergenomics_skip0' 
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

######
# Mouse consensus
# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'

# dataset='run_MouseOrgansgenomics_skip50'
# penalty_type=0
# admm_iters=10
# rho=1
# gauss_coef=5.0
# neighbors=10
# split_size=20
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip50'
# penalty_type=0
# admm_iters=20
# rho=1
# gauss_coef=5.0
# neighbors=10
# split_size=20
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip50'
# penalty_type=0
# admm_iters=50
# rho=1
# gauss_coef=5.0
# neighbors=10
# split_size=20
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip50'
# penalty_type=0
# admm_iters=10
# rho=1
# gauss_coef=5.0
# neighbors=25
# split_size=50
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip50'
# penalty_type=0
# admm_iters=20
# rho=1
# gauss_coef=5.0
# neighbors=25
# split_size=50
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip50'
# penalty_type=0
# admm_iters=50
# rho=1
# gauss_coef=5.0
# neighbors=25
# split_size=50
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

##
# dataset='run_MouseOrgansgenomics_skip25'
# penalty_type=0
# admm_iters=10
# rho=1
# gauss_coef=5.0
# neighbors=50
# split_size=100
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip25'
# penalty_type=0
# admm_iters=20
# rho=1
# gauss_coef=5.0
# neighbors=50
# split_size=100
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip25'
# penalty_type=0
# admm_iters=50
# rho=1
# gauss_coef=5.0
# neighbors=50
# split_size=100
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# # ##
# dataset='run_MouseOrgansgenomics_skip10'
# penalty_type=0
# admm_iters=10
# rho=1
# gauss_coef=5.0
# neighbors=50
# split_size=100
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip10'
# penalty_type=0
# admm_iters=20
# rho=1
# gauss_coef=5.0
# neighbors=50
# split_size=100
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip10'
# penalty_type=0
# admm_iters=50
# rho=1
# gauss_coef=5.0
# neighbors=50
# split_size=100
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# # ##
# dataset='run_MouseOrgansgenomics_skip10'
# penalty_type=8
# admm_iters=10
# rho=1
# gauss_coef=5.0
# neighbors=50
# split_size=100
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip10'
# penalty_type=8
# admm_iters=20
# rho=1
# gauss_coef=5.0
# neighbors=50
# split_size=100
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip10'
# penalty_type=8
# admm_iters=50
# rho=1
# gauss_coef=5.0
# neighbors=50
# split_size=100
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# # ##
# dataset='run_MouseOrgansgenomics_skip5'
# penalty_type=6
# admm_iters=10
# rho=1
# gauss_coef=5.0
# neighbors=50
# split_size=100
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip5'
# penalty_type=6
# admm_iters=20
# rho=1
# gauss_coef=5.0
# neighbors=50
# split_size=100
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip5'
# penalty_type=6
# admm_iters=50
# rho=1
# gauss_coef=5.0
# neighbors=50
# split_size=100
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# ##
# dataset='run_MouseOrgansgenomics_skip5'
# penalty_type=8
# admm_iters=10
# rho=1
# gauss_coef=5.0
# neighbors=100
# split_size=200
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip5'
# penalty_type=8
# admm_iters=20
# rho=1
# gauss_coef=5.0
# neighbors=100
# split_size=200
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip5'
# penalty_type=8
# admm_iters=50
# rho=1
# gauss_coef=5.0
# neighbors=50
# split_size=200
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# # # ##
# dataset='run_MouseOrgansgenomics_skip2'
# penalty_type=6
# admm_iters=10
# rho=1
# gauss_coef=5.0
# neighbors=50
# split_size=100
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=200gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip2'
# penalty_type=6
# admm_iters=20
# rho=1
# gauss_coef=5.0
# neighbors=50
# split_size=100
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip2'
# penalty_type=6
# admm_iters=50
# rho=1
# gauss_coef=5.0
# neighbors=50
# split_size=100
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# # # ##
# dataset='run_MouseOrgansgenomics_skip2'
# penalty_type=6
# admm_iters=10
# rho=1
# gauss_coef=5.0
# neighbors=150
# split_size=300
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=200gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip2'
# penalty_type=6
# admm_iters=20
# rho=1
# gauss_coef=5.0
# neighbors=150
# split_size=300
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip2'
# penalty_type=6
# admm_iters=50
# rho=1
# gauss_coef=5.0
# neighbors=150
# split_size=300
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

# dataset='run_MouseOrgansgenomics_skip2'
# penalty_type=6
# admm_iters=100
# rho=1
# gauss_coef=5.0
# neighbors=150
# split_size=300
# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
# sbatch -p scu-cpu --mem=200gb --cpus-per-task=48 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# ##
	# dataset='run_MouseOrgansgenomics_skip1'
	# penalty_type=0
	# admm_iters=10
	# rho=1
	# gauss_coef=5.0
	# neighbors=150
	# split_size=300
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=500gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

		dataset='run_MouseOrgansgenomics_skip1'
		penalty_type=0
		admm_iters=20
		rho=1
		gauss_coef=5.0
		neighbors=150
		split_size=300
		echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
		sbatch -p scu-cpu --mem=250gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

		dataset='run_MouseOrgansgenomics_skip1'
		penalty_type=0
		admm_iters=50
		rho=1
		gauss_coef=5.0
		neighbors=150
		split_size=300
		echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
		sbatch -p scu-cpu --mem=250gb --cpus-per-task=32 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip0smallP'
	# penalty_type=6
	# admm_iters=20
	# rho=1
	# gauss_coef=5.0
	# neighbors=50
	# split_size=100
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=200gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip0smallP'
	# penalty_type=6
	# admm_iters=50
	# rho=1
	# gauss_coef=5.0
	# neighbors=50
	# split_size=100
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=200gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip0smallP'
	# penalty_type=6
	# admm_iters=100
	# rho=1
	# gauss_coef=5.0
	# neighbors=50
	# split_size=100
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=200gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip50smallP'
	# penalty_type=0
	# admm_iters=10
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip50smallP'
	# penalty_type=0
	# admm_iters=20
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip50smallP'
	# penalty_type=0
	# admm_iters=50
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


	# dataset='run_MouseOrgansgenomics_skip25smallP'
	# penalty_type=0
	# admm_iters=10
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip25smallP'
	# penalty_type=0
	# admm_iters=20
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip25smallP'
	# penalty_type=0
	# admm_iters=50
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip10smallP'
	# penalty_type=0
	# admm_iters=10
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip10smallP'
	# penalty_type=0
	# admm_iters=20
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip10smallP'
	# penalty_type=0
	# admm_iters=50
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh


	# dataset='run_MouseOrgansgenomics_skip5smallP'
	# penalty_type=0
	# admm_iters=10
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip5smallP'
	# penalty_type=0
	# admm_iters=20
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip5smallP'
	# penalty_type=0
	# admm_iters=50
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip2smallP'
	# penalty_type=0
	# admm_iters=10
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip2smallP'
	# penalty_type=0
	# admm_iters=20
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip2smallP'
	# penalty_type=0
	# admm_iters=50
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=50gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip1'
	# penalty_type=0
	# admm_iters=10
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=100gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip1'
	# penalty_type=0
	# admm_iters=20
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=100gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh

	# dataset='run_MouseOrgansgenomics_skip1'
	# penalty_type=0
	# admm_iters=50
	# rho=1
	# gauss_coef=5.0
	# neighbors=25
	# split_size=50
	# echo sbatch --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh
	# sbatch -p scu-cpu --mem=100gb --cpus-per-task=16 --export=code_path=$code_path,gauss_coef=$gauss_coef,neighbors=$neighbors,split_size=$split_size,dataset=$dataset,admm_iters=$admm_iters,rho=$rho,penalty_type=$penalty_type,penalty_type=$penalty_type ${code_path}experiments/icml-2022/submit_genomics.sh




## AISTATS Genomics
	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# echo sbatch --export=code_path=$code_path ${code_path}experiments/icml-2022/submit_aistats_genomics_run_NCI.sh
	# sbatch --export=code_path=$code_path ${code_path}experiments/icml-2022/submit_aistats_genomics_run_NCI.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# echo sbatch --export=code_path=$code_path ${code_path}experiments/icml-2022/submit_aistats_genomics_run_GbmBreastLungCancer.sh
	# sbatch --export=code_path=$code_path ${code_path}experiments/icml-2022/submit_aistats_genomics_run_GbmBreastLungCancer.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# echo sbatch --export=code_path=$code_path ${code_path}experiments/icml-2022/submit_aistats_genomics_run_SRBCT.sh
	# sbatch --export=code_path=$code_path ${code_path}experiments/icml-2022/submit_aistats_genomics_run_SRBCT.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# echo sbatch --export=code_path=$code_path ${code_path}experiments/icml-2022/submit_aistats_genomics_run_14c.sh
	# sbatch --export=code_path=$code_path ${code_path}experiments/icml-2022/submit_aistats_genomics_run_14c.sh

	# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
	# echo sbatch --export=code_path=$code_path ${code_path}experiments/icml-2022/submit_aistats_genomics_run_MouseOrgans.sh
	# sbatch --export=code_path=$code_path ${code_path}experiments/icml-2022/submit_aistats_genomics_run_MouseOrgans.sh

# AISTATS Table

# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
# echo sbatch --export=code_path=$code_path ${code_path}experiments/icml-2022/submit_aistats_table_run_div_neighborsNone.sh
# sbatch --export=code_path=$code_path ${code_path}experiments/icml-2022/submit_aistats_table_run_div_neighborsNone.sh

# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
# echo sbatch --export=code_path=$code_path ${code_path}experiments/icml-2022/submit_aistats_table_run_div_neighbors25.sh
# sbatch --export=code_path=$code_path ${code_path}experiments/icml-2022/submit_aistats_table_run_div_neighbors25.sh

# code_path='/home/amb2022/clusterCCA_revision1/clusterCCA/'
# echo sbatch --export=code_path=$code_path ${code_path}experiments/icml-2022/submit_aistats_table_run_agglom.sh
# sbatch --export=code_path=$code_path ${code_path}experiments/icml-2022/submit_aistats_table_run_agglom.sh







