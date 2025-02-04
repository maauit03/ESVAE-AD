#!/bin/bash
#SBATCH --job-name=ESVAE_MVTEC       # Job name
#SBATCH --output=ESVAE_MVTEC_gpu.out    # Output file
#SBATCH --error=ESVAE_MVTEC_gpu.err     # Error file
#SBATCH --time=48:00:00                # Time limit (30 minutes)
#SBATCH --partition=gpu_4           # Partition name (dev_gpu_4)
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --cpus-per-task=10             # Request 10 CPUs per task (adjust as needed)
#SBATCH --gres=gpu:1                   # Request 1 GPU (you can adjust this up to 4)
#SBATCH --mem=94GB                     # Request memory (94GB per GPU based on job defaults)

# Load required modules
module load devel/python/3.8.6_gnu_10.2
module load devel/cuda/11.8

# Activate your virtual environment 
source /pfs/data5/home/es/es_es/es_maauit03/.venv/bin/activate


# Run the Python script
cd /pfs/data5/home/es/es_es/es_maauit03/ESVAE-main-original
srun python init_fid_stats.py
srun python main_esvae.py -name exp_name -config NetworkConfigs/esvae_configs/CIFAR10.yaml   