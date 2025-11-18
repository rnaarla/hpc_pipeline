#!/bin/bash
#SBATCH -N 16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --job-name=hpc_pipeline
#SBATCH --output=logs/%x-%j.out

module load cuda/12.2
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_TOPO_FILE=/etc/nccl_topo.xml

srun python orchestrator.py --config configs/default_config.yaml
# Ensure the script is executable
chmod +x configs/slurm_job.sh   