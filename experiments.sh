#!/bin/bash
#SBATCH --job-name=linquax         # Job name
#SBATCH --output=output.log        # Output file
#SBATCH --error=error.log          # Error file
#SBATCH --time=05:00:00            # Maximum run time (5 hours)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --cpus-per-task=64         # Number of CPU cores per task
#SBATCH --constraint=a100

export JAX_ENABLE_X64=True
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.03
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
env_ids=("inverted_pendulum")
for env_id in "${env_ids[@]}"; do
  echo "Running experiment for env_id: $env_id"
  python run_experiment.py --multirun hydra/launcher=joblib 'seed=range(64)' policy=OFULQ,TS,MED env_id=$env_id
done