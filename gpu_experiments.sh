#!/bin/bash
#SBATCH --job-name=linquax         # Job name
#SBATCH --output=output.log        # Output file
#SBATCH --error=error.log          # Error file
#SBATCH --time=05:00:00            # Maximum run time (5 hours)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --cpus-per-task=8         # Number of CPU cores per task
#SBATCH --constraint=a100

export JAX_ENABLE_X64=True
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.12
export CUDA_VISIBLE_DEVICES=0,1
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform

env_ids=("boeing747" "chained_integrator" "large_transient" "not_controllable" "uav" "unstable_laplacian")
strategies=("OFULQ" "TS" "MED")

IFS=',' env_ids_joined="${env_ids[*]}"
IFS=',' strategies_joined="${strategies[*]}"

python run_experiment_gpu.py --multirun hydra/launcher=joblib policy=$strategies_joined env_id=$env_ids_joined