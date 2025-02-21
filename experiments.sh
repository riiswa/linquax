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
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.075
export CUDA_VISIBLE_DEVICES=0,1
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform

env_ids=("inverted_pendulum" "boeing747" "chained_integrator" "large_transient" "not_controllable" "uav" "unstable_laplacian" "ac1" "ac3" "ac4" "ac6" "ac8" "bdt1" "dis1" "dis2" "he1" "he2" "je2" "psm")
strategies=("OFULQ" "TS" "MED")
output_file="processed_pairs.log"

# Clear the output file before starting
echo "Processed env/strategy pairs:" > "$output_file"

while [ ${#env_ids[@]} -gt 0 ]; do
  env_id=${env_ids[0]}  # Take the first environment
  for strategy in "${strategies[@]}"; do
    echo "Running experiment for env_id: $env_id with strategy: $strategy"
    python run_experiment.py --multirun hydra/launcher=joblib 'seed=range(64)' policy=$strategy env_id=$env_id
    echo "$env_id/$strategy" >> "$output_file"  # Log the processed pair
  done
  echo "End of $env_id" >> "$output_file"
  # Remove processed environment from the array (optional, keeping it static for now)
  # unset env_ids[0]
done