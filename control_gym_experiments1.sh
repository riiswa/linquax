#!/bin/bash
#SBATCH --job-name=linquax_control_gym1         # Job name
#SBATCH --output=output1_.log        # Output file
#SBATCH --error=error1_.log          # Error file
#SBATCH --time=24:00:00            # Maximum run time (5 hours)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --cpus-per-task=128         # Number of CPU cores per task
##SBATCH --constraint=a100
##SBATCH --exclusive

export JAX_ENABLE_X64=True
export JAX_COMPILATION_CACHE_DIR="./jax_cache"
#export XLA_PYTHON_CLIENT_PREALLOCATE=false
#export XLA_PYTHON_CLIENT_MEM_FRACTION=.075
#export CUDA_VISIBLE_DEVICES=0,1
# export XLA_PYTHON_CLIENT_ALLOCATOR=platform

python run_experiment.py --config-name=config1  --multirun hydra/launcher=joblib 'seed=range(32)' policy=OFULQ,TS,MED env_id==ac1,ac2,ac3,ac4,ac5,ac6,ac8,ac10,bdt1,cm1,dis1,dis2,he1,he2,he3,psm,je1,je2





#output_file="processed_pairs.log"
#
## Clear the output file before starting
#echo "Processed env/strategy pairs:" > "$output_file"
#
#for env_id in "${env_ids[@]}"; do
#  for strategy in "${strategies[@]}"; do
#    echo "Running experiment for env_id: $env_id with strategy: $strategy"
#    python run_experiment.py --multirun hydra/launcher=joblib 'seed=range(48)' policy=$strategy env_id=$env_id
#    echo "$env_id/$strategy" >> "$output_file"  # Log the processed pair
#  done
#  echo "End of $env_id" >> "$output_file"
#done