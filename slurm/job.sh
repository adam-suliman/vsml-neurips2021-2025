#!/bin/bash -l
#
#SBATCH --job-name=vsml
#SBATCH --time=04:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --signal=SIGINT@300

# Enable if compiled with CUDA compatible MPI
#export MPI4JAX_USE_CUDA_MPI=1

echo "Activate venv"
# TODO Set correct virtualenv path
source ~/path/to/venv/bin/activate

echo "Run job"
srun -X --wait=30 "$@"
