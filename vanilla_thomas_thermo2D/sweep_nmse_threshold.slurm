#!/bin/bash
#SBATCH --job-name=thermo2d_sweep
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/tcasartelli/thermoelastic2d_constrained_lvae/logs/sweep_%j.out
#SBATCH --error=/cluster/scratch/tcasartelli/thermoelastic2d_constrained_lvae/logs/sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tcasartelli@ethz.ch

# Create output directories on $SCRATCH
mkdir -p /cluster/scratch/tcasartelli/thermoelastic2d_constrained_lvae/logs
mkdir -p /cluster/scratch/tcasartelli/thermoelastic2d_constrained_lvae/images
mkdir -p /cluster/scratch/tcasartelli/thermoelastic2d_constrained_lvae/checkpoints

# Load modules
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.8.0
module load eth_proxy

# Activate virtual environment
VENV_DIR="/cluster/scratch/tcasartelli/engi-venv"
source "$VENV_DIR/bin/activate"

# W&B credentials
export no_proxy=$(echo "$no_proxy" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "$NO_PROXY" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
source ~/.secrets/ml_env.sh
export WANDB_MODE=online

cd ~/projects/EngiOpt/engiopt/vanilla_thomas_thermo2D

# Each array task picks up one run from the sweep queue
wandb agent --count 1 tcasartelli-eth-z-rich/engiopt-vanilla_thomas_thermo2D/cpvrcqbx
