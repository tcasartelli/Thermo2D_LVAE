#!/bin/bash
#SBATCH --job-name=thermo2d_lvae
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/tcasartelli/thermoelastic2d_lvae/logs/thermo2d_lvae_%j.out
#SBATCH --error=/cluster/scratch/tcasartelli/thermoelastic2d_lvae/logs/thermo2d_lvae_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tcasartelli@ethz.ch

# ------------------------------------------------------------------------------
# Directories on $SCRATCH
# ------------------------------------------------------------------------------
mkdir -p /cluster/scratch/tcasartelli/thermoelastic2d_lvae/logs
mkdir -p /cluster/scratch/tcasartelli/thermoelastic2d_lvae/images
mkdir -p /cluster/scratch/tcasartelli/thermoelastic2d_lvae/checkpoints

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.8.0
module load eth_proxy

# ------------------------------------------------------------------------------
# Virtual environment
# ------------------------------------------------------------------------------
source "$HOME/engi-venv/bin/activate"

# ------------------------------------------------------------------------------
# Secrets (W&B token etc.)
# ------------------------------------------------------------------------------
source ~/.secrets/ml_env.sh
export WANDB_MODE=online
export WANDB_PROJECT="engiopt"

# ------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------
python "$HOME/projects/EngiOpt/thermoelastic2d_vanilla_lvae_2d.py" \
    --problem_id thermoelastic2d \
    --seed 1 \
    --n_epochs 10000 \
    --batch_size 128 \
    --lr 1e-4 \
    --latent_dim 100 \
    --resize_dimensions 64 64 \
    --pruning_epoch 500 \
    --pruning_threshold 0.05 \
    --pruning_strategy plummet \
    --w_reconstruction 1.0 \
    --w_volume 0.01 \
    --save_model True \
    --track True
