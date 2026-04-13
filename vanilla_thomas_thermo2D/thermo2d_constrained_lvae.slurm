#!/bin/bash
#SBATCH --job-name=thermo2d_constrained
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/tcasartelli/thermoelastic2d_constrained_lvae/logs/thermo2d_constrained_%j.out
#SBATCH --error=/cluster/scratch/tcasartelli/thermoelastic2d_constrained_lvae/logs/thermo2d_constrained_%j.err
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
source ~/.secrets/ml_env.sh
export WANDB_MODE=online
export WANDB_PROJECT="thermoelastic2d"

# Run training
python ~/projects/EngiOpt/vanilla_thomas_thermo2D/thermoelastic2d_constrained_lvae_2d.py \
    --problem_id thermoelastic2d \
    --seed 1 \
    --n_epochs 10000 \
    --batch_size 128 \
    --lr 1e-4 \
    --latent_dim 100 \
    --resize_dimensions 100 100 \
    --nmse_threshold 0.05 \
    --pruning_epoch 500 \
    --pruning_threshold 0.05 \
    --pruning_strategy plummet \
    --save_model \
    --track
    