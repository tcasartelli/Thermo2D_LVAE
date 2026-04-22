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

# --------------------------------------------------------------------------
# Directories on $SCRATCH
# ------------------------------------------------------------------------------
mkdir -p /cluster/scratch/tcasartelli/thermoelastic2d_lvae/logs
mkdir -p /cluster/scratch/tcasartelli/thermoelastic2d_lvae/images
mkdir -p /cluster/scratch/tcasartelli/thermoelastic2d_lvae/checkpoints

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------
module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.8.0
module load eth_proxy

#------------------------------------------------------------------------------
# Activate virtual environment
#------------------------------------------------------------------------------
VENV_DIR="/cluster/scratch/tcasartelli/engi-venv"
source "$VENV_DIR/bin/activate"

#------------------------------------------------------------------------------
# W&B proxy fix (eth_proxy sets no_proxy which blocks api.wandb.ai)
#------------------------------------------------------------------------------
export no_proxy=$(echo "$no_proxy" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "$NO_PROXY" | sed 's/api\.wandb\.ai,\?//g; s/,$//')

#------------------------------------------------------------------------------
# W&B
#------------------------------------------------------------------------------
source ~/.secrets/ml_env.sh
export WANDB_MODE=online
export WANDB_PROJECT="thermoelastic2d"

#------------------------------------------------------------------------------
# Run
#------------------------------------------------------------------------------
python ~/projects/EngiOpt/engiopt/vanilla_thomas_thermo2D/thermoelastic2d_vanilla_lvae_2d.py \
    --problem_id thermoelastic2d \
    --seed 1 \
    --n_epochs 10000 \
    --batch_size 128 \
    --lr 1e-4 \
    --latent_dim 100 \
    --resize_dimensions 100 100 \
    --pruning_epoch 500 \
    --pruning_threshold 0.05 \
    --pruning_strategy plummet \
    --w_reconstruction 1.0 \
    --w_volume 0.01 \
    --save_model \
    --track
