#!/bin/bash
#SBATCH --job-name=thermo2d_clvae_vgate
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/tcasartelli/thermoelastic2d_constrained_lvae_val_gate/logs/thermo2d_clvae_vgate_%j.out
#SBATCH --error=/cluster/scratch/tcasartelli/thermoelastic2d_constrained_lvae_val_gate/logs/thermo2d_clvae_vgate_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tcasartelli@ethz.ch

mkdir -p /cluster/scratch/tcasartelli/thermoelastic2d_constrained_lvae_val_gate/logs
mkdir -p /cluster/scratch/tcasartelli/thermoelastic2d_constrained_lvae_val_gate/images
mkdir -p /cluster/scratch/tcasartelli/thermoelastic2d_constrained_lvae_val_gate/checkpoints

module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.8.0
module load eth_proxy

VENV_DIR="/cluster/scratch/tcasartelli/engi-venv"
source "$VENV_DIR/bin/activate"

export no_proxy=$(echo "$no_proxy" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
export NO_PROXY=$(echo "$NO_PROXY" | sed 's/api\.wandb\.ai,\?//g; s/,$//')
source ~/.secrets/ml_env.sh
export WANDB_MODE=online
export WANDB_PROJECT="spectral_norm_encoder"

python ~/projects/EngiOpt/engiopt/vanilla_thomas_thermo2D/thermoelastic2d_constrained_lvae_2d_val_gate.py \
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
    --gn_groups 8 \
    --wandb_project spectral_norm_encoder \
    --wandb_run_name thermo2d_constrained_lvae_val_gate \
    --early_stopping_start_epoch 2000 \
    --patience 50 \
    --save_model \
    --track
