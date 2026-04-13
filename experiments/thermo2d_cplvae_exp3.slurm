#!/bin/bash
#SBATCH --job-name=cplvae_exp3
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=/cluster/scratch/tcasartelli/thermoelastic2d_constrained_plvae/logs/cplvae_exp3_%j.out
#SBATCH --error=/cluster/scratch/tcasartelli/thermoelastic2d_constrained_plvae/logs/cplvae_exp3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tcasartelli@ethz.ch

mkdir -p /cluster/scratch/tcasartelli/thermoelastic2d_constrained_plvae/logs
mkdir -p /cluster/scratch/tcasartelli/thermoelastic2d_constrained_plvae/images
mkdir -p /cluster/scratch/tcasartelli/thermoelastic2d_constrained_plvae/checkpoints

module purge
module load stack/2024-06
module load gcc/12.2.0
module load python_cuda/3.11.6
module load cuda/12.8.0
module load eth_proxy

VENV_DIR="/cluster/scratch/tcasartelli/engi-venv"
source "$VENV_DIR/bin/activate"

source ~/.secrets/ml_env.sh
export WANDB_MODE=online
export WANDB_PROJECT="engiopt"

python ~/projects/EngiOpt/vanilla_thomas/experiments/thermoelastic2d_constrained_plvae_ablation.py \
    --problem_id thermoelastic2d \
    --seed 1 \
    --n_epochs 10000 \
    --batch_size 128 \
    --lr 1e-4 \
    --latent_dim 100 \
    --resize_dimensions 100 100 \
    --nmse_threshold_rec 0.05 \
    --nmse_threshold_perf 0.05 \
    --pruning_epoch 500 \
    --pruning_threshold 0.05 \
    --pruning_strategy plummet \
    --w_reconstruction 0.0 \
    --save_model \
    --track
    