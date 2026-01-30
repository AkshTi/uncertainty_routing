#!/bin/bash
#SBATCH --job-name=exp2_localization
#SBATCH --output=logs/exp2_%j.out
#SBATCH --error=logs/exp2_%j.err
#SBATCH --time=01:30:00
#SBATCH --partition=sched_mit_psfc_gpu_r8
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=A100
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load required modules
module load python/3.9
module load cuda/11.8

# Activate virtual environment if you have one
# source ~/venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Running Experiment 2: Gate Localization"
echo "=================================================="

# Run experiment
python experiment2_localization.py \
    --n_pairs 10

echo "=================================================="
echo "Experiment 2 completed at: $(date)"
echo "Check results in ./results/"
