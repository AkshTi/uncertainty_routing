#!/bin/bash
#SBATCH --job-name=exp6_cross_domain
#SBATCH --output=logs/exp6_%j.out
#SBATCH --error=logs/exp6_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
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
echo "Running Experiment 6: Cross-Domain Robustness"
echo "=================================================="

# Run experiment
python experiment6_robustness.py \
    --n_per_domain 50 \
    --epsilon 5.0

echo "=================================================="
echo "Experiment 6 completed at: $(date)"
echo "Check results in ./results/"
