#!/bin/bash
#SBATCH --job-name=exp2_pos
#SBATCH --output=logs/exp2_pos_%j.out
#SBATCH --error=logs/exp2_pos_%j.err
#SBATCH --time=00:15:00

#SBATCH -p mit_normal_gpu
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -G h200:1

# Load required modules
module load python/3.9
module load cuda/11.8

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Running Experiment 2: Position Sweep (FIXED)"
echo "=================================================="

# Run position sweep with fix
python rerun_position_sweep.py

# Regenerate summary
python regenerate_exp2_summary.py

echo "=================================================="
echo "Experiment 2 position sweep completed at: $(date)"
echo "Check results in ./results/"
