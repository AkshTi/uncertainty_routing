#!/bin/bash
#SBATCH --job-name=exp3_full
#SBATCH --output=logs/exp3_full_%j.out
#SBATCH --error=logs/exp3_full_%j.err
#SBATCH --time=06:00:00

#SBATCH -p mit_normal_gpu
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -G h200:1


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
echo "Running Experiment 3: Robust Steering Control (FULL MODE)"
echo "=================================================="

# Run full experiment (no --quick_test flag)
# This will take ~5 hours with the optimizations
python experiment3_steering_robust.py

echo "=================================================="
echo "Experiment 3 (FULL) completed at: $(date)"
echo "Check results in ./results/"
