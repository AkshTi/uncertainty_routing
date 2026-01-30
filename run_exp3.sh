#!/bin/bash
#SBATCH --job-name=exp2_localization
#SBATCH --output=logs/exp2_%j.out
#SBATCH --error=logs/exp2_%j.err
#SBATCH --time=01:30:00

#SBATCH -p mit_normal_gpu
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -G 1


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
echo "Running Experiment 3: Robust Steering Control"
echo "=================================================="

# Run experiment
python experiment3_steering_robust.py

echo "=================================================="
echo "Experiment 3 completed at: $(date)"
echo "Check results in ./results/"
