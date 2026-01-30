#!/bin/bash
#SBATCH --job-name=exp4
#SBATCH --output=logs/exp4_%j.out
#SBATCH --error=logs/exp4_%j.err
#SBATCH --time=01:30:00

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
echo "Running Experiment 4: Steering Selectivity"
echo "=================================================="

# Run experiment with 300 questions for robust statistics
python experiment4_selectivity.py \
    --n_questions 200

echo "=================================================="
echo "Experiment 4 completed at: $(date)"
echo "Check results in ./results/"
