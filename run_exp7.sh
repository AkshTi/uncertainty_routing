#!/bin/bash
#SBATCH --job-name=exp7_separability
#SBATCH --output=logs/exp7_%j.out
#SBATCH --error=logs/exp7_%j.err
#SBATCH --time=02:00:00

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
echo "Running Experiment 7: Separability from Refusal"
echo "=================================================="

# Run experiment
python experiment7_separability.py \
    --n_harmful 50 \
    --n_benign 50 \
    --n_per_risk 20

echo "=================================================="
echo "Experiment 7 completed at: $(date)"
echo "Check results in ./results/"
