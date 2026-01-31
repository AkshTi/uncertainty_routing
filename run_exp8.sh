#!/bin/bash
#SBATCH --job-name=exp8_scaling
#SBATCH --output=logs/exp8_%j.out
#SBATCH --error=logs/exp8_%j.err
#SBATCH --time=03:00:00

#SBATCH -p mit_normal_gpu
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -G h200:1

# Load required modules
module load python/3.9
module load cuda/11.8

# Activate virtual environment if you have one
# source ~/venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration (adjust these as needed)
N_QUESTIONS=${N_QUESTIONS:-50}       # Total questions per category (default: 50)
MIN_PER_SPLIT=${MIN_PER_SPLIT:-10}   # Minimum per split (default: 10)

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Running Experiment 8: Scaling Analysis"
echo "Testing models: Qwen2.5-1.5B, 3B, 7B"
echo "Configuration: ${N_QUESTIONS} questions per category, ${MIN_PER_SPLIT} min per split"
echo "=================================================="

# Run experiment with configurable parameters
python experiment8_scaling_analysis.py \
    --n_questions ${N_QUESTIONS} \
    --min_per_split ${MIN_PER_SPLIT}

echo "=================================================="
echo "Experiment 8 completed at: $(date)"
echo "Check results in ./results/"
echo "  - exp8_scaling_summary.csv"
echo "  - exp8_scaling_analysis.png"
echo "  - exp8_summary.json"
