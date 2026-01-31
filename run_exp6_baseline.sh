#!/bin/bash
#SBATCH --job-name=exp6_baseline
#SBATCH --output=logs/exp6_baseline_%j.out
#SBATCH --error=logs/exp6_baseline_%j.err
#SBATCH --time=01:30:00
#SBATCH -p mit_normal_gpu
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -G h200:1

module load python/3.9
module load cuda/11.8

mkdir -p logs

echo "Running Experiment 6 BASELINE (ε=0)"
echo "=================================================="

python experiment6_robustness.py \
    --n_per_domain 50 \
    --epsilon 0.0

echo "=================================================="
echo "Baseline experiment completed at: $(date)"
