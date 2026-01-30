#!/bin/bash
# Submit all experiments to SLURM
# Usage: ./submit_all_experiments.sh

echo "=========================================="
echo "Submitting all experiments to SLURM"
echo "=========================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Submit each experiment
for i in {1..7}; do
    echo "Submitting Experiment $i..."
    JOB_ID=$(sbatch run_exp${i}.sh | awk '{print $4}')
    echo "  → Job ID: $JOB_ID"
    echo ""
done

echo "=========================================="
echo "All experiments submitted!"
echo "=========================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check logs in: ./logs/"
echo "Check results in: ./results/"
echo ""
echo "To cancel all jobs: scancel -u \$USER"
