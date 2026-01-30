# SLURM Job Submission Guide

## Quick Start

### Submit All Experiments at Once
```bash
./submit_all_experiments.sh
```

### Submit Individual Experiments
```bash
sbatch run_exp1.sh  # Experiment 1: Behavior-Belief Dissociation
sbatch run_exp2.sh  # Experiment 2: Gate Localization
sbatch run_exp3.sh  # Experiment 3: Robust Steering Control
sbatch run_exp4.sh  # Experiment 4: Steering Selectivity
sbatch run_exp5.sh  # Experiment 5: Risk-Coverage Tradeoff
sbatch run_exp6.sh  # Experiment 6: Cross-Domain Robustness
sbatch run_exp7.sh  # Experiment 7: Separability from Refusal
```

## Resource Allocation

All jobs are configured with:
- **GPU**: 1 GPU
- **Memory**: 32GB RAM
- **CPUs**: 4 cores
- **Time limits**:
  - Exp 1: 4 hours
  - Exp 2: 3 hours
  - Exp 3: 4 hours
  - Exp 4: 5 hours (larger dataset)
  - Exp 5: 6 hours (entropy computation intensive)
  - Exp 6: 4 hours
  - Exp 7: 5 hours

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <JOB_ID>

# View detailed job info
scontrol show job <JOB_ID>

# Cancel a job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER
```

## Checking Results

### Logs
- **Standard output**: `logs/exp<N>_<JOB_ID>.out`
- **Error output**: `logs/exp<N>_<JOB_ID>.err`

```bash
# Tail the most recent log
tail -f logs/exp1_*.out

# Check for errors
cat logs/exp1_*.err
```

### Results
All experiment results are saved to `./results/`:
- CSV files with raw data
- PNG files with visualizations
- JSON files with summaries

```bash
ls -lh results/
```

## Customizing Parameters

Edit the respective `run_exp<N>.sh` file to modify experiment parameters:

### Example: Increase dataset size for Exp 1
```bash
# Edit run_exp1.sh
python experiment1_behavior_belief.py \
    --n_answerable 500 \    # Changed from 200
    --n_unanswerable 500    # Changed from 200
```

### Example: Quick test mode
```bash
# Add --quick_test flag to any experiment
python experiment3_steering_robust.py --quick_test
```

## Adjusting SLURM Resources

Edit the `#SBATCH` directives at the top of each script:

```bash
#SBATCH --time=08:00:00      # Increase to 8 hours
#SBATCH --mem=64G            # Increase to 64GB RAM
#SBATCH --gres=gpu:2         # Request 2 GPUs
#SBATCH --partition=gpu-long # Use different partition
```

## Dependencies

Make sure you have installed:
- torch
- transformers
- pandas
- matplotlib
- seaborn
- numpy
- sklearn
- tqdm
- **scipy** (for Experiment 6)
- **statsmodels** (for Experiment 4)

## Troubleshooting

### Out of Memory
Increase `#SBATCH --mem=` or reduce dataset size with `--quick_test`

### Out of Time
Increase `#SBATCH --time=` or reduce dataset size

### GPU Not Available
Check available partitions: `sinfo`
Change partition: `#SBATCH --partition=<partition_name>`

### Module Not Found
Update module names in script:
```bash
module load python/3.10  # Use your cluster's Python version
module load cuda/12.0    # Use your cluster's CUDA version
```

## Sequential Execution (Exp 3 → Exp 4,5,6,7)

Experiments 4-7 depend on Experiment 3's steering vectors. To run them sequentially:

```bash
# Submit Exp 3 first
EXP3_JOB=$(sbatch run_exp3.sh | awk '{print $4}')

# Submit Exp 4-7 with dependency
sbatch --dependency=afterok:$EXP3_JOB run_exp4.sh
sbatch --dependency=afterok:$EXP3_JOB run_exp5.sh
sbatch --dependency=afterok:$EXP3_JOB run_exp6.sh
sbatch --dependency=afterok:$EXP3_JOB run_exp7.sh
```
