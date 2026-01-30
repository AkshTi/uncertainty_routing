# SLURM Monitoring Guide for Experiment 7

## Quick Reference

### Submit the job:
```bash
sbatch run_exp7_slurm.sh
```

### Check job status:
```bash
squeue -u $USER
```

### View logs in real-time:
```bash
# Find your job ID first (from squeue output)
JOBID=<your_job_id>

# Watch output log
tail -f results/slurm_${JOBID}_exp7.out

# Watch error log (in another terminal)
tail -f results/slurm_${JOBID}_exp7.err
```

---

## Detailed Commands

### 1. **Submit Job**
```bash
sbatch run_exp7_slurm.sh
```

Output will be:
```
Submitted batch job 1234567
```

### 2. **Check Job Status**
```bash
# See your jobs
squeue -u $USER

# See specific job
squeue -j 1234567

# See detailed info
scontrol show job 1234567
```

Status codes:
- `PD` = Pending (waiting for resources)
- `R` = Running
- `CG` = Completing
- `CD` = Completed
- `F` = Failed

### 3. **Monitor Logs While Running**

#### Option A: Real-time tail (recommended)
```bash
# Replace 1234567 with your actual job ID
tail -f results/slurm_1234567_exp7.out
```

Press `Ctrl+C` to stop watching.

#### Option B: Continuous monitoring with auto-refresh
```bash
watch -n 5 "tail -n 50 results/slurm_1234567_exp7.out"
```

This refreshes every 5 seconds.

#### Option C: Both stdout and stderr together
```bash
tail -f results/slurm_1234567_exp7.out results/slurm_1234567_exp7.err
```

#### Option D: Use `less` for scrolling
```bash
# While job is running, refresh with Shift+F
less +F results/slurm_1234567_exp7.out
```

### 4. **After Job Completes**

```bash
# View full output
cat results/slurm_1234567_exp7.out

# View full errors
cat results/slurm_1234567_exp7.err

# Check last 100 lines
tail -n 100 results/slurm_1234567_exp7.out

# Search for specific text
grep "SAFETY PRESERVATION" results/slurm_1234567_exp7.out
```

### 5. **Check Job Details**

```bash
# Why is my job pending?
squeue -j 1234567 --start

# Job accounting info (after completion)
sacct -j 1234567 --format=JobID,JobName,Elapsed,State,ExitCode

# Detailed accounting
sacct -j 1234567 -l
```

### 6. **Cancel Job** (if needed)
```bash
scancel 1234567
```

---

## Expected Log Output

### During Startup:
```
=========================================
Experiment 7: Safety & Alignment Testing
Publication Version - SLURM Job
=========================================

Job ID: 1234567
Node: node042
Start time: Tue Jan 28 14:30:00 EST 2026

Python version:
Python 3.10.8

GPU info:
name, memory.total [MiB], memory.free [MiB]
NVIDIA A100-SXM4-40GB, 40960 MiB, 40000 MiB

Checking prerequisites...
✓ All prerequisites found
```

### During Experiment:
```
======================================================================
EXPERIMENT 7A: Safety Preservation
======================================================================
Testing if uncertainty steering (ε=-20.0) affects safety refusals
Hypothesis: Safety and uncertainty are orthogonal -> no effect expected

Testing safety: 100%|██████████| 10/10 [00:45<00:00,  4.5s/it]

======================================================================
EXPERIMENT 7B: Risk-Sensitive Abstention
======================================================================
...
```

### At Completion:
```
=========================================
✓ Experiment 7 completed successfully!
End time: Tue Jan 28 14:38:23 EST 2026
=========================================

QUICK SUMMARY:
-------------

1. SAFETY PRESERVATION:
   ✓ Safety PRESERVED (p=0.8234)
   - Safety violations: 0

2. RISK SENSITIVITY:
   HIGH-RISK: 33.3% → 83.3% (Δ=+50.0%)
   MEDIUM-RISK: 25.0% → 66.7% (Δ=+41.7%)
   LOW-RISK: 0.0% → 40.0% (Δ=+40.0%)
   ✓ Risk-sensitive behavior observed

3. SPURIOUS CORRELATIONS:
   Average consistency: 0.123
   ✓ Good semantic understanding

==================================================
OVERALL ASSESSMENT:
✓✓✓ EXCELLENT - All criteria met
```

---

## Troubleshooting

### Job stuck in PENDING?
```bash
# Check why
squeue -j 1234567 --start

# Common reasons:
# - Resources: Not enough GPUs available
# - Priority: Other jobs ahead in queue
# - Limits: Hit job/resource limits
```

### Job failed immediately?
```bash
# Check error log
cat results/slurm_1234567_exp7.err

# Common issues:
# - Module not loaded
# - Python environment not activated
# - Missing files (steering vectors)
# - GPU not available
```

### Can't find log files?
```bash
# List all slurm logs
ls -lt results/slurm_*_exp7.*

# If job failed before creating logs, check:
ls -lt slurm-*.out slurm-*.err
```

### Job killed due to time/memory?
```bash
# Check what happened
sacct -j 1234567 --format=JobID,State,ExitCode,MaxRSS,Elapsed

# If TIMEOUT: Increase --time in sbatch script
# If OUT_OF_MEMORY: Increase --mem in sbatch script
```

---

## Advanced Monitoring

### Get notified when job completes:
Add to sbatch script:
```bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@institution.edu
```

### Monitor GPU usage during job:
```bash
# SSH to the node (find node from squeue)
ssh node042

# Watch GPU
watch -n 1 nvidia-smi
```

### Monitor from outside:
```bash
# Create a monitoring script
cat > monitor_exp7.sh << 'EOF'
#!/bin/bash
JOBID=$1
while squeue -j $JOBID 2>/dev/null | grep -q $JOBID; do
    clear
    echo "=== Job Status ==="
    squeue -j $JOBID
    echo ""
    echo "=== Latest Output ==="
    tail -n 30 results/slurm_${JOBID}_exp7.out 2>/dev/null || echo "Log not created yet"
    sleep 10
done
echo "Job completed!"
EOF

chmod +x monitor_exp7.sh

# Use it:
./monitor_exp7.sh 1234567
```

---

## Quick One-Liners

### Submit and immediately watch:
```bash
JOBID=$(sbatch run_exp7_slurm.sh | awk '{print $4}') && \
echo "Job ID: $JOBID" && \
tail -f results/slurm_${JOBID}_exp7.out
```

### Check if job is done:
```bash
squeue -j 1234567 || echo "Job completed or not found"
```

### View summary only:
```bash
grep -A 30 "QUICK SUMMARY" results/slurm_1234567_exp7.out
```

### Check all your exp7 jobs:
```bash
squeue -u $USER -n exp7_safety
```

---

## File Locations

After job completes, find results in:
```
results/
├── slurm_1234567_exp7.out              # Main log
├── slurm_1234567_exp7.err              # Error log
├── exp7a_safety_preservation_paper.csv
├── exp7b_risk_sensitive_abstention_paper.csv
├── exp7c_spurious_correlations_paper.csv
├── exp7_summary_paper.json
└── exp7_safety_analysis_paper.png
```

---

## Example Workflow

```bash
# 1. Submit job
sbatch run_exp7_slurm.sh
# Output: Submitted batch job 1234567

# 2. Check it started
squeue -u $USER

# 3. Watch logs in real-time
tail -f results/slurm_1234567_exp7.out

# 4. When complete, view summary
cat results/slurm_1234567_exp7.out | grep -A 40 "QUICK SUMMARY"

# 5. View the figure (if using X11 forwarding)
display results/exp7_safety_analysis_paper.png

# Or copy to local machine:
# scp user@cluster:path/to/results/exp7_safety_analysis_paper.png .

# 6. Check detailed statistics
cat results/exp7_summary_paper.json | python -m json.tool
```

---

## Common Customizations

### Need more time?
Edit `run_exp7_slurm.sh`:
```bash
#SBATCH --time=04:00:00  # 4 hours instead of 2
```

### Need more memory?
```bash
#SBATCH --mem=64G  # 64GB instead of 32GB
```

### Use specific GPU type?
```bash
#SBATCH --gres=gpu:a100:1  # Request A100 specifically
```

### Multiple GPUs?
```bash
#SBATCH --gres=gpu:2  # Request 2 GPUs
```

### Higher priority queue?
```bash
#SBATCH --partition=high_priority
```

---

## Debugging Failed Jobs

### Step 1: Check exit code
```bash
sacct -j 1234567 --format=JobID,State,ExitCode
```

### Step 2: Read error log
```bash
cat results/slurm_1234567_exp7.err
```

### Step 3: Check last lines of output
```bash
tail -n 50 results/slurm_1234567_exp7.out
```

### Step 4: Test interactively
```bash
# Request interactive session
srun --pty --gres=gpu:1 --mem=32G --time=1:00:00 bash

# Then run manually
python -u experiment7_safety_alignment_paper.py
```

---

**Pro tip**: Keep the job ID handy! Set it as a variable:
```bash
export JOBID=1234567
tail -f results/slurm_${JOBID}_exp7.out
```
