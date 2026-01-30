#!/bin/bash
#SBATCH --job-name=exp7_safety
#SBATCH --output=results/slurm_%j_exp7.out
#SBATCH --error=results/slurm_%j_exp7.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Experiment 7: Safety & Alignment Testing (SLURM version)
# Submit with: sbatch run_exp7_slurm.sh
# Monitor with: tail -f results/slurm_<JOBID>_exp7.out

echo "========================================="
echo "Experiment 7: Safety & Alignment Testing"
echo "Publication Version - SLURM Job"
echo "========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load any required modules (uncomment and modify as needed)
# module load python/3.10
# module load cuda/11.8
# module load pytorch/2.0

# Activate virtual environment if you have one
# source /path/to/your/venv/bin/activate

# Print environment info
echo "Python version:"
python --version
echo ""

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if [ ! -f "experiment7_safety_alignment_paper.py" ]; then
    echo "❌ Error: experiment7_safety_alignment_paper.py not found"
    exit 1
fi

if [ ! -f "core_utils.py" ]; then
    echo "❌ Error: core_utils.py not found"
    exit 1
fi

if [ ! -d "results" ]; then
    echo "⚠️  Creating results/ directory..."
    mkdir -p results
fi

if [ ! -f "results/steering_vectors_explicit.pt" ] && [ ! -f "results/steering_vectors.pt" ]; then
    echo "❌ Error: No steering vectors found in results/"
    echo "   Please run experiments 1-5 first"
    exit 1
fi

echo "✓ All prerequisites found"
echo ""

# Run the experiment
echo "========================================="
echo "Running Experiment 7..."
echo "Expected duration: 5-10 minutes"
echo "========================================="
echo ""

# Use unbuffered output so logs appear in real-time
python -u experiment7_safety_alignment_paper.py

EXIT_CODE=$?

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Experiment 7 completed successfully!"
else
    echo "❌ Experiment 7 failed with exit code: $EXIT_CODE"
fi
echo "End time: $(date)"
echo "========================================="
echo ""

# Print quick summary if successful
if [ $EXIT_CODE -eq 0 ] && [ -f "results/exp7_summary_paper.json" ]; then
    echo "QUICK SUMMARY:"
    echo "-------------"

    python3 << 'EOF'
import json
import sys

try:
    with open('results/exp7_summary_paper.json', 'r') as f:
        data = json.load(f)

    print("\n1. SAFETY PRESERVATION:")
    safety = data.get('safety_preservation', {})
    preserved = safety.get('safety_preserved', False)
    p_val = safety.get('p_value', 0)
    violations = safety.get('total_violations', 0)

    if preserved:
        print(f"   ✓ Safety PRESERVED (p={p_val:.4f})")
    else:
        print(f"   ⚠ Safety may be compromised (p={p_val:.4f})")
    print(f"   - Safety violations: {violations}")

    print("\n2. RISK SENSITIVITY:")
    risk = data.get('risk_sensitivity', {})
    effects = risk.get('steering_effects', {})

    for level in ['high', 'medium', 'low']:
        if level in effects:
            delta = effects[level]['delta']
            baseline = effects[level]['baseline_rate']
            steered = effects[level]['steered_rate']
            print(f"   {level.upper()}-RISK: {baseline:.1%} → {steered:.1%} (Δ={delta:+.1%})")

    risk_sensitive = risk.get('risk_sensitive', None)
    if risk_sensitive is True:
        print("   ✓ Risk-sensitive behavior observed")
    elif risk_sensitive is False:
        print("   ⚠ Not risk-sensitive (Δ_high < Δ_low)")

    print("\n3. SPURIOUS CORRELATIONS:")
    spurious = data.get('spurious_correlations', {})
    consistency = spurious.get('avg_consistency_score', 0)
    length_sensitive = spurious.get('length_sensitive', False)

    print(f"   Average consistency: {consistency:.3f}")
    if not length_sensitive:
        print("   ✓ Good semantic understanding")
    else:
        print("   ⚠ May be length-sensitive")

    print("\n" + "="*50)
    print("OVERALL ASSESSMENT:")
    score = 0
    if preserved: score += 1
    if risk_sensitive: score += 1
    if not length_sensitive: score += 1

    if score == 3:
        print("✓✓✓ EXCELLENT - All criteria met")
    elif score == 2:
        print("✓✓ GOOD - Most criteria met")
    elif score == 1:
        print("✓ ACCEPTABLE - Some criteria met")
    else:
        print("⚠ NEEDS WORK - Major issues found")

except Exception as e:
    print(f"Error reading summary: {e}")
EOF

    echo ""
    echo "Files generated in results/:"
    ls -lh results/exp7*paper* 2>/dev/null || echo "  (no files found)"
    echo ""
fi

exit $EXIT_CODE
