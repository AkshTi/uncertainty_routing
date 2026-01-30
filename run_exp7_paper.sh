#!/bin/bash

# Experiment 7 - Publication Version Runner
# This script runs the fixed experiment and provides a quick summary

set -e  # Exit on error

echo "========================================="
echo "Experiment 7: Safety & Alignment Testing"
echo "Publication Version"
echo "========================================="
echo ""

# Check if required files exist
if [ ! -f "experiment7_safety_alignment_paper.py" ]; then
    echo "❌ Error: experiment7_safety_alignment_paper.py not found"
    exit 1
fi

if [ ! -f "core_utils.py" ]; then
    echo "❌ Error: core_utils.py not found"
    exit 1
fi

if [ ! -d "results" ]; then
    echo "⚠️  Warning: results/ directory not found, creating..."
    mkdir results
fi

# Check for steering vectors
if [ ! -f "results/steering_vectors_explicit.pt" ] && [ ! -f "results/steering_vectors.pt" ]; then
    echo "❌ Error: No steering vectors found in results/"
    echo "   Please run experiments 1-5 first to generate steering vectors"
    exit 1
fi

echo "✓ All prerequisites found"
echo ""

# Run the experiment
echo "Running Experiment 7..."
echo "This will take approximately 5-10 minutes..."
echo ""

python experiment7_safety_alignment_paper.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Experiment 7 completed successfully!"
    echo "========================================="
    echo ""

    # Quick summary from JSON
    if [ -f "results/exp7_summary_paper.json" ]; then
        echo "QUICK SUMMARY:"
        echo "-------------"

        # Extract key findings using Python
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
    else:
        print("   ? Unable to determine risk sensitivity")

    print("\n3. SPURIOUS CORRELATIONS:")
    spurious = data.get('spurious_correlations', {})
    consistency = spurious.get('avg_consistency_score', 0)
    length_sensitive = spurious.get('length_sensitive', False)

    print(f"   Average consistency: {consistency:.3f}")
    if not length_sensitive:
        print("   ✓ Good semantic understanding (not length-sensitive)")
    else:
        print("   ⚠ May be length-sensitive")

    print("\n" + "="*50)
    print("OVERALL ASSESSMENT:")
    print("="*50)

    # Determine overall quality
    score = 0
    if preserved:
        score += 1
    if risk_sensitive:
        score += 1
    if not length_sensitive:
        score += 1

    if score == 3:
        print("✓✓✓ EXCELLENT - All criteria met")
        print("    Ready for publication!")
    elif score == 2:
        print("✓✓ GOOD - Most criteria met")
        print("   Publishable with honest discussion of limitations")
    elif score == 1:
        print("✓ ACCEPTABLE - Some criteria met")
        print("  Publishable with careful framing")
    else:
        print("⚠ NEEDS WORK - Major issues found")
        print("  Consider revising approach")

    print("\nSee EXP7_FIXES_README.md for detailed interpretation guide")

except Exception as e:
    print(f"Error reading summary: {e}")
    sys.exit(1)
EOF

        echo ""
        echo "Files generated:"
        echo "  - results/exp7a_safety_preservation_paper.csv"
        echo "  - results/exp7b_risk_sensitive_abstention_paper.csv"
        echo "  - results/exp7c_spurious_correlations_paper.csv"
        echo "  - results/exp7_summary_paper.json"
        echo "  - results/exp7_safety_analysis_paper.png"
        echo ""
        echo "Next steps:"
        echo "  1. View the figure: open results/exp7_safety_analysis_paper.png"
        echo "  2. Read full analysis: cat results/exp7_summary_paper.json | python -m json.tool"
        echo "  3. Consult interpretation guide: cat EXP7_FIXES_README.md"

    else
        echo "⚠️  Warning: Summary file not generated"
    fi

else
    echo ""
    echo "❌ Experiment failed!"
    echo "Check error messages above"
    exit 1
fi
