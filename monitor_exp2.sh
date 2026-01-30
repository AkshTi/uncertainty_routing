#!/bin/bash
# Monitor Experiment 2 progress

echo "=== Experiment 2 Monitor ==="
echo ""

# Find the most recent exp2 job
EXP2_LOG=$(ls -t logs/exp2_*.out 2>/dev/null | head -1)

if [ -z "$EXP2_LOG" ]; then
    echo "No exp2 log found yet. Job may not have started."
    echo ""
    echo "Check job queue:"
    squeue -u $USER | grep exp2
    exit 0
fi

echo "Latest log: $EXP2_LOG"
echo ""

# Check for errors
if grep -i "error\|exception\|traceback" "$EXP2_LOG" > /dev/null 2>&1; then
    echo "⚠️  ERRORS DETECTED:"
    echo ""
    grep -i "error\|exception" "$EXP2_LOG" | tail -5
    echo ""
    echo "Full error log:"
    tail -30 "$EXP2_LOG"
    exit 1
fi

# Check progress
echo "Progress indicators:"
echo ""

if grep -q "Initializing model" "$EXP2_LOG"; then
    echo "✓ Model initialized"
fi

if grep -q "Loading datasets" "$EXP2_LOG"; then
    echo "✓ Datasets loaded"
fi

if grep -q "STEP 1" "$EXP2_LOG"; then
    echo "✓ Window sweep started"
fi

if grep -q "STEP 2" "$EXP2_LOG"; then
    echo "✓ Position sweep started"
fi

if grep -q "STEP 3" "$EXP2_LOG"; then
    echo "✓ Negative controls started"
fi

if grep -q "Experiment 2 complete" "$EXP2_LOG"; then
    echo "✓ COMPLETED!"
    echo ""
    echo "Results saved to:"
    ls -lh results/exp2_* 2>/dev/null
    exit 0
fi

# Show last few lines
echo ""
echo "Recent activity:"
tail -10 "$EXP2_LOG"

echo ""
echo "---"
echo "To follow live: tail -f $EXP2_LOG"
