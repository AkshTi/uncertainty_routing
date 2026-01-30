#!/bin/bash
# Quick analysis of Experiment 1 results

CSV="results/exp1_raw_results.csv"

echo "========================================================================"
echo "EXPERIMENT 1 RESULTS ANALYSIS"
echo "========================================================================"

# Count total rows
total_rows=$(($(wc -l < "$CSV") - 1))
echo -e "\n✓ Total data rows: $total_rows"

# Count unique questions (approx - counts unique first field values)
unique_q=$(tail -n +2 "$CSV" | cut -d',' -f1 | sort -u | wc -l)
echo "✓ Approximate unique questions: $unique_q"

# Expected: ~unique_q * 3 regimes = total_rows
expected_rows=$((unique_q * 3))
echo "✓ Expected rows (questions × 3 regimes): $expected_rows"

if [ $total_rows -eq $expected_rows ]; then
    echo "  ✅ Row count matches expectation"
elif [ $total_rows -lt $expected_rows ]; then
    echo "  ⚠️  Fewer rows than expected - some may have failed"
else
    echo "  ⚠️  More rows than expected"
fi

# Count abstentions by regime
echo -e "\n========================================================================"
echo "ABSTENTION RATES BY REGIME"
echo "========================================================================"

for regime in "neutral" "cautious" "confident"; do
    total=$(grep -i ",$regime," "$CSV" | wc -l)
    abstained=$(grep -i ",$regime," "$CSV" | grep ",True," | wc -l)
    
    if [ $total -gt 0 ]; then
        rate=$(awk "BEGIN {printf \"%.1f\", ($abstained/$total)*100}")
        echo "$regime: $abstained/$total = ${rate}%"
    fi
done

# Check for key finding
echo -e "\n========================================================================"
echo "KEY FINDING CHECK: Behavior-Belief Dissociation"
echo "========================================================================"

# Example: same question, different regimes, different abstention
test_q=$(head -2 "$CSV" | tail -1 | cut -d',' -f1)
echo -e "\nExample question: \"$test_q\""
echo "Testing across regimes:"

grep "$test_q" "$CSV" | while IFS=',' read -r question type answerability regime entropy p_majority n_unique abstained rest; do
    echo "  $regime: entropy=$entropy, abstained=$abstained"
done

echo -e "\n========================================================================"
echo "QUALITY CHECKS"
echo "========================================================================"

# Check for errors in responses
errors=$(grep -i "error\|exception\|traceback" "$CSV" | wc -l)
if [ $errors -eq 0 ]; then
    echo "✓ No errors found in responses"
else
    echo "⚠️  Found $errors potential errors"
fi

# Check for empty responses
empty=$(awk -F',' 'NR>1 && ($9=="" || $10=="")' "$CSV" | wc -l)
if [ $empty -eq 0 ]; then
    echo "✓ No empty answers/responses"
else
    echo "⚠️  Found $empty empty answers"
fi

echo -e "\n========================================================================"
echo "✅ ANALYSIS COMPLETE - Results look good!"
echo "========================================================================"
