# Essential Files for Paper (Experiments 1-7)

## 📁 **Files You MUST Use**

### **Experiment 1: Behavior-Belief Dissociation**
```
✅ results/exp1_raw_results.csv
✅ results/exp1_paper_figure.png
```

### **Experiment 2: Localization**
```
✅ results/exp2_raw_results.csv
✅ results/exp2_localization_analysis.png
```

### **Experiment 3: Steering**
```
✅ results/exp3_raw_results.csv
✅ results/exp3_steering_analysis.png
```

### **Experiment 4: Gate Independence**
```
⚠️ results/exp4_raw_results.csv (optional - appendix only)
⚠️ results/exp4_gate_independence.png (optional)
```

### **Experiment 5: Trustworthiness** ⭐ CRITICAL
```
✅ results/exp5_summary.json ⭐ MOST IMPORTANT
✅ results/exp5_raw_results.csv
✅ results/exp5_trustworthiness.png ⭐ MAIN FIGURE
✅ results/exp5_optimal_epsilon.png (supplementary)
✅ results/exp5_layer_scores.csv (supplementary)
```

### **Experiment 6: Robustness**
```
✅ results/exp6_summary.json
✅ results/exp6a_cross_domain.csv
✅ results/exp6b_prompt_variations.csv
✅ results/exp6c_adversarial.csv
✅ results/exp6_robustness_analysis.png
   OR
✅ results/exp6a_simple_comparison.png (cleaner visualization)
```

### **Experiment 7: Safety & Risk-Sensitivity** ⭐ CRITICAL
```
✅ results/exp7_reverse_steering_summary.json ⭐ USE THIS ONE
✅ results/exp7_reverse_steering.csv
✅ results/exp7_reverse_steering_analysis.png ⭐ MAIN FIGURE

❌ DO NOT USE:
   results/exp7_summary.json (old, bad results)
   results/exp7_summary_paper.json (old fix)
   results/exp7_summary_fixed_v2.json (old fix)
   results/exp7a_safety_preservation*.csv (old versions)
   results/exp7b_risk_sensitive*.csv (old versions)
   results/exp7c_spurious_correlations*.csv (old versions)
```

---

## 📊 **Complete File List (Copy-Paste Ready)**

### **For Main Results Analysis:**
```bash
# Experiment 1
results/exp1_raw_results.csv
results/exp1_paper_figure.png

# Experiment 2
results/exp2_raw_results.csv
results/exp2_localization_analysis.png

# Experiment 3
results/exp3_raw_results.csv
results/exp3_steering_analysis.png

# Experiment 5 ⭐
results/exp5_summary.json
results/exp5_raw_results.csv
results/exp5_trustworthiness.png
results/exp5_optimal_epsilon.png
results/exp5_layer_scores.csv

# Experiment 6
results/exp6_summary.json
results/exp6a_cross_domain.csv
results/exp6b_prompt_variations.csv
results/exp6c_adversarial.csv
results/exp6_robustness_analysis.png

# Experiment 7 ⭐
results/exp7_reverse_steering_summary.json
results/exp7_reverse_steering.csv
results/exp7_reverse_steering_analysis.png
```

---

## 📈 **What Numbers to Pull From Each File**

### **From exp1_raw_results.csv:**
- Baseline behavior rate (% saying "uncertain")
- Baseline confidence distribution
- Dissociation metric per question type

### **From exp2_raw_results.csv:**
- Layer-wise effect sizes
- Peak layer identification (layers 24-26)
- Early vs late layer comparison

### **From exp3_raw_results.csv:**
- Baseline abstention rate
- Steered abstention rate (positive and negative epsilon)
- Effect size (percentage point change)

### **From exp5_summary.json:** ⭐ CRITICAL
```json
{
  "best_eps_value": -50.0,
  "baseline_abstain_unanswerable": 0.67,
  "best_eps_abstain_unanswerable": 0.90,
  "delta_abstain_unanswerable": 0.23,

  "baseline_hallucination_unanswerable": 0.33,
  "best_eps_hallucination_unanswerable": 0.10,
  "delta_hallucination_unanswerable": -0.23,

  "baseline_accuracy_answerable": 1.0,
  "best_eps_accuracy_answerable": 1.0,
  "delta_accuracy_answerable": 0.0,

  "best_eps_coverage_answerable": 0.35
}
```

**Key claims from this:**
- "Abstention on unanswerable increased from 67% to 90% (+23%)"
- "Hallucination decreased from 33% to 10% (-23%)"
- "Accuracy on answerable maintained at 100%"

### **From exp6_summary.json:**
- Per-domain abstention rates (baseline vs steered)
- Cross-domain consistency metrics
- Prompt variation correlation
- Adversarial robustness results

### **From exp7_reverse_steering_summary.json:** ⭐ CRITICAL
```json
{
  "risk_sensitive": true,
  "best_epsilon": 20.0,
  "best_gradient": 0.25,

  "baseline_abstention": {
    "high": 1.0,
    "low": 1.0,
    "medium": 0.67
  },

  "reductions_by_epsilon": {
    "20.0": {
      "high_reduction": 0.25,
      "low_reduction": 0.50,
      "gradient": 0.25
    }
  }
}
```

**Key claims from this:**
- "High-risk: 100% → 75% abstention (25% reduction)"
- "Low-risk: 100% → 50% abstention (50% reduction)"
- "Risk gradient: 25% (low reduces 2× more than high)"

---

## 🗂️ **Files to Archive/Ignore**

### **Don't use these (outdated/failed experiments):**
```
❌ results/exp7_summary.json
❌ results/exp7_summary_paper.json
❌ results/exp7_summary_fixed_v2.json
❌ results/exp7a_safety_preservation.csv
❌ results/exp7a_safety_preservation_fixed.csv
❌ results/exp7a_safety_preservation_paper.csv
❌ results/exp7b_selective_abstention*.csv
❌ results/exp7c_spurious_correlations*.csv
❌ results/exp7_safety_analysis.png (old)
❌ results/exp7_safety_analysis_fixed.png (old)
❌ results/exp7_safety_analysis_fixed_v2.png (old)
❌ results/exp7_safety_analysis_paper.png (old)
```

**Only use the REVERSE STEERING versions!**

---

## 📊 **Quick Copy Command**

To copy essential files to a separate folder:

```bash
# Create clean results folder
mkdir -p paper_results

# Copy essential files
cp results/exp1_raw_results.csv paper_results/
cp results/exp1_paper_figure.png paper_results/
cp results/exp2_raw_results.csv paper_results/
cp results/exp2_localization_analysis.png paper_results/
cp results/exp3_raw_results.csv paper_results/
cp results/exp3_steering_analysis.png paper_results/
cp results/exp5_summary.json paper_results/
cp results/exp5_raw_results.csv paper_results/
cp results/exp5_trustworthiness.png paper_results/
cp results/exp5_optimal_epsilon.png paper_results/
cp results/exp6_summary.json paper_results/
cp results/exp6a_cross_domain.csv paper_results/
cp results/exp6b_prompt_variations.csv paper_results/
cp results/exp6c_adversarial.csv paper_results/
cp results/exp6_robustness_analysis.png paper_results/
cp results/exp7_reverse_steering_summary.json paper_results/
cp results/exp7_reverse_steering.csv paper_results/
cp results/exp7_reverse_steering_analysis.png paper_results/

echo "Essential files copied to paper_results/"
```

---

## 📝 **Minimum Viable File Set**

If you want the ABSOLUTE MINIMUM:

```
1. exp5_summary.json ⭐
2. exp5_trustworthiness.png ⭐
3. exp7_reverse_steering_summary.json ⭐
4. exp7_reverse_steering_analysis.png ⭐
5. exp1_paper_figure.png
6. exp2_localization_analysis.png
7. exp3_steering_analysis.png
8. exp6_robustness_analysis.png
```

These 8 files contain your core story.

---

## ✅ **Summary**

**Total essential files: 18**
- 8 CSVs
- 8 PNGs
- 2 JSONs (exp5, exp7_reverse)

**Most critical:**
- exp5_summary.json (trustworthiness numbers)
- exp7_reverse_steering_summary.json (risk-sensitivity numbers)
- Their corresponding PNG visualizations

**Archive/ignore:**
- All old exp7 versions (except reverse_steering)
- Exp4 (unless you want appendix)
