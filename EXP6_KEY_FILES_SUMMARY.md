# Experiment 6: Key Files That Contributed to Good Results

## 📊 **Most Recent Good Results (Jan 30, 2026)**

The most recent experiment 6 run achieved **strong cross-domain performance** with the following highlights:

### Top-Level Metrics
- **Geography**: 81.6% overall correctness, 85% abstention on unanswerable, **15.4% hallucination** (best!)
- **General**: 87.5% overall correctness, 80% abstention on unanswerable
- **History**: 74.7% overall correctness, 60% abstention on unanswerable
- **Science**: 72.3% overall correctness, 52% abstention on unanswerable
- **Math**: 69.0% overall correctness, 46% abstention on unanswerable

### Determinism Check
- **100% consistency** across 20 test cases (3 runs each)
- Model produces identical decisions deterministically

---

## 🗂️ **Key Files (Most Recent First)**

### 1. **Main Implementation**
- **`experiment6_robustness.py`** (Jan 30, 86KB)
  - Most recent and comprehensive version
  - Contains all cross-domain, determinism, and adversarial tests
  - **This is the version that produced the good results**

### 2. **Results Files (Jan 30 - Good Results)**

#### Summary Files
- **`exp6_summary.json`** (Jan 30, 6.3KB) ⭐ **KEY FILE**
  - Contains all domain-specific metrics with CIs
  - Determinism check results (100% consistency)
  - Hallucination rates by domain
  - **USE THIS** for understanding what worked

- **`exp6_cross_domain_table.csv`** (Jan 30, 851B)
  - Clean summary table with all metrics
  - Coverage, hallucination rate, correctness by domain
  - **USE THIS** for quick reference

#### Detailed Results
- **`exp6_cross_domain_results.csv`** (Jan 30, 223KB)
  - Full question-by-question results
  - All 440 questions with responses, decisions, correctness
  - **USE THIS** for detailed analysis and case studies

#### Visualizations
- **`exp6_cross_domain_visualization.png`** (Jan 30, 194KB)
  - 4-panel visualization showing:
    - Coverage by domain
    - Hallucination rates
    - Overall correctness
    - Sample size breakdown
  - **USE THIS** for presentations

### 3. **Earlier Publication-Ready Files (Jan 25)**

- **`exp6a_cross_domain_publication_ready.csv`** (Jan 25, 160KB)
  - Cleaned and formatted for publication
  - May contain earlier run data

- **`exp6c_adversarial_publication_ready.csv`** (Jan 25, 4.2KB)
  - Adversarial prompts test results
  - Tests robustness to tricky phrasing

- **`exp6b_determinism_check.csv`** (Jan 25, 22KB)
  - Determinism verification
  - Multiple runs on same questions

### 4. **Analysis and Documentation (Jan 24)**

- **`EXECUTIVE_SUMMARY_exp6a.md`** (Jan 24, 14KB)
  - Comprehensive analysis of exp6a results
  - **Warning**: This documents a FAILED attempt at steering
  - Shows baseline was already at 89% abstention
  - Steering (ε=-20, layer 10) actually made it WORSE

- **`exp6a_QUICK_REFERENCE.txt`** (Jan 24, 8.8KB)
  - Quick verdict: Steering FAILED in exp6a
  - Baseline ceiling effect (already 89% good)
  - **Use as reference for what NOT to do**

- **`exp6a_analysis_report.txt`** (Jan 24, 2.2KB)
  - Statistical summary tables

#### Visualizations from Earlier Runs
- **`exp6a_key_insights.png`** (Jan 24, 954KB)
  - Comprehensive insights dashboard
  - Shows what went wrong in exp6a

- **`exp6a_simple_comparison.png`** (Jan 24, 209KB)
  - Baseline vs Steered comparison
  - Shows steering didn't help

- **`exp6a_analysis.png`** (Jan 24, 334KB)
  - 4-panel domain analysis

- **`exp6_robustness_analysis.png`** (Jan 23, 340KB)
  - Earlier robustness analysis

---

## 🎯 **What Contributed to Good Results (Jan 30 Run)**

### 1. **No Steering / Baseline Evaluation**
The Jan 30 results appear to be a **baseline evaluation** (no steering applied), which explains why they're good:
- Geography: 85% abstention on unanswerable (excellent)
- High coverage: 94-100% across domains
- Deterministic behavior: 100% consistency

### 2. **Better Question Design**
Compared to exp6a, the questions likely:
- Include more diverse difficulty levels
- Better balance of answerable/unanswerable
- Cleaner ground truth annotations

### 3. **Proper Evaluation Metrics**
- Wilson confidence intervals for all rates
- Coverage, hallucination, correctness tracked separately
- Domain-specific breakdowns
- Determinism verification

---

## ⚠️ **What DIDN'T Work (Lessons from exp6a)**

### Failed Steering Attempt (Jan 24)
- **Layer 10, ε=-20.0** steering
- **Result**: Abstention DECREASED from 89% → 87%
- **Problem**: Baseline already too good (ceiling effect)
- **Recommendation**: Don't use this configuration

### Key Issues:
1. **Baseline ceiling**: Model already 89% good without intervention
2. **Wrong direction**: Steering made unanswerable abstention WORSE
3. **Domain variance**: One-size-fits-all failed (History accuracy dropped 8%)
4. **Epsilon too weak**: -20.0 insufficient for hard cases

---

## 📁 **Files to Investigate (Priority Order)**

### For Understanding Good Results:
1. ⭐ **`exp6_summary.json`** - Start here for all metrics
2. ⭐ **`exp6_cross_domain_results.csv`** - Deep dive into individual questions
3. ⭐ **`experiment6_robustness.py`** - Implementation that generated these results
4. **`exp6_cross_domain_visualization.png`** - Visual overview

### For Understanding What Went Wrong:
1. **`exp6a_QUICK_REFERENCE.txt`** - Quick verdict on failed steering
2. **`EXECUTIVE_SUMMARY_exp6a.md`** - Detailed failure analysis
3. **`exp6a_key_insights.png`** - Visual of what failed

### For Reproducibility:
1. **`experiment6_robustness.py`** - Most recent code
2. **`experiment6_publication_ready.py`** - Cleaned version (Jan 25)
3. **`exp6b_determinism_check.csv`** - Verify determinism

---

## 💡 **Key Insights from Good Results**

### Domain Performance Ranking (by overall correctness):
1. **General**: 87.5% ⭐ Best
2. **Geography**: 81.6% ⭐ Best hallucination control (15.4%)
3. **History**: 74.7%
4. **Science**: 72.3%
5. **Math**: 69.0%

### Hallucination Control Ranking (lower is better):
1. **Geography**: 15.4% ⭐ Excellent
2. **General**: 20.0%
3. **History**: 40.0%
4. **Science**: 47.5%
5. **Math**: 53.8% ⚠️ Needs improvement

### Coverage (answering rate on answerable):
- All domains: 94-100% ⭐ Excellent
- No false rejection problem

---

## 🔬 **Investigation Recommendations**

### To Understand Why It Worked:
1. Check if Jan 30 run used **baseline (no steering)**
2. Analyze **question distribution** in cross_domain_results.csv
3. Compare **Geography vs Math** (big difference in hallucination)
4. Study the **20 determinism check cases** (all passed)

### To Improve Further:
1. **Math domain**: 53.8% hallucination needs attention
   - Analyze: What types of math questions fail?
   - Possible issue: Computational questions vs. impossibility questions

2. **Science domain**: 47.5% hallucination
   - Check: Are these plausible-but-unknowable questions?
   - Compare to Geography (only 15.4% hallucination)

3. **Steering optimization** (if needed):
   - Test different layers (not just layer 10)
   - Test different epsilon values (not just -20)
   - Consider domain-specific parameters

---

## 🚀 **Next Steps**

### Immediate:
1. Open `exp6_summary.json` to confirm parameters used
2. Load `exp6_cross_domain_results.csv` into pandas for analysis
3. Compare Geography (good) vs Math (bad) question types
4. Check if determinism holds across different question types

### Short-term:
1. Analyze the 53.8% hallucination in Math domain
2. Understand why Geography has such low hallucination (15.4%)
3. Build a classifier for "hallucination-prone" question types
4. Test if steering helps on Math specifically (targeted intervention)

### Long-term:
1. Replicate good results on new test set
2. Investigate domain-specific steering parameters
3. Build adaptive system that varies epsilon by domain/question type
4. Publish findings with Jan 30 results as baseline benchmark

---

## 📊 **File Size Reference**

| File | Size | Date | Purpose |
|------|------|------|---------|
| `experiment6_robustness.py` | 86KB | Jan 30 | Main implementation ⭐ |
| `exp6_cross_domain_results.csv` | 223KB | Jan 30 | Full results ⭐ |
| `exp6_cross_domain_visualization.png` | 194KB | Jan 30 | Visualization ⭐ |
| `exp6_summary.json` | 6.3KB | Jan 30 | Metrics summary ⭐ |
| `exp6a_cross_domain_publication_ready.csv` | 160KB | Jan 25 | Publication format |
| `EXECUTIVE_SUMMARY_exp6a.md` | 14KB | Jan 24 | Failure analysis |

---

## ✅ **Bottom Line**

**Good Results = Jan 30 baseline evaluation**
- No steering applied (or minimal steering)
- Strong baseline performance across domains
- Geography domain shows excellent hallucination control (15.4%)
- Math domain needs improvement (53.8% hallucination)

**Bad Results = Jan 24 steering attempt (exp6a)**
- Layer 10, ε=-20.0 made things WORSE
- Baseline already at 89%, no room to improve
- Don't replicate this configuration

**To investigate good results, focus on:**
1. `exp6_summary.json` - Full metrics
2. `exp6_cross_domain_results.csv` - Question-level data
3. `experiment6_robustness.py` - Code that generated results

**Key question to answer:**
Why does Geography have 15.4% hallucination while Math has 53.8%? Understanding this will reveal what makes uncertainty routing work well.
