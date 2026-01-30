# Experiment 3 Critical Fix: Format Prompt Issue

## Problem Found

**Exp3 completely failed with 0% flip rate across ALL conditions.**

### Root Cause
The `format_prompt()` function used mode "neutral" which just says:
```
"Answer the following question concisely."
```

This produces normal answers like:
- `"49."`
- `"Tokyo."`
- `"25."`

But Exp3's abstention detection expects:
- `"ABSTAIN: [reasoning]"`
- `"ANSWER: [answer]"`

**Result**: 0% abstentions detected → 0% flips possible → experiment completely invalid!

## The Fix

### 1. Added new prompt mode: `"abstain_or_answer"`

**Location**: `data_preparation.py` line 294-301

```python
"abstain_or_answer": (
    "You must begin your response with EXACTLY one of these two options:\n"
    "1. 'ABSTAIN:' if the question cannot be answered with certainty\n"
    "2. 'ANSWER:' if you can provide a definite answer\n\n"
    "After the prefix, provide your reasoning or answer. "
    "You MUST start with either 'ABSTAIN:' or 'ANSWER:' - no other format is acceptable."
)
```

### 2. Updated all format_prompt calls in Exp3

**Changed**: `format_prompt(question, "neutral", context)`
**To**: `format_prompt(question, "abstain_or_answer", context)`

**Files modified**:
- `experiment3_steering_robust.py` lines 369-371, 404, 773, 793

## Expected Impact

### Before Fix:
- Responses: `"49."`, `"Tokyo."`
- Abstentions detected: 0%
- Flips: 0%
- **Experiment completely broken**

### After Fix:
- Responses: `"ABSTAIN: Cannot determine..."` or `"ANSWER: Tokyo."`
- Abstentions detected: 20-60% (expected range)
- Flips: Should see difference between main estimators and controls
- **Experiment should work correctly**

## How to Rerun

```bash
# Quick test (1 hour) - WITH FIX
sbatch run_exp3.sh

# Or test locally first (5 min)
python experiment3_steering_robust.py --quick_test
```

## What to Check After Rerun

1. **Log file should show**:
   ```
   [Sanity] Starts with ABSTAIN/ANSWER: True  ✅
   ```
   (NOT "False" like before!)

2. **Results should show**:
   - Baseline abstention rate: 20-60% (not 0%!)
   - Some direction types have >0% flip rate
   - Main estimators different from controls

3. **If still 0% flip rate**:
   - Model might be too small/task too hard
   - Try different layers (14-18 instead of 24-27, based on Exp2)
   - Try higher epsilon values

## Files Modified

1. **data_preparation.py**: Added "abstain_or_answer" mode
2. **experiment3_steering_robust.py**: Changed all "neutral" → "abstain_or_answer"

## Confidence: 🟢 VERY HIGH

This was a **fundamental formatting bug** - the model literally couldn't produce the required output format. With this fix:
- ✅ Model will be forced to use ABSTAIN/ANSWER prefix
- ✅ Abstention detection will work
- ✅ Flip metric will be measurable
- ✅ Experiment will produce valid results

**Status**: ✅ CRITICAL FIX APPLIED - Ready to rerun!
