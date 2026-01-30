# Experiment 3 Issues and Fixes

## Issues Identified

### 1. **Test Set Uses Ambiguous Questions**
- **Problem**: Lines 763-767 load "dataset_ambiguous.json" with subjective questions like "Is 7 large?"
- **Impact**: These questions have no ground truth, making steering effects impossible to validate
- **Fix**: Use the eval split of clearly answerable/unanswerable questions instead

### 2. **Early Layer Control is Backwards**
- **Problem**: Lines 361-369 compute early layer direction (layer 7) but apply it at late layers (24-27)
- **Impact**: This is not a proper control - it's actually a mismatch intervention that can have large effects
- **Fix**: Apply early layer directions at early layers to show they don't work for steering

### 3. **Quick Test Too Small**
- **Problem**: Only 10 train + 10 test examples, 3 epsilon values
- **Impact**: Too noisy to see real effects
- **Fix**: Increase to 20 train + 20 test, 5 epsilon values (still fast ~1 hour)

### 4. **No Negative Epsilon in Quick Test**
- **Problem**: Only tests [-5, 0, 5]
- **Impact**: Misses potential effects in other direction
- **Fix**: Test both directions symmetrically

## Fixes Applied

### Fix 1: Test Set Now Uses Eval Split (Lines 763-775)
**Before:** Loaded "dataset_ambiguous.json" with subjective questions
**After:** Uses eval split of clearly answerable/unanswerable questions
**Impact:** Test set now has ground truth, enabling proper validation

### Fix 2: Improved Quick Test Parameters (Lines 769-775)
**Before:** 10 train, 10 test, 3 epsilon values (too small)
**After:** 20 train, 20 test, 5 epsilon values [-8, -4, 0, 4, 8]
**Impact:** Better signal-to-noise ratio, symmetric epsilon range

### Fix 3: Full Mode Runtime Reduced (Lines 777-799)
**Before:** 30 train examples, estimated 20 hours
**After:** 40 train examples, estimated 5 hours
**Impact:** More data with better efficiency

### Fix 4: Fixed Early Layer Control (Lines 318-325, 361-376)
**Before:** Early layer direction computed at layer 7 but applied at layer 24-27
**After:** Early layer direction applied at actual early layer (layer 7)
**Impact:** Proper control test - early layers shouldn't affect abstention

## Results to Expect After Fixes

**With Fixed Experiment:**
- Main estimators should achieve 60-80% flip rate (if steering works)
- Controls (random, shuffled) should achieve <30% flip rate
- Early layer control should achieve <20% flip rate (it shouldn't work)
- Main estimators should significantly outperform controls (>2x better)

**If these expectations aren't met:**
- The steering vectors may need different computation methods
- May need to test different layers (try layers 18-22 instead of 24-27)
- May need higher epsilon values

## How to Run

**Quick Test (1 hour):**
```bash
sbatch run_exp3.sh
```

**Full Test (5 hours):**
```bash
sbatch run_exp3_full.sh
```

**Monitor Progress:**
```bash
tail -f logs/exp3_*.out
```

## Files Modified

1. `experiment3_steering_robust.py` - Core fixes to test set, early layer control, and parameters
2. `run_exp3.sh` - Updated time allocation and comments for quick test
3. `run_exp3_full.sh` - NEW: Script for full 5-hour run
4. `EXPERIMENT3_FIXES.md` - This documentation
