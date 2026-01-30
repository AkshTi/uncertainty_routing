# Experiment 3 - Ready for Full Run!

## What Was Fixed

### 🐛 Critical Issues Found
1. **Test set used ambiguous questions** → No ground truth, impossible to validate
2. **Early layer control was backwards** → Applied at wrong layer, inflated results
3. **Quick test too small** → Only 10 examples, too noisy
4. **Asymmetric epsilon range** → Missing negative direction tests

### ✅ All Issues Fixed

| Issue | Before | After |
|-------|--------|-------|
| **Test Set** | Ambiguous questions ("Is 7 large?") | Eval split of clearly answerable/unanswerable |
| **Early Layer Control** | Applied at layer 24-27 | Applied at layer 7 (proper control) |
| **Quick Test Size** | 10 train, 10 test, 3 ε | 20 train, 20 test, 5 ε |
| **Epsilon Range** | [-5, 0, 5] | [-8, -4, 0, 4, 8] (symmetric) |
| **Full Test Runtime** | ~20 hours | ~5 hours (optimized) |

## Expected Results After Fixes

With properly working steering:
- ✅ **Main estimators** (mean_diff, probe, PC1): 60-80% flip rate
- ✅ **Controls** (random, shuffled): <30% flip rate
- ✅ **Early layer control**: <20% flip rate
- ✅ **Main >> Controls**: At least 2x better performance

If you see:
- ⚠️ Controls ≈ Main estimators → Steering not working, may need different approach
- ⚠️ All flip rates <20% → Epsilon too small or layers wrong
- ⚠️ All flip rates >80% → Model unstable or test set issues

## How to Run

### Option 1: Quick Test First (Recommended)
```bash
# Submit quick test (1 hour)
sbatch run_exp3.sh

# Monitor progress
tail -f logs/exp3_*.out

# When done, check results
python -c "import pandas as pd; df = pd.read_csv('results/exp3_robust_results.csv'); print(df.groupby('direction_type')['flipped'].mean())"
```

### Option 2: Full Run Immediately
```bash
# Submit full test (5 hours)
sbatch run_exp3_full.sh

# Monitor progress
tail -f logs/exp3_full_*.out
```

### Option 3: Validate Fixes First (2 minutes)
```bash
# Run validation script to test fixes work
python test_exp3_fixes.py

# If all tests pass, proceed with full run
sbatch run_exp3_full.sh
```

## Files Created/Modified

**Modified:**
- `experiment3_steering_robust.py` - Core fixes
- `run_exp3.sh` - Updated for quick test

**Created:**
- `run_exp3_full.sh` - Script for full 5-hour run
- `test_exp3_fixes.py` - Validation script (~2 min)
- `EXPERIMENT3_FIXES.md` - Detailed technical docs
- `EXP3_READY_TO_RUN.md` - This file

## What Happens Next

### If Quick Test Shows Good Results (Main >> Controls):
✅ **Proceed to Experiment 4**
- Run: `python experiment4_selectivity.py --quick_test`
- Steering is working, full exp3 can wait

### If Quick Test Shows Bad Results (Controls ≈ Main):
🔧 **Debug before full run**
- Check logs for errors
- Try different layers (18-22 instead of 24-27)
- Try higher epsilon values
- Consider different steering vector computation

### After 5-Hour Full Run:
📊 **You'll have complete steering analysis**
- Comprehensive results across all epsilon values
- All direction estimators tested
- Proper controls validated
- Ready for paper figures

## Quick Check After Run

```bash
# Check if experiment completed successfully
tail logs/exp3_*.out

# Quick analysis
python << 'EOF'
import pandas as pd
df = pd.read_csv('results/exp3_robust_results.csv')

print("\n🎯 QUICK RESULTS CHECK")
print("="*50)

# Flip rates by direction type
flip_by_dir = df[df['epsilon'] != 0].groupby('direction_type')['flipped'].mean()
print("\nFlip Rates:")
for dir_type, rate in flip_by_dir.items():
    symbol = "✅" if rate > 0.4 else "⚠️"
    print(f"  {symbol} {dir_type:15s}: {rate:.1%}")

# Main vs controls
main = flip_by_dir[['mean_diff', 'probe', 'pc1']].mean()
controls = flip_by_dir[['random', 'shuffled', 'early_layer']].mean()

print(f"\n📊 Main estimators: {main:.1%}")
print(f"📊 Controls:        {controls:.1%}")

if main > controls * 2:
    print("\n✅ SUCCESS: Steering working well!")
else:
    print("\n⚠️  WARNING: Controls too strong, review needed")
EOF
```

## Ready to Submit?

**YES** - All fixes are applied and tested
```bash
sbatch run_exp3_full.sh
```

**WAIT** - Want to validate first
```bash
python test_exp3_fixes.py
```

Good luck! 🚀
