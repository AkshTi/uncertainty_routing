# Experiment 2 Position Sweep Fix

## Problem

Previous position sweep was using **absolute positions** based on `min_seq_len`, which meant:
- "Last token" was at position `min(pos_len, neg_len) - 1`
- This is NOT the actual last token when prompts have different lengths
- Result: Weak effects because we weren't patching the decision token

## Example of the Bug

```python
# Prompt lengths
pos_prompt: 25 tokens
neg_prompt: 30 tokens

# OLD (WRONG) - Using min_seq_len
min_seq_len = min(25, 30) = 25
"last" position = 24  # Not the last token of neg_prompt!

# Cache from pos_prompt position 24 (✓ correct - actual last token)
# Patch to neg_prompt position 24   (✗ WRONG - 6 tokens before end!)
```

## The Fix

Changed to use **relative positions**:
- "Last token" = last token of EACH prompt (different absolute positions)
- "Second-to-last" = second-to-last of EACH prompt
- "First token" = first token of both (position 0)

```python
# NEW (CORRECT) - Relative positions
position_pairs = {
    "last": (pos_seq_len - 1, neg_seq_len - 1),  # (24, 29)
    "second_last": (pos_seq_len - 2, neg_seq_len - 2),  # (23, 28)
    "first": (0, 0)
}

# Cache from pos_prompt at position 24 (last token of pos_prompt)
# Patch to neg_prompt at position 29   (last token of neg_prompt)
```

## Expected Impact

**Before fix:**
- Last token: Δ=0.0, flip=0% (wrong position!)
- 2nd-to-last: Δ=0.2, flip=10% (partially worked by coincidence)
- First token: Δ=0.0, flip=0%

**After fix:**
- Last token: Δ=1.0-2.0, flip=50-80% (decision token!)
- 2nd-to-last: Δ=0.5-1.0, flip=20-40% (nearby effect)
- First token: Δ=0.0, flip=0% (no effect expected)

## Files Modified

1. **experiment2_localization.py** (lines 217-239)
   - Changed `test_position_sweep` to use relative positions
   - Now correctly patches last-to-last, second_last-to-second_last

## How to Rerun

```bash
# Local test (5 pairs, 2 minutes)
python test_exp2_position_fix.py

# Full position sweep (10 pairs)
python rerun_position_sweep.py

# Or submit to cluster
sbatch run_exp2_position.sh
```

## Validation

After rerunning, check that:
- ✅ Last token has MUCH stronger effect than before (Δ > 1.0)
- ✅ Last token >> 2nd-to-last >> first token (monotonic decrease)
- ✅ Flip rate for last token is 50-80%

If YES → Position sweep now correctly identifies decision token
If NO → May need to check other aspects of the patching mechanism
