# Experiment 7 Upgrade Summary

## Completed Improvements

### ✅ 1. Deterministic ResponseJudge
- Replaced keyword-based heuristics with `ResponseJudge` class
- Three labels: `REFUSAL`, `SAFE_COMPLETION`, `COMPLIANCE`
- Pattern-based detection with threshold logic
- Used throughout `test_refusal_with_steering()` and `run_reverse_steering_experiment()`

### ✅ 2. Statistical Utilities
- `wilson_confidence_interval()`: Wilson score CI for binomial proportions
- `bootstrap_difference_ci()`: Bootstrap CI for difference in proportions
- Both return (point_estimate, lower, upper) tuples

### ✅ 3. Enhanced Direction Computation
- Updated to use **≥200 samples** (default n_samples=200)
- `compute_split_half_stability()`: Computes cosine similarity between directions from random data splits
- Returns stability info dict with mean/std of split-half cosines
- Both `compute_refusal_direction()` and `compute_abstention_direction()` now return `(direction, stability_info)`

### ✅ 4. Safe Steering Infrastructure
- `generate_with_steering()`: Centralized safe steering method
- Features:
  - `clone()` before editing hidden states
  - Hook guard flag (applies only once per forward pass)
  - `try/finally` for guaranteed cleanup
  - Consistent layer accessor (`self.model.model.layers[layer_idx]`)
- Used in all steering contexts

### ✅ 5. Epsilon Sweeps
- `run_cross_effect_test()` now performs epsilon sweeps (default: [-10, -5, -2, 0, 2, 5, 10])
- Returns dataframes instead of summary dict
- Saves:
  - `exp7_cross_effect_abstention_to_refusal.csv`
  - `exp7_cross_effect_refusal_to_abstention.csv`

### ✅ 6. Ground Truth for Low-Risk Questions
- Low-risk questions now include `"answer"` field
- `run_reverse_steering_experiment()` tracks `correct` for low-risk questions
- Allows accuracy computation in addition to abstention/refusal rates

### ✅ 7. Updated Reverse Steering Eval
- Uses `ResponseJudge` for policy classification
- Tracks `judgment`, `refused`, `complied_harmful`, `correct` (for low-risk)
- Saves detailed CSV with all metrics

---

## Remaining Tasks

### ❌ 8. Control Experiments
**Need to implement:**
```python
def run_control_experiments(self, ...):
    # Control 1: Random direction
    random_direction = torch.randn_like(abstention_direction)
    random_direction = random_direction / torch.norm(random_direction)

    # Test refusal/abstention with random steering
    ...

    # Control 2: Label-shuffled direction
    # Shuffle labels and recompute direction
    ...

    return control_results_df
```

### ❌ 9. Comprehensive Plotting Suite
**Need to create:**
- `plot_cosine_vs_layer()`: Direction similarity across layers
- `plot_refusal_vs_epsilon()`: Refusal/compliance rates vs epsilon
- `plot_abstention_vs_epsilon()`: Abstention rate vs epsilon
- `plot_reverse_steering_by_risk()`: Three curves (low/medium/high risk)
- `plot_controls()`: Show random/shuffled don't reproduce effects

### ❌ 10. Update `analyze()` Function
**Changes needed:**
- Accept dataframes instead of summary dict for cross-effect results
- Compute Wilson CIs for all rates
- Compute bootstrap CIs for key comparisons (steered - baseline)
- Report statistical significance
- Update all print statements to show CIs

### ❌ 11. Update `main()` Function
**Changes needed:**
- Handle new return signatures (tuple unpacking for directions)
- Call control experiments
- Pass epsilon_values to cross_effect_test
- Handle new plotting functions
- Save stability info to JSON

---

## Key Type Signature Changes

### Before → After
```python
# Direction computation
compute_refusal_direction(...) -> torch.Tensor
→ compute_refusal_direction(..., n_samples=200) -> Tuple[torch.Tensor, Dict]

compute_abstention_direction(...) -> torch.Tensor
→ compute_abstention_direction(..., n_samples=200) -> Tuple[torch.Tensor, Dict]

# Cross-effect test
run_cross_effect_test(...) -> Dict
→ run_cross_effect_test(..., epsilon_values=None) -> Tuple[pd.DataFrame, pd.DataFrame]

# Reverse steering (already updated)
run_reverse_steering_experiment(...) -> pd.DataFrame  # Now with judgment, correct fields
```

---

## Integration Checklist

1. **main() function:**
   - [ ] Unpack direction tuples: `direction, stability_info = compute_...()`
   - [ ] Save stability_info to JSON
   - [ ] Pass epsilon_values to cross_effect_test
   - [ ] Unpack cross_effect dataframes
   - [ ] Call control experiments
   - [ ] Call new plotting functions

2. **analyze() function:**
   - [ ] Update signature to accept dataframes
   - [ ] Add Wilson CI computation for all rates
   - [ ] Add bootstrap CI for differences
   - [ ] Update all visualizations
   - [ ] Save statistical tables

3. **Plotting:**
   - [ ] Implement all 5 plotting functions
   - [ ] Use seaborn for confidence bands
   - [ ] Save all plots to results_dir

4. **Documentation:**
   - [ ] Update docstrings for changed functions
   - [ ] Add usage examples in main

---

## Quick Fix Template for main()

```python
# Instead of:
refusal_direction = exp7.compute_refusal_direction(harmful, benign, layer)
abstention_direction = exp7.compute_abstention_direction(answerable, unanswerable, layer)

# Use:
refusal_direction, refusal_stability = exp7.compute_refusal_direction(
    harmful, benign, layer, n_samples=200
)
abstention_direction, abstention_stability = exp7.compute_abstention_direction(
    answerable, unanswerable, layer, n_samples=200
)

# Save stability
with open(config.results_dir / "exp7_direction_stability.json", 'w') as f:
    json.dump({
        "refusal": refusal_stability,
        "abstention": abstention_stability
    }, f, indent=2)

# Instead of:
cross_effect_results = exp7.run_cross_effect_test(...)

# Use:
abs_to_ref_df, ref_to_abs_df = exp7.run_cross_effect_test(
    harmful, unanswerable, abstention_direction, refusal_direction, layer,
    epsilon_values=[-10, -5, -2, 0, 2, 5, 10]
)
```

---

## Testing Strategy

1. **Quick test mode** (`--quick_test`):
   - 10 harmful, 10 benign
   - 5 per risk level
   - epsilon_values = [-5, 0, 5]
   - n_samples = 20 (instead of 200)

2. **Full run**:
   - 200+ samples for directions
   - 50+ per risk level
   - Full epsilon sweep
   - All plots and controls

---

## Files Modified
- [experiment7_separability.py](experiment7_separability.py:1) - Core implementation

## Files To Create
- Control experiment plots
- Cosine vs layer plots
- Epsilon sweep plots with CIs
- Statistical significance tables

