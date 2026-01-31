"""
Create simple table with coverage, abstention, and delta
"""
import pandas as pd
from pathlib import Path

results_dir = Path("./results")

# ============================================================================
# VERSION 1: Using exp6a data (has baseline, smaller n)
# ============================================================================

df6a = pd.read_csv(results_dir / "exp6a_cross_domain.csv")

def compute_metrics_v1():
    """Compute from exp6a data"""

    baseline = df6a[df6a['epsilon'] == 0.0]
    steered = df6a[df6a['epsilon'] == -50.0]

    results = []
    for domain in ['mathematics', 'science', 'history', 'geography']:
        # Baseline
        base_dom = baseline[baseline['domain'] == domain]
        base_ans = base_dom[base_dom['is_unanswerable'] == False]
        base_unans = base_dom[base_dom['is_unanswerable'] == True]

        base_coverage = (1 - base_ans['abstained'].mean()) * 100 if len(base_ans) > 0 else 0
        base_abstention = base_unans['abstained'].mean() * 100 if len(base_unans) > 0 else 0

        # Steered
        steer_dom = steered[steered['domain'] == domain]
        steer_ans = steer_dom[steer_dom['is_unanswerable'] == False]
        steer_unans = steer_dom[steer_dom['is_unanswerable'] == True]

        steer_coverage = (1 - steer_ans['abstained'].mean()) * 100 if len(steer_ans) > 0 else 0
        steer_abstention = steer_unans['abstained'].mean() * 100 if len(steer_unans) > 0 else 0

        # Delta
        abs_delta = steer_abstention - base_abstention

        results.append({
            'Domain': domain.capitalize(),
            'Coverage (%)': f"{steer_coverage:.1f}",
            'Abstention (%)': f"{steer_abstention:.1f}",
            'Abstention Δ': f"{abs_delta:+.1f}",
            'n': f"{len(steer_ans)}/{len(steer_unans)}"
        })

    return pd.DataFrame(results)

# ============================================================================
# VERSION 2: Using main exp6 data (no baseline available)
# ============================================================================

df6_main = pd.read_csv(results_dir / "exp6_cross_domain_results.csv")

def compute_metrics_v2():
    """Compute from main exp6 data (steered only)"""

    results = []
    for domain in ['math', 'science', 'history', 'geography', 'general']:
        dom_data = df6_main[df6_main['domain'] == domain]

        answerable = dom_data[dom_data['true_answerability'] == 'answerable']
        unanswerable = dom_data[dom_data['true_answerability'] == 'unanswerable']

        coverage = (1 - answerable['abstained'].mean()) * 100 if len(answerable) > 0 else 0
        abstention = unanswerable['abstained'].mean() * 100 if len(unanswerable) > 0 else 0

        results.append({
            'Domain': domain.capitalize(),
            'Coverage (%)': f"{coverage:.1f}",
            'Abstention (%)': f"{abstention:.1f}",
            'Abstention Δ': '—',  # No baseline available
            'n': f"{len(answerable)}/{len(unanswerable)}"
        })

    return pd.DataFrame(results)

# ============================================================================
# Generate both versions
# ============================================================================

print("="*70)
print("VERSION 1: exp6a data (HAS BASELINE, n=5)")
print("="*70)
table_v1 = compute_metrics_v1()
print(table_v1.to_string(index=False))
table_v1.to_csv(results_dir / "table_selective_prediction_with_delta_v1.csv", index=False)
print("\n✓ Saved: table_selective_prediction_with_delta_v1.csv")

print("\n" + "="*70)
print("VERSION 2: Main exp6 data (NO BASELINE, n=50)")
print("="*70)
table_v2 = compute_metrics_v2()
print(table_v2.to_string(index=False))
table_v2.to_csv(results_dir / "table_selective_prediction_with_delta_v2.csv", index=False)
print("\n✓ Saved: table_selective_prediction_with_delta_v2.csv")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("Version 1: Shows delta but has smaller sample size (n=5)")
print("Version 2: Matches your Table 2 but can't show delta without baseline")
print("\nTo get delta for Version 2, you'd need to run:")
print("  python experiment6_robustness.py --epsilon 0.0 --n_per_domain 50")
