"""
Merge baseline and steered exp6 results to create table with deltas

Run this AFTER you have:
1. results/exp6_cross_domain_results.csv (baseline, ε=0)
2. results/exp6_cross_domain_results_steered.csv (steered, ε=-10)
"""
import pandas as pd
from pathlib import Path

results_dir = Path("./results")

def compute_metrics(df, label):
    """Compute metrics for a given dataframe"""
    results = []

    for domain in ['math', 'science', 'history', 'geography', 'general']:
        dom_data = df[df['domain'] == domain]

        # Coverage on answerable (among valid)
        answerable = dom_data[dom_data['true_answerability'] == 'answerable']
        answerable_valid = answerable[answerable['valid'] == True]
        coverage = (1 - answerable_valid['abstained'].mean()) * 100 if len(answerable_valid) > 0 else 0

        # Abstention on unanswerable (among valid)
        unanswerable = dom_data[dom_data['true_answerability'] == 'unanswerable']
        unanswerable_valid = unanswerable[unanswerable['valid'] == True]
        abstention = unanswerable_valid['abstained'].mean() * 100 if len(unanswerable_valid) > 0 else 0

        results.append({
            'domain': domain,
            f'coverage_{label}': coverage,
            f'abstention_{label}': abstention,
            'n_answerable': len(answerable_valid),
            'n_unanswerable': len(unanswerable_valid)
        })

    return pd.DataFrame(results)

def main():
    print("\n" + "="*70)
    print("MERGING BASELINE AND STEERED RESULTS")
    print("="*70)

    # Load data
    try:
        df_baseline = pd.read_csv(results_dir / "exp6_cross_domain_results.csv")
        print("✓ Loaded baseline results (ε=0)")
    except FileNotFoundError:
        print("❌ ERROR: Could not find results/exp6_cross_domain_results.csv")
        print("   Run: python experiment6_robustness.py --epsilon 0.0 --n_per_domain 50")
        return

    try:
        df_steered = pd.read_csv(results_dir / "exp6_cross_domain_results_steered.csv")
        print("✓ Loaded steered results (ε=-10)")
    except FileNotFoundError:
        print("❌ ERROR: Could not find results/exp6_cross_domain_results_steered.csv")
        print("   First save your current results:")
        print("   mv results/exp6_cross_domain_results.csv results/exp6_cross_domain_results_steered.csv")
        return

    # Compute metrics
    baseline_metrics = compute_metrics(df_baseline, 'baseline')
    steered_metrics = compute_metrics(df_steered, 'steered')

    # Merge
    merged = baseline_metrics.merge(steered_metrics, on='domain', suffixes=('_base', '_steer'))

    # Calculate deltas
    merged['abstention_delta'] = merged['abstention_steered'] - merged['abstention_baseline']
    merged['coverage_delta'] = merged['coverage_steered'] - merged['coverage_baseline']

    # Create final table
    final_table = []
    for _, row in merged.iterrows():
        final_table.append({
            'Domain': row['domain'].capitalize(),
            'Coverage (%)': f"{row['coverage_steered']:.1f}",
            'Abstention (%)': f"{row['abstention_steered']:.1f}",
            'Abstention Δ': f"{row['abstention_delta']:+.1f}%",
            'n': f"{row['n_answerable']}/{row['n_unanswerable']}"
        })

    final_df = pd.DataFrame(final_table)

    # Save
    output_path = results_dir / "table_selective_prediction_with_delta_FINAL.csv"
    final_df.to_csv(output_path, index=False)

    print("\n" + "="*70)
    print("FINAL TABLE WITH DELTAS")
    print("="*70)
    print(final_df.to_string(index=False))

    print("\n" + "="*70)
    print(f"✓ Saved to: {output_path}")
    print("="*70)

    # Also print detailed comparison
    print("\n" + "="*70)
    print("DETAILED COMPARISON")
    print("="*70)
    for _, row in merged.iterrows():
        domain = row['domain'].capitalize()
        print(f"\n{domain}:")
        print(f"  Coverage:   {row['coverage_baseline']:.1f}% → {row['coverage_steered']:.1f}% (Δ {row['coverage_delta']:+.1f}%)")
        print(f"  Abstention: {row['abstention_baseline']:.1f}% → {row['abstention_steered']:.1f}% (Δ {row['abstention_delta']:+.1f}%)")

if __name__ == "__main__":
    main()
