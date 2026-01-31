"""
Generate risk-aware coverage figure showing baseline comparison (delta)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set publication style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

results_dir = Path("./results")

def create_risk_aware_with_baseline():
    """
    Shows abstention on unanswerable by risk level
    With baseline (ε=0) comparison to show the delta
    """
    # Load reverse steering data
    df = pd.read_csv(results_dir / "exp7_reverse_steering.csv")

    # Compute abstention rate by risk and epsilon
    summary = df.groupby(['epsilon', 'risk_level']).agg({
        'epistemic_abstained': 'mean'
    }).reset_index()
    summary.rename(columns={'epistemic_abstained': 'abstention_rate'}, inplace=True)

    # Get baseline (ε=0) values
    baseline = summary[summary['epsilon'] == 0.0].set_index('risk_level')['abstention_rate'].to_dict()

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'}
    markers = {'low': 'o', 'medium': 's', 'high': '^'}

    # ========== PANEL 1: Abstention Rate Across Steering ==========
    for risk in ['low', 'medium', 'high']:
        risk_data = summary[summary['risk_level'] == risk].sort_values('epsilon')
        ax1.plot(risk_data['epsilon'], risk_data['abstention_rate'],
                marker=markers[risk], linestyle='-', linewidth=2.5, markersize=8,
                label=f'{risk.capitalize()} Risk', color=colors[risk],
                alpha=0.9)

    # Add vertical line at ε=0
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5,
               label='Baseline (ε=0)')

    # Add shaded regions
    ax1.axvspan(-4, 0, alpha=0.08, color='orange')
    ax1.axvspan(0, 4, alpha=0.08, color='blue')

    ax1.set_xlabel('Steering Strength (ε)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Abstention Rate on Unanswerable Questions', fontsize=13, fontweight='bold')
    ax1.set_title('(A) Abstention vs Steering Strength', fontsize=14, fontweight='bold', pad=10)

    ax1.set_ylim([0, 1.05])
    ax1.set_xlim([-4.5, 4.5])
    ax1.grid(alpha=0.3, linestyle=':', linewidth=1)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95,
              edgecolor='gray', fancybox=True)

    # Add annotations for steering direction
    ax1.text(-2, 0.05, 'Reduce\nAbstention', ha='center', fontsize=9,
            style='italic', color='darkorange', fontweight='bold')
    ax1.text(2, 0.05, 'Increase\nAbstention', ha='center', fontsize=9,
            style='italic', color='darkblue', fontweight='bold')

    # ========== PANEL 2: Baseline vs Steered Comparison ==========
    # Show baseline (ε=0) vs best reverse steering (ε=-4) vs best forward steering (ε=4)

    epsilon_values = [0.0, -4.0, 4.0]
    epsilon_labels = ['Baseline\n(ε=0)', 'Reverse\n(ε=-4)', 'Forward\n(ε=+4)']

    x = np.arange(len(epsilon_labels))
    width = 0.25

    for i, risk in enumerate(['low', 'medium', 'high']):
        risk_values = []
        for eps in epsilon_values:
            val = summary[(summary['epsilon'] == eps) &
                         (summary['risk_level'] == risk)]['abstention_rate'].values
            risk_values.append(val[0] if len(val) > 0 else 0)

        offset = (i - 1) * width
        bars = ax2.bar(x + offset, risk_values, width,
                      label=f'{risk.capitalize()} Risk',
                      color=colors[risk], alpha=0.85, edgecolor='black', linewidth=1.2)

        # Add value labels on bars
        for j, (bar, val) in enumerate(zip(bars, risk_values)):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                        f'{val:.0%}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

    ax2.set_ylabel('Abstention Rate', fontsize=13, fontweight='bold')
    ax2.set_title('(B) Baseline vs Steered Abstention', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(epsilon_labels, fontsize=11)
    ax2.set_ylim([0, 1.15])
    ax2.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')

    # Add annotation showing risk-aware gap
    if baseline:
        # Draw arrow showing gap at baseline
        low_baseline = baseline.get('low', 0)
        high_baseline = baseline.get('high', 0)
        gap = high_baseline - low_baseline

        y_low = summary[(summary['epsilon'] == 0.0) &
                       (summary['risk_level'] == 'low')]['abstention_rate'].values[0]
        y_high = summary[(summary['epsilon'] == 0.0) &
                        (summary['risk_level'] == 'high')]['abstention_rate'].values[0]

        # Add bracket showing gap
        ax2.plot([0.35, 0.35], [y_low + 0.03, y_high - 0.03], 'k-', linewidth=2, alpha=0.6)
        ax2.plot([0.33, 0.35], [y_low + 0.03, y_low + 0.03], 'k-', linewidth=2, alpha=0.6)
        ax2.plot([0.33, 0.35], [y_high - 0.03, y_high - 0.03], 'k-', linewidth=2, alpha=0.6)

        ax2.text(0.42, (y_low + y_high) / 2, f'Gap:\n{gap:.1%}',
                fontsize=9, fontweight='bold', color='purple',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lavender',
                         edgecolor='purple', alpha=0.6))

    plt.suptitle('Risk-Aware Selective Prediction: Baseline and Steering Effects',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    plt.savefig(results_dir / "fig_risk_aware_with_baseline.png", dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / "fig_risk_aware_with_baseline.pdf", bbox_inches='tight')
    print("✓ Saved: fig_risk_aware_with_baseline.png/pdf")
    plt.close()


def create_risk_aware_coverage_with_delta():
    """
    Alternative version: Shows coverage with delta annotations from baseline
    """
    # Load reverse steering data
    df = pd.read_csv(results_dir / "exp7_reverse_steering.csv")

    # Compute coverage (1 - abstention) by risk and epsilon
    summary = df.groupby(['epsilon', 'risk_level']).agg({
        'epistemic_abstained': lambda x: 1 - x.mean()
    }).reset_index()
    summary.rename(columns={'epistemic_abstained': 'coverage'}, inplace=True)

    # Get baseline (ε=0) values
    baseline = summary[summary['epsilon'] == 0.0].set_index('risk_level')['coverage'].to_dict()

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))

    colors = {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'}
    markers = {'low': 'o', 'medium': 's', 'high': '^'}

    for risk in ['low', 'medium', 'high']:
        risk_data = summary[summary['risk_level'] == risk].sort_values('epsilon')
        ax.plot(risk_data['epsilon'], risk_data['coverage'],
               marker=markers[risk], linestyle='-', linewidth=2.5, markersize=8,
               label=f'{risk.capitalize()} Risk', color=colors[risk],
               alpha=0.9, zorder=3)

        # Add baseline marker with emphasis
        baseline_val = baseline.get(risk, 0)
        ax.scatter([0], [baseline_val], s=250, marker=markers[risk],
                  edgecolor='black', linewidth=3, facecolor=colors[risk],
                  zorder=5, alpha=0.9)

    # Add vertical line at ε=0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.7,
              label='Baseline (no steering)', zorder=2)

    # Shaded regions
    ax.axvspan(-4, 0, alpha=0.1, color='orange', label='↓ Reduce Abstention')
    ax.axvspan(0, 4, alpha=0.1, color='blue', label='↑ Increase Abstention')

    # Add delta annotations at specific epsilon values
    for eps_show in [-4, 4]:
        for risk in ['low', 'high']:  # Only show low and high for clarity
            eps_data = summary[(summary['epsilon'] == eps_show) &
                              (summary['risk_level'] == risk)]
            if len(eps_data) > 0:
                coverage_eps = eps_data['coverage'].values[0]
                coverage_baseline = baseline.get(risk, 0)
                delta = coverage_eps - coverage_baseline

                # Add arrow from baseline to steered
                if abs(delta) > 0.01:
                    y_pos = coverage_eps
                    x_offset = 0.3 if eps_show < 0 else -0.3

                    ax.annotate(f'Δ={delta:+.1%}',
                              xy=(eps_show, y_pos),
                              xytext=(eps_show + x_offset, y_pos),
                              fontsize=8, fontweight='bold',
                              color=colors[risk],
                              bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white', edgecolor=colors[risk],
                                      alpha=0.8, linewidth=1.5),
                              arrowprops=dict(arrowstyle='->', color=colors[risk],
                                            lw=1.5, alpha=0.7))

    ax.set_xlabel('Steering Strength (ε)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Coverage (1 - Abstention Rate)', fontsize=13, fontweight='bold')
    ax.set_title('Risk-Aware Coverage: Baseline and Steering Effects\n' +
                'Large baseline markers show no-steering performance',
                fontsize=14, fontweight='bold', pad=15)

    ax.set_ylim([0.55, 1.08])
    ax.set_xlim([-4.5, 4.5])
    ax.grid(alpha=0.3, linestyle=':', linewidth=1)

    # Legend
    ax.legend(loc='lower left', fontsize=9, framealpha=0.95,
             edgecolor='gray', fancybox=True, ncol=2)

    plt.tight_layout()
    plt.savefig(results_dir / "fig_risk_aware_coverage_with_delta.png", dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / "fig_risk_aware_coverage_with_delta.pdf", bbox_inches='tight')
    print("✓ Saved: fig_risk_aware_coverage_with_delta.png/pdf")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("GENERATING RISK-AWARE FIGURES WITH BASELINE COMPARISON")
    print("="*60)

    print("\n[1/2] Two-panel version (abstention rates + bar comparison)...")
    create_risk_aware_with_baseline()

    print("\n[2/2] Single-panel version (coverage with delta annotations)...")
    create_risk_aware_coverage_with_delta()

    print("\n" + "="*60)
    print("✓ FIGURES GENERATED")
    print("="*60)
    print("\nSaved to ./results/:")
    print("  - fig_risk_aware_with_baseline.png/pdf (RECOMMENDED)")
    print("  - fig_risk_aware_coverage_with_delta.png/pdf")
    print("\nThese show the delta from baseline (ε=0) performance!")
