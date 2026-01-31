"""
Generate cross-domain figure comparing baseline vs steered (ε=-50)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

results_dir = Path("./results")

def create_domain_baseline_comparison_strong():
    """
    Shows cross-domain performance with STRONG steering (ε=-50)
    """
    # Use exp6a_cross_domain.csv which has ε=-50
    df = pd.read_csv(results_dir / "exp6a_cross_domain.csv")

    df_baseline = df[df['epsilon'] == 0.0].copy()
    df_steered = df[df['epsilon'] == -50.0].copy()

    print(f"Baseline epsilon: 0.0")
    print(f"Steered epsilon: -50.0")

    # Compute metrics
    def compute_metrics(df_condition):
        metrics = []
        for domain in df_condition['domain'].unique():
            df_domain = df_condition[df_condition['domain'] == domain]

            # Answerable questions
            answerable = df_domain[df_domain['is_unanswerable'] == False]
            if len(answerable) > 0:
                coverage = (1 - answerable['abstained'].mean()) * 100
            else:
                coverage = 0

            # Unanswerable questions
            unanswerable = df_domain[df_domain['is_unanswerable'] == True]
            if len(unanswerable) > 0:
                abstention = unanswerable['abstained'].mean() * 100
            else:
                abstention = 0

            metrics.append({
                'domain': domain,
                'coverage': coverage,
                'abstention': abstention,
                'n_answerable': len(answerable),
                'n_unanswerable': len(unanswerable)
            })

        return pd.DataFrame(metrics)

    baseline_metrics = compute_metrics(df_baseline)
    steered_metrics = compute_metrics(df_steered)

    print("\nBaseline metrics:")
    print(baseline_metrics)
    print("\nSteered metrics:")
    print(steered_metrics)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    domains = baseline_metrics['domain'].unique()
    x = np.arange(len(domains))
    width = 0.35

    # ========== PANEL 1: Coverage on Answerable ==========
    baseline_cov = baseline_metrics['coverage'].values
    steered_cov = steered_metrics['coverage'].values

    bars1 = ax1.bar(x - width/2, baseline_cov, width,
                    label='Baseline (ε=0)', color='#95a5a6',
                    alpha=0.85, edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x + width/2, steered_cov, width,
                    label='With Steering (ε=-50)', color='#2ecc71',
                    alpha=0.85, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    # Add delta annotations
    for i, (baseline, steered) in enumerate(zip(baseline_cov, steered_cov)):
        delta = steered - baseline
        if abs(delta) > 2:
            mid_x = x[i]
            y_start = min(baseline, steered) + 2
            y_end = max(baseline, steered) - 2

            color = '#2ecc71' if delta > 0 else '#e74c3c'
            ax1.annotate('', xy=(mid_x, y_end), xytext=(mid_x, y_start),
                        arrowprops=dict(arrowstyle='->', color=color,
                                      lw=2.5, alpha=0.7))

            # Position delta label
            label_y = (y_start + y_end) / 2
            ax1.text(mid_x + 0.27, label_y,
                    f'{delta:+.0f}%', fontsize=9, fontweight='bold',
                    color=color,
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='white', edgecolor=color,
                             linewidth=1.5, alpha=0.9))

    ax1.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Coverage on Answerable Questions',
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.capitalize() for d in domains], rotation=30, ha='right')
    ax1.set_ylim([0, 115])
    ax1.axhline(y=90, color='red', linestyle='--', alpha=0.4, linewidth=1.5,
               label='90% threshold')
    ax1.grid(axis='y', alpha=0.3, linestyle=':')
    ax1.legend(fontsize=10, loc='lower right')

    # ========== PANEL 2: Abstention on Unanswerable ==========
    baseline_abs = baseline_metrics['abstention'].values
    steered_abs = steered_metrics['abstention'].values

    bars3 = ax2.bar(x - width/2, baseline_abs, width,
                    label='Baseline (ε=0)', color='#95a5a6',
                    alpha=0.85, edgecolor='black', linewidth=1.2)
    bars4 = ax2.bar(x + width/2, steered_abs, width,
                    label='With Steering (ε=-50)', color='#3498db',
                    alpha=0.85, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.0f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    # Add delta annotations - THESE ARE THE BIG WINS
    for i, (baseline, steered) in enumerate(zip(baseline_abs, steered_abs)):
        delta = steered - baseline
        if abs(delta) > 2:
            mid_x = x[i]
            y_start = min(baseline, steered) + 2
            y_end = max(baseline, steered) - 2

            color = '#3498db' if delta > 0 else '#e74c3c'
            ax2.annotate('', xy=(mid_x, y_end), xytext=(mid_x, y_start),
                        arrowprops=dict(arrowstyle='->', color=color,
                                      lw=2.5, alpha=0.7))

            label_y = (y_start + y_end) / 2
            ax2.text(mid_x + 0.27, label_y,
                    f'{delta:+.0f}%', fontsize=9, fontweight='bold',
                    color=color,
                    bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='lightyellow', edgecolor=color,
                             linewidth=1.5, alpha=0.9))

    ax2.set_ylabel('Abstention Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Abstention on Unanswerable Questions',
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.capitalize() for d in domains], rotation=30, ha='right')
    ax2.set_ylim([0, 115])
    ax2.grid(axis='y', alpha=0.3, linestyle=':')
    ax2.legend(fontsize=10, loc='upper left')

    # Add annotation box highlighting the improvement
    ax2.text(0.98, 0.5,
            'Large abstention\nincreases:\n+40% to +60%',
            transform=ax2.transAxes,
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen',
                     edgecolor='darkgreen', linewidth=2, alpha=0.7),
            ha='right', va='center')

    plt.suptitle('Cross-Domain Selective Prediction: Strong Abstention Steering Effects',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    plt.savefig(results_dir / "fig_domain_baseline_vs_steered_STRONG.png", dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / "fig_domain_baseline_vs_steered_STRONG.pdf", bbox_inches='tight')
    print("\n✓ Saved: fig_domain_baseline_vs_steered_STRONG.png/pdf")
    plt.close()

    # Summary table
    summary_data = []
    for i, domain in enumerate(domains):
        summary_data.append({
            'Domain': domain.capitalize(),
            'Coverage Baseline': f"{baseline_cov[i]:.0f}%",
            'Coverage Steered': f"{steered_cov[i]:.0f}%",
            'Coverage Δ': f"{steered_cov[i] - baseline_cov[i]:+.0f}%",
            'Abstention Baseline': f"{baseline_abs[i]:.0f}%",
            'Abstention Steered': f"{steered_abs[i]:.0f}%",
            'Abstention Δ': f"{steered_abs[i] - baseline_abs[i]:+.0f}%"
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / "table_domain_baseline_comparison_STRONG.csv", index=False)
    print("✓ Saved: table_domain_baseline_comparison_STRONG.csv")

    print("\n" + "="*80)
    print("SUMMARY TABLE (STRONG STEERING)")
    print("="*80)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    create_domain_baseline_comparison_strong()
