"""
Generate publication-quality figures for trustworthiness results
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
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

results_dir = Path("./results")

# ============================================================================
# FIGURE 1: Risk-Aware Coverage Tradeoff (Most Impactful)
# ============================================================================

def create_risk_aware_coverage_figure():
    """
    Shows coverage by risk level across epsilon values
    Demonstrates the system is risk-aware in its selective prediction
    """
    # Load reverse steering data
    df = pd.read_csv(results_dir / "exp7_reverse_steering.csv")

    # Compute coverage (1 - epistemic_abstained) by risk and epsilon
    summary = df.groupby(['epsilon', 'risk_level']).agg({
        'epistemic_abstained': lambda x: 1 - x.mean()  # Coverage = 1 - abstention
    }).reset_index()
    summary.rename(columns={'epistemic_abstained': 'coverage'}, inplace=True)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    colors = {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'}
    markers = {'low': 'o', 'medium': 's', 'high': '^'}

    for risk in ['low', 'medium', 'high']:
        risk_data = summary[summary['risk_level'] == risk].sort_values('epsilon')
        ax.plot(risk_data['epsilon'], risk_data['coverage'],
               marker=markers[risk], linestyle='-', linewidth=2.5, markersize=8,
               label=f'{risk.capitalize()} Risk', color=colors[risk],
               alpha=0.9)

    # Add shaded region for positive epsilon (increase abstention)
    ax.axvspan(0, 4, alpha=0.1, color='blue', label='Increase Abstention')
    # Add shaded region for negative epsilon (reverse steering - reduce abstention)
    ax.axvspan(-4, 0, alpha=0.1, color='orange', label='Reduce Abstention')

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    ax.set_xlabel('Steering Strength (ε)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Coverage (1 - Abstention Rate)', fontsize=13, fontweight='bold')
    ax.set_title('Risk-Aware Selective Prediction:\nCoverage Varies by Question Risk',
                fontsize=14, fontweight='bold', pad=15)

    ax.set_ylim([0, 1.05])
    ax.set_xlim([-4.5, 4.5])
    ax.grid(alpha=0.3, linestyle=':', linewidth=1)

    # Legend with better positioning
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95,
             edgecolor='gray', fancybox=True)

    # Add annotation for key finding
    ax.annotate('Risk-aware gap:\nMore abstention on\nhigh-risk questions',
               xy=(-2, 0.85), fontsize=9,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                        edgecolor='gray', alpha=0.8),
               ha='center')

    plt.tight_layout()
    plt.savefig(results_dir / "fig_risk_aware_coverage.png", dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / "fig_risk_aware_coverage.pdf", bbox_inches='tight')
    print("✓ Saved: fig_risk_aware_coverage.png/pdf")
    plt.close()


# ============================================================================
# FIGURE 2: Cross-Domain Selective Prediction
# ============================================================================

def create_cross_domain_figure():
    """
    Shows coverage vs abstention across domains
    Clean bar chart showing domain generalization
    """
    # Load exp6 data
    df = pd.read_csv(results_dir / "exp6_cross_domain_table.csv")

    domains = df['Domain'].values
    coverage = [float(x.strip('%')) / 100 for x in df['Coverage'].values]

    # Extract abstention from the Note column (e.g., "Unans abstain:46%")
    abstention = []
    for note in df['Note'].values:
        # Parse "Unans abstain:46% (18/39)"
        abs_part = note.split('Unans abstain:')[1].split('%')[0]
        abstention.append(float(abs_part) / 100)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Coverage on answerable
    x = np.arange(len(domains))
    bars1 = ax1.bar(x, coverage, color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, coverage)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.0%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Coverage', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Coverage on Answerable Questions', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(domains, rotation=30, ha='right')
    ax1.set_ylim([0, 1.1])
    ax1.axhline(y=0.9, color='red', linestyle='--', alpha=0.4, linewidth=1.5,
               label='90% threshold')
    ax1.grid(axis='y', alpha=0.3, linestyle=':')
    ax1.legend(fontsize=9)

    # Panel B: Abstention on unanswerable
    bars2 = ax2.bar(x, abstention, color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, abstention)):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.0%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Abstention Rate', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Abstention on Unanswerable Questions', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(domains, rotation=30, ha='right')
    ax2.set_ylim([0, 1.1])
    ax2.grid(axis='y', alpha=0.3, linestyle=':')

    plt.suptitle('Cross-Domain Selective Prediction Performance',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(results_dir / "fig_cross_domain_selective_prediction.png", dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / "fig_cross_domain_selective_prediction.pdf", bbox_inches='tight')
    print("✓ Saved: fig_cross_domain_selective_prediction.png/pdf")
    plt.close()


# ============================================================================
# FIGURE 3: Combined Multi-Panel (Most Complete)
# ============================================================================

def create_combined_figure():
    """
    Three-panel figure showing:
    (A) Cross-domain coverage/abstention
    (B) Risk-aware gap
    (C) Direction orthogonality
    """
    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 0.8])

    # ========== PANEL A: Cross-Domain Performance ==========
    ax1 = fig.add_subplot(gs[0])

    # Load exp6 data
    df6 = pd.read_csv(results_dir / "exp6_cross_domain_table.csv")
    domains = df6['Domain'].values
    coverage = [float(x.strip('%')) / 100 for x in df6['Coverage'].values]
    abstention = []
    for note in df6['Note'].values:
        abs_part = note.split('Unans abstain:')[1].split('%')[0]
        abstention.append(float(abs_part) / 100)

    x = np.arange(len(domains))
    width = 0.35

    bars1 = ax1.bar(x - width/2, coverage, width, label='Coverage (Answerable)',
                   color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, abstention, width, label='Abstention (Unanswerable)',
                   color='#3498db', alpha=0.85, edgecolor='black', linewidth=1)

    ax1.set_ylabel('Rate', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Cross-Domain Generalization', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(domains, rotation=30, ha='right', fontsize=9)
    ax1.set_ylim([0, 1.1])
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(axis='y', alpha=0.3, linestyle=':')

    # ========== PANEL B: Risk-Aware Gap ==========
    ax2 = fig.add_subplot(gs[1])

    # Load reverse steering data
    df7 = pd.read_csv(results_dir / "exp7_reverse_steering.csv")
    summary = df7.groupby(['epsilon', 'risk_level']).agg({
        'epistemic_abstained': lambda x: 1 - x.mean()
    }).reset_index()
    summary.rename(columns={'epistemic_abstained': 'coverage'}, inplace=True)

    colors = {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'}

    for risk in ['low', 'high']:  # Only show low and high for clarity
        risk_data = summary[summary['risk_level'] == risk].sort_values('epsilon')
        ax2.plot(risk_data['epsilon'], risk_data['coverage'],
                marker='o', linestyle='-', linewidth=2, markersize=6,
                label=f'{risk.capitalize()} Risk', color=colors[risk], alpha=0.9)

    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Steering Strength (ε)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Coverage', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Risk-Aware Behavior', fontsize=12, fontweight='bold')
    ax2.set_ylim([0.6, 1.05])
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, linestyle=':')

    # Add arrow showing gap
    ax2.annotate('', xy=(-2, 0.9), xytext=(-2, 1.0),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax2.text(-2.8, 0.95, 'Gap', fontsize=9, color='purple', fontweight='bold')

    # ========== PANEL C: Direction Orthogonality ==========
    ax3 = fig.add_subplot(gs[2])

    # Load direction similarity
    import json
    with open(results_dir / "exp7_summary.json", 'r') as f:
        exp7_summary = json.load(f)

    cosine_sim = exp7_summary['direction_similarity']['cosine_similarity']
    angle = exp7_summary['direction_similarity']['angle_degrees']

    # Create a simple visual representation
    ax3.text(0.5, 0.7, f"Cosine Similarity", ha='center', fontsize=11, fontweight='bold')
    ax3.text(0.5, 0.55, f"{cosine_sim:.3f}", ha='center', fontsize=20, fontweight='bold',
            color='#3498db')

    ax3.text(0.5, 0.35, f"Angle", ha='center', fontsize=11, fontweight='bold')
    ax3.text(0.5, 0.2, f"{angle:.1f}°", ha='center', fontsize=20, fontweight='bold',
            color='#e74c3c')

    ax3.text(0.5, 0.05, "(Near-orthogonal)", ha='center', fontsize=9,
            style='italic', color='gray')

    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.set_title('(C) Direction Separation', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Add box around metrics
    from matplotlib.patches import FancyBboxPatch
    rect = FancyBboxPatch((0.15, 0.15), 0.7, 0.7,
                          boxstyle="round,pad=0.05",
                          edgecolor='gray', facecolor='white',
                          linewidth=2, alpha=0.3)
    ax3.add_patch(rect)

    plt.suptitle('Mechanistic Abstention Control Supports Trustworthiness Objectives',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(results_dir / "fig_trustworthiness_combined.png", dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / "fig_trustworthiness_combined.pdf", bbox_inches='tight')
    print("✓ Saved: fig_trustworthiness_combined.png/pdf")
    plt.close()


# ============================================================================
# FIGURE 4: Coverage-Accuracy Scatter (Bonus)
# ============================================================================

def create_coverage_accuracy_scatter():
    """
    Shows that high coverage doesn't sacrifice accuracy
    """
    # Load exp6 data
    df = pd.read_csv(results_dir / "exp6_cross_domain_table.csv")

    domains = df['Domain'].values
    coverage = [float(x.strip('%')) / 100 for x in df['Coverage'].values]

    # Extract accuracy from Note (e.g., "Checkable ans:88%")
    accuracy = []
    for note in df['Note'].values:
        acc_part = note.split('Checkable ans:')[1].split('%')[0]
        accuracy.append(float(acc_part) / 100)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    colors_map = {'Math': '#e74c3c', 'Science': '#3498db', 'History': '#9b59b6',
                  'Geography': '#2ecc71', 'General': '#f39c12'}

    for i, domain in enumerate(domains):
        ax.scatter(coverage[i], accuracy[i], s=200, alpha=0.7,
                  color=colors_map.get(domain, 'gray'),
                  edgecolor='black', linewidth=1.5,
                  label=domain, zorder=3)

        # Add domain labels
        ax.annotate(domain, (coverage[i], accuracy[i]),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, fontweight='bold')

    # Add reference lines
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.3, linewidth=1.5,
              label='80% accuracy threshold')
    ax.axvline(x=0.9, color='blue', linestyle='--', alpha=0.3, linewidth=1.5,
              label='90% coverage threshold')

    # Shade the "good" region
    ax.axhspan(0.8, 1.0, 0.9, 1.0, alpha=0.1, color='green', zorder=1)
    ax.text(0.97, 0.92, 'High Coverage\n& High Accuracy',
           ha='center', va='center', fontsize=9, style='italic',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen',
                    edgecolor='green', alpha=0.3))

    ax.set_xlabel('Coverage on Answerable Questions', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (on Answered Questions)', fontsize=12, fontweight='bold')
    ax.set_title('Coverage-Accuracy Tradeoff:\nHigh Coverage Without Sacrificing Accuracy',
                fontsize=13, fontweight='bold', pad=15)

    ax.set_xlim([0.88, 1.02])
    ax.set_ylim([0.75, 1.0])
    ax.grid(alpha=0.3, linestyle=':', linewidth=1)

    ax.legend(loc='lower left', fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(results_dir / "fig_coverage_accuracy_scatter.png", dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / "fig_coverage_accuracy_scatter.pdf", bbox_inches='tight')
    print("✓ Saved: fig_coverage_accuracy_scatter.png/pdf")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*60)

    print("\n[1/4] Risk-aware coverage tradeoff...")
    create_risk_aware_coverage_figure()

    print("\n[2/4] Cross-domain selective prediction...")
    create_cross_domain_figure()

    print("\n[3/4] Combined multi-panel figure...")
    create_combined_figure()

    print("\n[4/4] Coverage-accuracy scatter...")
    create_coverage_accuracy_scatter()

    print("\n" + "="*60)
    print("✓ ALL FIGURES GENERATED")
    print("="*60)
    print("\nSaved to ./results/:")
    print("  - fig_risk_aware_coverage.png/pdf (RECOMMENDED)")
    print("  - fig_cross_domain_selective_prediction.png/pdf")
    print("  - fig_trustworthiness_combined.png/pdf")
    print("  - fig_coverage_accuracy_scatter.png/pdf")
    print("\nRecommendation: Use 'fig_risk_aware_coverage' or 'fig_trustworthiness_combined'")
    print("for maximum impact in your paper!")
