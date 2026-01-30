"""Regenerate exp2_summary.json from existing CSV files"""
import pandas as pd
import json
from pathlib import Path

def regenerate_summary():
    """Regenerate the summary.json file from existing CSVs"""

    results_dir = Path("results")
    summary = {}

    # ===== WINDOW SWEEP ANALYSIS =====
    window_csv = results_dir / "exp2_window_sweep.csv"
    if window_csv.exists():
        print("Processing window sweep results...")
        window_df = pd.read_csv(window_csv)

        # Group by window size
        window_stats = window_df.groupby("window_size").agg({
            "delta_margin": ["mean", "std", "max"],
            "flipped": "mean"
        }).round(3)

        # Convert MultiIndex columns to JSON-serializable format
        window_stats_dict = {}
        for col in window_stats.columns:
            if isinstance(col, tuple):
                # Flatten tuple column names
                key = '_'.join(str(x) for x in col)
                window_stats_dict[key] = window_stats[col].to_dict()
            else:
                window_stats_dict[str(col)] = window_stats[col].to_dict()

        summary["window_stats"] = window_stats_dict

        # Find best window for each size
        best_windows = {}
        for k in window_df['window_size'].unique():
            k_df = window_df[window_df['window_size'] == k]
            best_idx = k_df['delta_margin'].idxmax()
            best_row = k_df.loc[best_idx]
            best_windows[str(int(k))] = {  # Convert key to string for JSON
                "window": f"[{int(best_row['window_start'])}-{int(best_row['window_end'])}]",
                "delta": float(best_row['delta_margin']),
                "flip_rate": float(best_row['flipped'])
            }

        summary["best_windows"] = best_windows
        print(f"  ✓ Window stats processed")

    # ===== POSITION SWEEP ANALYSIS =====
    position_csv = results_dir / "exp2_position_sweep.csv"
    if position_csv.exists():
        try:
            position_df = pd.read_csv(position_csv)
            if len(position_df) > 0:
                print("Processing position sweep results...")

                positions = ['last', 'second_last', 'first']
                position_means = {}
                position_flips = {}

                for pos in positions:
                    delta_col = f"{pos}_delta"
                    flipped_col = f"{pos}_flipped"
                    if delta_col in position_df.columns and flipped_col in position_df.columns:
                        position_means[pos] = float(position_df[delta_col].mean())
                        position_flips[pos] = float(position_df[flipped_col].mean())

                summary["position_effects"] = position_means
                summary["position_flips"] = position_flips
                print(f"  ✓ Position stats processed")
            else:
                print("  ⚠ Position sweep CSV is empty")
        except Exception as e:
            print(f"  ⚠ Error processing position sweep: {e}")

    # ===== NEGATIVE CONTROLS ANALYSIS =====
    control_csv = results_dir / "exp2_negative_controls.csv"
    if control_csv.exists():
        print("Processing negative controls...")
        control_df = pd.read_csv(control_csv)

        control_types = ['correct_patch', 'random_prompt', 'wrong_position']
        control_means = {}
        control_flips = {}

        for ct in control_types:
            delta_col = f"{ct}_delta"
            flipped_col = f"{ct}_flipped"
            if delta_col in control_df.columns and flipped_col in control_df.columns:
                control_means[ct] = float(control_df[delta_col].mean())
                control_flips[ct] = float(control_df[flipped_col].mean())

        summary["control_effects"] = control_means
        summary["control_flips"] = control_flips
        print(f"  ✓ Control stats processed")

    # ===== MIXING SWEEP ANALYSIS =====
    mixing_csv = results_dir / "exp2_mixing_sweep.csv"
    if mixing_csv.exists():
        print("Processing mixing sweep results...")
        mixing_df = pd.read_csv(mixing_csv)

        mixing_stats = mixing_df.groupby("alpha").agg({
            "delta_margin": "mean",
            "flipped": "mean"
        }).round(3)

        # Convert to JSON-serializable format
        mixing_stats_dict = {}
        for col in mixing_stats.columns:
            if isinstance(col, tuple):
                key = '_'.join(str(x) for x in col)
                mixing_stats_dict[key] = mixing_stats[col].to_dict()
            else:
                mixing_stats_dict[str(col)] = mixing_stats[col].to_dict()

        summary["mixing_stats"] = mixing_stats_dict

        # Check for monotonicity
        alphas_sorted = sorted(mixing_df['alpha'].unique())
        deltas = [mixing_df[mixing_df['alpha'] == a]['delta_margin'].mean() for a in alphas_sorted]
        monotonic = all(deltas[i] <= deltas[i+1] for i in range(len(deltas)-1))
        summary["monotonic"] = monotonic
        print(f"  ✓ Mixing stats processed")

    # Save summary
    output_path = results_dir / "exp2_summary.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary regenerated successfully: {output_path}")
    print(f"\nSummary contains:")
    for key in summary.keys():
        print(f"  - {key}")

    return summary

if __name__ == "__main__":
    summary = regenerate_summary()
