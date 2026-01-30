"""Rerun just the position sweep for Experiment 2"""
import json
from core_utils import ModelWrapper, ExperimentConfig
from experiment2_localization import Experiment2

def main():
    """Rerun position sweep only"""

    # Setup
    config = ExperimentConfig()

    print("Initializing model...")
    model = ModelWrapper(config)

    # Load data
    print("\nLoading datasets...")
    try:
        with open("./data/dataset_clearly_answerable_expanded.json", 'r') as f:
            answerable = json.load(f)
    except FileNotFoundError:
        with open("./data/dataset_clearly_answerable.json", 'r') as f:
            answerable = json.load(f)

    try:
        with open("./data/dataset_clearly_unanswerable_expanded.json", 'r') as f:
            unanswerable = json.load(f)
    except FileNotFoundError:
        with open("./data/dataset_clearly_unanswerable.json", 'r') as f:
            unanswerable = json.load(f)

    # Initialize experiment
    exp2 = Experiment2(model, config)

    # Determine best layer from previous results
    # You can change this if you want to use a different layer
    best_layer = int(model.model.config.num_hidden_layers * 0.75)

    print(f"\n{'='*60}")
    print(f"RERUNNING: POSITION SWEEP")
    print(f"{'='*60}")
    print(f"Using layer: {best_layer}")

    # Run position sweep
    n_pairs = 10
    position_df = exp2.run_position_sweep(
        answerable[:n_pairs*2],
        unanswerable[:n_pairs*2],
        n_pairs=n_pairs,
        best_layer=best_layer
    )

    print(f"\n✓ Position sweep complete!")
    print(f"\nResults saved to: {config.results_dir / 'exp2_position_sweep.csv'}")

    # Show summary
    if len(position_df) > 0:
        print(f"\n{'='*60}")
        print("POSITION SWEEP SUMMARY")
        print(f"{'='*60}")

        positions = ['first', 'second_last', 'last']
        pos_labels = {'first': 'First Token', 'second_last': '2nd-to-Last', 'last': 'Last Token'}

        print("\nEffect by token position:")
        for pos in positions:
            delta_col = f"{pos}_delta"
            flipped_col = f"{pos}_flipped"
            if delta_col in position_df.columns and flipped_col in position_df.columns:
                mean_delta = position_df[delta_col].mean()
                mean_flip = position_df[flipped_col].mean()
                print(f"  {pos_labels[pos]:15s}: Δ={mean_delta:.3f}, flip={mean_flip:.1%}")

    return position_df

if __name__ == "__main__":
    main()
