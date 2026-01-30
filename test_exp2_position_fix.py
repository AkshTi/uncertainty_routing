"""Quick test to verify position sweep fix works correctly"""
import json
from core_utils import ModelWrapper, ExperimentConfig
from experiment2_localization import Experiment2

def main():
    config = ExperimentConfig()
    print("Loading model...")
    model = ModelWrapper(config)

    # Load data
    with open("./data/dataset_clearly_answerable.json", 'r') as f:
        answerable = json.load(f)
    with open("./data/dataset_clearly_unanswerable.json", 'r') as f:
        unanswerable = json.load(f)

    exp2 = Experiment2(model, config)

    # Test just 5 pairs to verify the fix
    print("\nTesting position sweep with RELATIVE position fix...")
    print("Testing 5 example pairs at layer 21")

    position_df = exp2.run_position_sweep(
        answerable[:10],
        unanswerable[:10],
        n_pairs=5,
        best_layer=21
    )

    print("\n" + "="*60)
    print("POSITION SWEEP RESULTS (with fix)")
    print("="*60)

    positions = ['last', 'second_last', 'first']
    pos_labels = {'last': 'Last Token', 'second_last': '2nd-to-Last', 'first': 'First Token'}

    for pos in positions:
        delta_col = f"{pos}_delta"
        flipped_col = f"{pos}_flipped"
        if delta_col in position_df.columns:
            mean_delta = position_df[delta_col].mean()
            mean_flip = position_df[flipped_col].mean()
            print(f"  {pos_labels[pos]:15s}: Δ={mean_delta:+.3f}, flip={mean_flip:.1%}")

    print("\n✅ Position sweep completed!")
    print("Expected: Last token should have STRONGEST effect (Δ >> 0.2)")

if __name__ == "__main__":
    main()
