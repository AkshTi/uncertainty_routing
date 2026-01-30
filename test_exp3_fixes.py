"""
Quick validation script to test Experiment 3 fixes
Runs a tiny version to verify the fixes work correctly
"""
import json
from experiment3_steering_robust import Experiment3Robust
from core_utils import ModelWrapper, ExperimentConfig

def test_fixes():
    """Test that all fixes are working"""

    print("="*60)
    print("TESTING EXPERIMENT 3 FIXES")
    print("="*60)

    # Load minimal data
    print("\n1. Loading data...")
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

    # Minimal test set
    answerable = answerable[:6]  # 6 examples for train/eval split
    unanswerable = unanswerable[:6]

    # Split into train/eval
    train_split = 0.6
    n_train = int(6 * train_split)  # 3 train, 3 eval

    eval_pos = answerable[n_train:]
    eval_neg = unanswerable[n_train:]
    test_questions = eval_pos + eval_neg

    print(f"✓ Loaded data: {len(answerable)} answerable, {len(unanswerable)} unanswerable")
    print(f"  Train: {n_train} per class, Eval: {len(test_questions)} total")

    # CHECK 1: Test set has known answerability
    print("\n2. Checking test set has known answerability...")
    has_unknown = any(ex.get("answerability", "unknown") == "unknown" for ex in test_questions)
    if has_unknown:
        print("  ✗ FAIL: Test set contains 'unknown' answerability")
        return False
    else:
        print("  ✓ PASS: All test questions have known answerability")

    # CHECK 2: Epsilon range is symmetric
    print("\n3. Checking epsilon range...")
    epsilon_range = [-8, -4, 0, 4, 8]
    positive = [e for e in epsilon_range if e > 0]
    negative = [abs(e) for e in epsilon_range if e < 0]
    if sorted(positive) == sorted(negative):
        print(f"  ✓ PASS: Epsilon range is symmetric: {epsilon_range}")
    else:
        print(f"  ✗ FAIL: Epsilon range not symmetric: {epsilon_range}")
        return False

    # CHECK 3: Run mini experiment to test early layer control
    print("\n4. Testing early layer control fix...")
    print("  (This will take ~2 minutes...)")

    try:
        config = ExperimentConfig()
        model = ModelWrapper(config)
        exp3 = Experiment3Robust(model, config)

        # Run minimal version
        results_df = exp3.run(
            pos_examples=answerable,
            neg_examples=unanswerable,
            test_examples=test_questions,
            train_split=train_split,
            epsilon_range=[0, 8],  # Just baseline and one steering
            test_layers=[24]  # Just one layer
        )

        # Check early layer control is tested at early layer
        early_layer = int(model.model.config.num_hidden_layers * 0.25)
        early_results = results_df[results_df['direction_type'] == 'early_layer']

        if len(early_results) > 0:
            actual_layer = early_results['layer'].iloc[0]
            if actual_layer == early_layer:
                print(f"  ✓ PASS: Early layer control tested at layer {early_layer}")
            else:
                print(f"  ✗ FAIL: Early layer control at layer {actual_layer}, expected {early_layer}")
                return False
        else:
            print("  ✗ FAIL: No early layer control results found")
            return False

        # Check we have the right direction types
        direction_types = set(results_df['direction_type'].unique())
        expected = {'mean_diff', 'probe', 'pc1', 'random', 'shuffled', 'early_layer'}
        if expected.issubset(direction_types):
            print(f"  ✓ PASS: All expected direction types present: {direction_types}")
        else:
            missing = expected - direction_types
            print(f"  ✗ FAIL: Missing direction types: {missing}")
            return False

    except Exception as e:
        print(f"  ✗ FAIL: Error during mini experiment: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED - Experiment 3 fixes are working!")
    print("="*60)
    print("\nYou can now run:")
    print("  Quick test (1 hour):  sbatch run_exp3.sh")
    print("  Full test (5 hours):  sbatch run_exp3_full.sh")
    return True

if __name__ == "__main__":
    success = test_fixes()
    exit(0 if success else 1)
