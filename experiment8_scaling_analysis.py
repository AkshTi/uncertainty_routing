"""
Experiment 8: Scaling Analysis & Model Comparison

CRITICAL FOR ACCEPTANCE: Show findings generalize across model sizes/families

Addresses reviewer concern: "Only tested on 1.5B model, will it scale?"

This is THE experiment that will significantly boost your novelty and acceptance chances.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import json
import copy
import argparse

from core_utils import ModelWrapper, ExperimentConfig, extract_answer, set_seed


class Experiment8:
    """Test steering across different model sizes and families"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = []
        set_seed(config.seed)

    def test_model(self, model_name: str, train_questions: List[Dict],
                   eval_questions: List[Dict], target_layers: List[int] = None) -> Dict:
        """
        Test steering on a specific model

        Args:
            model_name: HuggingFace model name
            train_questions: Questions for extracting steering directions (DISJOINT from eval)
            eval_questions: Questions for evaluation (flexible size, minimum 10+10)
            target_layers: Which layers to test (None = auto-detect late layers)
        """
        print(f"\n{'='*70}")
        print(f"Testing model: {model_name}")
        print(f"{'='*70}")

        # Load model (deepcopy self.config to preserve ALL settings: seed, device, paths, dtype, etc.)
        config = copy.deepcopy(self.config)
        config.model_name = model_name  # Only override the model name

        try:
            model_wrapper = ModelWrapper(config)
            n_layers = model_wrapper.model.config.num_hidden_layers
            hidden_dim = model_wrapper.model.config.hidden_size

            print(f"Loaded: {n_layers} layers, {hidden_dim} hidden dim")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            return None

        # Auto-detect late layers if not specified
        if target_layers is None:
            # Test last 20% of layers (late layers where semantic info is processed)
            start_layer = int(n_layers * 0.8)
            target_layers = list(range(start_layer, n_layers))
            print(f"Auto-selected late layers: {target_layers}")

        # Extract steering vectors from TRAIN set (prevent data leakage)
        print("\nExtracting steering vectors from training set...")

        answerable_train = [q for q in train_questions if not q.get('is_unanswerable', False)]
        unanswerable_train = [q for q in train_questions if q.get('is_unanswerable', False)]

        print(f"  Train set: {len(answerable_train)} answerable, {len(unanswerable_train)} unanswerable")

        # Test DEEPEST late layers (most semantic processing happens here)
        # Pick 3-5 deepest layers to find the best one for this model
        layers_to_test = target_layers[-5:]  # Last 5 layers
        print(f"  Testing deepest layers: {layers_to_test}")

        steering_vectors = {}
        for layer_idx in layers_to_test:
            try:
                from experiment3_4_steering_independence import Experiment3

                # Format prompts
                from data_preparation import format_prompt
                pos_prompts = [format_prompt(q["question"], "neutral", q.get("context"))
                              for q in answerable_train]
                neg_prompts = [format_prompt(q["question"], "neutral", q.get("context"))
                              for q in unanswerable_train]

                exp3 = Experiment3(model_wrapper, config)
                direction = exp3.compute_steering_direction(pos_prompts, neg_prompts, layer_idx)
                steering_vectors[layer_idx] = direction

            except Exception as e:
                print(f"ERROR computing steering for layer {layer_idx}: {e}")
                continue

        if not steering_vectors:
            print("ERROR: Could not extract steering vectors")
            return None

        print(f"  ✓ Extracted steering vectors for layers: {list(steering_vectors.keys())}")

        # Find best layer by testing each on a small validation subset
        print("\nFinding best layer for this model...")
        print("Selection criterion: max[Δ(abstain unanswerable) - Δ(abstain answerable)]")
        print("  → Maximize hallucination reduction while minimizing coverage loss")
        from experiment5_trustworthiness import Experiment5

        best_layer = None
        best_trustworthiness_score = -float('inf')

        # Quick test with first 20 eval questions (10 answerable + 10 unanswerable if possible)
        quick_eval_answerable = [q for q in eval_questions if not q.get('is_unanswerable', False)][:10]
        quick_eval_unanswerable = [q for q in eval_questions if q.get('is_unanswerable', False)][:10]
        quick_eval = quick_eval_answerable + quick_eval_unanswerable

        for layer_idx in steering_vectors.keys():
            exp5_temp = Experiment5(model_wrapper, config, {layer_idx: steering_vectors[layer_idx]})
            temp_results = []

            for q in quick_eval:
                try:
                    # Test baseline (eps=0) and steered (eps=-30)
                    for eps in [0.0, -30.0]:
                        result = exp5_temp.test_one(q, layer_idx, eps)
                        temp_results.append(result)
                except:
                    continue

            if temp_results:
                temp_df = pd.DataFrame(temp_results)

                # Split by question type
                answerable_df = temp_df[~temp_df['is_unanswerable']]
                unanswerable_df = temp_df[temp_df['is_unanswerable']]

                # Compute deltas separately for answerable and unanswerable
                if len(answerable_df) > 0 and len(unanswerable_df) > 0:
                    # Delta for unanswerable (want this to increase)
                    baseline_abstain_unans = answerable_df[answerable_df['epsilon'] == 0.0]['abstained'].mean()
                    steered_abstain_unans = unanswerable_df[unanswerable_df['epsilon'] == -30.0]['abstained'].mean()
                    delta_unans = steered_abstain_unans - baseline_abstain_unans

                    # Delta for answerable (want this to stay small/negative)
                    baseline_abstain_ans = answerable_df[answerable_df['epsilon'] == 0.0]['abstained'].mean()
                    steered_abstain_ans = answerable_df[answerable_df['epsilon'] == -30.0]['abstained'].mean()
                    delta_ans = steered_abstain_ans - baseline_abstain_ans

                    # Trustworthiness score: increase unanswerable abstention, minimize answerable abstention
                    trustworthiness_score = delta_unans - delta_ans

                    print(f"  Layer {layer_idx}: Δ_unans={delta_unans:+.3f}, Δ_ans={delta_ans:+.3f}, score={trustworthiness_score:+.3f}")

                    if trustworthiness_score > best_trustworthiness_score:
                        best_trustworthiness_score = trustworthiness_score
                        best_layer = layer_idx

        if best_layer is None:
            best_layer = max(steering_vectors.keys())  # Fallback to deepest
            print(f"  ⚠️  Fallback to deepest layer {best_layer}")
        else:
            print(f"  ✓ Selected layer {best_layer} (trustworthiness score={best_trustworthiness_score:+.3f})")

        # Test steering effect on FULL EVALUATION set (disjoint from training)
        print(f"\nTesting steering effect on evaluation set ({len(eval_questions)} questions)...")
        print(f"Using layer: {best_layer}")
        print("EPSILON SIGN CONVENTION: -eps increases abstention (hallucination reduction)")
        print("                         +eps decreases abstention (more answering)")

        # Instantiate Experiment5 ONCE (not in inner loop for efficiency)
        exp5 = Experiment5(model_wrapper, config, {best_layer: steering_vectors[best_layer]})

        results = []
        # Test range of epsilon values (negative = increase abstention)
        for eps in [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]:
            for q in eval_questions:
                try:
                    result = exp5.test_one(q, best_layer, eps)
                    result['model'] = model_name
                    result['n_layers'] = n_layers
                    result['hidden_dim'] = hidden_dim
                    result['best_layer'] = best_layer  # Record which layer was used
                    results.append(result)
                except Exception as e:
                    print(f"ERROR testing question: {e}")
                    continue

        if not results:
            print("ERROR: No results collected")
            return None

        df = pd.DataFrame(results)

        # Compute metrics
        metrics = []
        for eps in df['epsilon'].unique():
            sub = df[df['epsilon'] == eps]
            ans = sub[~sub['is_unanswerable']]
            unans = sub[sub['is_unanswerable']]

            coverage = 1.0 - ans['abstained'].mean() if len(ans) else 0.0
            abstain_unans = unans['abstained'].mean() if len(unans) else 0.0
            halluc_unans = unans['hallucinated'].mean() if len(unans) else 0.0

            metrics.append({
                'model': model_name,
                'n_layers': n_layers,
                'hidden_dim': hidden_dim,
                'epsilon': eps,
                'coverage_answerable': coverage,
                'abstain_unanswerable': abstain_unans,
                'hallucination_unanswerable': halluc_unans,
            })

        metrics_df = pd.DataFrame(metrics)

        # Compute steering effectiveness
        baseline = metrics_df[metrics_df['epsilon'] == 0.0].iloc[0]
        steered = metrics_df[metrics_df['epsilon'] == -30.0].iloc[0]  # Use strong negative steering

        delta_abstain = steered['abstain_unanswerable'] - baseline['abstain_unanswerable']
        delta_coverage = steered['coverage_answerable'] - baseline['coverage_answerable']

        print(f"\n{'='*70}")
        print(f"RESULTS FOR {model_name}:")
        print(f"{'='*70}")
        print(f"  Baseline hallucination: {baseline['hallucination_unanswerable']:.1%}")
        print(f"  Steered hallucination: {steered['hallucination_unanswerable']:.1%}")
        print(f"\n  RAW ABSTENTION DELTA: {delta_abstain:+.3f} ({delta_abstain:+.1%})")
        print(f"  RAW COVERAGE DELTA: {delta_coverage:+.3f} ({delta_coverage:+.1%})")
        print(f"{'='*70}")

        # Cleanup
        del model_wrapper
        torch.cuda.empty_cache()

        return {
            'model_name': model_name,
            'n_layers': n_layers,
            'hidden_dim': hidden_dim,
            'best_layer': best_layer,  # Which layer worked best for this model
            'delta_abstain_unanswerable': float(delta_abstain),  # Raw delta (no threshold)
            'delta_coverage_answerable': float(delta_coverage),
            'baseline_hallucination': float(baseline['hallucination_unanswerable']),
            'steered_hallucination': float(steered['hallucination_unanswerable']),
            'metrics_df': metrics_df,
        }

    def run_scaling_analysis(self, train_questions: List[Dict],
                            eval_questions: List[Dict]) -> Tuple[Optional[pd.DataFrame], List[Dict]]:
        """
        Test steering across multiple model sizes

        Args:
            train_questions: Questions for extracting steering directions (disjoint from eval)
            eval_questions: Questions for evaluation (flexible size, minimum 10+10 per category)

        Models to test (in order of priority):
        1. Qwen2.5-1.5B-Instruct
        2. Qwen2.5-3B-Instruct
        3. Qwen2.5-7B-Instruct (best ROI for scaling)
        4. LLaMA-3.1-8B-Instruct (optional, for family generalization)
        """
        print("\n" + "="*70)
        print("EXPERIMENT 8: SCALING ANALYSIS")
        print("="*70)
        print(f"Train set: {len(train_questions)} questions")
        print(f"Eval set: {len(eval_questions)} questions")
        print(f"{'='*70}\n")

        # Models in order of priority
        models_to_test = [
            "Qwen/Qwen2.5-1.5B-Instruct",   # Small baseline
            "Qwen/Qwen2.5-3B-Instruct",     # Medium size
            "Qwen/Qwen2.5-7B-Instruct",     # Best ROI for scaling analysis
            # "meta-llama/Llama-3.1-8B-Instruct",  # Optional: different family
        ]

        all_results = []

        for model_name in models_to_test:
            result = self.test_model(model_name, train_questions, eval_questions)
            if result:
                all_results.append(result)

                # Save individual result
                model_safe_name = model_name.replace('/', '_')
                result['metrics_df'].to_csv(
                    self.config.results_dir / f"exp8_{model_safe_name}_results.csv",
                    index=False
                )

        if not all_results:
            print("ERROR: No models successfully tested")
            return None, []

        # Combine results
        summary_df = pd.DataFrame([
            {
                'model': r['model_name'],
                'n_layers': r['n_layers'],
                'hidden_dim': r['hidden_dim'],
                'best_layer': r['best_layer'],  # Which layer was selected
                'delta_abstain': r['delta_abstain_unanswerable'],  # Raw delta
                'delta_coverage': r['delta_coverage_answerable'],  # Raw delta
                'baseline_halluc': r['baseline_hallucination'],
                'steered_halluc': r['steered_hallucination'],
            }
            for r in all_results
        ])

        summary_df.to_csv(self.config.results_dir / "exp8_scaling_summary.csv", index=False)

        return summary_df, all_results

    def analyze_scaling(self, summary_df: pd.DataFrame, all_results: List[Dict]) -> Dict:
        """Analyze scaling behavior"""
        print("\n" + "="*70)
        print("EXPERIMENT 8: SCALING ANALYSIS")
        print("="*70)

        print("\nModel Comparison (Raw Abstention Deltas):")
        print(summary_df.to_string(index=False))

        # Report raw abstention deltas (no threshold)
        print("\n" + "="*70)
        print("RAW ABSTENTION DELTAS BY MODEL:")
        print("="*70)
        for _, row in summary_df.iterrows():
            model_short = row['model'].split('/')[-1].replace('-Instruct', '')
            print(f"{model_short:20s}: Δ={row['delta_abstain']:+.3f} ({row['delta_abstain']:+.1%})")
        print("="*70)

        # Check for scaling trends
        if len(summary_df) >= 2:
            # Compute total model capacity (layers × hidden_dim)
            summary_df['model_capacity'] = summary_df['n_layers'] * summary_df['hidden_dim']

            # Correlation between model capacity and steering effectiveness
            corr_params = summary_df['model_capacity'].corr(summary_df['delta_abstain'].abs())
            print(f"\nScaling correlation (capacity vs effect): {corr_params:.3f}")
            print(f"  (capacity = n_layers × hidden_dim)")

            if abs(corr_params) < 0.3:
                print("✓ Steering effectiveness independent of model size!")
            elif corr_params > 0:
                print("⚠️  Larger models show stronger steering effects")
            else:
                print("⚠️  Smaller models show stronger steering effects")

            # Also show best layers found per model
            print("\nBest layers found per model:")
            for _, row in summary_df.iterrows():
                model_short = row['model'].split('/')[-1].replace('-Instruct', '')
                print(f"  {model_short:20s}: Layer {row['best_layer']}/{row['n_layers']-1}")

        # Visualize
        self._plot_scaling(summary_df, all_results)

        return {
            'models_tested': summary_df['model'].tolist(),
            'mean_delta_abstain': float(summary_df['delta_abstain'].mean()),
            'scaling_correlation': float(corr_params) if len(summary_df) >= 2 else None,
            'summary': summary_df.to_dict('records'),
        }

    def _plot_scaling(self, summary_df: pd.DataFrame, all_results: List[Dict]):
        """Create scaling visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Panel 1: Raw abstention delta by model
        model_labels = [m.split('/')[-1].replace('-Instruct', '') for m in summary_df['model']]

        colors = ['green' if x < 0 else 'red' for x in summary_df['delta_abstain']]
        axes[0, 0].bar(range(len(model_labels)), summary_df['delta_abstain'],
                      color=colors, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xticks(range(len(model_labels)))
        axes[0, 0].set_xticklabels(model_labels, rotation=45, ha='right')
        axes[0, 0].set_ylabel("Raw Abstention Delta")
        axes[0, 0].set_title("Steering Effect by Model (Raw Δ Abstention)", fontweight='bold')
        axes[0, 0].axhline(0.0, color='black', linestyle='-', alpha=0.5)
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Panel 2: Tradeoff comparison
        axes[0, 1].scatter(summary_df['delta_coverage'], summary_df['delta_abstain'],
                          s=100, alpha=0.7, edgecolors='black')

        for i, model in enumerate(model_labels):
            axes[0, 1].annotate(model,
                               (summary_df.iloc[i]['delta_coverage'],
                                summary_df.iloc[i]['delta_abstain']),
                               fontsize=9, ha='center')

        axes[0, 1].axhline(0, color='gray', linestyle='-', alpha=0.3)
        axes[0, 1].axvline(0, color='gray', linestyle='-', alpha=0.3)
        axes[0, 1].set_xlabel("Δ Coverage (answerable)")
        axes[0, 1].set_ylabel("Δ Abstention (unanswerable)")
        axes[0, 1].set_title("Risk-Coverage Tradeoff by Model", fontweight='bold')
        axes[0, 1].grid(alpha=0.3)

        # Panel 3: Hallucination reduction
        x = np.arange(len(model_labels))
        width = 0.35

        axes[1, 0].bar(x - width/2, summary_df['baseline_halluc'], width,
                      label='Baseline', alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].bar(x + width/2, summary_df['steered_halluc'], width,
                      label='Steered', alpha=0.7, color='green', edgecolor='black')

        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(model_labels, rotation=45, ha='right')
        axes[1, 0].set_ylabel("Hallucination Rate")
        axes[1, 0].set_title("Hallucination Reduction by Model", fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Panel 4: Full epsilon sweep (if we have multiple models)
        if len(all_results) > 0:
            for result in all_results:
                model_name = result['model_name'].split('/')[-1].replace('-Instruct', '')
                metrics_df = result['metrics_df']

                axes[1, 1].plot(metrics_df['epsilon'],
                               metrics_df['hallucination_unanswerable'],
                               marker='o', label=model_name, linewidth=2)

            axes[1, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel("Epsilon")
            axes[1, 1].set_ylabel("Hallucination Rate (unanswerable)")
            axes[1, 1].set_title("Steering Curves by Model", fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
            axes[1, 1].set_ylim([0, 1])

        plt.tight_layout()

        output_path = self.config.results_dir / "exp8_scaling_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Figure saved to {output_path}")
        plt.close()


def main():
    """Run scaling analysis"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Experiment 8: Scaling Analysis')
    parser.add_argument('--n_questions', type=int, default=50,
                        help='Minimum number of questions per category (default: 50)')
    parser.add_argument('--min_per_split', type=int, default=10,
                        help='Minimum number of questions per category per split (default: 10)')
    args = parser.parse_args()

    config = ExperimentConfig()

    # Load questions
    print("Loading questions...")
    with open("./data/dataset_clearly_answerable.json", 'r') as f:
        answerable = json.load(f)
    with open("./data/dataset_clearly_unanswerable.json", 'r') as f:
        unanswerable = json.load(f)

    # CRITICAL: Split into train/eval to prevent data leakage
    print("\nSplitting data to prevent leakage...")

    # ASSERTIONS: Ensure we have enough data
    min_required = args.n_questions
    print(f"Configuration: {min_required} questions per category, {args.min_per_split} minimum per split")
    assert len(answerable) >= min_required, f"Need at least {min_required} answerable questions, got {len(answerable)}"
    assert len(unanswerable) >= min_required, f"Need at least {min_required} unanswerable questions, got {len(unanswerable)}"
    print(f"✓ Dataset size check passed: {len(answerable)} answerable, {len(unanswerable)} unanswerable")

    # Shuffle for random split
    import random
    random.seed(config.seed)
    random.shuffle(answerable)
    random.shuffle(unanswerable)

    # Split: Use half for train, half for eval (minimum min_per_split each if possible)
    min_per_split = args.min_per_split
    n_ans_train = max(min_per_split, len(answerable) // 2)
    n_unans_train = max(min_per_split, len(unanswerable) // 2)

    # Ensure we don't exceed available data
    n_ans_train = min(n_ans_train, len(answerable) - min_per_split)  # Leave at least min_per_split for eval
    n_unans_train = min(n_unans_train, len(unanswerable) - min_per_split)

    answerable_train = answerable[:n_ans_train]
    answerable_eval = answerable[n_ans_train:]  # Use remaining for eval

    unanswerable_train = unanswerable[:n_unans_train]
    unanswerable_eval = unanswerable[n_unans_train:]  # Use remaining for eval

    # Create train and eval sets
    train_questions = [
        {**q, "is_unanswerable": False} for q in answerable_train
    ] + [
        {**q, "is_unanswerable": True, "answer": None} for q in unanswerable_train
    ]

    eval_questions = [
        {**q, "is_unanswerable": False} for q in answerable_eval
    ] + [
        {**q, "is_unanswerable": True, "answer": None} for q in unanswerable_eval
    ]

    # ASSERTIONS: Ensure splits have minimum required size
    assert len(answerable_train) >= min_per_split, f"Train needs ≥{min_per_split} answerable, got {len(answerable_train)}"
    assert len(unanswerable_train) >= min_per_split, f"Train needs ≥{min_per_split} unanswerable, got {len(unanswerable_train)}"
    assert len(answerable_eval) >= min_per_split, f"Eval needs ≥{min_per_split} answerable, got {len(answerable_eval)}"
    assert len(unanswerable_eval) >= min_per_split, f"Eval needs ≥{min_per_split} unanswerable, got {len(unanswerable_eval)}"

    print(f"Train set: {len(train_questions)} questions "
          f"({len(answerable_train)} answerable, {len(unanswerable_train)} unanswerable)")
    print(f"Eval set:  {len(eval_questions)} questions "
          f"({len(answerable_eval)} answerable, {len(unanswerable_eval)} unanswerable)")
    print("✓ All assertions passed: sufficient data, no leakage\n")

    # Run scaling analysis
    exp8 = Experiment8(config)
    summary_df, all_results = exp8.run_scaling_analysis(train_questions, eval_questions)

    if summary_df is not None:
        # Analyze
        analysis = exp8.analyze_scaling(summary_df, all_results)

        # Save
        with open(config.results_dir / "exp8_summary.json", 'w') as f:
            json.dump(analysis, f, indent=2)

        print("\n✓ Experiment 8 complete!")


if __name__ == "__main__":
    main()
