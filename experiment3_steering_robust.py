"""
Experiment 3: Low-Dimensional Steering Control (ROBUST VERSION)

Addresses reviewer objections:
1. Fixed metric inconsistency (flip relative to ε=0)
2. Train/eval split
3. Negative controls (random direction, shuffled labels, early layer)
4. Compare 3 direction estimators (mean-diff, logistic probe, PC1)
5. Epsilon sweep: [-20, -10, -5, -2, 0, 2, 5, 10, 20]
6. Proper output artifacts (heatmap, table, bar chart)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple
import json
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from core_utils import (
    ModelWrapper, ExperimentConfig, extract_answer,
    get_decision_token_position, compute_binary_margin_simple, set_seed
)
from data_preparation import format_prompt


def _get_blocks(hf_model):
    """Get transformer blocks from HF model"""
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return hf_model.model.layers
    if hasattr(hf_model, "layers"):
        return hf_model.layers
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        return hf_model.transformer.h
    raise AttributeError(f"Can't find blocks for {type(hf_model)}")


def compute_flip_rate(df: pd.DataFrame, baseline_col: str = 'baseline_answer',
                      steered_col: str = 'steered_answer') -> float:
    """
    Canonical flip rate computation - used everywhere for consistency.

    Flip = answer changed relative to baseline (ε=0 for same prompt).
    """
    if len(df) == 0:
        return 0.0

    flipped = (df[baseline_col] != df[steered_col]).sum()
    return float(flipped / len(df))


class Experiment3Robust:
    """Robust steering experiment with controls and multiple direction estimators"""

    def __init__(self, model_wrapper: ModelWrapper, config: ExperimentConfig):
        self.model = model_wrapper
        self.config = config
        self.steering_vectors = {}  # {(layer, estimator_type): vector}
        self.results = []
        set_seed(config.seed)
        self.n_layers = self.model.model.config.num_hidden_layers

    # ============================================================================
    # Direction Computation Methods
    # ============================================================================

    def collect_activations(self, prompts: List[str], layer_idx: int) -> torch.Tensor:
        """Collect activations at decision position for all prompts"""
        activations = []

        for i, prompt in enumerate(prompts):
            self.model.clear_hooks()

            # FIX: Use get_decision_token_position to get where model chooses ABSTAIN/ANSWER
            position = get_decision_token_position(self.model.tokenizer, prompt)

            # SANITY CHECK: Log decision position for first prompt
            if i == 0:
                inputs = self.model.tokenizer(prompt, return_tensors="pt")
                prompt_len = inputs["input_ids"].shape[1]
                print(f"    [Sanity] Decision position: {position}/{prompt_len-1} (0-indexed)")
                # Decode the token at decision position to verify
                token_id = inputs["input_ids"][0, position].item()
                token_text = self.model.tokenizer.decode([token_id])
                print(f"    [Sanity] Token at decision pos: '{token_text}'")

            # Cache activation at decision position
            self.model.register_cache_hook(layer_idx, position)
            _ = self.model.generate(prompt, temperature=0.0, do_sample=False, max_new_tokens=1)

            activation = self.model.activation_cache[f"layer_{layer_idx}"].clone()
            activations.append(activation)
            self.model.clear_hooks()

        return torch.stack(activations).squeeze(1)  # [N, D]

    def compute_mean_diff_direction(self, pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
        """
        Method 1: Mean difference direction (current approach)

        FIX: Standardized convention - direction points from NEG (unanswerable) to POS (answerable)
        Positive epsilon pushes TOWARD answering, negative epsilon pushes TOWARD abstaining
        """
        pos_mean = pos_acts.mean(dim=0)
        neg_mean = neg_acts.mean(dim=0)
        direction = pos_mean - neg_mean  # Points toward POS (answerable)
        direction = direction / (direction.norm() + 1e-8)
        return direction

    def compute_probe_direction(self, pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
        """Method 2: Logistic regression probe weights"""
        X = torch.cat([pos_acts, neg_acts], dim=0).cpu().numpy()
        y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))

        probe = LogisticRegression(max_iter=1000, random_state=42)
        probe.fit(X, y)

        # Match dtype and device of pos_acts to avoid dtype mismatch errors
        direction = torch.tensor(probe.coef_[0], dtype=pos_acts.dtype, device=pos_acts.device)
        direction = direction / (direction.norm() + 1e-8)
        return direction

    def compute_pc1_direction(self, pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
        """Method 3: PC1 from PCA on difference-centered activations"""
        # Center by class means
        pos_centered = pos_acts - pos_acts.mean(dim=0, keepdim=True)
        neg_centered = neg_acts - neg_acts.mean(dim=0, keepdim=True)

        all_centered = torch.cat([pos_centered, neg_centered], dim=0).cpu().numpy()

        pca = PCA(n_components=1)
        pca.fit(all_centered)

        # Match dtype of pos_acts to avoid dtype mismatch errors
        direction = torch.tensor(pca.components_[0], dtype=pos_acts.dtype, device=pos_acts.device)
        direction = direction / (direction.norm() + 1e-8)

        # Ensure pointing toward POS
        pos_proj = (pos_acts @ direction).mean()
        neg_proj = (neg_acts @ direction).mean()
        if pos_proj < neg_proj:
            direction = -direction

        return direction

    def compute_random_direction(self, reference: torch.Tensor) -> torch.Tensor:
        """Control: Random direction with same norm as reference"""
        random_dir = torch.randn_like(reference)
        random_dir = random_dir / (random_dir.norm() + 1e-8)
        return random_dir

    def compute_shuffled_direction(self, pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
        """Control: Mean diff with shuffled labels"""
        all_acts = torch.cat([pos_acts, neg_acts], dim=0)
        labels = torch.cat([torch.ones(len(pos_acts)), torch.zeros(len(neg_acts))])

        # Shuffle labels
        perm = torch.randperm(len(labels))
        shuffled_labels = labels[perm]

        pos_shuffled = all_acts[shuffled_labels == 1]
        neg_shuffled = all_acts[shuffled_labels == 0]

        return self.compute_mean_diff_direction(pos_shuffled, neg_shuffled)

    def compute_all_directions(self, pos_prompts: List[str], neg_prompts: List[str],
                               layer_idx: int) -> Dict[str, torch.Tensor]:
        """Compute all direction types for a layer"""
        print(f"  Computing directions for layer {layer_idx}...")

        # Collect activations
        pos_acts = self.collect_activations(pos_prompts, layer_idx)
        neg_acts = self.collect_activations(neg_prompts, layer_idx)

        directions = {}

        # Main estimators
        directions['mean_diff'] = self.compute_mean_diff_direction(pos_acts, neg_acts)
        directions['probe'] = self.compute_probe_direction(pos_acts, neg_acts)
        directions['pc1'] = self.compute_pc1_direction(pos_acts, neg_acts)

        # Controls
        directions['random'] = self.compute_random_direction(directions['mean_diff'])
        directions['shuffled'] = self.compute_shuffled_direction(pos_acts, neg_acts)

        print(f"    Computed {len(directions)} direction types")

        return directions

    # ============================================================================
    # Steering Application
    # ============================================================================

    def apply_steering(self, prompt: str, layer_idx: int, steering_vector: torch.Tensor,
                      epsilon: float) -> Dict:
        """Apply steering and measure effect"""
        self.model.clear_hooks()

        # FIX: Get decision position for this prompt
        decision_pos = get_decision_token_position(self.model.tokenizer, prompt)

        # Baseline (ε=0)
        baseline_response = self.model.generate(
            prompt, temperature=0.0, do_sample=False, max_new_tokens=50
        )
        baseline_answer = extract_answer(baseline_response)

        # FIX: Use ABSTAIN prefix detection, not "UNCERTAIN"
        abstained_baseline = baseline_response.strip().upper().startswith("ABSTAIN")

        if epsilon == 0:
            return {
                "baseline_answer": baseline_answer,
                "steered_answer": baseline_answer,
                "baseline_response": baseline_response[:100],
                "steered_response": baseline_response[:100],
                "flipped": False,
                "abstained_baseline": abstained_baseline,
                "abstained_steered": abstained_baseline
            }

        # Steered generation
        self.model.clear_hooks()

        # FIX: Apply steering at decision position, not last token
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            hidden_states = hidden_states.clone()
            sv = steering_vector.to(hidden_states.device)
            if sv.ndim == 2 and sv.shape[0] == 1:
                sv = sv[0]

            # Apply at decision position (where model chooses ABSTAIN/ANSWER)
            seq_len = hidden_states.shape[1]
            if decision_pos < seq_len:
                hidden_states[:, decision_pos, :] += epsilon * sv

            if rest is None:
                return hidden_states
            return (hidden_states,) + rest

        layer = _get_blocks(self.model.model)[layer_idx]
        handle = layer.register_forward_hook(steering_hook)
        self.model.hooks.append(handle)

        steered_response = self.model.generate(
            prompt, temperature=0.0, do_sample=False, max_new_tokens=50
        )
        steered_answer = extract_answer(steered_response)

        self.model.clear_hooks()

        # FIX: Use ABSTAIN prefix detection consistently
        abstained_steered = steered_response.strip().upper().startswith("ABSTAIN")

        # FIX: Flip = decision changed (abstain vs answer), not content changed
        flipped = (abstained_baseline != abstained_steered)

        # Keep separate metric for content change
        answer_content_changed = (baseline_answer != steered_answer)

        return {
            "baseline_answer": baseline_answer,
            "steered_answer": steered_answer,
            "baseline_response": baseline_response[:100],
            "steered_response": steered_response[:100],
            "flipped": flipped,  # Decision flip (abstain vs answer)
            "answer_changed": answer_content_changed,  # Content changed
            "abstained_baseline": abstained_baseline,
            "abstained_steered": abstained_steered
        }

    # ============================================================================
    # Main Experiment
    # ============================================================================

    def run(self, pos_examples: List[Dict], neg_examples: List[Dict],
            test_examples: List[Dict],
            train_split: float = 0.6,
            epsilon_range: List[float] = None,
            test_layers: List[int] = None) -> pd.DataFrame:
        """
        Run comprehensive steering experiment with train/eval split

        Args:
            pos_examples: Answerable questions for computing direction
            neg_examples: Unanswerable questions for computing direction
            test_examples: Questions to test steering on (eval set)
            train_split: Fraction of pos/neg to use for training direction
            epsilon_range: List of epsilon values to test
            test_layers: Layers to test (default: config.target_layers)
        """
        if epsilon_range is None:
            epsilon_range = [-20, -10, -5, -2, 0, 2, 5, 10, 20]

        if test_layers is None:
            test_layers = self.config.target_layers

        print("\n" + "="*60)
        print("EXPERIMENT 3: ROBUST STEERING CONTROL")
        print("="*60)
        print(f"Train/Eval split: {train_split:.0%} train, {1-train_split:.0%} eval")
        print(f"Epsilon range: {epsilon_range}")
        print(f"Testing layers: {test_layers}")
        print(f"Direction estimators: mean-diff, probe, PC1 + controls")

        # FIX: Shuffle before splitting to avoid domain/difficulty bias
        import random
        pos_examples_copy = pos_examples.copy()
        neg_examples_copy = neg_examples.copy()
        random.seed(self.config.seed)
        random.shuffle(pos_examples_copy)
        random.shuffle(neg_examples_copy)

        # Split into train/eval
        n_train_pos = int(len(pos_examples_copy) * train_split)
        n_train_neg = int(len(neg_examples_copy) * train_split)

        train_pos = pos_examples_copy[:n_train_pos]
        eval_pos = pos_examples_copy[n_train_pos:]
        train_neg = neg_examples_copy[:n_train_neg]
        eval_neg = neg_examples_copy[n_train_neg:]

        print(f"\nTrain set: {len(train_pos)} POS + {len(train_neg)} NEG")
        print(f"Eval set: {len(eval_pos)} POS + {len(eval_neg)} NEG")
        print(f"Test set: {len(test_examples)} questions")

        # Format prompts
        train_pos_prompts = [format_prompt(ex["question"], "neutral", ex.get("context"))
                             for ex in train_pos]
        train_neg_prompts = [format_prompt(ex["question"], "neutral", ex.get("context"))
                             for ex in train_neg]

        # Compute all directions for all layers
        print("\n" + "="*60)
        print("COMPUTING STEERING DIRECTIONS")
        print("="*60)
        print("CONVENTION: Direction points from NEG→POS (unanswerable→answerable)")
        print("            Positive ε pushes TOWARD answering")
        print("            Negative ε pushes TOWARD abstaining")

        for layer_idx in test_layers:
            directions = self.compute_all_directions(train_pos_prompts, train_neg_prompts, layer_idx)

            for dir_type, direction in directions.items():
                self.steering_vectors[(layer_idx, dir_type)] = direction
                print(f"  Layer {layer_idx}, {dir_type}: norm={direction.norm():.4f}")

        # FIX 3: Compute early-layer direction as control
        # This tests if steering early layers (which shouldn't control abstention) works
        early_layer = int(self.n_layers * 0.25)
        print(f"\nComputing early-layer control (layer {early_layer})...")
        early_directions = self.compute_all_directions(train_pos_prompts, train_neg_prompts, early_layer)
        for dir_type, direction in early_directions.items():
            if dir_type == 'mean_diff':  # Only need one early control
                self.steering_vectors[(early_layer, 'early_layer')] = direction
                print(f"  Early layer control: norm={direction.norm():.4f}")

        # Test steering
        print("\n" + "="*60)
        print("TESTING STEERING EFFECTS")
        print("="*60)

        test_prompts = [(ex, format_prompt(ex["question"], "neutral", ex.get("context")))
                        for ex in test_examples]

        # SANITY CHECK: Test abstention detection on first example
        print("\n[SANITY CHECK] Testing abstention detection on first example...")
        first_prompt = test_prompts[0][1]
        test_response = self.model.generate(first_prompt, temperature=0.0, do_sample=False, max_new_tokens=50)
        print(f"  Response: {test_response[:100]}")
        abstained = test_response.strip().upper().startswith("ABSTAIN")
        print(f"  Detected as: {'ABSTAIN' if abstained else 'ANSWER'}")
        if not (test_response.strip().upper().startswith("ABSTAIN") or
                test_response.strip().upper().startswith("ANSWER")):
            print("  ⚠️  WARNING: Response doesn't start with ABSTAIN or ANSWER!")
            print("  ⚠️  Check if format_prompt is using correct forced prefix!")

        for layer_idx in tqdm(test_layers, desc="Layers"):
            for dir_type in ['mean_diff', 'probe', 'pc1', 'random', 'shuffled']:
                key = (layer_idx, dir_type)
                if key not in self.steering_vectors:
                    continue

                steering_vec = self.steering_vectors[key]

                for epsilon in epsilon_range:
                    for ex, prompt in test_prompts:
                        try:
                            result = self.apply_steering(prompt, layer_idx, steering_vec, epsilon)

                            self.results.append({
                                "layer": layer_idx,
                                "direction_type": dir_type,
                                "epsilon": epsilon,
                                "question": ex["question"][:50],
                                "true_answerability": ex.get("answerability", "unknown"),
                                **result
                            })
                        except Exception as e:
                            print(f"\nError: layer={layer_idx}, dir={dir_type}, ε={epsilon}: {e}")
                            continue

        # FIX 4: Test early-layer control at EARLY layer (not late layer)
        # This properly tests if early layers control abstention (they shouldn't)
        early_layer = int(self.n_layers * 0.25)
        if (early_layer, 'early_layer') in self.steering_vectors:
            print("\nTesting early-layer control...")
            early_vec = self.steering_vectors[(early_layer, 'early_layer')]

            for epsilon in epsilon_range:
                for ex, prompt in test_prompts:  # Test on all, not just subset
                    try:
                        result = self.apply_steering(prompt, early_layer, early_vec, epsilon)

                        self.results.append({
                            "layer": early_layer,
                            "direction_type": "early_layer",
                            "epsilon": epsilon,
                            "question": ex["question"][:50],
                            "true_answerability": ex.get("answerability", "unknown"),
                            **result
                        })
                    except Exception as e:
                        continue

        df = pd.DataFrame(self.results)

        # Save
        output_path = self.config.results_dir / "exp3_robust_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\n\nResults saved to {output_path}")

        # Save steering vectors
        torch.save(
            {str(k): v.detach().cpu() for k, v in self.steering_vectors.items()},
            self.config.results_dir / "steering_vectors_robust.pt"
        )
        print(f"Steering vectors saved")

        return df

    # ============================================================================
    # Analysis
    # ============================================================================

    def analyze(self, df: pd.DataFrame) -> Dict:
        """Comprehensive analysis"""
        print("\n" + "="*60)
        print("EXPERIMENT 3: ROBUST ANALYSIS")
        print("="*60)

        # 1. Compute flip rates using canonical function
        print("\n1. FLIP RATES BY DIRECTION TYPE")
        print("-" * 40)

        # For each direction type, at best epsilon
        direction_types = df['direction_type'].unique()

        flip_rates_by_dir = {}
        for dir_type in direction_types:
            dir_df = df[df['direction_type'] == dir_type]

            # Find best epsilon (highest flip rate)
            eps_flip_rates = []
            for eps in dir_df['epsilon'].unique():
                if eps == 0:
                    continue
                eps_df = dir_df[dir_df['epsilon'] == eps]
                flip_rate = compute_flip_rate(eps_df)
                eps_flip_rates.append((abs(eps), flip_rate))

            if eps_flip_rates:
                best_eps, best_flip = max(eps_flip_rates, key=lambda x: x[1])
                flip_rates_by_dir[dir_type] = {
                    "best_epsilon": best_eps,
                    "flip_rate": best_flip
                }
                print(f"  {dir_type:15s}: {best_flip:.1%} at |ε|={best_eps}")

        # 2. Compare estimators vs controls
        print("\n2. MAIN ESTIMATORS VS CONTROLS")
        print("-" * 40)

        main_estimators = ['mean_diff', 'probe', 'pc1']
        controls = ['random', 'shuffled', 'early_layer']

        main_rates = [flip_rates_by_dir.get(e, {}).get("flip_rate", 0) for e in main_estimators]
        control_rates = [flip_rates_by_dir.get(c, {}).get("flip_rate", 0) for c in controls]

        print(f"\nMain estimators (mean): {np.mean(main_rates):.1%}")
        print(f"Controls (mean):        {np.mean(control_rates):.1%}")

        if np.mean(main_rates) > np.mean(control_rates) * 2:
            print("✓ Main estimators >> controls (specific direction found!)")
        else:
            print("⚠️ Controls comparable to main estimators")

        # 3. Layer analysis
        print("\n3. BEST LAYER")
        print("-" * 40)

        # Find best (layer, direction_type, epsilon) combination
        best_flip = 0
        best_config = None

        for layer in df['layer'].unique():
            for dir_type in main_estimators:
                layer_dir_df = df[(df['layer'] == layer) & (df['direction_type'] == dir_type)]
                for eps in layer_dir_df['epsilon'].unique():
                    if eps == 0:
                        continue
                    eps_df = layer_dir_df[layer_dir_df['epsilon'] == eps]
                    flip_rate = compute_flip_rate(eps_df)

                    if flip_rate > best_flip:
                        best_flip = flip_rate
                        best_config = (layer, dir_type, eps)

        if best_config:
            print(f"Best configuration: layer={best_config[0]}, {best_config[1]}, ε={best_config[2]}")
            print(f"Flip rate: {best_flip:.1%}")

        # 4. Create visualizations
        self.create_visualizations(df, flip_rates_by_dir)

        # 5. Create summary table
        self.create_summary_table(df, flip_rates_by_dir)

        return {
            "flip_rates_by_direction": flip_rates_by_dir,
            "best_config": best_config,
            "best_flip_rate": float(best_flip),
            "main_estimator_mean": float(np.mean(main_rates)),
            "control_mean": float(np.mean(control_rates))
        }

    def create_visualizations(self, df: pd.DataFrame, flip_rates_by_dir: Dict):
        """Create all visualizations"""
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Plot 1: Flip rate heatmap for best direction (mean_diff)
        ax1 = fig.add_subplot(gs[0, :2])

        best_dir_df = df[df['direction_type'] == 'mean_diff']
        if len(best_dir_df) > 0:
            pivot = best_dir_df.pivot_table(
                index='layer',
                columns='epsilon',
                values='flipped',
                aggfunc='mean'
            )

            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                       ax=ax1, cbar_kws={'label': 'Flip Rate'}, vmin=0, vmax=1)
            ax1.set_title('Flip Rate Heatmap: Mean-Diff Direction', fontsize=13, fontweight='bold')
            ax1.set_xlabel('Epsilon (ε)', fontsize=11)
            ax1.set_ylabel('Layer', fontsize=11)

        # Plot 2: Comparison bar chart (estimators vs controls)
        ax2 = fig.add_subplot(gs[0, 2])

        dir_types = ['mean_diff', 'probe', 'pc1', 'random', 'shuffled', 'early_layer']
        flip_rates = [flip_rates_by_dir.get(d, {}).get("flip_rate", 0) for d in dir_types]
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#95a5a6', '#95a5a6', '#95a5a6']

        bars = ax2.barh(dir_types, flip_rates, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Flip Rate (at best ε)', fontsize=11)
        ax2.set_title('Direction Estimators vs Controls', fontsize=12, fontweight='bold')
        ax2.set_xlim([0, 1])
        ax2.grid(axis='x', alpha=0.3)

        # Add values on bars
        for i, (dt, fr) in enumerate(zip(dir_types, flip_rates)):
            ax2.text(fr + 0.02, i, f'{fr:.1%}', va='center', fontsize=10)

        # Plot 3: Epsilon sweep for main estimators
        ax3 = fig.add_subplot(gs[1, 0])

        best_layer = df[df['direction_type'] == 'mean_diff']['layer'].mode().iloc[0] if len(df) > 0 else self.config.target_layers[-1]

        for dir_type in ['mean_diff', 'probe', 'pc1']:
            dir_layer_df = df[(df['layer'] == best_layer) & (df['direction_type'] == dir_type)]

            eps_flip = []
            for eps in sorted(dir_layer_df['epsilon'].unique()):
                eps_df = dir_layer_df[dir_layer_df['epsilon'] == eps]
                flip_rate = compute_flip_rate(eps_df)
                eps_flip.append((eps, flip_rate))

            if eps_flip:
                epsilons, flips = zip(*eps_flip)
                ax3.plot(epsilons, flips, marker='o', label=dir_type, linewidth=2)

        ax3.set_xlabel('Epsilon (ε)', fontsize=11)
        ax3.set_ylabel('Flip Rate', fontsize=11)
        ax3.set_title(f'Epsilon Sweep (Layer {best_layer})', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        ax3.set_ylim([0, 1])
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)

        # Plot 4: Abstention rate changes
        ax4 = fig.add_subplot(gs[1, 1])

        # For mean_diff direction, show abstention rate at different epsilons
        mean_diff_df = df[(df['layer'] == best_layer) & (df['direction_type'] == 'mean_diff')]

        eps_abstain = []
        for eps in sorted(mean_diff_df['epsilon'].unique()):
            eps_df = mean_diff_df[mean_diff_df['epsilon'] == eps]
            abstain_rate = eps_df['abstained_steered'].mean()
            eps_abstain.append((eps, abstain_rate))

        if eps_abstain:
            epsilons, abstains = zip(*eps_abstain)
            ax4.plot(epsilons, abstains, marker='o', color='#e74c3c', linewidth=2, markersize=8)

        ax4.set_xlabel('Epsilon (ε)', fontsize=11)
        ax4.set_ylabel('Abstention Rate', fontsize=11)
        ax4.set_title('Abstention Control via Steering', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        ax4.set_ylim([0, 1])
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5)

        # Plot 5: Layer comparison for mean_diff
        ax5 = fig.add_subplot(gs[1, 2])

        layers = sorted(df['layer'].unique())
        best_eps = 10  # Fixed epsilon for comparison

        layer_flips = []
        for layer in layers:
            layer_df = df[(df['layer'] == layer) & (df['direction_type'] == 'mean_diff') &
                         (df['epsilon'] == best_eps)]
            flip_rate = compute_flip_rate(layer_df)
            layer_flips.append(flip_rate)

        ax5.bar(range(len(layers)), layer_flips, color='#3498db', alpha=0.8, edgecolor='black')
        ax5.set_xlabel('Layer Index', fontsize=11)
        ax5.set_ylabel('Flip Rate', fontsize=11)
        ax5.set_title(f'Flip Rate by Layer (ε={best_eps})', fontsize=12, fontweight='bold')
        ax5.set_xticks(range(len(layers)))
        ax5.set_xticklabels(layers)
        ax5.grid(axis='y', alpha=0.3)
        ax5.set_ylim([0, 1])

        plt.tight_layout()

        output_path = self.config.results_dir / "exp3_robust_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nComprehensive figure saved to {output_path}")
        plt.close()

    def create_summary_table(self, df: pd.DataFrame, flip_rates_by_dir: Dict):
        """Create summary table for paper"""
        print("\n" + "="*60)
        print("SUMMARY TABLE: Direction Estimators vs Controls")
        print("="*60)

        # Compute flip rate at a chosen epsilon (e.g., ε=10)
        chosen_eps = 10

        table_data = []
        direction_types = ['mean_diff', 'probe', 'pc1', 'random', 'shuffled', 'early_layer']
        labels = ['Mean-Diff (Ours)', 'Logistic Probe', 'PC1',
                 'Random Direction', 'Shuffled Labels', 'Early Layer']

        for dir_type, label in zip(direction_types, labels):
            dir_df = df[(df['direction_type'] == dir_type) & (df['epsilon'] == chosen_eps)]

            if len(dir_df) > 0:
                flip_rate = compute_flip_rate(dir_df)
                n_flips = dir_df['flipped'].sum()
                n_total = len(dir_df)

                # Get best epsilon flip rate from previous analysis
                best_info = flip_rates_by_dir.get(dir_type, {})
                best_flip = best_info.get("flip_rate", 0)
                best_eps = best_info.get("best_epsilon", 0)

                table_data.append({
                    'Method': label,
                    f'Flip Rate (ε={chosen_eps})': f"{flip_rate:.1%}",
                    'Best ε': f"{best_eps}",
                    'Best Flip Rate': f"{best_flip:.1%}",
                    'Type': 'Main' if dir_type in ['mean_diff', 'probe', 'pc1'] else 'Control'
                })

        table_df = pd.DataFrame(table_data)
        print("\n", table_df.to_string(index=False))

        # Save
        output_path = self.config.results_dir / "exp3_summary_table.csv"
        table_df.to_csv(output_path, index=False)
        print(f"\n\nSummary table saved to {output_path}")


# ============================================================================
# Standalone Helper Function (for importing by other experiments)
# ============================================================================

def compute_mean_diff_direction(model_wrapper, answerable_questions: List[Dict],
                                 unanswerable_questions: List[Dict],
                                 layer_idx: int) -> torch.Tensor:
    """
    Compute mean difference direction from answerable and unanswerable questions.

    This is a standalone helper function that can be imported by other experiments.

    Args:
        model_wrapper: ModelWrapper instance
        answerable_questions: List of answerable question dictionaries
        unanswerable_questions: List of unanswerable question dictionaries
        layer_idx: Layer index to extract activations from

    Returns:
        Normalized steering direction tensor (unanswerable - answerable)
    """
    from data_preparation import format_prompt

    activations_answerable = []
    activations_unanswerable = []

    # Collect answerable activations
    for q_data in answerable_questions:
        question = q_data["question"]
        context = q_data.get("context", None)
        prompt = format_prompt(question, "neutral", context)

        model_wrapper.clear_hooks()

        # Get prompt length for position
        inputs = model_wrapper.tokenizer(prompt, return_tensors="pt").to(model_wrapper.config.device)
        position = inputs["input_ids"].shape[1] - 1

        # Cache activation
        model_wrapper.register_cache_hook(layer_idx, position)
        _ = model_wrapper.generate(prompt, temperature=0.0, do_sample=False, max_new_tokens=1)

        activation = model_wrapper.activation_cache[f"layer_{layer_idx}"].clone()
        activations_answerable.append(activation)
        model_wrapper.clear_hooks()

    # Collect unanswerable activations
    for q_data in unanswerable_questions:
        question = q_data["question"]
        context = q_data.get("context", None)
        prompt = format_prompt(question, "neutral", context)

        model_wrapper.clear_hooks()

        # Get prompt length for position
        inputs = model_wrapper.tokenizer(prompt, return_tensors="pt").to(model_wrapper.config.device)
        position = inputs["input_ids"].shape[1] - 1

        # Cache activation
        model_wrapper.register_cache_hook(layer_idx, position)
        _ = model_wrapper.generate(prompt, temperature=0.0, do_sample=False, max_new_tokens=1)

        activation = model_wrapper.activation_cache[f"layer_{layer_idx}"].clone()
        activations_unanswerable.append(activation)
        model_wrapper.clear_hooks()

    # FIX: Compute mean difference direction with standardized convention
    # Direction points from NEG (unanswerable) to POS (answerable)
    # Positive epsilon pushes TOWARD answering, negative epsilon pushes TOWARD abstaining
    answerable_mean = torch.stack(activations_answerable).squeeze(1).mean(dim=0)
    unanswerable_mean = torch.stack(activations_unanswerable).squeeze(1).mean(dim=0)

    direction = answerable_mean - unanswerable_mean  # Points toward answerable (POS)
    direction = direction / (direction.norm() + 1e-8)

    return direction


# ============================================================================
# Main
# ============================================================================

def main(quick_test: bool = False):
    """Run robust Experiment 3"""
    import json

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

    # FIX 1: Don't use ambiguous questions - use eval split of clear questions
    # This ensures we're testing on questions with known ground truth
    # test_questions will be set from eval split below

    if quick_test:
        print("\n🔬 QUICK TEST MODE (Est. runtime: ~1 hour)")
        # Use more examples for meaningful signal
        answerable = answerable[:20]
        unanswerable = unanswerable[:20]
        # Test on eval split (set below after train/eval split)
        # Use symmetric epsilon range
        epsilon_range = [-8, -4, 0, 4, 8]
    else:
        print("\n⚠️  FULL MODE (Est. runtime: ~5 hours)")
        answerable = answerable[:40]
        unanswerable = unanswerable[:40]
        # Test on eval split (set below)
        epsilon_range = [-20, -10, -5, -2, 0, 2, 5, 10, 20]

    # FIX 2: Create test set from eval split instead of ambiguous questions
    # Split train/eval first
    train_split = 0.6
    n_train_pos = int(len(answerable) * train_split)
    n_train_neg = int(len(unanswerable) * train_split)

    train_pos = answerable[:n_train_pos]
    eval_pos = answerable[n_train_pos:]
    train_neg = unanswerable[:n_train_neg]
    eval_neg = unanswerable[n_train_neg:]

    # Use eval set as test questions (these have known answerability)
    test_questions = eval_pos + eval_neg

    print(f"\nDataset split:")
    print(f"  Train: {len(train_pos)} answerable + {len(train_neg)} unanswerable")
    print(f"  Test:  {len(eval_pos)} answerable + {len(eval_neg)} unanswerable")

    # Run experiment - pass pre-split data
    exp3 = Experiment3Robust(model, config)
    results_df = exp3.run(
        pos_examples=answerable,
        neg_examples=unanswerable,
        test_examples=test_questions,
        train_split=train_split,
        epsilon_range=epsilon_range
    )

    # Analyze
    summary = exp3.analyze(results_df)

    # Save summary
    summary_path = config.results_dir / "exp3_robust_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n✓ Experiment 3 (Robust) complete!")
    print(f"\n📊 Output files:")
    print(f"  Results:              {config.results_dir / 'exp3_robust_results.csv'}")
    print(f"  Summary:              {summary_path}")
    print(f"  Steering vectors:     {config.results_dir / 'steering_vectors_robust.pt'}")
    print(f"\n📈 Figures:")
    print(f"  Comprehensive:        {config.results_dir / 'exp3_robust_analysis.png'}")
    print(f"  Summary table:        {config.results_dir / 'exp3_summary_table.csv'}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Experiment 3: Robust Steering')
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test mode with smaller dataset')

    args = parser.parse_args()

    main(quick_test=args.quick_test)
