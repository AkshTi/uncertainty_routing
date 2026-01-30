"""
Experiment 2: Gate Localization via Activation Patching
Save as: experiment2_localization.py

Identifies which layers causally control the abstention decision
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple
import json

from core_utils import (
    ModelWrapper, ExperimentConfig, extract_answer,
    get_decision_token_position, compute_binary_margin_simple, set_seed
)
from data_preparation import format_prompt


class Experiment2:
    """Activation Patching to Localize Decision Control - Enhanced with Window Sweeps"""

    def __init__(self, model_wrapper: ModelWrapper, config: ExperimentConfig):
        self.model = model_wrapper
        self.config = config
        self.results = []
        self.window_results = []
        self.position_results = []
        self.control_results = []
        self.mixing_results = []
        set_seed(config.seed)
        self.n_layers = self.model.model.config.num_hidden_layers
    
    def get_baseline_margin(self, prompt: str) -> float:
        """Get baseline answerable/unanswerable tendency"""
        self.model.clear_hooks()
        response = self.model.generate(prompt, temperature=0.0, do_sample=False)
        return compute_binary_margin_simple(response)
    
    def cache_activation(self, prompt: str, layer_idx: int) -> torch.Tensor:
        """Cache activation at specific layer for a prompt"""
        self.model.clear_hooks()
        
        # Get decision token position
        position = get_decision_token_position(self.model.tokenizer, prompt)
        
        # Register hook to cache
        self.model.register_cache_hook(layer_idx, position)
        
        # Run forward pass
        _ = self.model.generate(prompt, temperature=0.0, do_sample=False)
        
        # Extract cached activation
        activation = self.model.activation_cache[f"layer_{layer_idx}"].clone()
        
        self.model.clear_hooks()
        return activation
    
    def patch_and_measure(self, target_prompt: str, donor_activation: torch.Tensor,
                         layer_idx: int, position: int = None) -> float:
        """Patch donor activation into target and measure result"""
        self.model.clear_hooks()

        # Get decision position for target (if not specified)
        if position is None:
            position = get_decision_token_position(self.model.tokenizer, target_prompt)

        # Register patching hook
        self.model.register_patch_hook(layer_idx, position, donor_activation)

        # Generate with patch
        response = self.model.generate(target_prompt, temperature=0.0, do_sample=False)
        margin = compute_binary_margin_simple(response)

        self.model.clear_hooks()
        return margin

    def patch_window_and_measure(self, target_prompt: str, donor_activations: List[torch.Tensor],
                                  layer_window: List[int], position: int = None,
                                  alpha: float = 1.0) -> float:
        """
        Patch multiple layers (window) and measure result

        Args:
            target_prompt: Target prompt to patch into
            donor_activations: List of activations to patch (one per layer in window)
            layer_window: List of layer indices to patch
            position: Token position to patch at (None = decision token)
            alpha: Mixing coefficient (1.0 = full replacement, <1.0 = interpolation)
        """
        self.model.clear_hooks()

        # Get decision position for target (if not specified)
        if position is None:
            position = get_decision_token_position(self.model.tokenizer, target_prompt)

        # Register patching hooks for all layers in window
        for layer_idx, donor_act in zip(layer_window, donor_activations):
            if alpha < 1.0:
                # Will need to mix during forward pass - register with mixing
                self.model.register_patch_hook(layer_idx, position, donor_act, alpha=alpha)
            else:
                self.model.register_patch_hook(layer_idx, position, donor_act)

        # Generate with patches
        response = self.model.generate(target_prompt, temperature=0.0, do_sample=False)
        margin = compute_binary_margin_simple(response)

        self.model.clear_hooks()
        return margin

    def cache_activation_at_position(self, prompt: str, layer_idx: int, position: int) -> torch.Tensor:
        """Cache activation at specific layer and position"""
        self.model.clear_hooks()

        # Register hook to cache at specific position
        self.model.register_cache_hook(layer_idx, position)

        # Run forward pass
        _ = self.model.generate(prompt, temperature=0.0, do_sample=False)

        # Extract cached activation
        activation = self.model.activation_cache[f"layer_{layer_idx}"].clone()

        self.model.clear_hooks()
        return activation
    
    def test_single_layer_patch(self, pos_prompt: str, neg_prompt: str,
                                layer_idx: int) -> Dict:
        """
        Test patching at single layer:
        1. Cache activation from POS (answerable) example
        2. Patch into NEG (unanswerable) example
        3. Measure effect
        """
        # Get baseline for NEG
        baseline_margin = self.get_baseline_margin(neg_prompt)

        # Cache POS activation
        pos_activation = self.cache_activation(pos_prompt, layer_idx)

        # Patch into NEG and measure
        patched_margin = self.patch_and_measure(neg_prompt, pos_activation, layer_idx)

        # Calculate effect
        delta = patched_margin - baseline_margin
        flipped = (baseline_margin < 0 and patched_margin > 0)

        return {
            "layer": layer_idx,
            "baseline_margin": float(baseline_margin),
            "patched_margin": float(patched_margin),
            "delta_margin": float(delta),
            "flipped": flipped
        }

    def test_layer_window_patch(self, pos_prompt: str, neg_prompt: str,
                                 layer_window: List[int], alpha: float = 1.0) -> Dict:
        """
        Test patching a window of k consecutive layers

        Args:
            pos_prompt: Source prompt (answerable)
            neg_prompt: Target prompt (unanswerable)
            layer_window: List of consecutive layer indices
            alpha: Mixing coefficient (1.0 = full replacement)
        """
        # Get baseline for NEG
        baseline_margin = self.get_baseline_margin(neg_prompt)

        # Cache POS activations for all layers in window
        donor_activations = []
        for layer_idx in layer_window:
            pos_activation = self.cache_activation(pos_prompt, layer_idx)
            donor_activations.append(pos_activation)

        # Patch window into NEG and measure
        patched_margin = self.patch_window_and_measure(
            neg_prompt, donor_activations, layer_window, alpha=alpha
        )

        # Calculate effect
        delta = patched_margin - baseline_margin
        flipped = (baseline_margin < 0 and patched_margin > 0)

        return {
            "window_start": layer_window[0],
            "window_end": layer_window[-1],
            "window_size": len(layer_window),
            "alpha": alpha,
            "baseline_margin": float(baseline_margin),
            "patched_margin": float(patched_margin),
            "delta_margin": float(delta),
            "flipped": flipped
        }

    def test_position_sweep(self, pos_prompt: str, neg_prompt: str,
                            layer_idx: int) -> Dict:
        """
        Test patching at different token positions:
        - Last token (decision position)
        - Second-to-last token
        - First token

        Args:
            pos_prompt: Source prompt
            neg_prompt: Target prompt
            layer_idx: Layer to patch at
        """
        # Get baseline
        baseline_margin = self.get_baseline_margin(neg_prompt)

        # Tokenize both prompts to get positions
        pos_tokens = self.model.tokenizer(pos_prompt, return_tensors="pt")
        neg_tokens = self.model.tokenizer(neg_prompt, return_tensors="pt")
        pos_seq_len = pos_tokens['input_ids'].shape[1]
        neg_seq_len = neg_tokens['input_ids'].shape[1]

        # Use the minimum sequence length to ensure positions are valid for both
        min_seq_len = min(pos_seq_len, neg_seq_len)

        positions = {
            "last": min_seq_len - 1,
            "second_last": max(0, min_seq_len - 2),
            "first": 0
        }

        results = {"baseline_margin": float(baseline_margin), "layer": layer_idx}

        for pos_name, pos_idx in positions.items():
            # Cache activation from pos_prompt at this position
            pos_activation = self.cache_activation_at_position(pos_prompt, layer_idx, pos_idx)

            # Patch into neg_prompt at same position
            patched_margin = self.patch_and_measure(neg_prompt, pos_activation, layer_idx, position=pos_idx)

            delta = patched_margin - baseline_margin
            flipped = (baseline_margin < 0 and patched_margin > 0)

            results[f"{pos_name}_margin"] = float(patched_margin)
            results[f"{pos_name}_delta"] = float(delta)
            results[f"{pos_name}_flipped"] = flipped

        return results

    def test_negative_controls(self, pos_prompt: str, neg_prompt: str,
                               random_prompt: str, layer_idx: int) -> Dict:
        """
        Test negative controls:
        1. Random prompt patch (should have ~zero effect)
        2. Wrong position patch (should be weaker)

        Args:
            pos_prompt: Source prompt (answerable)
            neg_prompt: Target prompt (unanswerable)
            random_prompt: Unrelated random prompt
            layer_idx: Layer to test
        """
        # Get baseline
        baseline_margin = self.get_baseline_margin(neg_prompt)

        results = {"baseline_margin": float(baseline_margin), "layer": layer_idx}

        # Control 1: Patch from random unrelated prompt
        random_activation = self.cache_activation(random_prompt, layer_idx)
        random_patched = self.patch_and_measure(neg_prompt, random_activation, layer_idx)
        results["random_prompt_delta"] = float(random_patched - baseline_margin)
        results["random_prompt_flipped"] = (baseline_margin < 0 and random_patched > 0)

        # Control 2: Patch from correct prompt but wrong position
        # Get decision position
        decision_pos = get_decision_token_position(self.model.tokenizer, neg_prompt)
        wrong_pos = 0  # First token instead of decision token

        pos_activation_wrong = self.cache_activation_at_position(pos_prompt, layer_idx, wrong_pos)
        wrong_pos_patched = self.patch_and_measure(neg_prompt, pos_activation_wrong, layer_idx,
                                                    position=decision_pos)
        results["wrong_position_delta"] = float(wrong_pos_patched - baseline_margin)
        results["wrong_position_flipped"] = (baseline_margin < 0 and wrong_pos_patched > 0)

        # Main effect (correct patch for comparison)
        pos_activation = self.cache_activation(pos_prompt, layer_idx)
        correct_patched = self.patch_and_measure(neg_prompt, pos_activation, layer_idx)
        results["correct_patch_delta"] = float(correct_patched - baseline_margin)
        results["correct_patch_flipped"] = (baseline_margin < 0 and correct_patched > 0)

        return results
    
    def run(self, pos_examples: List[Dict], neg_examples: List[Dict],
            n_pairs: int = 10, layer_stride: int = 1) -> pd.DataFrame:
        """
        Run activation patching across all layers
        
        Args:
            pos_examples: Questions model tends to answer
            neg_examples: Questions model tends to abstain on
            n_pairs: Number of example pairs to test
            layer_stride: Test every Nth layer (1 = all layers, 2 = every other, etc.)
        """
        print("\n" + "="*60)
        print("EXPERIMENT 2: Gate Localization")
        print("="*60)
        print(f"Testing {n_pairs} example pairs")
        print(f"Across {self.n_layers // layer_stride} layers (stride={layer_stride})")
        print()
        
        # Prepare example pairs
        pairs = self._prepare_pairs(pos_examples, neg_examples, n_pairs)
        
        # Test each layer
        layers_to_test = range(0, self.n_layers, layer_stride)
        
        for layer_idx in tqdm(layers_to_test, desc="Layers"):
            for pair_idx, (pos_prompt, neg_prompt, pos_q, neg_q) in enumerate(pairs):
                try:
                    result = self.test_single_layer_patch(pos_prompt, neg_prompt, layer_idx)
                    
                    # Add metadata
                    result.update({
                        "pair_idx": pair_idx,
                        "pos_question": pos_q["question"][:50],
                        "neg_question": neg_q["question"][:50]
                    })
                    
                    self.results.append(result)
                    
                except Exception as e:
                    print(f"\nError at layer {layer_idx}, pair {pair_idx}: {e}")
                    continue
        
        df = pd.DataFrame(self.results)
        
        # Save results
        output_path = self.config.results_dir / "exp2_raw_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
        return df
    
    def _prepare_pairs(self, pos_examples: List[Dict], neg_examples: List[Dict],
                      n_pairs: int) -> List[Tuple]:
        """Prepare (POS, NEG) prompt pairs"""
        pairs = []

        n_available = min(len(pos_examples), len(neg_examples), n_pairs)

        for i in range(n_available):
            pos_q = pos_examples[i]
            neg_q = neg_examples[i]

            pos_prompt = format_prompt(
                pos_q["question"],
                "neutral",
                pos_q.get("context")
            )

            neg_prompt = format_prompt(
                neg_q["question"],
                "neutral",
                neg_q.get("context")
            )

            pairs.append((pos_prompt, neg_prompt, pos_q, neg_q))

        return pairs

    def run_window_sweep(self, pos_examples: List[Dict], neg_examples: List[Dict],
                         n_pairs: int = 10, window_sizes: List[int] = None) -> pd.DataFrame:
        """
        Run minimal-sufficient window sweep: test k-layer windows

        Args:
            pos_examples: Answerable questions
            neg_examples: Unanswerable questions
            n_pairs: Number of example pairs to test
            window_sizes: List of window sizes to test (default: [1, 2, 3])
        """
        if window_sizes is None:
            window_sizes = [1, 2, 3]

        print("\n" + "="*60)
        print("WINDOW SWEEP: Minimal-Sufficient Layer Windows")
        print("="*60)
        print(f"Testing window sizes: {window_sizes}")
        print(f"Testing {n_pairs} example pairs")

        pairs = self._prepare_pairs(pos_examples, neg_examples, n_pairs)

        for window_size in window_sizes:
            print(f"\n--- Testing {window_size}-layer windows ---")

            # Test sliding windows of this size
            for start_layer in tqdm(range(0, self.n_layers - window_size + 1, 2),
                                   desc=f"k={window_size}"):
                layer_window = list(range(start_layer, start_layer + window_size))

                for pair_idx, (pos_prompt, neg_prompt, pos_q, neg_q) in enumerate(pairs):
                    try:
                        result = self.test_layer_window_patch(pos_prompt, neg_prompt, layer_window)

                        # Add metadata
                        result.update({
                            "pair_idx": pair_idx,
                            "pos_question": pos_q["question"][:50],
                            "neg_question": neg_q["question"][:50]
                        })

                        self.window_results.append(result)

                    except Exception as e:
                        print(f"\nError at window {layer_window}, pair {pair_idx}: {e}")
                        continue

        df = pd.DataFrame(self.window_results)
        output_path = self.config.results_dir / "exp2_window_sweep.csv"
        df.to_csv(output_path, index=False)
        print(f"\nWindow sweep results saved to {output_path}")

        return df

    def run_position_sweep(self, pos_examples: List[Dict], neg_examples: List[Dict],
                           n_pairs: int = 10, best_layer: int = None) -> pd.DataFrame:
        """
        Run position sweep: test patching at different token positions

        Args:
            pos_examples: Answerable questions
            neg_examples: Unanswerable questions
            n_pairs: Number of example pairs
            best_layer: Layer to test (if None, uses layer 3/4 through network)
        """
        if best_layer is None:
            best_layer = int(self.n_layers * 0.75)

        print("\n" + "="*60)
        print("POSITION SWEEP: Token Position Sensitivity")
        print("="*60)
        print(f"Testing layer {best_layer}")
        print(f"Positions: last token, second-to-last, first token")

        pairs = self._prepare_pairs(pos_examples, neg_examples, n_pairs)

        for pair_idx, (pos_prompt, neg_prompt, pos_q, neg_q) in enumerate(tqdm(pairs, desc="Position sweep")):
            try:
                result = self.test_position_sweep(pos_prompt, neg_prompt, best_layer)

                # Add metadata
                result.update({
                    "pair_idx": pair_idx,
                    "pos_question": pos_q["question"][:50],
                    "neg_question": neg_q["question"][:50]
                })

                self.position_results.append(result)

            except Exception as e:
                print(f"\nError at pair {pair_idx}: {e}")
                continue

        df = pd.DataFrame(self.position_results)
        output_path = self.config.results_dir / "exp2_position_sweep.csv"
        df.to_csv(output_path, index=False)
        print(f"\nPosition sweep results saved to {output_path}")

        return df

    def run_negative_controls(self, pos_examples: List[Dict], neg_examples: List[Dict],
                              n_pairs: int = 10, best_layer: int = None) -> pd.DataFrame:
        """
        Run negative controls: random prompt and wrong position patches

        Args:
            pos_examples: Answerable questions
            neg_examples: Unanswerable questions
            n_pairs: Number of example pairs
            best_layer: Layer to test
        """
        if best_layer is None:
            best_layer = int(self.n_layers * 0.75)

        print("\n" + "="*60)
        print("NEGATIVE CONTROLS")
        print("="*60)
        print(f"Testing layer {best_layer}")

        pairs = self._prepare_pairs(pos_examples, neg_examples, n_pairs)

        # Use remaining examples as random prompts
        random_examples = pos_examples[n_pairs:n_pairs*2] if len(pos_examples) > n_pairs*2 else pos_examples

        for pair_idx, (pos_prompt, neg_prompt, pos_q, neg_q) in enumerate(tqdm(pairs, desc="Controls")):
            try:
                # Get a random prompt
                random_q = random_examples[pair_idx % len(random_examples)]
                random_prompt = format_prompt(random_q["question"], "neutral", random_q.get("context"))

                result = self.test_negative_controls(pos_prompt, neg_prompt, random_prompt, best_layer)

                # Add metadata
                result.update({
                    "pair_idx": pair_idx,
                    "pos_question": pos_q["question"][:50],
                    "neg_question": neg_q["question"][:50]
                })

                self.control_results.append(result)

            except Exception as e:
                print(f"\nError at pair {pair_idx}: {e}")
                continue

        df = pd.DataFrame(self.control_results)
        output_path = self.config.results_dir / "exp2_negative_controls.csv"
        df.to_csv(output_path, index=False)
        print(f"\nNegative control results saved to {output_path}")

        return df

    def run_mixing_sweep(self, pos_examples: List[Dict], neg_examples: List[Dict],
                         n_pairs: int = 10, best_window: List[int] = None,
                         alphas: List[float] = None) -> pd.DataFrame:
        """
        Run mixing/interpolation sweep: test different alpha values

        Args:
            pos_examples: Answerable questions
            neg_examples: Unanswerable questions
            n_pairs: Number of example pairs
            best_window: Best layer window to test (if None, uses top 3 layers)
            alphas: Mixing coefficients to test
        """
        if best_window is None:
            best_window = list(range(int(self.n_layers * 0.7), int(self.n_layers * 0.7) + 3))

        if alphas is None:
            alphas = [0.25, 0.5, 0.75, 1.0]

        print("\n" + "="*60)
        print("MIXING SWEEP: Patch Magnitude Interpolation")
        print("="*60)
        print(f"Testing window: {best_window}")
        print(f"Alpha values: {alphas}")

        pairs = self._prepare_pairs(pos_examples, neg_examples, n_pairs)

        for alpha in alphas:
            print(f"\n--- Testing alpha={alpha} ---")

            for pair_idx, (pos_prompt, neg_prompt, pos_q, neg_q) in enumerate(tqdm(pairs, desc=f"α={alpha}")):
                try:
                    result = self.test_layer_window_patch(pos_prompt, neg_prompt, best_window, alpha=alpha)

                    # Add metadata
                    result.update({
                        "pair_idx": pair_idx,
                        "pos_question": pos_q["question"][:50],
                        "neg_question": neg_q["question"][:50]
                    })

                    self.mixing_results.append(result)

                except Exception as e:
                    print(f"\nError at alpha={alpha}, pair {pair_idx}: {e}")
                    continue

        df = pd.DataFrame(self.mixing_results)
        output_path = self.config.results_dir / "exp2_mixing_sweep.csv"
        df.to_csv(output_path, index=False)
        print(f"\nMixing sweep results saved to {output_path}")

        return df
    
    def analyze(self, df: pd.DataFrame = None,
                window_df: pd.DataFrame = None,
                position_df: pd.DataFrame = None,
                control_df: pd.DataFrame = None,
                mixing_df: pd.DataFrame = None) -> Dict:
        """
        Comprehensive analysis of all patching experiments

        Args:
            df: Single-layer patching results (optional, from old run)
            window_df: Window sweep results
            position_df: Position sweep results
            control_df: Negative control results
            mixing_df: Mixing/interpolation results
        """
        print("\n" + "="*60)
        print("EXPERIMENT 2: COMPREHENSIVE ANALYSIS")
        print("="*60)

        analysis_results = {}

        # ===== WINDOW SWEEP ANALYSIS =====
        if window_df is not None and len(window_df) > 0:
            print("\n--- WINDOW SWEEP ANALYSIS ---")

            # Group by window size
            window_stats = window_df.groupby("window_size").agg({
                "delta_margin": ["mean", "std", "max"],
                "flipped": "mean"
            }).round(3)

            print("\nEffect by window size:")
            print(window_stats)

            # Find best window for each size
            best_windows = {}
            for k in window_df['window_size'].unique():
                k_df = window_df[window_df['window_size'] == k]
                best_idx = k_df['delta_margin'].idxmax()
                best_row = k_df.loc[best_idx]
                best_windows[k] = {
                    "window": f"[{best_row['window_start']}-{best_row['window_end']}]",
                    "delta": float(best_row['delta_margin']),
                    "flip_rate": float(best_row['flipped'])
                }

            print("\nBest window for each size:")
            for k, info in best_windows.items():
                print(f"  k={k}: {info['window']} → Δ={info['delta']:.3f}, flip={info['flip_rate']:.1%}")

            # Convert MultiIndex columns to JSON-serializable format
            window_stats_dict = {}
            for col in window_stats.columns:
                if isinstance(col, tuple):
                    # Flatten tuple column names
                    key = '_'.join(str(x) for x in col)
                    window_stats_dict[key] = window_stats[col].to_dict()
                else:
                    window_stats_dict[str(col)] = window_stats[col].to_dict()

            analysis_results["window_stats"] = window_stats_dict
            analysis_results["best_windows"] = best_windows

            # Compute what % of max effect each window size achieves
            if window_df['delta_margin'].max() > 0:
                max_effect = window_df['delta_margin'].max()
                for k in sorted(window_df['window_size'].unique()):
                    k_max = window_df[window_df['window_size'] == k]['delta_margin'].max()
                    pct = (k_max / max_effect) * 100
                    print(f"  k={k}-layer window achieves {pct:.1f}% of max effect")

        # ===== POSITION SWEEP ANALYSIS =====
        if position_df is not None and len(position_df) > 0:
            print("\n--- POSITION SWEEP ANALYSIS ---")

            positions = ['last', 'second_last', 'first']
            position_means = {pos: position_df[f"{pos}_delta"].mean() for pos in positions}
            position_flips = {pos: position_df[f"{pos}_flipped"].mean() for pos in positions}

            print("\nEffect by token position:")
            for pos in positions:
                print(f"  {pos:15s}: Δ={position_means[pos]:.3f}, flip={position_flips[pos]:.1%}")

            analysis_results["position_effects"] = position_means
            analysis_results["position_flips"] = position_flips

        # ===== NEGATIVE CONTROLS ANALYSIS =====
        if control_df is not None and len(control_df) > 0:
            print("\n--- NEGATIVE CONTROLS ANALYSIS ---")

            control_types = ['correct_patch', 'random_prompt', 'wrong_position']
            control_means = {ct: control_df[f"{ct}_delta"].mean() for ct in control_types}
            control_flips = {ct: control_df[f"{ct}_flipped"].mean() for ct in control_types}

            print("\nEffect by patch type:")
            for ct in control_types:
                print(f"  {ct:20s}: Δ={control_means[ct]:.3f}, flip={control_flips[ct]:.1%}")

            # Check if controls are near zero as expected
            random_effect = abs(control_means['random_prompt'])
            correct_effect = abs(control_means['correct_patch'])
            if correct_effect > 0:
                ratio = random_effect / correct_effect
                print(f"\nRandom/Correct ratio: {ratio:.2%} (should be near 0)")

            analysis_results["control_effects"] = control_means
            analysis_results["control_flips"] = control_flips

        # ===== MIXING SWEEP ANALYSIS =====
        if mixing_df is not None and len(mixing_df) > 0:
            print("\n--- MIXING SWEEP ANALYSIS ---")

            mixing_stats = mixing_df.groupby("alpha").agg({
                "delta_margin": "mean",
                "flipped": "mean"
            }).round(3)

            print("\nEffect by mixing coefficient:")
            print(mixing_stats)

            # Check for monotonicity
            alphas_sorted = sorted(mixing_df['alpha'].unique())
            deltas = [mixing_df[mixing_df['alpha'] == a]['delta_margin'].mean() for a in alphas_sorted]
            monotonic = all(deltas[i] <= deltas[i+1] for i in range(len(deltas)-1))

            print(f"\nMonotonic increase: {monotonic} (expected for real gating)")

            # Convert MultiIndex columns to JSON-serializable format
            mixing_stats_dict = {}
            for col in mixing_stats.columns:
                if isinstance(col, tuple):
                    # Flatten tuple column names
                    key = '_'.join(str(x) for x in col)
                    mixing_stats_dict[key] = mixing_stats[col].to_dict()
                else:
                    mixing_stats_dict[str(col)] = mixing_stats[col].to_dict()

            analysis_results["mixing_stats"] = mixing_stats_dict
            analysis_results["monotonic"] = monotonic

        # ===== CREATE VISUALIZATIONS =====
        self.plot_comprehensive_results(window_df, position_df, control_df, mixing_df)

        # ===== CREATE SUMMARY TABLE =====
        self.create_summary_table(window_df, position_df, control_df)

        return analysis_results
    
    def plot_comprehensive_results(self, window_df: pd.DataFrame = None,
                                    position_df: pd.DataFrame = None,
                                    control_df: pd.DataFrame = None,
                                    mixing_df: pd.DataFrame = None):
        """Create comprehensive visualization of all experiments"""

        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # ===== PLOT 1: Window size vs flip rate =====
        if window_df is not None and len(window_df) > 0:
            ax1 = fig.add_subplot(gs[0, 0])

            window_flip = window_df.groupby('window_size')['flipped'].mean()
            window_delta = window_df.groupby('window_size')['delta_margin'].mean()

            ax1.bar(window_flip.index, window_flip.values, alpha=0.7, color='#3498db',
                   label='Flip Rate')
            ax1.set_xlabel('Window Size (k layers)', fontsize=11)
            ax1.set_ylabel('Flip Rate', fontsize=11, color='#3498db')
            ax1.set_title('Minimal-Sufficient Window', fontsize=12, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor='#3498db')
            ax1.set_ylim([0, 1])

            # Add delta margin on secondary axis
            ax1_twin = ax1.twinx()
            ax1_twin.plot(window_delta.index, window_delta.values, 'o-',
                         color='#e74c3c', linewidth=2, markersize=8, label='Δ Margin')
            ax1_twin.set_ylabel('Δ Margin', fontsize=11, color='#e74c3c')
            ax1_twin.tick_params(axis='y', labelcolor='#e74c3c')

            ax1.grid(alpha=0.3)

        # ===== PLOT 2: Flip rate by window start position =====
        if window_df is not None and len(window_df) > 0:
            ax2 = fig.add_subplot(gs[0, 1])

            # For each window size, show effect vs position
            for k in sorted(window_df['window_size'].unique()):
                k_df = window_df[window_df['window_size'] == k]
                grouped = k_df.groupby('window_start')['flipped'].mean()
                ax2.plot(grouped.index, grouped.values, 'o-', label=f'k={k}', alpha=0.7)

            ax2.set_xlabel('Window Start Layer', fontsize=11)
            ax2.set_ylabel('Flip Rate', fontsize=11)
            ax2.set_title('Window Position Sensitivity', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.set_ylim([0, 1])
            ax2.grid(alpha=0.3)

        # ===== PLOT 3: Position sweep =====
        if position_df is not None and len(position_df) > 0:
            ax3 = fig.add_subplot(gs[0, 2])

            positions = ['first', 'second_last', 'last']
            pos_labels = ['First Token', '2nd-to-Last', 'Last Token']
            deltas = [position_df[f"{pos}_delta"].mean() for pos in positions]
            flips = [position_df[f"{pos}_flipped"].mean() for pos in positions]

            x = np.arange(len(positions))
            width = 0.35

            ax3.bar(x - width/2, deltas, width, label='Δ Margin', color='#2ecc71', alpha=0.8)
            ax3_twin = ax3.twinx()
            ax3_twin.bar(x + width/2, flips, width, label='Flip Rate', color='#e74c3c', alpha=0.8)

            ax3.set_xlabel('Token Position', fontsize=11)
            ax3.set_ylabel('Δ Margin', fontsize=11, color='#2ecc71')
            ax3_twin.set_ylabel('Flip Rate', fontsize=11, color='#e74c3c')
            ax3.set_title('Token Position Localization', fontsize=12, fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(pos_labels, rotation=15)
            ax3_twin.set_ylim([0, 1])
            ax3.tick_params(axis='y', labelcolor='#2ecc71')
            ax3_twin.tick_params(axis='y', labelcolor='#e74c3c')
            ax3.grid(alpha=0.3)

        # ===== PLOT 4: Negative controls comparison =====
        if control_df is not None and len(control_df) > 0:
            ax4 = fig.add_subplot(gs[1, 0])

            control_types = ['correct_patch', 'wrong_position', 'random_prompt']
            labels = ['Correct Patch', 'Wrong Position', 'Random Prompt']
            deltas = [control_df[f"{ct}_delta"].mean() for ct in control_types]
            stds = [control_df[f"{ct}_delta"].std() for ct in control_types]

            colors = ['#2ecc71', '#f39c12', '#95a5a6']
            ax4.bar(labels, deltas, yerr=stds, capsize=5, color=colors, alpha=0.8)
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax4.set_ylabel('Δ Margin', fontsize=11)
            ax4.set_title('Negative Controls', fontsize=12, fontweight='bold')
            ax4.set_xticklabels(labels, rotation=20, ha='right')
            ax4.grid(axis='y', alpha=0.3)

        # ===== PLOT 5: Mixing/interpolation monotonicity =====
        if mixing_df is not None and len(mixing_df) > 0:
            ax5 = fig.add_subplot(gs[1, 1])

            mixing_stats = mixing_df.groupby('alpha').agg({
                'delta_margin': ['mean', 'std'],
                'flipped': 'mean'
            })

            alphas = sorted(mixing_df['alpha'].unique())
            means = [mixing_stats.loc[a, ('delta_margin', 'mean')] for a in alphas]
            stds = [mixing_stats.loc[a, ('delta_margin', 'std')] for a in alphas]

            ax5.plot(alphas, means, 'o-', linewidth=2, markersize=8, color='#9b59b6')
            ax5.fill_between(alphas,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.3, color='#9b59b6')
            ax5.set_xlabel('Mixing Coefficient (α)', fontsize=11)
            ax5.set_ylabel('Δ Margin', fontsize=11)
            ax5.set_title('Patch Magnitude Monotonicity', fontsize=12, fontweight='bold')
            ax5.grid(alpha=0.3)
            ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        # ===== PLOT 6: Best window heatmap =====
        if window_df is not None and len(window_df) > 0:
            ax6 = fig.add_subplot(gs[1, 2])

            # Create heatmap of flip rates by window start and size
            pivot_data = window_df.pivot_table(
                index='window_start',
                columns='window_size',
                values='flipped',
                aggfunc='mean'
            )

            sns.heatmap(pivot_data, ax=ax6, cmap='RdYlGn', vmin=0, vmax=1,
                       cbar_kws={'label': 'Flip Rate'})
            ax6.set_xlabel('Window Size', fontsize=11)
            ax6.set_ylabel('Window Start Layer', fontsize=11)
            ax6.set_title('Window Configuration Heatmap', fontsize=12, fontweight='bold')

        plt.tight_layout()

        # Save
        output_path = self.config.results_dir / "exp2_comprehensive_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nComprehensive figure saved to {output_path}")
        plt.close()

    def create_summary_table(self, window_df: pd.DataFrame = None,
                             position_df: pd.DataFrame = None,
                             control_df: pd.DataFrame = None):
        """Create summary table comparing best window with controls"""

        print("\n" + "="*60)
        print("SUMMARY TABLE: Best Window vs Controls")
        print("="*60)

        table_data = []

        # Best window for each size
        if window_df is not None:
            for k in sorted(window_df['window_size'].unique()):
                k_df = window_df[window_df['window_size'] == k]
                best_row = k_df.loc[k_df['delta_margin'].idxmax()]
                table_data.append({
                    'Configuration': f'{k}-layer window [{best_row["window_start"]}-{best_row["window_end"]}]',
                    'Δ Margin': f"{best_row['delta_margin']:.3f}",
                    'Flip Rate': f"{best_row['flipped']:.1%}",
                    'Type': 'Window'
                })

        # Position effects
        if position_df is not None and len(position_df) > 0:
            for pos in ['last', 'second_last', 'first']:
                # Check if columns exist before accessing
                delta_col = f'{pos}_delta'
                flipped_col = f'{pos}_flipped'
                if delta_col in position_df.columns and flipped_col in position_df.columns:
                    pos_label = {'last': 'Last token', 'second_last': '2nd-to-last', 'first': 'First token'}
                    table_data.append({
                        'Configuration': pos_label[pos],
                        'Δ Margin': f"{position_df[delta_col].mean():.3f}",
                        'Flip Rate': f"{position_df[flipped_col].mean():.1%}",
                        'Type': 'Position'
                    })

        # Controls
        if control_df is not None and len(control_df) > 0:
            for ct in ['correct_patch', 'wrong_position', 'random_prompt']:
                # Check if columns exist before accessing
                delta_col = f'{ct}_delta'
                flipped_col = f'{ct}_flipped'
                if delta_col in control_df.columns and flipped_col in control_df.columns:
                    ct_label = {'correct_patch': 'Correct patch', 'wrong_position': 'Wrong position',
                               'random_prompt': 'Random prompt'}
                    table_data.append({
                        'Configuration': ct_label[ct],
                        'Δ Margin': f"{control_df[delta_col].mean():.3f}",
                        'Flip Rate': f"{control_df[flipped_col].mean():.1%}",
                        'Type': 'Control'
                    })

        table_df = pd.DataFrame(table_data)
        print("\n", table_df.to_string(index=False))

        # Save as CSV
        output_path = self.config.results_dir / "exp2_summary_table.csv"
        table_df.to_csv(output_path, index=False)
        print(f"\n\nSummary table saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main(n_pairs: int = 10, quick_test: bool = False):
    """
    Run Experiment 2: Comprehensive Causal Localization

    Args:
        n_pairs: Number of example pairs to test
        quick_test: If True, run minimal tests for quick validation
    """
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

    # Initialize experiment
    exp2 = Experiment2(model, config)

    if quick_test:
        print("\n🔬 QUICK TEST MODE - Running minimal experiments")
        n_pairs = 3
        window_sizes = [1, 2]
        alphas = [0.5, 1.0]
    else:
        window_sizes = [1, 2, 3]
        alphas = [0.25, 0.5, 0.75, 1.0]

    # ===== RUN ALL EXPERIMENTS =====

    # 1. Window Sweep
    print("\n" + "="*60)
    print("STEP 1: MINIMAL-SUFFICIENT WINDOW SWEEP")
    print("="*60)
    window_df = exp2.run_window_sweep(
        answerable[:n_pairs*2],
        unanswerable[:n_pairs*2],
        n_pairs=n_pairs,
        window_sizes=window_sizes
    )

    # Find best window from results
    best_window_row = window_df.loc[window_df['delta_margin'].idxmax()]
    best_window = list(range(best_window_row['window_start'], best_window_row['window_end'] + 1))
    best_layer = best_window_row['window_start'] + len(best_window) // 2

    print(f"\nBest window identified: {best_window} (center: layer {best_layer})")

    # 2. Position Sweep
    print("\n" + "="*60)
    print("STEP 2: PATCH POSITION SWEEP")
    print("="*60)
    position_df = exp2.run_position_sweep(
        answerable[:n_pairs*2],
        unanswerable[:n_pairs*2],
        n_pairs=n_pairs,
        best_layer=best_layer
    )

    # 3. Negative Controls
    print("\n" + "="*60)
    print("STEP 3: NEGATIVE CONTROLS")
    print("="*60)
    control_df = exp2.run_negative_controls(
        answerable[:n_pairs*2],
        unanswerable[:n_pairs*2],
        n_pairs=n_pairs,
        best_layer=best_layer
    )

    # 4. Mixing Sweep (optional, only if not quick test)
    mixing_df = None
    if not quick_test:
        print("\n" + "="*60)
        print("STEP 4: MIXING/INTERPOLATION SWEEP")
        print("="*60)
        mixing_df = exp2.run_mixing_sweep(
            answerable[:n_pairs*2],
            unanswerable[:n_pairs*2],
            n_pairs=n_pairs,
            best_window=best_window,
            alphas=alphas
        )

    # ===== ANALYZE =====
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS")
    print("="*60)

    summary = exp2.analyze(
        window_df=window_df,
        position_df=position_df,
        control_df=control_df,
        mixing_df=mixing_df
    )

    # Save summary
    summary_path = config.results_dir / "exp2_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n✓ Experiment 2 complete!")
    print(f"\n📊 Output files:")
    print(f"  Window sweep:         {config.results_dir / 'exp2_window_sweep.csv'}")
    print(f"  Position sweep:       {config.results_dir / 'exp2_position_sweep.csv'}")
    print(f"  Negative controls:    {config.results_dir / 'exp2_negative_controls.csv'}")
    if mixing_df is not None:
        print(f"  Mixing sweep:         {config.results_dir / 'exp2_mixing_sweep.csv'}")
    print(f"  Summary:              {summary_path}")
    print(f"\n📈 Figures:")
    print(f"  Comprehensive:        {config.results_dir / 'exp2_comprehensive_analysis.png'}")
    print(f"  Summary table:        {config.results_dir / 'exp2_summary_table.csv'}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Experiment 2: Causal Localization')
    parser.add_argument('--n_pairs', type=int, default=10,
                       help='Number of example pairs to test')
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test mode with minimal experiments')

    args = parser.parse_args()

    main(n_pairs=args.n_pairs, quick_test=args.quick_test)
