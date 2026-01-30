"""
Experiment 5: Risk-Coverage Tradeoff Analysis

Addresses reviewer objection: "Prompt engineering already does this."

Compares multiple intervention methods on risk-coverage frontier:
1. Activation steering (ε sweep from Exp 3)
2. Prompt-only engineering (4-6 caution variants)
3. Semantic entropy threshold baseline (abstain if entropy > τ)
4. (Optional) Temperature/top-p sweep

Risk = hallucination rate on unanswerables (% answered when should abstain)
Coverage = % of answerables that get answered

Output:
- Risk-coverage frontier plot (all methods)
- Matched-risk comparison table (e.g., coverage at 5% risk)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import json
import torch
from collections import Counter

from core_utils import (
    ModelWrapper, ExperimentConfig, extract_answer,
    set_seed
)
from data_preparation import format_prompt


class Experiment5:
    """Risk-Coverage Tradeoff Experiment"""

    def __init__(self, model_wrapper: ModelWrapper, config: ExperimentConfig):
        self.model = model_wrapper
        self.config = config
        self.results = []
        set_seed(config.seed)
        self.n_layers = self.model.model.config.num_hidden_layers

    @staticmethod
    def parse_abstention(response: str) -> tuple:
        """
        Parse whether model abstained based on abstain_or_answer format

        Returns:
            (abstained: bool, valid: bool) - valid=False if format is invalid
        """
        response_stripped = response.strip().upper()

        # Check for valid format
        if response_stripped.startswith("ABSTAIN"):
            return (True, True)
        elif response_stripped.startswith("ANSWER"):
            return (False, True)
        else:
            # Invalid format - doesn't start with ABSTAIN or ANSWER
            return (False, False)

    # =========================================================================
    # Semantic Entropy Computation (for baseline and optional use)
    # =========================================================================

    def compute_semantic_entropy(self, question: str, context: str = None,
                                  n_samples: int = 15, temperature: float = 1.0) -> float:
        """Compute semantic entropy for a question"""
        prompt = format_prompt(question, "force_guess", context)

        answers = []
        for _ in range(n_samples):
            response = self.model.generate(
                prompt,
                temperature=temperature,
                do_sample=True,
                max_new_tokens=50
            )
            answer = extract_answer(response)
            answers.append(answer)

        # Compute entropy
        answer_counts = Counter(answers)
        total = len(answers)
        probs = np.array([count/total for count in answer_counts.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        return float(entropy)

    # =========================================================================
    # Method 1: Activation Steering
    # =========================================================================

    def run_steering_sweep(self, questions: List[Dict], steering_direction: torch.Tensor,
                           layer_idx: int, epsilon_values: List[float]) -> pd.DataFrame:
        """
        Run activation steering sweep across epsilon values

        Args:
            questions: List of questions (both answerable and unanswerable)
            steering_direction: Pre-computed steering vector
            layer_idx: Layer to apply steering
            epsilon_values: List of epsilon values to sweep

        Returns:
            DataFrame with columns: epsilon, question, true_answerability, abstained, answer
        """
        print("\n" + "="*60)
        print("METHOD 1: ACTIVATION STEERING")
        print("="*60)
        print(f"Sweeping epsilon: {epsilon_values}")

        results = []

        for epsilon in tqdm(epsilon_values, desc="Steering sweep"):
            for q_data in questions:
                question = q_data["question"]
                context = q_data.get("context", None)
                true_answerability = q_data.get("answerability", "unknown")

                # Use abstain_or_answer format
                prompt = format_prompt(question, "abstain_or_answer", context)

                self.model.clear_hooks()

                # Register steering hook - steer ONCE at last prompt token only
                direction_norm = steering_direction / (torch.norm(steering_direction) + 1e-8)
                steered = [False]  # Track if we've already steered

                def steering_hook(module, input, output):
                    if steered[0]:
                        # Already steered, don't modify anything else
                        return output

                    # Extract hidden states and clone to avoid in-place mutation issues
                    if isinstance(output, tuple):
                        hs = output[0].clone()
                        rest = output[1:]
                    else:
                        hs = output.clone()
                        rest = None

                    # Steer the last token of the prompt (once only)
                    hs[:, -1, :] = hs[:, -1, :] + epsilon * direction_norm.to(hs.device).to(hs.dtype)
                    steered[0] = True

                    # Return modified tensor explicitly
                    if rest is not None:
                        return (hs,) + rest
                    return hs

                layer = self.model.model.layers[layer_idx]
                handle = layer.register_forward_hook(steering_hook)

                response = self.model.generate(prompt, temperature=0.0, do_sample=False)

                handle.remove()
                self.model.clear_hooks()

                # Parse abstention using consistent method
                abstained, valid = self.parse_abstention(response)

                # Extract answer
                answer = extract_answer(response)

                results.append({
                    "method": "steering",
                    "parameter": float(epsilon),
                    "question": question,
                    "true_answerability": true_answerability,
                    "abstained": abstained,
                    "valid_format": valid,
                    "answer": answer,
                    "response": response
                })

        df = pd.DataFrame(results)

        # Report format validation stats
        if 'valid_format' in df.columns:
            n_valid = df['valid_format'].sum()
            n_total = len(df)
            n_invalid = n_total - n_valid
            print(f"\nFormat validation:")
            print(f"  Valid:   {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
            if n_invalid > 0:
                print(f"  Invalid: {n_invalid}/{n_total} ({100*n_invalid/n_total:.1f}%)")
                print(f"  → These will be excluded from risk-coverage analysis")

        output_path = self.config.results_dir / "exp5_steering_sweep.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSteering sweep results saved to {output_path}")

        return df

    # =========================================================================
    # Method 2: Prompt Engineering
    # =========================================================================

    def run_prompt_sweep(self, questions: List[Dict]) -> pd.DataFrame:
        """
        Run prompt engineering sweep with varying caution levels

        All variants use abstain_or_answer format with different system instructions

        Args:
            questions: List of questions (both answerable and unanswerable)

        Returns:
            DataFrame with columns: prompt_variant, question, true_answerability, abstained, answer
        """
        print("\n" + "="*60)
        print("METHOD 2: PROMPT ENGINEERING")
        print("="*60)

        # Define 6 prompt variants from "force answer" to "extremely cautious"
        # All use abstain_or_answer base format with prepended instructions
        prompt_variants = {
            "force_answer": "You must ALWAYS answer with 'ANSWER: <your answer>'. Never abstain.",
            "default": None,  # Use default abstain_or_answer prompt
            "slight_caution": "If you're not certain, you may abstain.",
            "moderate_caution": "Be careful. If you don't have enough information, abstain.",
            "high_caution": "Be very cautious. Only answer if confident. Otherwise abstain.",
            "extreme_caution": "Abstain unless absolutely certain. Better to abstain than guess."
        }

        print(f"Testing {len(prompt_variants)} prompt variants")

        results = []

        for variant_name, system_instruction in tqdm(list(prompt_variants.items()), desc="Prompt variants"):
            for q_data in questions:
                question = q_data["question"]
                context = q_data.get("context", None)
                true_answerability = q_data.get("answerability", "unknown")

                # Use abstain_or_answer format
                base_prompt = format_prompt(question, "abstain_or_answer", context)

                # Prepend system instruction if provided
                if system_instruction is not None:
                    prompt = f"{system_instruction}\n\n{base_prompt}"
                else:
                    prompt = base_prompt

                self.model.clear_hooks()
                response = self.model.generate(prompt, temperature=0.0, do_sample=False)

                # Parse abstention using consistent method
                abstained, valid = self.parse_abstention(response)

                # Extract answer
                answer = extract_answer(response)

                results.append({
                    "method": "prompt",
                    "parameter": variant_name,
                    "question": question,
                    "true_answerability": true_answerability,
                    "abstained": abstained,
                    "valid_format": valid,
                    "answer": answer,
                    "response": response
                })

        df = pd.DataFrame(results)

        # Report format validation stats
        if 'valid_format' in df.columns:
            n_valid = df['valid_format'].sum()
            n_total = len(df)
            n_invalid = n_total - n_valid
            print(f"\nFormat validation:")
            print(f"  Valid:   {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
            if n_invalid > 0:
                print(f"  Invalid: {n_invalid}/{n_total} ({100*n_invalid/n_total:.1f}%)")
                print(f"  → These will be excluded from risk-coverage analysis")

        output_path = self.config.results_dir / "exp5_prompt_sweep.csv"
        df.to_csv(output_path, index=False)
        print(f"\nPrompt sweep results saved to {output_path}")

        return df

    # =========================================================================
    # Method 3: Semantic Entropy Threshold
    # =========================================================================

    def run_entropy_threshold_sweep(self, questions: List[Dict],
                                     tau_values: List[float] = None) -> pd.DataFrame:
        """
        Run semantic entropy threshold sweep

        Unlike the old version which hard-abstained, this version always generates
        with abstain_or_answer format, but adds a system instruction based on entropy.

        Args:
            questions: List of questions
            tau_values: Threshold values to sweep (suggest abstention if entropy > τ)

        Returns:
            DataFrame with results
        """
        print("\n" + "="*60)
        print("METHOD 3: SEMANTIC ENTROPY THRESHOLD")
        print("="*60)

        if tau_values is None:
            tau_values = np.linspace(0.1, 2.0, 10).tolist()

        print(f"Sweeping tau: {[f'{t:.2f}' for t in tau_values]}")

        # First, compute entropy for all questions (cache to avoid recomputation)
        print("\nComputing semantic entropy for all questions...")
        entropy_cache = {}
        for q_data in tqdm(questions, desc="Computing entropy"):
            question = q_data["question"]
            context = q_data.get("context", None)
            entropy = self.compute_semantic_entropy(question, context)
            entropy_cache[question] = entropy

        # Now sweep thresholds
        results = []
        for tau in tqdm(tau_values, desc="Threshold sweep"):
            for q_data in questions:
                question = q_data["question"]
                context = q_data.get("context", None)
                true_answerability = q_data.get("answerability", "unknown")

                entropy = entropy_cache[question]

                # Use abstain_or_answer format with entropy-based instruction
                base_prompt = format_prompt(question, "abstain_or_answer", context)

                if entropy > tau:
                    # High entropy: suggest abstention
                    system_instruction = (
                        f"Your uncertainty is high (entropy={entropy:.2f} > threshold={tau:.2f}). "
                        "You should strongly consider abstaining."
                    )
                    prompt = f"{system_instruction}\n\n{base_prompt}"
                else:
                    # Low entropy: encourage answering
                    system_instruction = (
                        f"Your uncertainty is acceptable (entropy={entropy:.2f} \u2264 threshold={tau:.2f}). "
                        "You may answer if you have information."
                    )
                    prompt = f"{system_instruction}\n\n{base_prompt}"

                self.model.clear_hooks()
                response = self.model.generate(prompt, temperature=0.0, do_sample=False)

                # Parse abstention using consistent method
                abstained, valid = self.parse_abstention(response)

                # Extract answer
                answer = extract_answer(response)

                results.append({
                    "method": "entropy_threshold",
                    "parameter": float(tau),
                    "question": question,
                    "true_answerability": true_answerability,
                    "abstained": abstained,
                    "valid_format": valid,
                    "answer": answer,
                    "response": response,
                    "entropy": float(entropy)
                })

        df = pd.DataFrame(results)

        # Report format validation stats
        if 'valid_format' in df.columns:
            n_valid = df['valid_format'].sum()
            n_total = len(df)
            n_invalid = n_total - n_valid
            print(f"\nFormat validation:")
            print(f"  Valid:   {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
            if n_invalid > 0:
                print(f"  Invalid: {n_invalid}/{n_total} ({100*n_invalid/n_total:.1f}%)")
                print(f"  → These will be excluded from risk-coverage analysis")

        output_path = self.config.results_dir / "exp5_entropy_threshold_sweep.csv"
        df.to_csv(output_path, index=False)
        print(f"\nEntropy threshold sweep results saved to {output_path}")

        return df

    # =========================================================================
    # Method 4: Temperature/Top-p Sweep (Optional)
    # =========================================================================

    def run_temperature_sweep(self, questions: List[Dict],
                              temperature_values: List[float] = None) -> pd.DataFrame:
        """
        Run temperature sweep (higher T → more diverse → potentially more abstention)

        Args:
            questions: List of questions
            temperature_values: Temperature values to sweep

        Returns:
            DataFrame with results
        """
        print("\n" + "="*60)
        print("METHOD 4: TEMPERATURE SWEEP (OPTIONAL)")
        print("="*60)

        if temperature_values is None:
            temperature_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5]

        print(f"Sweeping temperature: {temperature_values}")

        results = []

        for temp in tqdm(temperature_values, desc="Temperature sweep"):
            for q_data in questions:
                question = q_data["question"]
                context = q_data.get("context", None)
                true_answerability = q_data.get("answerability", "unknown")

                # Use abstain_or_answer format
                prompt = format_prompt(question, "abstain_or_answer", context)
                self.model.clear_hooks()
                response = self.model.generate(prompt, temperature=temp, do_sample=(temp > 0.0))

                # Parse abstention using consistent method
                abstained, valid = self.parse_abstention(response)

                # Extract answer
                answer = extract_answer(response)

                results.append({
                    "method": "temperature",
                    "parameter": float(temp),
                    "question": question,
                    "true_answerability": true_answerability,
                    "abstained": abstained,
                    "valid_format": valid,
                    "answer": answer,
                    "response": response
                })

        df = pd.DataFrame(results)

        # Report format validation stats
        if 'valid_format' in df.columns:
            n_valid = df['valid_format'].sum()
            n_total = len(df)
            n_invalid = n_total - n_valid
            print(f"\nFormat validation:")
            print(f"  Valid:   {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
            if n_invalid > 0:
                print(f"  Invalid: {n_invalid}/{n_total} ({100*n_invalid/n_total:.1f}%)")
                print(f"  → These will be excluded from risk-coverage analysis")

        output_path = self.config.results_dir / "exp5_temperature_sweep.csv"
        df.to_csv(output_path, index=False)
        print(f"\nTemperature sweep results saved to {output_path}")

        return df

    # =========================================================================
    # Analysis: Compute Risk-Coverage Curves
    # =========================================================================

    def compute_risk_coverage_curve(self, df: pd.DataFrame, method_name: str) -> pd.DataFrame:
        """
        Compute risk-coverage curve from results

        Risk = hallucination rate on unanswerables (% answered when should abstain)
        Coverage = % of answerables that get answered

        Args:
            df: Results dataframe with columns: parameter, true_answerability, abstained, valid_format
            method_name: Name of method for labeling

        Returns:
            DataFrame with columns: method, parameter, risk, coverage
        """
        # Filter out invalid format responses
        if 'valid_format' in df.columns:
            n_total = len(df)
            df = df[df['valid_format'] == True].copy()
            n_valid = len(df)
            n_invalid = n_total - n_valid
            if n_invalid > 0:
                print(f"  ⚠️  {method_name}: Dropped {n_invalid}/{n_total} ({100*n_invalid/n_total:.1f}%) invalid format responses")

        curve_points = []

        for param_value in df['parameter'].unique():
            param_df = df[df['parameter'] == param_value]

            # Split by true answerability
            answerable = param_df[param_df['true_answerability'] == 'answerable']
            unanswerable = param_df[param_df['true_answerability'] == 'unanswerable']

            if len(answerable) == 0 or len(unanswerable) == 0:
                continue

            # Coverage = % of answerables that were answered (not abstained)
            coverage = (1 - answerable['abstained'].mean())

            # Risk = % of unanswerables that were answered (hallucination rate)
            risk = (1 - unanswerable['abstained'].mean())

            curve_points.append({
                'method': method_name,
                'parameter': param_value,
                'risk': float(risk),
                'coverage': float(coverage),
                'n_answerable': len(answerable),
                'n_unanswerable': len(unanswerable)
            })

        return pd.DataFrame(curve_points)

    def analyze(self, steering_df: pd.DataFrame = None,
                prompt_df: pd.DataFrame = None,
                entropy_df: pd.DataFrame = None,
                temperature_df: pd.DataFrame = None,
                matched_risk_levels: List[float] = None) -> Dict:
        """
        Analyze all methods and create risk-coverage frontier

        Args:
            steering_df: Steering sweep results
            prompt_df: Prompt sweep results
            entropy_df: Entropy threshold sweep results
            temperature_df: Temperature sweep results (optional)
            matched_risk_levels: Risk levels for matched comparison (e.g., [0.05, 0.10])

        Returns:
            Analysis summary dictionary
        """
        print("\n" + "="*60)
        print("EXPERIMENT 5: RISK-COVERAGE ANALYSIS")
        print("="*60)

        if matched_risk_levels is None:
            matched_risk_levels = [0.05, 0.10, 0.15]

        # Compute risk-coverage curves for each method
        all_curves = []

        if steering_df is not None and len(steering_df) > 0:
            print("\nComputing steering risk-coverage curve...")
            steering_curve = self.compute_risk_coverage_curve(steering_df, "Activation Steering")
            all_curves.append(steering_curve)

        if prompt_df is not None and len(prompt_df) > 0:
            print("Computing prompt risk-coverage curve...")
            prompt_curve = self.compute_risk_coverage_curve(prompt_df, "Prompt Engineering")
            all_curves.append(prompt_curve)

        if entropy_df is not None and len(entropy_df) > 0:
            print("Computing entropy threshold risk-coverage curve...")
            entropy_curve = self.compute_risk_coverage_curve(entropy_df, "Entropy Threshold")
            all_curves.append(entropy_curve)

        if temperature_df is not None and len(temperature_df) > 0:
            print("Computing temperature risk-coverage curve...")
            temp_curve = self.compute_risk_coverage_curve(temperature_df, "Temperature")
            all_curves.append(temp_curve)

        # Combine all curves
        if len(all_curves) == 0:
            print("ERROR: No valid curves to analyze")
            return {}

        all_curves_df = pd.concat(all_curves, ignore_index=True)

        # Save combined curves
        output_path = self.config.results_dir / "exp5_risk_coverage_curves.csv"
        all_curves_df.to_csv(output_path, index=False)
        print(f"\nRisk-coverage curves saved to {output_path}")

        # Print summary statistics
        print("\n" + "="*60)
        print("RISK-COVERAGE SUMMARY")
        print("="*60)

        for method in all_curves_df['method'].unique():
            method_df = all_curves_df[all_curves_df['method'] == method]
            print(f"\n{method}:")
            print(f"  Risk range: [{method_df['risk'].min():.3f}, {method_df['risk'].max():.3f}]")
            print(f"  Coverage range: [{method_df['coverage'].min():.3f}, {method_df['coverage'].max():.3f}]")

            # Find pareto-optimal point (max coverage for given risk constraint)
            best_point = method_df.loc[method_df['coverage'].idxmax()]
            print(f"  Best point: Risk={best_point['risk']:.3f}, Coverage={best_point['coverage']:.3f}")

        # Matched-risk comparison
        matched_results = self.compute_matched_risk_comparison(all_curves_df, matched_risk_levels)

        # Create visualizations
        self.plot_risk_coverage_frontier(all_curves_df)
        self.create_matched_risk_table(matched_results)

        return {
            "curves": all_curves_df.to_dict(orient='records'),
            "matched_risk_comparison": matched_results
        }

    def compute_matched_risk_comparison(self, curves_df: pd.DataFrame,
                                        risk_levels: List[float]) -> List[Dict]:
        """
        For each risk level, find the maximum coverage each method achieves
        while keeping risk ≤ target

        Args:
            curves_df: Combined risk-coverage curves
            risk_levels: Risk levels to compare at (e.g., [0.05, 0.10])

        Returns:
            List of comparison dictionaries
        """
        print("\n" + "="*60)
        print("MATCHED-RISK COMPARISON")
        print("="*60)

        results = []

        for target_risk in risk_levels:
            print(f"\nAt target risk \u2264 {target_risk:.1%}:")

            comparison = {"target_risk": target_risk}

            for method in curves_df['method'].unique():
                method_df = curves_df[curves_df['method'] == method].copy()

                # Find all points with risk ≤ target
                valid_points = method_df[method_df['risk'] <= target_risk]

                if len(valid_points) == 0:
                    # No valid points - method can't achieve this risk level
                    print(f"  {method:25s}: No valid points (minimum risk = {method_df['risk'].min():.1%})")
                    comparison[f"{method}_coverage"] = float('nan')
                    comparison[f"{method}_actual_risk"] = float('nan')
                else:
                    # Find max coverage among valid points
                    best_idx = valid_points['coverage'].idxmax()
                    best_point = valid_points.loc[best_idx]

                    actual_risk = best_point['risk']
                    coverage = best_point['coverage']

                    print(f"  {method:25s}: Coverage = {coverage:.1%} (actual risk = {actual_risk:.1%})")

                    comparison[f"{method}_coverage"] = float(coverage)
                    comparison[f"{method}_actual_risk"] = float(actual_risk)

            results.append(comparison)

        return results

    def plot_risk_coverage_frontier(self, curves_df: pd.DataFrame):
        """Create risk-coverage frontier plot"""

        print("\n" + "="*60)
        print("CREATING RISK-COVERAGE FRONTIER PLOT")
        print("="*60)

        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 7))

        # Color palette
        colors = {
            "Activation Steering": "#e74c3c",
            "Prompt Engineering": "#3498db",
            "Entropy Threshold": "#2ecc71",
            "Temperature": "#9b59b6"
        }

        markers = {
            "Activation Steering": "o",
            "Prompt Engineering": "s",
            "Entropy Threshold": "^",
            "Temperature": "D"
        }

        # Plot each method
        for method in curves_df['method'].unique():
            method_df = curves_df[curves_df['method'] == method].sort_values('risk')

            ax.plot(method_df['risk'], method_df['coverage'],
                   marker=markers.get(method, 'o'),
                   markersize=8,
                   linewidth=2.5,
                   label=method,
                   color=colors.get(method, 'gray'),
                   alpha=0.8)

        # Add reference line (y=x)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Reference (Risk = Coverage)')

        # Styling
        ax.set_xlabel('Risk (Hallucination Rate on Unanswerables)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Coverage (Fraction of Answerables Answered)', fontsize=13, fontweight='bold')
        ax.set_title('Risk-Coverage Tradeoff: Steering vs. Baselines',
                    fontsize=15, fontweight='bold', pad=15)

        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])

        ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
        ax.grid(alpha=0.3)

        # Add annotation for ideal region
        ax.annotate('Ideal: High Coverage,\nLow Risk',
                   xy=(0.05, 0.95), fontsize=10, color='green',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        plt.tight_layout()

        # Save
        output_path = self.config.results_dir / "exp5_risk_coverage_frontier.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nRisk-coverage frontier saved to {output_path}")
        plt.close()

    def create_matched_risk_table(self, matched_results: List[Dict]):
        """Create table showing coverage at matched risk levels"""

        print("\n" + "="*60)
        print("CREATING MATCHED-RISK TABLE")
        print("="*60)

        # Convert to DataFrame
        table_data = []

        for result in matched_results:
            target_risk = result['target_risk']

            row = {'Target Risk': f"{target_risk:.1%}"}

            # Extract coverage for each method
            methods = [k.replace('_coverage', '') for k in result.keys() if k.endswith('_coverage')]

            for method in methods:
                coverage_key = f"{method}_coverage"
                if coverage_key in result:
                    row[method] = f"{result[coverage_key]:.1%}"

            table_data.append(row)

        table_df = pd.DataFrame(table_data)

        print("\n", table_df.to_string(index=False))

        # Save as CSV
        output_path = self.config.results_dir / "exp5_matched_risk_table.csv"
        table_df.to_csv(output_path, index=False)
        print(f"\n\nMatched-risk table saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main(n_questions: int = 100, quick_test: bool = False,
         run_temperature: bool = False):
    """
    Run Experiment 5: Risk-Coverage Tradeoff

    Args:
        n_questions: Number of questions to test (split evenly between answerable/unanswerable)
        quick_test: If True, run minimal tests for quick validation
        run_temperature: Whether to include temperature sweep (optional)
    """
    import json

    # Setup
    config = ExperimentConfig()

    print("Initializing model...")
    model = ModelWrapper(config)

    # Load data (balanced answerable/unanswerable)
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

    # Add answerability labels
    for q in answerable:
        q['answerability'] = 'answerable'
    for q in unanswerable:
        q['answerability'] = 'unanswerable'

    # Sample balanced dataset
    if quick_test:
        n_per_class = 5
    else:
        n_per_class = n_questions // 2

    import random
    random.seed(42)
    sampled_answerable = random.sample(answerable, min(n_per_class, len(answerable)))
    sampled_unanswerable = random.sample(unanswerable, min(n_per_class, len(unanswerable)))

    questions = sampled_answerable + sampled_unanswerable
    random.shuffle(questions)

    print(f"\nDataset: {len(sampled_answerable)} answerable + {len(sampled_unanswerable)} unanswerable")

    # Initialize experiment
    exp5 = Experiment5(model, config)

    # ===== LOAD OR COMPUTE STEERING DIRECTION =====
    # Try to load from Experiment 3 results
    steering_direction = None
    best_layer = None

    try:
        # Look for saved steering direction from Exp 3
        exp3_results_path = config.results_dir / "exp3_steering_directions.pt"
        if exp3_results_path.exists():
            print("\nLoading steering direction from Experiment 3...")
            saved_data = torch.load(exp3_results_path)
            steering_direction = saved_data['mean_diff_direction']
            best_layer = saved_data.get('best_layer', int(exp5.n_layers * 0.75))
            print(f"Loaded steering direction (layer {best_layer})")
        else:
            print("\nWARNING: No saved steering direction found from Exp 3")
            print("Computing steering direction from scratch...")

            # Compute mean-diff direction quickly
            from experiment3_steering_robust import compute_mean_diff_direction

            best_layer = int(exp5.n_layers * 0.75)
            steering_direction = compute_mean_diff_direction(
                model, sampled_answerable[:20], sampled_unanswerable[:20], best_layer
            )

            # Save for future use
            torch.save({
                'mean_diff_direction': steering_direction,
                'best_layer': best_layer
            }, exp3_results_path)

    except Exception as e:
        print(f"ERROR loading steering direction: {e}")
        print("Skipping steering sweep...")

    # ===== RUN ALL METHODS =====

    # 1. Activation Steering
    steering_df = None
    if steering_direction is not None:
        if quick_test:
            epsilon_values = [-10, -5, 0, 5, 10]
        else:
            epsilon_values = [-20, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20]

        steering_df = exp5.run_steering_sweep(
            questions, steering_direction, best_layer, epsilon_values
        )

    # 2. Prompt Engineering
    prompt_df = exp5.run_prompt_sweep(questions)

    # 3. Semantic Entropy Threshold
    if quick_test:
        tau_values = np.linspace(0.2, 1.5, 5).tolist()
    else:
        tau_values = np.linspace(0.1, 2.0, 15).tolist()

    entropy_df = exp5.run_entropy_threshold_sweep(questions, tau_values)

    # 4. Temperature Sweep (optional)
    temperature_df = None
    if run_temperature:
        if quick_test:
            temp_values = [0.3, 0.7, 1.0]
        else:
            temp_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5]

        temperature_df = exp5.run_temperature_sweep(questions, temp_values)

    # ===== ANALYZE =====
    print("\n" + "="*60)
    print("COMPREHENSIVE RISK-COVERAGE ANALYSIS")
    print("="*60)

    summary = exp5.analyze(
        steering_df=steering_df,
        prompt_df=prompt_df,
        entropy_df=entropy_df,
        temperature_df=temperature_df,
        matched_risk_levels=[0.05, 0.10, 0.15, 0.20]
    )

    # Save summary
    summary_path = config.results_dir / "exp5_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n✓ Experiment 5 complete!")
    print(f"\n📊 Output files:")
    if steering_df is not None:
        print(f"  Steering sweep:       {config.results_dir / 'exp5_steering_sweep.csv'}")
    print(f"  Prompt sweep:         {config.results_dir / 'exp5_prompt_sweep.csv'}")
    print(f"  Entropy threshold:    {config.results_dir / 'exp5_entropy_threshold_sweep.csv'}")
    if temperature_df is not None:
        print(f"  Temperature sweep:    {config.results_dir / 'exp5_temperature_sweep.csv'}")
    print(f"  Risk-coverage curves: {config.results_dir / 'exp5_risk_coverage_curves.csv'}")
    print(f"  Summary:              {summary_path}")
    print(f"\n📈 Figures:")
    print(f"  Frontier plot:        {config.results_dir / 'exp5_risk_coverage_frontier.png'}")
    print(f"  Matched-risk table:   {config.results_dir / 'exp5_matched_risk_table.csv'}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Experiment 5: Risk-Coverage Tradeoff')
    parser.add_argument('--n_questions', type=int, default=100,
                       help='Number of questions (split evenly answerable/unanswerable)')
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test mode with minimal experiments')
    parser.add_argument('--run_temperature', action='store_true',
                       help='Include temperature sweep (optional)')

    args = parser.parse_args()

    main(
        n_questions=args.n_questions,
        quick_test=args.quick_test,
        run_temperature=args.run_temperature
    )
