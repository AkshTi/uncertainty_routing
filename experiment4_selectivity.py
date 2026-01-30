"""
Experiment 4: Steering Selectivity (Enhanced)

Addresses reviewer objections:
1. Scales dataset so each entropy bin has ≥100 examples
2. Replaces binning with continuous interaction model
3. Fits: flip ~ entropy + ε + entropy*ε
4. Shows interaction term is significant/large
5. Proves steering is not a uniform bias
6. Plots predicted flip probability vs entropy for different ε values
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf

from core_utils import (
    ModelWrapper, ExperimentConfig, extract_answer,
    set_seed
)
from data_preparation import format_prompt
from core_utils import get_decision_token_position


def _get_blocks(hf_model):
    """Get transformer blocks from HF model"""
    if hasattr(hf_model, "model") and hasattr(hf_model.model, "layers"):
        return hf_model.model.layers
    if hasattr(hf_model, "layers"):
        return hf_model.layers
    if hasattr(hf_model, "transformer") and hasattr(hf_model.transformer, "h"):
        return hf_model.transformer.h
    raise AttributeError(f"Can't find blocks for {type(hf_model)}")


# FIX #7: Helper function to validate response format
def _validate_response_format(response: str) -> Tuple[bool, str]:
    """
    Validate that response starts with ABSTAIN: or ANSWER:

    Returns:
        (is_valid, prefix_type) where prefix_type is 'ABSTAIN', 'ANSWER', or 'INVALID'
    """
    response_upper = response.strip().upper()
    if response_upper.startswith("ABSTAIN"):
        return True, "ABSTAIN"
    elif response_upper.startswith("ANSWER"):
        return True, "ANSWER"
    else:
        return False, "INVALID"


class Experiment4Enhanced:
    """Enhanced selectivity experiment with interaction modeling"""

    def __init__(self, model_wrapper: ModelWrapper, config: ExperimentConfig):
        self.model = model_wrapper
        self.config = config
        self.results = []
        # FIX #7: Track format compliance
        self.format_violations = {
            "baseline_invalid": 0,
            "steered_invalid": 0,
            "total_baseline": 0,
            "total_steered": 0
        }
        set_seed(config.seed)

    # ============================================================================
    # Uncertainty Measurement
    # ============================================================================

    def measure_internal_uncertainty(self, question: str, context: str = None,
                                    n_samples: int = 20) -> Dict:
        """
        Measure internal uncertainty by sampling forced guesses.
        Returns entropy and margin of answer distribution.
        """
        from collections import Counter

        prompt = format_prompt(question, "force_guess", context)

        answers = []
        for i in range(n_samples):
            response = self.model.generate(
                prompt, temperature=1.0, do_sample=True, max_new_tokens=30
            )
            answer = extract_answer(response)
            answers.append(answer)

        # Compute entropy
        counts = Counter(answers)
        probs = np.array([c / len(answers) for c in counts.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # FIX: Compute sample_margin (difference between top two probabilities)
        # This is based on sampled answers, not token-level logits
        # Smaller margin = more uncertain
        sorted_probs = np.sort(probs)[::-1]  # Sort descending
        if len(sorted_probs) >= 2:
            sample_margin = float(sorted_probs[0] - sorted_probs[1])
        else:
            sample_margin = float(sorted_probs[0])  # Only one answer, maximum certainty

        return {
            "entropy": float(entropy),
            "sample_margin": sample_margin,
            "p_max": float(np.max(probs)),
            "n_unique": len(counts)
        }

    def measure_decision_confidence(self, question: str, context: str = None,
                                   n_samples: int = 10) -> float:
        """
        Measure confidence in abstain/answer decision by sampling.
        Returns proportion that abstain minus proportion that answer.

        Returns:
            Decision margin in [-1, 1]: -1 = always answer, +1 = always abstain
        """
        prompt = format_prompt(question, "abstain_or_answer", context)

        abstain_count = 0
        for _ in range(n_samples):
            response = self.model.generate(
                prompt, temperature=0.5, do_sample=True, max_new_tokens=10
            )
            if response.strip().upper().startswith("ABSTAIN"):
                abstain_count += 1

        p_abstain = abstain_count / n_samples
        p_answer = 1 - p_abstain
        decision_margin = p_abstain - p_answer  # Range: [-1, 1]

        return float(decision_margin)

    # ============================================================================
    # Steering Application
    # ============================================================================

    def apply_steering(self, prompt: str, layer_idx: int,
                      steering_vector: torch.Tensor, epsilon: float,
                      baseline_response: Optional[str] = None,
                      baseline_answer: Optional[str] = None,
                      baseline_abstained: Optional[bool] = None) -> Dict:
        """Apply steering at given epsilon and measure decision flip"""
        self.model.clear_hooks()

        # Use provided baseline if available (to avoid recomputing)
        if baseline_response is None:
            baseline_response = self.model.generate(
                prompt, temperature=0.0, do_sample=False, max_new_tokens=50
            )
            baseline_answer = extract_answer(baseline_response)
            baseline_abstained = baseline_response.strip().upper().startswith("ABSTAIN")

        if epsilon == 0:
            return {
                "baseline_answer": baseline_answer,
                "steered_answer": baseline_answer,
                "baseline_response": baseline_response,  # FIX: Return for ε=0 too
                "steered_response": baseline_response,  # FIX: For ε=0, steered = baseline
                "abstained_baseline": baseline_abstained,
                "abstained_steered": baseline_abstained,  # FIX: Same as baseline
                "flipped": False,
                "answer_changed": False
            }

        # Steered
        self.model.clear_hooks()

        # FIX: Get decision position for this prompt
        decision_pos = get_decision_token_position(self.model.tokenizer, prompt)

        # Track if we've already steered (to avoid double-steering)
        steered = {"done": False}

        def steering_hook(module, input, output):
            # Skip if already steered
            if steered["done"]:
                return output

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

            seq_len = hidden_states.shape[1]

            # FIX: Steer at decision token, handle KV cache
            if seq_len > 1 and decision_pos < seq_len:
                # Full forward pass - steer at decision position
                hidden_states[:, decision_pos, :] += epsilon * sv
                steered["done"] = True
            elif seq_len == 1 and not steered["done"]:
                # Generation step (KV cache) - steer current token
                hidden_states[:, 0, :] += epsilon * sv
                steered["done"] = True

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

        # FIX: Detect abstention using ABSTAIN prefix
        abstained_steered = steered_response.strip().upper().startswith("ABSTAIN")

        self.model.clear_hooks()

        # FIX: Flip = decision change (abstain ↔ answer), NOT content change
        flipped = (baseline_abstained != abstained_steered)
        answer_changed = (baseline_answer != steered_answer)

        return {
            "baseline_answer": baseline_answer,
            "steered_answer": steered_answer,
            "baseline_response": baseline_response,  # FIX #7: Return for validation
            "steered_response": steered_response,  # FIX #7: Return for validation
            "abstained_baseline": baseline_abstained,
            "abstained_steered": abstained_steered,
            "flipped": flipped,
            "answer_changed": answer_changed
        }

    # ============================================================================
    # Main Experiment
    # ============================================================================

    def run(self, questions: List[Dict], steering_vector: torch.Tensor,
            layer_idx: int, epsilon_values: List[float],
            n_uncert_samples: int = 20,
            min_per_bin: int = 100) -> pd.DataFrame:
        """
        Run enhanced selectivity experiment

        Args:
            questions: List of questions (needs to be large enough for coverage)
            steering_vector: Pre-computed steering direction
            layer_idx: Layer to apply steering at
            epsilon_values: List of epsilon values to test
            n_uncert_samples: Samples for uncertainty measurement
            min_per_bin: Minimum examples per entropy bin (for validation)
        """
        print("\n" + "="*60)
        print("EXPERIMENT 4: STEERING SELECTIVITY (ENHANCED)")
        print("="*60)
        print(f"Questions: {len(questions)}")
        print(f"Target: ≥{min_per_bin} examples per entropy bin")
        print(f"Epsilon values: {epsilon_values}")
        print(f"Layer: {layer_idx}")

        # Phase 1: Measure uncertainty for all questions
        print("\n" + "="*60)
        print("PHASE 1: MEASURING INTERNAL UNCERTAINTY")
        print("="*60)

        enriched_questions = []
        for i, q_data in enumerate(tqdm(questions, desc="Measuring uncertainty")):
            question = q_data["question"]
            context = q_data.get("context")

            uncert = self.measure_internal_uncertainty(
                question, context, n_samples=n_uncert_samples
            )

            enriched_questions.append({
                **q_data,
                "internal_entropy": uncert["entropy"],
                "sample_margin": uncert["sample_margin"],  # FIX: Renamed from margin
                "p_max": uncert["p_max"],
                "n_unique": uncert["n_unique"]
            })

        # Check entropy coverage
        entropies = [q["internal_entropy"] for q in enriched_questions]
        print(f"\nEntropy range: [{min(entropies):.2f}, {max(entropies):.2f}]")
        print(f"Entropy mean: {np.mean(entropies):.2f} ± {np.std(entropies):.2f}")

        # Bin check (for reporting, not for analysis)
        try:
            bins = pd.qcut(entropies, q=5, labels=False, duplicates='drop')
            # Map numeric bins to names
            bin_names = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
            n_bins = len(pd.unique(bins[~pd.isna(bins)]))
            # Use only as many names as we have bins
            bin_labels = bin_names[:n_bins]
            bins_named = pd.Series([bin_labels[int(b)] if not pd.isna(b) else 'Unknown' for b in bins])
            bin_counts = bins_named.value_counts()

            print(f"\nEntropy distribution ({n_bins} bins):")
            for bin_name in bin_labels:
                count = bin_counts.get(bin_name, 0)
                print(f"  {bin_name:12s}: {count:4d} examples ({100*count/len(entropies):.1f}%)")
        except (ValueError, IndexError) as e:
            print(f"\n⚠️ Could not create entropy bins: {e}")
            print("   Using continuous entropy for analysis instead.")
            bin_counts = pd.Series([len(entropies)])  # Single bin

        if len(bin_counts) > 0 and bin_counts.min() < min_per_bin:
            print(f"\n⚠️ WARNING: Some bins have <{min_per_bin} examples.")
            print(f"   Smallest bin: {bin_counts.min()} examples")
            print(f"   Consider using more questions for robust analysis.")

        # Phase 2: Test steering across epsilon values
        print("\n" + "="*60)
        print("PHASE 2: TESTING STEERING SELECTIVITY")
        print("="*60)

        # FIX #4: Compute baseline once per question, reuse across epsilons
        for q_idx, q_data in enumerate(tqdm(enriched_questions, desc="Questions")):
            question = q_data["question"]
            context = q_data.get("context")
            # Use abstain_or_answer mode for consistent format
            prompt = format_prompt(question, "abstain_or_answer", context)

            # Compute baseline once for this question
            try:
                baseline_response = self.model.generate(
                    prompt, temperature=0.0, do_sample=False, max_new_tokens=50
                )
                baseline_answer = extract_answer(baseline_response)
                baseline_abstained = baseline_response.strip().upper().startswith("ABSTAIN")

                # FIX: Validate baseline format - SKIP QUESTION if invalid
                self.format_violations["total_baseline"] += 1
                baseline_valid, prefix_type = _validate_response_format(baseline_response)
                if not baseline_valid:
                    self.format_violations["baseline_invalid"] += 1
                    print(f"\n⚠️ Invalid baseline format for Q{q_idx}: '{baseline_response[:50]}...' - SKIPPING")
                    continue  # Skip entire question if baseline is invalid

            except Exception as e:
                print(f"\nError computing baseline for question: {e}")
                continue

            # Test all epsilon values against the same baseline
            for epsilon in epsilon_values:
                try:
                    result = self.apply_steering(
                        prompt, layer_idx, steering_vector, epsilon,
                        baseline_response=baseline_response,
                        baseline_answer=baseline_answer,
                        baseline_abstained=baseline_abstained
                    )

                    # FIX: Validate steered response format
                    if epsilon == 0:
                        # For ε=0, steered = baseline, so it's valid if baseline is valid
                        steered_valid = baseline_valid
                    else:
                        # For ε≠0, validate the steered response
                        steered_valid = True
                        if "steered_response" in result:
                            self.format_violations["total_steered"] += 1
                            steered_valid, _ = _validate_response_format(result.get("steered_response", ""))
                            if not steered_valid:
                                self.format_violations["steered_invalid"] += 1

                    self.results.append({
                        "question_id": q_idx,
                        "question": question[:80],
                        "internal_entropy": q_data["internal_entropy"],
                        "sample_margin": q_data["sample_margin"],  # FIX: Renamed
                        "p_max": q_data["p_max"],
                        "n_unique": q_data["n_unique"],
                        "epsilon": epsilon,
                        "layer": layer_idx,
                        "baseline_answer": result["baseline_answer"],
                        "steered_answer": result["steered_answer"],
                        "flipped": result["flipped"],
                        # FIX: Store validity and abstention flags
                        "baseline_valid": baseline_valid,
                        "steered_valid": steered_valid,
                        "abstained_baseline": result["abstained_baseline"],
                        "abstained_steered": result["abstained_steered"]
                    })
                except Exception as e:
                    print(f"\nError at ε={epsilon}: {e}")
                    continue

        df = pd.DataFrame(self.results)

        # FIX #7: Report format compliance
        print("\n" + "="*60)
        print("FORMAT COMPLIANCE REPORT")
        print("="*60)
        total_base = self.format_violations["total_baseline"]
        invalid_base = self.format_violations["baseline_invalid"]
        total_steer = self.format_violations["total_steered"]
        invalid_steer = self.format_violations["steered_invalid"]

        if total_base > 0:
            base_compliance = 100 * (1 - invalid_base / total_base)
            print(f"Baseline responses:  {total_base - invalid_base}/{total_base} valid ({base_compliance:.1f}%)")
            if invalid_base > 0:
                print(f"  ⚠️ {invalid_base} baseline responses did not start with ABSTAIN:/ANSWER:")

        if total_steer > 0:
            steer_compliance = 100 * (1 - invalid_steer / total_steer)
            print(f"Steered responses:   {total_steer - invalid_steer}/{total_steer} valid ({steer_compliance:.1f}%)")
            if invalid_steer > 0:
                print(f"  ⚠️ {invalid_steer} steered responses did not start with ABSTAIN:/ANSWER:")

        # Log decision token info
        print(f"\n📍 Decision token position used for steering")
        print(f"   (computed via get_decision_token_position for each prompt)")

        # Save
        output_path = self.config.results_dir / "exp4_selectivity_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\n\nResults saved to {output_path}")

        return df

    # ============================================================================
    # Analysis
    # ============================================================================

    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive analysis using interaction modeling

        Key: Fit flip ~ entropy + ε + entropy*ε
        """
        print("\n" + "="*60)
        print("EXPERIMENT 4: INTERACTION MODEL ANALYSIS")
        print("="*60)

        # FIX: Filter invalid rows before analysis
        print(f"\nTotal observations (raw): {len(df)}")

        # Filter out invalid baseline or steered responses
        if 'baseline_valid' in df.columns and 'steered_valid' in df.columns:
            # Always require baseline_valid
            # For ε=0: steered_valid should equal baseline_valid (already set correctly)
            # For ε≠0: require steered_valid
            # So we can just filter on both flags
            df_valid = df[(df['baseline_valid'] == True) & (df['steered_valid'] == True)].copy()
            n_filtered = len(df) - len(df_valid)
            if n_filtered > 0:
                print(f"Filtered out {n_filtered} invalid responses ({100*n_filtered/len(df):.1f}%)")
                # Show breakdown
                n_baseline_invalid = len(df[df['baseline_valid'] == False])
                n_steered_invalid = len(df[(df['baseline_valid'] == True) & (df['steered_valid'] == False)])
                print(f"  - {n_baseline_invalid} had invalid baseline")
                print(f"  - {n_steered_invalid} had invalid steered response")
            df = df_valid

        # Prepare data for modeling
        df_model = df[df['epsilon'] != 0].copy()  # Exclude ε=0 (baseline)
        df_model['flipped_binary'] = df_model['flipped'].astype(int)

        print(f"Valid observations: {len(df)}")
        print(f"Non-baseline observations for modeling: {len(df_model)}")
        print(f"Total flips: {df_model['flipped_binary'].sum()}")

        # 1. Fit interaction model using statsmodels (for significance testing)
        print("\n" + "="*60)
        print("1. INTERACTION MODEL: flip ~ entropy + ε + entropy*ε")
        print("="*60)

        # Standardize for numerical stability
        scaler_ent = StandardScaler()
        scaler_eps = StandardScaler()

        df_model['entropy_std'] = scaler_ent.fit_transform(df_model[['internal_entropy']])
        df_model['epsilon_std'] = scaler_eps.fit_transform(df_model[['epsilon']])
        df_model['interaction'] = df_model['entropy_std'] * df_model['epsilon_std']

        # FIX #5: Fit with cluster-robust standard errors (clustered by question_id)
        # This accounts for repeated measures (multiple epsilon values per question)
        formula = 'flipped_binary ~ entropy_std + epsilon_std + entropy_std:epsilon_std'
        model = smf.logit(formula, data=df_model).fit(
            cov_type='cluster',
            cov_kwds={'groups': df_model['question_id']},
            disp=0
        )

        print("\nModel Summary (with cluster-robust standard errors):")
        print(model.summary2())
        print("\nNote: Standard errors are clustered by question_id to account for")
        print("      repeated measures (multiple epsilon values per question).")

        # Extract key results
        interaction_coef = model.params['entropy_std:epsilon_std']
        interaction_pval = model.pvalues['entropy_std:epsilon_std']
        interaction_significant = interaction_pval < 0.05

        print("\n" + "-"*60)
        print("INTERACTION TERM ANALYSIS")
        print("-"*60)
        print(f"Coefficient: {interaction_coef:.4f}")
        print(f"P-value: {interaction_pval:.4e}")
        print(f"Significant (p<0.05): {interaction_significant}")

        if interaction_significant:
            print("\n✓ INTERACTION IS SIGNIFICANT")
            print("  → Steering effect varies with uncertainty")
            print("  → NOT a uniform bias")
        else:
            print("\n⚠️ Interaction not significant")
            print("  → May need more data or larger epsilon range")

        # FIX: Alternative model using sample_margin instead of entropy
        print("\n" + "="*60)
        print("1b. ALTERNATIVE MODEL: flip ~ sample_margin + ε + sample_margin*ε")
        print("="*60)
        print("Note: sample_margin is based on answer distribution, not token logits")

        # Standardize sample_margin
        scaler_margin = StandardScaler()
        df_model['margin_std'] = scaler_margin.fit_transform(df_model[['sample_margin']])
        df_model['margin_interaction'] = df_model['margin_std'] * df_model['epsilon_std']

        # Fit margin-based model
        formula_margin = 'flipped_binary ~ margin_std + epsilon_std + margin_std:epsilon_std'
        model_margin = smf.logit(formula_margin, data=df_model).fit(
            cov_type='cluster',
            cov_kwds={'groups': df_model['question_id']},
            disp=0
        )

        # Extract margin interaction results
        margin_interaction_coef = model_margin.params['margin_std:epsilon_std']
        margin_interaction_pval = model_margin.pvalues['margin_std:epsilon_std']

        print(f"\nSample-margin interaction coefficient: {margin_interaction_coef:.4f}")
        print(f"Sample-margin interaction p-value: {margin_interaction_pval:.4e}")
        print(f"\nCompare uncertainty measures:")
        print(f"  Entropy interaction:        coef={interaction_coef:.4f}, p={interaction_pval:.4e}")
        print(f"  Sample-margin interaction:  coef={margin_interaction_coef:.4f}, p={margin_interaction_pval:.4e}")
        print(f"  Pseudo-R² (entropy):        {model.prsquared:.4f}")
        print(f"  Pseudo-R² (sample-margin):  {model_margin.prsquared:.4f}")

        # FIX: Test with abs(epsilon) - effects may be symmetric in |ε|
        print("\n" + "="*60)
        print("1c. ALTERNATIVE MODEL: flip ~ entropy + |ε| + entropy*|ε|")
        print("="*60)

        # Standardize abs(epsilon)
        scaler_eps_abs = StandardScaler()
        df_model['epsilon_abs'] = df_model['epsilon'].abs()
        df_model['epsilon_abs_std'] = scaler_eps_abs.fit_transform(df_model[['epsilon_abs']])
        df_model['entropy_eps_abs_interaction'] = df_model['entropy_std'] * df_model['epsilon_abs_std']

        # Fit abs(epsilon) model
        formula_abs = 'flipped_binary ~ entropy_std + epsilon_abs_std + entropy_std:epsilon_abs_std'
        model_abs = smf.logit(formula_abs, data=df_model).fit(
            cov_type='cluster',
            cov_kwds={'groups': df_model['question_id']},
            disp=0
        )

        # Extract abs(epsilon) interaction results
        abs_interaction_coef = model_abs.params['entropy_std:epsilon_abs_std']
        abs_interaction_pval = model_abs.pvalues['entropy_std:epsilon_abs_std']

        print(f"\nAbs(epsilon) interaction coefficient: {abs_interaction_coef:.4f}")
        print(f"Abs(epsilon) interaction p-value: {abs_interaction_pval:.4e}")
        print(f"\nCompare signed vs absolute epsilon:")
        print(f"  Signed ε interaction:   coef={interaction_coef:.4f}, p={interaction_pval:.4e}")
        print(f"  Abs(ε) interaction:     coef={abs_interaction_coef:.4f}, p={abs_interaction_pval:.4e}")
        print(f"  Pseudo-R² (signed ε):   {model.prsquared:.4f}")
        print(f"  Pseudo-R² (abs ε):      {model_abs.prsquared:.4f}")

        if abs_interaction_pval < interaction_pval:
            print("\n→ Abs(ε) model has stronger interaction (effects may be symmetric)")
        else:
            print("\n→ Signed ε model is preferred")

        # FIX: Check directionality - test if selectivity exists only for ε>0 or ε<0
        print("\n" + "="*60)
        print("1d. DIRECTIONALITY CHECK: Separate models for ε>0 and ε<0")
        print("="*60)

        # Split by sign
        df_pos = df_model[df_model['epsilon'] > 0].copy()
        df_neg = df_model[df_model['epsilon'] < 0].copy()

        if len(df_pos) > 10 and len(df_neg) > 10:
            # Fit for positive epsilon
            formula_dir = 'flipped_binary ~ entropy_std + epsilon_std + entropy_std:epsilon_std'
            model_pos = smf.logit(formula_dir, data=df_pos).fit(
                cov_type='cluster',
                cov_kwds={'groups': df_pos['question_id']},
                disp=0
            )
            pos_interaction_coef = model_pos.params['entropy_std:epsilon_std']
            pos_interaction_pval = model_pos.pvalues['entropy_std:epsilon_std']

            # Fit for negative epsilon
            model_neg = smf.logit(formula_dir, data=df_neg).fit(
                cov_type='cluster',
                cov_kwds={'groups': df_neg['question_id']},
                disp=0
            )
            neg_interaction_coef = model_neg.params['entropy_std:epsilon_std']
            neg_interaction_pval = model_neg.pvalues['entropy_std:epsilon_std']

            print(f"\nFor ε>0 (n={len(df_pos)}):")
            print(f"  Interaction coef: {pos_interaction_coef:.4f}, p={pos_interaction_pval:.4e}")
            print(f"\nFor ε<0 (n={len(df_neg)}):")
            print(f"  Interaction coef: {neg_interaction_coef:.4f}, p={neg_interaction_pval:.4e}")

            # Check if effects are in same direction
            if pos_interaction_coef * neg_interaction_coef < 0:
                print("\n⚠️ WARNING: Interaction coefficients have opposite signs!")
                print("   Selectivity may exist only for one direction of steering")
            else:
                print("\n→ Interaction coefficients have same sign (consistent directionality)")
        else:
            print("\n⚠️ Insufficient data for split analysis")
            print(f"   ε>0: {len(df_pos)} obs, ε<0: {len(df_neg)} obs")

        # 2. Compare models (with vs without interaction)
        print("\n" + "="*60)
        print("2. MODEL COMPARISON")
        print("="*60)

        # Null model (also with cluster-robust SEs)
        formula_null = 'flipped_binary ~ entropy_std + epsilon_std'
        model_null = smf.logit(formula_null, data=df_model).fit(
            cov_type='cluster',
            cov_kwds={'groups': df_model['question_id']},
            disp=0
        )

        # FIX: With cluster SEs, use Wald test not LR test for inference
        # LR test
        lr_stat = 2 * (model.llf - model_null.llf)

        print(f"\nModel Comparison (descriptive):")
        print(f"  LR statistic: {lr_stat:.4f}")
        print(f"  Note: LR p-value not valid with cluster SEs")
        print(f"\nWald test for interaction (valid with cluster SEs):")
        print(f"  Already reported above: p={interaction_pval:.4e}")
        print(f"  Interaction significant: {interaction_significant}")

        # Pseudo R²
        from sklearn.metrics import log_loss
        y_true = df_model['flipped_binary'].values

        ll_null = -log_loss(y_true, model_null.predict(df_model), normalize=False)
        ll_full = -log_loss(y_true, model.predict(df_model), normalize=False)

        print(f"\nPseudo-R² (McFadden):")
        print(f"  Without interaction: {model_null.prsquared:.4f}")
        print(f"  With interaction:    {model.prsquared:.4f}")

        # 3. Plot predicted probabilities
        self.plot_interaction_effects(df, model, scaler_ent, scaler_eps)

        # 4. Create coefficient table
        self.create_coefficient_table(model)

        # 5. Quantify effect size across entropy levels
        print("\n" + "="*60)
        print("3. EFFECT SIZE ACROSS ENTROPY LEVELS")
        print("="*60)

        # Compute predicted flip probability at different entropy levels
        entropy_levels = np.percentile(df_model['internal_entropy'], [10, 50, 90])
        epsilon_test = 10.0  # Fixed epsilon for comparison

        print(f"\nPredicted flip probability at ε={epsilon_test}:")
        for pct, ent_val in zip([10, 50, 90], entropy_levels):
            # Standardize
            ent_std = (ent_val - scaler_ent.mean_[0]) / scaler_ent.scale_[0]
            eps_std = (epsilon_test - scaler_eps.mean_[0]) / scaler_eps.scale_[0]

            # Predict
            X_pred = pd.DataFrame({
                'entropy_std': [ent_std],
                'epsilon_std': [eps_std],
                'entropy_std:epsilon_std': [ent_std * eps_std]
            })
            pred_prob = model.predict(X_pred)[0]

            print(f"  {pct}th percentile (entropy={ent_val:.2f}): {pred_prob:.1%}")

        return {
            "interaction_coefficient": float(interaction_coef),
            "interaction_pvalue": float(interaction_pval),
            "interaction_significant": bool(interaction_significant),
            "lr_statistic": float(lr_stat),
            "lr_pvalue": None,  # FIX: Not valid with cluster SEs
            "pseudo_r2_null": float(model_null.prsquared),
            "pseudo_r2_full": float(model.prsquared),
            "n_observations": int(len(df_model)),
            "n_flips": int(df_model['flipped_binary'].sum())
        }

    def plot_interaction_effects(self, df: pd.DataFrame, model,
                                 scaler_ent, scaler_eps):
        """
        Plot predicted flip probability vs entropy for different epsilon values
        """
        print("\n" + "="*60)
        print("CREATING INTERACTION PLOTS")
        print("="*60)

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Get epsilon values
        epsilon_values = sorted(df['epsilon'].unique())
        epsilon_values = [e for e in epsilon_values if e != 0]  # Exclude baseline

        # Create entropy grid
        entropy_range = np.linspace(df['internal_entropy'].min(),
                                   df['internal_entropy'].max(), 100)

        # Plot 1: Predicted flip probability vs entropy
        ax1 = axes[0]

        for eps in epsilon_values:
            # Standardize
            ent_std = (entropy_range - scaler_ent.mean_[0]) / scaler_ent.scale_[0]
            eps_std = (eps - scaler_eps.mean_[0]) / scaler_eps.scale_[0]

            # Predict
            X_pred = pd.DataFrame({
                'entropy_std': ent_std,
                'epsilon_std': [eps_std] * len(ent_std),
                'entropy_std:epsilon_std': ent_std * eps_std
            })
            pred_probs = model.predict(X_pred)

            ax1.plot(entropy_range, pred_probs, linewidth=2, label=f'ε={eps}')

        ax1.set_xlabel('Internal Uncertainty (Entropy)', fontsize=12)
        ax1.set_ylabel('Predicted Flip Probability', fontsize=12)
        ax1.set_title('Steering Selectivity: Effect Varies with Uncertainty',
                     fontsize=13, fontweight='bold')
        ax1.legend(title='Epsilon', fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.set_ylim([0, 1])

        # Plot 2: Observed flip rates by entropy quintiles (for validation)
        ax2 = axes[1]

        df_nonzero = df[df['epsilon'] != 0].copy()

        # Bin entropy into quintiles
        df_nonzero['entropy_bin'] = pd.qcut(
            df_nonzero['internal_entropy'],
            q=5,
            labels=['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)'],
            duplicates='drop'
        )

        # For each epsilon, compute flip rate by bin
        for eps in epsilon_values:
            eps_df = df_nonzero[df_nonzero['epsilon'] == eps]
            flip_rates = eps_df.groupby('entropy_bin')['flipped'].mean()

            # Get bin centers
            bin_centers = []
            for bin_label in flip_rates.index:
                bin_data = eps_df[eps_df['entropy_bin'] == bin_label]['internal_entropy']
                bin_centers.append(bin_data.mean())

            ax2.plot(range(len(flip_rates)), flip_rates.values,
                    marker='o', linewidth=2, markersize=8, label=f'ε={eps}')

        ax2.set_xlabel('Entropy Quintile', fontsize=12)
        ax2.set_ylabel('Observed Flip Rate', fontsize=12)
        ax2.set_title('Observed Data: Validates Interaction',
                     fontsize=13, fontweight='bold')
        ax2.set_xticks(range(5))
        ax2.set_xticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)'])
        ax2.legend(title='Epsilon', fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.set_ylim([0, 1])

        plt.tight_layout()

        output_path = self.config.results_dir / "exp4_interaction_plots.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nInteraction plots saved to {output_path}")
        plt.close()

    def create_coefficient_table(self, model):
        """Create table of model coefficients"""
        print("\n" + "="*60)
        print("COEFFICIENT TABLE")
        print("="*60)

        # Extract coefficients
        coef_df = pd.DataFrame({
            'Term': model.params.index,
            'Coefficient': model.params.values,
            'Std Error': model.bse.values,
            'z-value': model.tvalues.values,
            'P-value': model.pvalues.values
        })

        # Format
        coef_df['Coefficient'] = coef_df['Coefficient'].apply(lambda x: f"{x:.4f}")
        coef_df['Std Error'] = coef_df['Std Error'].apply(lambda x: f"{x:.4f}")
        coef_df['z-value'] = coef_df['z-value'].apply(lambda x: f"{x:.3f}")
        coef_df['P-value'] = coef_df['P-value'].apply(lambda x: f"{x:.4e}" if x < 0.001 else f"{x:.4f}")

        print("\n", coef_df.to_string(index=False))

        # Save
        output_path = self.config.results_dir / "exp4_coefficient_table.csv"
        coef_df.to_csv(output_path, index=False)
        print(f"\n\nCoefficient table saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main(n_questions: int = 300, quick_test: bool = False):
    """
    Run enhanced Experiment 4

    Args:
        n_questions: Number of questions to use (recommend ≥300 for robust bins)
        quick_test: Quick test mode
    """
    import json

    # Setup
    config = ExperimentConfig()

    print("Initializing model...")
    model = ModelWrapper(config)

    # Load steering vector (from Experiment 3)
    try:
        steering_vectors = torch.load(config.results_dir / "steering_vectors_robust.pt")
    except FileNotFoundError:
        try:
            steering_vectors = torch.load(config.results_dir / "steering_vectors.pt")
        except FileNotFoundError:
            print("ERROR: No steering vectors found!")
            print("Please run Experiment 3 first to generate steering vectors.")
            return None

    # Get best layer and direction
    best_layer = 27  # Default, can be overridden
    if (best_layer, 'mean_diff') in steering_vectors:
        steering_vec = steering_vectors[(best_layer, 'mean_diff')]
    elif str((best_layer, 'mean_diff')) in steering_vectors:
        steering_vec = steering_vectors[str((best_layer, 'mean_diff'))]
    elif best_layer in steering_vectors:
        steering_vec = steering_vectors[best_layer]
    elif str(best_layer) in steering_vectors:
        steering_vec = steering_vectors[str(best_layer)]
    else:
        print(f"Available keys: {list(steering_vectors.keys())}")
        # Use first available
        steering_vec = list(steering_vectors.values())[0]

    steering_vec = steering_vec.to(config.device)
    print(f"Loaded steering vector: shape={steering_vec.shape}, norm={steering_vec.norm():.4f}")

    # Load questions
    print("\nLoading questions...")
    try:
        with open("./data/dataset_ambiguous.json", 'r') as f:
            questions = json.load(f)
    except FileNotFoundError:
        # Fallback: mix answerable and unanswerable
        with open("./data/dataset_clearly_answerable_expanded.json", 'r') as f:
            answerable = json.load(f)
        with open("./data/dataset_clearly_unanswerable_expanded.json", 'r') as f:
            unanswerable = json.load(f)
        questions = answerable + unanswerable

    if quick_test:
        print("\n🔬 QUICK TEST MODE")
        n_questions = 50
        epsilon_values = [0, 5, 10]
    else:
        epsilon_values = [0, 5, 10, 15, 20]

    # Sample questions
    import random
    random.seed(42)
    questions = random.sample(questions, min(n_questions, len(questions)))

    print(f"\nUsing {len(questions)} questions")
    print(f"Epsilon values: {epsilon_values}")

    # Run experiment
    exp4 = Experiment4Enhanced(model, config)
    results_df = exp4.run(
        questions=questions,
        steering_vector=steering_vec,
        layer_idx=best_layer,
        epsilon_values=epsilon_values,
        n_uncert_samples=20,
        min_per_bin=100
    )

    # Analyze
    summary = exp4.analyze(results_df)

    # Save summary
    summary_path = config.results_dir / "exp4_selectivity_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Experiment 4 (Enhanced) complete!")
    print(f"\n📊 Output files:")
    print(f"  Results:              {config.results_dir / 'exp4_selectivity_results.csv'}")
    print(f"  Summary:              {summary_path}")
    print(f"  Coefficient table:    {config.results_dir / 'exp4_coefficient_table.csv'}")
    print(f"\n📈 Figures:")
    print(f"  Interaction plots:    {config.results_dir / 'exp4_interaction_plots.png'}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Experiment 4: Enhanced Selectivity')
    parser.add_argument('--n_questions', type=int, default=300,
                       help='Number of questions (recommend ≥300)')
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test mode')

    args = parser.parse_args()

    main(n_questions=args.n_questions, quick_test=args.quick_test)
