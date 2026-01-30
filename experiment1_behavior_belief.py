"""
Experiment 1: Behavior-Belief Dissociation
Save as: experiment1_behavior_belief.py

Shows that instruction regime changes abstention behavior
while internal uncertainty remains stable
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import torch

from core_utils import (
    ModelWrapper, ExperimentConfig, extract_answer,
    set_seed, compute_binary_margin_simple
)
from data_preparation import format_prompt


def load_scaled_dataset(n_answerable: int = 200, n_unanswerable: int = 200) -> List[Dict]:
    """
    Load and scale dataset to include multiple domains (math, science, history, geo)
    with specified number of answerable and unanswerable questions.

    Args:
        n_answerable: Number of answerable questions to include (200-500)
        n_unanswerable: Number of unanswerable questions to include (200-500)

    Returns:
        List of question dictionaries with domain labels
    """
    import random

    # Load existing datasets
    with open("./data/dataset_clearly_answerable_expanded.json", 'r') as f:
        answerable_base = json.load(f)

    with open("./data/dataset_clearly_unanswerable_expanded.json", 'r') as f:
        unanswerable_base = json.load(f)

    # Multi-domain question generation
    answerable_multi_domain = []
    unanswerable_multi_domain = []

    # Math domain - Answerable
    math_answerable = [
        {"question": "What is 15 multiplied by 8?", "answer": "120", "domain": "math", "answerability": "answerable"},
        {"question": "What is the square root of 144?", "answer": "12", "domain": "math", "answerability": "answerable"},
        {"question": "What is 25% of 200?", "answer": "50", "domain": "math", "answerability": "answerable"},
        {"question": "What is the sum of angles in a triangle?", "answer": "180 degrees", "domain": "math", "answerability": "answerable"},
        {"question": "What is 7 factorial (7!)?", "answer": "5040", "domain": "math", "answerability": "answerable"},
    ]

    # Science domain - Answerable
    science_answerable = [
        {"question": "What is the chemical symbol for gold?", "answer": "Au", "domain": "science", "answerability": "answerable"},
        {"question": "How many planets are in our solar system?", "answer": "8", "domain": "science", "answerability": "answerable"},
        {"question": "What is the speed of light in vacuum?", "answer": "299,792,458 m/s", "domain": "science", "answerability": "answerable"},
        {"question": "What is the atomic number of carbon?", "answer": "6", "domain": "science", "answerability": "answerable"},
        {"question": "What is the boiling point of water at sea level?", "answer": "100°C or 212°F", "domain": "science", "answerability": "answerable"},
    ]

    # History domain - Answerable
    history_answerable = [
        {"question": "In what year did World War II end?", "answer": "1945", "domain": "history", "answerability": "answerable"},
        {"question": "Who was the first President of the United States?", "answer": "George Washington", "domain": "history", "answerability": "answerable"},
        {"question": "What year did the Berlin Wall fall?", "answer": "1989", "domain": "history", "answerability": "answerable"},
        {"question": "In what year did Columbus reach the Americas?", "answer": "1492", "domain": "history", "answerability": "answerable"},
        {"question": "Who was the first person to walk on the moon?", "answer": "Neil Armstrong", "domain": "history", "answerability": "answerable"},
    ]

    # Geography domain - Answerable
    geo_answerable = [
        {"question": "What is the capital of Japan?", "answer": "Tokyo", "domain": "geography", "answerability": "answerable"},
        {"question": "What is the largest ocean on Earth?", "answer": "Pacific Ocean", "domain": "geography", "answerability": "answerable"},
        {"question": "What is the longest river in the world?", "answer": "Nile River", "domain": "geography", "answerability": "answerable"},
        {"question": "How many continents are there?", "answer": "7", "domain": "geography", "answerability": "answerable"},
        {"question": "What is the tallest mountain in the world?", "answer": "Mount Everest", "domain": "geography", "answerability": "answerable"},
    ]

    # Math domain - Unanswerable
    math_unanswerable = [
        {"question": "What is the best number?", "domain": "math", "answerability": "unanswerable"},
        {"question": "How many grains of sand should I count?", "domain": "math", "answerability": "unanswerable"},
        {"question": "What's a good score on a test?", "domain": "math", "answerability": "unanswerable"},
    ]

    # Science domain - Unanswerable
    science_unanswerable = [
        {"question": "What is the exact weight of happiness?", "domain": "science", "answerability": "unanswerable"},
        {"question": "What color is the smell of roses?", "domain": "science", "answerability": "unanswerable"},
        {"question": "How many atoms are in a thought?", "domain": "science", "answerability": "unanswerable"},
    ]

    # History domain - Unanswerable
    history_unanswerable = [
        {"question": "What was Julius Caesar's favorite color?", "domain": "history", "answerability": "unanswerable"},
        {"question": "How many times did Napoleon sneeze in his lifetime?", "domain": "history", "answerability": "unanswerable"},
        {"question": "What was Shakespeare's cat's name?", "domain": "history", "answerability": "unanswerable"},
    ]

    # Geography domain - Unanswerable
    geo_unanswerable = [
        {"question": "What is the prettiest city in the world?", "domain": "geography", "answerability": "unanswerable"},
        {"question": "Which country has the nicest people?", "domain": "geography", "answerability": "unanswerable"},
        {"question": "What is the best place to live?", "domain": "geography", "answerability": "unanswerable"},
    ]

    # Combine all multi-domain questions
    answerable_multi_domain = (math_answerable + science_answerable +
                               history_answerable + geo_answerable)
    unanswerable_multi_domain = (math_unanswerable + science_unanswerable +
                                 history_unanswerable + geo_unanswerable)

    # Add domain labels to base datasets
    for q in answerable_base:
        if 'domain' not in q:
            q['domain'] = 'general'

    for q in unanswerable_base:
        if 'domain' not in q:
            q['domain'] = 'general'

    # Combine with base datasets
    all_answerable = answerable_base + answerable_multi_domain
    all_unanswerable = unanswerable_base + unanswerable_multi_domain

    # Sample to requested sizes
    random.seed(42)
    if len(all_answerable) < n_answerable:
        # Repeat if needed
        sampled_answerable = random.choices(all_answerable, k=n_answerable)
    else:
        sampled_answerable = random.sample(all_answerable, n_answerable)

    if len(all_unanswerable) < n_unanswerable:
        sampled_unanswerable = random.choices(all_unanswerable, k=n_unanswerable)
    else:
        sampled_unanswerable = random.sample(all_unanswerable, n_unanswerable)

    # Combine and shuffle
    combined = sampled_answerable + sampled_unanswerable
    random.shuffle(combined)

    print(f"\nDataset loaded: {n_answerable} answerable + {n_unanswerable} unanswerable")
    print(f"Total questions: {len(combined)}")

    # Print domain distribution
    domain_dist = {}
    for q in combined:
        domain = q.get('domain', 'unknown')
        domain_dist[domain] = domain_dist.get(domain, 0) + 1
    print(f"Domain distribution: {domain_dist}")

    return combined


class Experiment1:
    """Behavior-Belief Dissociation Experiment"""
    
    def __init__(self, model_wrapper: ModelWrapper, config: ExperimentConfig):
        self.model = model_wrapper
        self.config = config
        self.results = []
        set_seed(config.seed)
    
    def measure_internal_uncertainty(self, question: str,
                                     context: str = None,
                                     temperature: float = 1.0,
                                     n_samples: int = None) -> Dict:
        """
        Measure internal uncertainty using multiple belief proxies:
        1. Semantic entropy (existing)
        2. Self-consistency disagreement rate
        3. Max token probability / predictive entropy on first answer token
        """
        if n_samples is None:
            n_samples = self.config.n_force_guess_samples

        prompt = format_prompt(question, "force_guess", context)

        answers = []
        first_token_probs = []
        print(f"  Measuring internal uncertainty ({n_samples} samples, T={temperature})...")

        for i in range(n_samples):
            # Generate response with logging for first token probability
            response = self.model.generate(
                prompt,
                temperature=temperature,
                do_sample=True,
                max_new_tokens=50
            )
            answer = extract_answer(response)
            answers.append(answer)

            # Get first token probability if possible
            try:
                # Get logits for first answer token
                inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.config.device)
                with torch.no_grad():
                    outputs = self.model.model(**inputs, output_attentions=False)
                    logits = outputs.logits[0, -1, :]  # Last position logits
                    probs = torch.softmax(logits, dim=-1)
                    max_prob = float(probs.max().cpu())
                    first_token_probs.append(max_prob)
            except:
                # If extraction fails, skip
                pass

        # 1. Semantic entropy (existing)
        answer_counts = Counter(answers)
        total = len(answers)
        probs = np.array([count/total for count in answer_counts.values()])
        semantic_entropy = -np.sum(probs * np.log(probs + 1e-10))

        # 2. Self-consistency disagreement rate
        # Fraction of sampled answers NOT in top cluster
        most_common_count = answer_counts.most_common(1)[0][1]
        top_cluster_fraction = most_common_count / total
        disagreement_rate = 1.0 - top_cluster_fraction

        # 3. Max token probability / predictive entropy on first token
        avg_max_token_prob = np.mean(first_token_probs) if first_token_probs else 0.0
        # Predictive entropy: -sum(p_i * log(p_i)) for first token distribution
        # Approximation: higher max_prob -> lower entropy
        predictive_entropy = -np.log(avg_max_token_prob + 1e-10) if avg_max_token_prob > 0 else 0.0

        # Most common answer
        most_common = answer_counts.most_common(1)[0][0]

        return {
            # Semantic entropy (original proxy)
            "semantic_entropy": float(semantic_entropy),
            "entropy": float(semantic_entropy),  # Keep for backward compatibility

            # Self-consistency disagreement
            "disagreement_rate": float(disagreement_rate),
            "self_consistency": float(top_cluster_fraction),

            # Token-level probability
            "max_token_prob": float(avg_max_token_prob),
            "predictive_entropy": float(predictive_entropy),

            # Legacy fields
            "p_majority": float(np.max(probs)),
            "n_unique_answers": len(answer_counts),
            "most_common_answer": most_common,
            "all_answers": answers,
            "distribution": dict(answer_counts)
        }
    
    def test_instruction_regime(self, question: str, regime: str,
                               context: str = None) -> Dict:
        """Test question under specific instruction regime"""
        prompt = format_prompt(question, regime, context)
        
        # Generate with temperature 0 for deterministic output
        response = self.model.generate(
            prompt,
            temperature=0.0,
            do_sample=False,
            max_new_tokens=50
        )
        
        answer = extract_answer(response)
        abstained = (answer == "UNCERTAIN")
        
        return {
            "regime": regime,
            "response": response,
            "answer": answer,
            "abstained": abstained
        }
    
    def run(self, questions: List[Dict], 
            regimes: List[str] = None) -> pd.DataFrame:
        """
        Run full experiment:
        1. Measure internal uncertainty once per question
        2. Test each question under each regime
        """
        if regimes is None:
            regimes = ["neutral", "cautious", "confident"]
        
        print("\n" + "="*60)
        print("EXPERIMENT 1: Behavior-Belief Dissociation")
        print("="*60)
        print(f"Testing {len(questions)} questions across {len(regimes)} regimes")
        print()
        
        for i, q_data in enumerate(tqdm(questions, desc="Questions")):
            question = q_data["question"]
            context = q_data.get("context", None)
            
            print(f"\nQuestion {i+1}/{len(questions)}: {question[:60]}...")
            
            # Step 1: Measure internal uncertainty (once)
            internal = self.measure_internal_uncertainty(question, context)
            
            # Step 2: Test each regime
            for regime in regimes:
                print(f"  Testing regime: {regime}")
                result = self.test_instruction_regime(question, regime, context)
                
                # Store combined result with all belief proxies
                self.results.append({
                    "question": question,
                    "question_type": q_data.get("type", "unknown"),
                    "domain": q_data.get("domain", "unknown"),
                    "true_answerability": q_data.get(
                        "answerability",
                        q_data.get("true_answerability", "unknown")
                    ),
                    "regime": regime,

                    # Multiple belief proxies (same for all regimes within a question)
                    "semantic_entropy": internal["semantic_entropy"],
                    "disagreement_rate": internal["disagreement_rate"],
                    "max_token_prob": internal["max_token_prob"],
                    "predictive_entropy": internal["predictive_entropy"],

                    # Legacy fields
                    "internal_entropy": internal["entropy"],
                    "internal_p_majority": internal["p_majority"],
                    "internal_n_unique": internal["n_unique_answers"],

                    # External behavior (varies by regime)
                    "abstained": result["abstained"],
                    "answer": result["answer"],
                    "response": result["response"]
                })
        
        df = pd.DataFrame(self.results)
        
        # Save results
        output_path = self.config.results_dir / "exp1_raw_results.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
        
        return df

    def test_proxy_robustness(self, questions: List[Dict], subset_size: int = 50) -> pd.DataFrame:
        """
        Test proxy robustness across different (K, T) settings.
        K = number of samples, T = temperature
        """
        print("\n" + "="*60)
        print("PROXY ROBUSTNESS CHECK")
        print("="*60)

        settings = [
            (10, 0.7),
            (15, 1.0),
            (30, 1.0)
        ]

        # Use a subset for speed
        import random
        random.seed(42)
        subset = random.sample(questions, min(subset_size, len(questions)))

        robustness_results = []

        for k, t in settings:
            print(f"\nTesting K={k}, T={t}")
            for q_data in tqdm(subset, desc=f"K={k}, T={t}"):
                question = q_data["question"]
                context = q_data.get("context", None)

                # Measure with this setting
                internal = self.measure_internal_uncertainty(
                    question, context, temperature=t, n_samples=k
                )

                robustness_results.append({
                    "question": question,
                    "K": k,
                    "T": t,
                    "semantic_entropy": internal["semantic_entropy"],
                    "disagreement_rate": internal["disagreement_rate"],
                    "max_token_prob": internal["max_token_prob"],
                    "true_answerability": q_data.get("answerability", "unknown")
                })

        robustness_df = pd.DataFrame(robustness_results)

        # Save results
        output_path = self.config.results_dir / "exp1_proxy_robustness.csv"
        robustness_df.to_csv(output_path, index=False)
        print(f"\nProxy robustness results saved to {output_path}")

        return robustness_df

    def quantitative_dissociation_test(self, df: pd.DataFrame) -> Dict:
        """
        Quantitative dissociation test using logistic regression:
        - Fit: abstain ~ instruction + proxy
        - Report: Δ log-likelihood / pseudo-R² when adding instruction vs adding proxy
        - Also create stratified plots: bin proxy into terciles, show abstention differs by instruction
        """
        print("\n" + "="*60)
        print("QUANTITATIVE DISSOCIATION TEST")
        print("="*60)

        # Prepare data
        df_clean = df.dropna(subset=['abstained', 'semantic_entropy']).copy()

        # Encode instruction regime as binary/categorical
        # Use dummy encoding: cautious vs. neutral vs. confident
        df_clean['instruction_cautious'] = (df_clean['regime'] == 'cautious').astype(int)
        df_clean['instruction_confident'] = (df_clean['regime'] == 'confident').astype(int)
        # neutral is reference category

        y = df_clean['abstained'].astype(int).values

        results = {}

        # Test each proxy
        proxies = ['semantic_entropy', 'disagreement_rate', 'predictive_entropy']

        for proxy in proxies:
            if proxy not in df_clean.columns:
                print(f"Skipping {proxy} - not in dataframe")
                continue

            print(f"\n--- Testing proxy: {proxy} ---")

            # Standardize proxy for better numerical stability
            scaler = StandardScaler()
            proxy_std = scaler.fit_transform(df_clean[[proxy]]).flatten()

            # Model 1: Null model (intercept only)
            X_null = np.ones((len(y), 1))
            model_null = LogisticRegression(max_iter=1000, random_state=42)
            model_null.fit(X_null, y)
            ll_null = -log_loss(y, model_null.predict_proba(X_null)[:, 1], normalize=False)

            # Model 2: Instruction only
            X_instruction = df_clean[['instruction_cautious', 'instruction_confident']].values
            model_instruction = LogisticRegression(max_iter=1000, random_state=42)
            model_instruction.fit(X_instruction, y)
            ll_instruction = -log_loss(y, model_instruction.predict_proba(X_instruction)[:, 1], normalize=False)

            # Model 3: Proxy only
            X_proxy = proxy_std.reshape(-1, 1)
            model_proxy = LogisticRegression(max_iter=1000, random_state=42)
            model_proxy.fit(X_proxy, y)
            ll_proxy = -log_loss(y, model_proxy.predict_proba(X_proxy)[:, 1], normalize=False)

            # Model 4: Both instruction and proxy
            X_both = np.column_stack([X_instruction, proxy_std])
            model_both = LogisticRegression(max_iter=1000, random_state=42)
            model_both.fit(X_both, y)
            ll_both = -log_loss(y, model_both.predict_proba(X_both)[:, 1], normalize=False)

            # Compute pseudo-R² (McFadden)
            pseudo_r2_instruction = 1 - (ll_instruction / ll_null)
            pseudo_r2_proxy = 1 - (ll_proxy / ll_null)
            pseudo_r2_both = 1 - (ll_both / ll_null)

            # Δ log-likelihood
            delta_ll_instruction = ll_instruction - ll_null
            delta_ll_proxy = ll_proxy - ll_null
            delta_ll_both = ll_both - ll_null

            print(f"\nPseudo-R² (McFadden):")
            print(f"  Instruction only: {pseudo_r2_instruction:.4f}")
            print(f"  Proxy only:       {pseudo_r2_proxy:.4f}")
            print(f"  Both:             {pseudo_r2_both:.4f}")

            print(f"\nΔ Log-Likelihood (vs null):")
            print(f"  Instruction only: {delta_ll_instruction:.2f}")
            print(f"  Proxy only:       {delta_ll_proxy:.2f}")
            print(f"  Both:             {delta_ll_both:.2f}")

            print(f"\nCoefficients (Both model):")
            print(f"  Cautious: {model_both.coef_[0][0]:.4f}")
            print(f"  Confident: {model_both.coef_[0][1]:.4f}")
            print(f"  {proxy}: {model_both.coef_[0][2]:.4f}")

            results[proxy] = {
                "pseudo_r2_instruction": float(pseudo_r2_instruction),
                "pseudo_r2_proxy": float(pseudo_r2_proxy),
                "pseudo_r2_both": float(pseudo_r2_both),
                "delta_ll_instruction": float(delta_ll_instruction),
                "delta_ll_proxy": float(delta_ll_proxy),
                "delta_ll_both": float(delta_ll_both),
                "coef_cautious": float(model_both.coef_[0][0]),
                "coef_confident": float(model_both.coef_[0][1]),
                "coef_proxy": float(model_both.coef_[0][2])
            }

        # Save quantitative test results
        results_path = self.config.results_dir / "exp1_quantitative_dissociation.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n\nQuantitative dissociation results saved to {results_path}")

        return results

    def analyze(self, df: pd.DataFrame) -> Dict:
        """Analyze and visualize results for Experiment 1."""
        import numpy as np
        import pandas as pd
    
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: ANALYSIS")
        print("=" * 60)
    
        # -------------------------
        # 1) Aggregate by regime
        # -------------------------
        abstention_by_regime = df.groupby("regime")["abstained"].agg(
            rate="mean",
            std="std",
            count="size",
        )
    
        print("\nAbstention Rates by Instruction Regime:")
        print(abstention_by_regime)
    
        entropy_by_regime = df.groupby("regime")["internal_entropy"].agg(
            mean="mean",
            std="std",
        )
    
        print("\nInternal Uncertainty (Entropy) by Regime:")
        print(entropy_by_regime)
    
        # -------------------------
        # 2) Key finding: Behavior vs Belief
        # -------------------------
        def safe_mean(regime: str, col: str) -> float:
            s = df.loc[df["regime"] == regime, col]
            return float(s.mean()) if len(s) else float("nan")
    
        cautious_abstain = safe_mean("cautious", "abstained")
        neutral_abstain = safe_mean("neutral", "abstained")
        confident_abstain = safe_mean("confident", "abstained")
    
        behavior_gap = cautious_abstain - confident_abstain
    
        entropy_cautious = safe_mean("cautious", "internal_entropy")
        entropy_confident = safe_mean("confident", "internal_entropy")
        belief_gap = float(abs(entropy_cautious - entropy_confident))
    
        ratio = float(behavior_gap / (belief_gap + 1e-6))
    
        print("\n" + "=" * 60)
        print("KEY FINDING: Behavior ≠ Belief")
        print("=" * 60)
        print(f"Abstention rate (cautious):   {cautious_abstain:.3f}")
        print(f"Abstention rate (neutral):    {neutral_abstain:.3f}")
        print(f"Abstention rate (confident):  {confident_abstain:.3f}")
        print(f"\nBehavior gap (cautious - confident): {behavior_gap:.3f}")
        print(f"Belief gap (entropy difference):     {belief_gap:.3f}")
        print(f"\nBehavior/Belief ratio: {ratio:.1f}x")
    
        # -------------------------
        # 3) Per-question "interesting" examples
        # -------------------------
        per_question = df.pivot_table(
            index="question",
            columns="regime",
            values=["abstained", "internal_entropy"],
            aggfunc="first",
        )
    
        # Flatten MultiIndex columns: ('internal_entropy','cautious') -> 'internal_entropy__cautious'
        per_question.columns = [f"{a}__{b}" for (a, b) in per_question.columns]
    
        regimes = ["neutral", "cautious", "confident"]
    
        abstain_cols = [f"abstained__{r}" for r in regimes if f"abstained__{r}" in per_question.columns]
        entropy_cols = [f"internal_entropy__{r}" for r in regimes if f"internal_entropy__{r}" in per_question.columns]
    
        # Make abstained numeric (bool -> 0/1) to avoid dtype weirdness
        if abstain_cols:
            per_question[abstain_cols] = per_question[abstain_cols].astype(float)
    
        per_question["abstain_variance"] = per_question[abstain_cols].std(axis=1) if abstain_cols else np.nan
        per_question["entropy_variance"] = per_question[entropy_cols].std(axis=1) if entropy_cols else np.nan
    
        interesting = per_question[
            (per_question["abstain_variance"] > 0.4) &
            (per_question["entropy_variance"] < 0.1)
        ].head(5)
    
        if len(interesting) > 0:
            print("\nExample questions where behavior changes but entropy doesn't:")
            for q, row in interesting.iterrows():
                q_preview = (q[:60] + "...") if isinstance(q, str) and len(q) > 60 else q
                print(f"\n  Q: {q_preview}")
                print(f"     Entropy variance: {float(row['entropy_variance']):.3f}")
                print(f"     Abstain variance: {float(row['abstain_variance']):.3f}")
        else:
            print("\nNo high-variance behavior / low-variance entropy examples found (in this sample).")
    
        # -------------------------
        # 4) Quantitative dissociation test
        # -------------------------
        quant_results = self.quantitative_dissociation_test(df)

        # -------------------------
        # 5) Stratified plots
        # -------------------------
        self.create_stratified_plots(df)

        # -------------------------
        # 6) Original plots
        # -------------------------
        self.plot_results(df, abstention_by_regime, entropy_by_regime)

        # -------------------------
        # 7) Create summary table
        # -------------------------
        self.create_summary_table(quant_results)

        # -------------------------
        # 8) Return JSON-serializable summary
        # -------------------------
        return {
            "abstention_by_regime": abstention_by_regime.to_dict(),
            "entropy_by_regime": entropy_by_regime.to_dict(),
            "behavior_gap": float(behavior_gap),
            "belief_gap": float(belief_gap),
            "behavior_belief_ratio": float(ratio),
            "quantitative_results": quant_results,
            "n_rows": int(len(df)),
            "n_questions": int(df["question"].nunique()) if "question" in df.columns else None,
        }
    
        
    def plot_results(self, df: pd.DataFrame, abstention_stats, entropy_stats):
        """Create publication-quality figures"""
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Abstention rates
        regimes = ["confident", "neutral", "cautious"]
        abstention_means = [abstention_stats.loc[r, 'rate'] for r in regimes]
        abstention_stds = [abstention_stats.loc[r, 'std'] for r in regimes]
        
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        axes[0].bar(regimes, abstention_means, yerr=abstention_stds, 
                   capsize=5, color=colors, alpha=0.8)
        axes[0].set_ylabel("Abstention Rate", fontsize=12)
        axes[0].set_xlabel("Instruction Regime", fontsize=12)
        axes[0].set_title("Behavior Changes with Instruction", fontsize=13, fontweight='bold')
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Internal entropy (should be stable)
        entropy_means = [entropy_stats.loc[r, 'mean'] for r in regimes]
        entropy_stds = [entropy_stats.loc[r, 'std'] for r in regimes]
        
        axes[1].bar(regimes, entropy_means, yerr=entropy_stds,
                   capsize=5, color='#95a5a6', alpha=0.8)
        axes[1].set_ylabel("Internal Entropy", fontsize=12)
        axes[1].set_xlabel("Instruction Regime", fontsize=12)
        axes[1].set_title("Internal Uncertainty Remains Stable", fontsize=13, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Scatter plot of entropy vs abstention
        for regime, color in zip(["confident", "neutral", "cautious"], colors):
            regime_data = df[df["regime"] == regime]
            axes[2].scatter(
                regime_data["internal_entropy"],
                regime_data["abstained"].astype(float),
                alpha=0.6,
                s=50,
                label=regime.capitalize(),
                color=color
            )
        
        axes[2].set_xlabel("Internal Entropy", fontsize=12)
        axes[2].set_ylabel("Abstained (0 or 1)", fontsize=12)
        axes[2].set_title("Behavior Decoupled from Belief", fontsize=13, fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.config.results_dir / "exp1_behavior_belief_dissociation.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {output_path}")
        plt.close()
        
        # Also create a combined figure for paper
        self._create_paper_figure(df)
    
    def _create_paper_figure(self, df: pd.DataFrame):
        """Create single figure for paper"""
        
        fig = plt.figure(figsize=(10, 4))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[0])
        
        regimes = ["confident", "neutral", "cautious"]
        regime_labels = ["Confident", "Neutral", "Cautious"]
        colors_behavior = ['#2ecc71', '#3498db', '#e74c3c']
        
        # Dual y-axis plot
        abstention_means = [df[df["regime"] == r]["abstained"].mean() for r in regimes]
        entropy_means = [df[df["regime"] == r]["internal_entropy"].mean() for r in regimes]
        
        x = np.arange(len(regimes))
        width = 0.35
        
        # Abstention bars
        ax1.bar(x - width/2, abstention_means, width, 
               label='Abstention Rate (Behavior)', color=colors_behavior, alpha=0.8)
        ax1.set_ylabel('Abstention Rate', fontsize=11)
        ax1.set_ylim([0, 1.0])
        ax1.set_xticks(x)
        ax1.set_xticklabels(regime_labels)
        
        # Entropy bars on secondary axis
        ax2 = ax1.twinx()
        ax2.bar(x + width/2, entropy_means, width,
               label='Internal Entropy (Belief)', color='gray', alpha=0.6)
        ax2.set_ylabel('Internal Entropy', fontsize=11)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
        
        ax1.set_title('Behavior Changes, Belief Stays Constant', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.config.results_dir / "exp1_paper_figure.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Paper figure saved to {output_path}")
        plt.close()

    def create_stratified_plots(self, df: pd.DataFrame):
        """
        Create stratified plots: bin proxy into terciles,
        within each bin show abstention differs by instruction
        """
        print("\n" + "="*60)
        print("CREATING STRATIFIED PLOTS")
        print("="*60)

        proxies = ['semantic_entropy', 'disagreement_rate', 'predictive_entropy']
        df_clean = df.dropna(subset=['abstained']).copy()

        fig, axes = plt.subplots(1, len(proxies), figsize=(15, 4))
        if len(proxies) == 1:
            axes = [axes]

        for idx, proxy in enumerate(proxies):
            if proxy not in df_clean.columns:
                print(f"Skipping {proxy} - not in dataframe")
                continue

            # Bin proxy into terciles
            df_clean[f'{proxy}_tercile'] = pd.qcut(
                df_clean[proxy],
                q=3,
                labels=['Low', 'Medium', 'High'],
                duplicates='drop'
            )

            # Calculate abstention rate by regime within each tercile
            grouped = df_clean.groupby([f'{proxy}_tercile', 'regime'])['abstained'].mean().reset_index()

            # Plot
            terciles = ['Low', 'Medium', 'High']
            regimes = ['confident', 'neutral', 'cautious']
            colors = {'confident': '#2ecc71', 'neutral': '#3498db', 'cautious': '#e74c3c'}

            x = np.arange(len(terciles))
            width = 0.25

            for i, regime in enumerate(regimes):
                regime_data = grouped[grouped['regime'] == regime]
                values = [
                    regime_data[regime_data[f'{proxy}_tercile'] == t]['abstained'].values[0]
                    if len(regime_data[regime_data[f'{proxy}_tercile'] == t]) > 0 else 0
                    for t in terciles
                ]
                axes[idx].bar(x + i*width, values, width, label=regime.capitalize(),
                             color=colors[regime], alpha=0.8)

            axes[idx].set_xlabel(f'{proxy.replace("_", " ").title()} Tercile', fontsize=10)
            axes[idx].set_ylabel('Abstention Rate', fontsize=10)
            axes[idx].set_title(f'Dissociation within {proxy.replace("_", " ").title()} bins',
                               fontsize=11, fontweight='bold')
            axes[idx].set_xticks(x + width)
            axes[idx].set_xticklabels(terciles)
            axes[idx].legend(fontsize=8)
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        output_path = self.config.results_dir / "exp1_stratified_plots.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nStratified plots saved to {output_path}")
        plt.close()

    def create_summary_table(self, quant_results: Dict):
        """
        Create a summary table of predictive power: instruction vs proxy
        """
        print("\n" + "="*60)
        print("SUMMARY TABLE: Predictive Power")
        print("="*60)

        # Create DataFrame for table
        table_data = []
        for proxy, results in quant_results.items():
            table_data.append({
                'Proxy': proxy.replace('_', ' ').title(),
                'Pseudo-R² (Instruction)': f"{results['pseudo_r2_instruction']:.4f}",
                'Pseudo-R² (Proxy)': f"{results['pseudo_r2_proxy']:.4f}",
                'Pseudo-R² (Both)': f"{results['pseudo_r2_both']:.4f}",
                'Coef. Cautious': f"{results['coef_cautious']:.4f}",
                'Coef. Confident': f"{results['coef_confident']:.4f}",
                'Coef. Proxy': f"{results['coef_proxy']:.4f}"
            })

        table_df = pd.DataFrame(table_data)
        print("\n", table_df.to_string(index=False))

        # Save as CSV
        output_path = self.config.results_dir / "exp1_summary_table.csv"
        table_df.to_csv(output_path, index=False)
        print(f"\n\nSummary table saved to {output_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main(n_answerable: int = 200, n_unanswerable: int = 200,
         run_robustness: bool = True, quick_test: bool = False):
    """
    Run Experiment 1 with scaled dataset and multiple belief proxies

    Args:
        n_answerable: Number of answerable questions (200-500)
        n_unanswerable: Number of unanswerable questions (200-500)
        run_robustness: Whether to run proxy robustness checks
        quick_test: If True, use small subset for quick testing
    """
    from pathlib import Path
    import json

    # Setup
    config = ExperimentConfig()
    config.n_force_guess_samples = 15  # K=15, T=1.0 as baseline

    print("Initializing model...")
    model = ModelWrapper(config)

    # Load scaled dataset with multiple domains
    print("\nLoading scaled dataset...")
    if quick_test:
        # Quick test mode
        questions = load_scaled_dataset(n_answerable=10, n_unanswerable=10)
    else:
        # Full experiment
        questions = load_scaled_dataset(n_answerable=n_answerable, n_unanswerable=n_unanswerable)

    # Run main experiment
    exp1 = Experiment1(model, config)
    results_df = exp1.run(questions)

    # Run proxy robustness checks (optional)
    if run_robustness:
        print("\n\nRunning proxy robustness checks...")
        robustness_df = exp1.test_proxy_robustness(questions, subset_size=50)

        # Analyze robustness
        print("\n" + "="*60)
        print("ROBUSTNESS ANALYSIS")
        print("="*60)
        for setting in [(10, 0.7), (15, 1.0), (30, 1.0)]:
            k, t = setting
            subset = robustness_df[(robustness_df['K'] == k) & (robustness_df['T'] == t)]
            mean_entropy = subset['semantic_entropy'].mean()
            std_entropy = subset['semantic_entropy'].std()
            print(f"K={k}, T={t}: Semantic Entropy = {mean_entropy:.3f} ± {std_entropy:.3f}")

    # Analyze results
    summary = exp1.analyze(results_df)

    # Save summary
    summary_path = config.results_dir / "exp1_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Experiment 1 complete!")
    print(f"\n📊 Output files:")
    print(f"  Raw results:          {config.results_dir / 'exp1_raw_results.csv'}")
    print(f"  Summary:              {summary_path}")
    print(f"  Quant. dissociation:  {config.results_dir / 'exp1_quantitative_dissociation.json'}")
    print(f"  Summary table:        {config.results_dir / 'exp1_summary_table.csv'}")
    if run_robustness:
        print(f"  Proxy robustness:     {config.results_dir / 'exp1_proxy_robustness.csv'}")
    print(f"\n📈 Figures:")
    print(f"  Main figure:          {config.results_dir / 'exp1_behavior_belief_dissociation.png'}")
    print(f"  Paper figure:         {config.results_dir / 'exp1_paper_figure.png'}")
    print(f"  Stratified plots:     {config.results_dir / 'exp1_stratified_plots.png'}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Experiment 1: Behavior-Belief Dissociation')
    parser.add_argument('--n_answerable', type=int, default=200,
                       help='Number of answerable questions (200-500)')
    parser.add_argument('--n_unanswerable', type=int, default=200,
                       help='Number of unanswerable questions (200-500)')
    parser.add_argument('--no_robustness', action='store_true',
                       help='Skip proxy robustness checks')
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test mode with small dataset')

    args = parser.parse_args()

    main(
        n_answerable=args.n_answerable,
        n_unanswerable=args.n_unanswerable,
        run_robustness=not args.no_robustness,
        quick_test=args.quick_test
    )
