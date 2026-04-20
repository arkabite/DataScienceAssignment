# """
# Reflector Agent - Students must extend this significantly

# The reflector evaluates execution results, identifies issues, and suggests improvements.
# Your task is to implement sophisticated analysis that goes beyond simple threshold checks.

# TODO: Extend this module with:
# 1. Statistical significance testing between models
# 2. Per-class performance analysis
# 3. Root cause diagnosis (data quality, preprocessing, model issues)
# 4. Actionable, prioritized suggestions
# 5. Learning from past reflections (meta-learning)
# """

# from typing import Any, Dict, List, Tuple


# def reflect(
#     dataset_profile: Dict[str, Any],
#     evaluation: Dict[str, Any],
#     all_metrics: List[Dict[str, Any]]
# ) -> Dict[str, Any]:
#     """
#     Analyze results and generate reflection with issues and suggestions.
    
#     This is a basic implementation. Students should extend this significantly.
    
#     Args:
#         dataset_profile: Dataset characteristics
#         evaluation: Best model's metrics
#         all_metrics: Metrics for all trained models
    
#     Returns:
#         Dictionary with:
#             - status: str ("ok" or "needs_attention")
#             - best_model: str (model name)
#             - issues: List[str] (identified problems)
#             - suggestions: List[str] (improvement recommendations)
#             - replan_recommended: bool (should we replan?)
    
#     TODO for students:
#     - Implement statistical tests (paired t-tests, Wilcoxon tests)
#     - Add per-class performance analysis
#     - Detect overfitting vs underfitting
#     - Analyze confusion matrix patterns
#     - Check for data quality issues
#     - Prioritize suggestions by expected impact
#     - Learn which suggestions work from memory
#     """
    
#     best_model = evaluation.get("model")
#     bal_acc = float(evaluation.get("balanced_accuracy", 0.0))
#     f1_macro = float(evaluation.get("f1_macro", 0.0))
#     imb = float(dataset_profile.get("imbalance_ratio") or 1.0)
    
#     issues: List[str] = []
#     suggestions: List[str] = []
    
#     # Basic comparison with dummy baseline
#     dummy = next((m for m in all_metrics if "Dummy" in m.get("model", "")), None)
    
#     if dummy is not None:
#         dummy_ba = float(dummy.get("balanced_accuracy", 0.0))
#         improvement = bal_acc - dummy_ba
        
#         # TODO: Make this more sophisticated
#         # Consider: confidence intervals, effect sizes, etc.
#         if improvement < 0.05:
#             issues.append(
#                 f"Best model only {improvement:.3f} better than baseline. "
#                 "Weak signal or pipeline issues."
#             )
#             suggestions.append(
#                 "Check for target leakage, verify target quality, "
#                 "or improve feature engineering."
#             )
    
#     # TODO: Add more sophisticated checks
    
#     # Check F1 score
#     # TODO: Make threshold adaptive based on problem difficulty
#     if f1_macro < 0.60:
#         issues.append("Macro F1 score is modest (<0.60).")
#         suggestions.append(
#             "Try different models, tune hyperparameters, "
#             "or improve preprocessing."
#         )
    
#     # TODO: Add imbalance-specific analysis
#     if imb >= 3.0:
#         suggestions.append(
#             "Imbalance detected: consider class_weight, "
#             "threshold tuning, or SMOTE."
#         )
    
#     # TODO: Add checks for:
#     # - Model diversity (are all models performing similarly?)
#     # - Per-class performance (which classes are problematic?)
#     # - Precision-recall tradeoff
#     # - High-cardinality categorical features
#     # - Feature importance patterns
#     # - Learning curves (overfitting/underfitting)
    
#     # Determine status
#     status = "needs_attention" if issues else "ok"
    
#     # Simple replanning trigger
#     # TODO: Make this more sophisticated
#     replan_recommended = bool(issues and f1_macro < 0.60)
    
#     return {
#         "status": status,
#         "best_model": best_model,
#         "issues": issues,
#         "suggestions": suggestions,
#         "replan_recommended": replan_recommended,
#     }


# def should_replan(reflection: Dict[str, Any]) -> bool:
#     """
#     Decide whether to trigger replanning based on reflection.
    
#     This is a simple policy. Students should implement more sophisticated logic.
    
#     TODO for students:
#     - Consider multiple factors (performance, confidence, resource budget)
#     - Implement diminishing returns detection
#     - Use memory to avoid repeating failed strategies
#     - Set adaptive thresholds based on problem difficulty
#     """
#     return bool(reflection.get("replan_recommended", False))


# def apply_replan_strategy(
#     plan: List[str],
#     dataset_profile: Dict[str, Any],
#     reflection: Dict[str, Any],
# ) -> Tuple[List[str], Dict[str, Any]]:
#     """
#     Modify the plan and dataset profile based on reflection.
    
#     This is a very basic implementation. Students should make this sophisticated.
    
#     Args:
#         plan: Current execution plan
#         dataset_profile: Current dataset profile
#         reflection: Reflection results
    
#     Returns:
#         Tuple of (modified_plan, modified_profile)
    
#     TODO for students:
#     - Implement specific strategies for specific issues
#     - Add preprocessing steps based on identified problems
#     - Modify model selection based on performance patterns
#     - Adjust hyperparameters
#     - Try ensemble methods
#     - Implement different replan strategies (aggressive, conservative)
#     """
    
#     # Copy to avoid modifying originals
#     new_plan = list(plan)
#     new_profile = dict(dataset_profile)
    
#     # Basic strategy: add a note
#     # TODO: Implement actual strategy changes
#     notes = list(new_profile.get("notes", []))
#     notes.append("Replan: adjusting strategy after reflection.")
#     new_profile["notes"] = notes
    
#     new_plan.append("replan_attempt")
    
#     # TODO: Implement sophisticated replan strategies:
#     # - If low performance: try ensemble methods
#     # - If imbalance issues: add SMOTE or adjust thresholds
#     # - If overfitting: add regularization
#     # - If underfitting: increase model complexity
#     # - If feature issues: add feature engineering steps
    
#     return new_plan, new_profile


# # TODO: Add helper functions for reflection
# # def compare_models_statistically(...):
# # def analyze_per_class_performance(...):
# # def detect_overfitting(...):
# # def detect_data_quality_issues(...):
# # def prioritize_suggestions(...):
# # def generate_explanation(...):

"""
Reflector Agent - Extended Version (Diagnostic + Replanning)

This version upgrades the reflector into a diagnostic engine that:
- Identifies WHY performance is poor
- Suggests targeted improvements
- Drives intelligent replanning

Safe for runtime: includes fallbacks and defensive checks.
"""

from typing import Any, Dict, List, Tuple


# ------------------------------------------------------------------
# DIAGNOSTIC HELPERS
# ------------------------------------------------------------------

def _check_dummy_gap(best_metrics: Dict[str, Any], all_metrics: List[Dict[str, Any]]):
    dummy = next((m for m in all_metrics if "Dummy" in m.get("model", "")), None)
    if not dummy:
        return None

    try:
        gap = float(best_metrics.get("balanced_accuracy", 0.0)) - float(dummy.get("balanced_accuracy", 0.0))
        return gap
    except Exception:
        return None


def _check_class_bias(evaluation: Dict[str, Any]) -> bool:
    try:
        acc = float(evaluation.get("accuracy", 0.0))
        bal_acc = float(evaluation.get("balanced_accuracy", 0.0))
        return acc > 0.85 and bal_acc < 0.60
    except Exception:
        return False


def _check_low_performance(evaluation: Dict[str, Any]) -> bool:
    try:
        return float(evaluation.get("f1_macro", 0.0)) < 0.60
    except Exception:
        return False


# ------------------------------------------------------------------
# MAIN REFLECTION FUNCTION
# ------------------------------------------------------------------

def reflect(
    dataset_profile: Dict[str, Any],
    evaluation: Dict[str, Any],
    all_metrics: List[Dict[str, Any]]
) -> Dict[str, Any]:

    try:
        best_model = evaluation.get("model")
        imb = float(dataset_profile.get("imbalance_ratio", 1.0))

        issues: List[str] = []
        suggestions: List[str] = []
        diagnostics: Dict[str, Any] = {}

        # -----------------------------
        # DIAGNOSTICS
        # -----------------------------
        dummy_gap = _check_dummy_gap(evaluation, all_metrics)
        class_bias = _check_class_bias(evaluation)
        low_perf = _check_low_performance(evaluation)

        diagnostics["dummy_gap"] = dummy_gap
        diagnostics["class_bias"] = class_bias
        diagnostics["low_performance"] = low_perf

        # -----------------------------
        # INTERPRETATION (WHY)
        # -----------------------------

        if dummy_gap is not None and dummy_gap < 0.05:
            issues.append("Model barely outperforms baseline (weak signal or poor features).")
            suggestions.append("Improve feature engineering or try more complex models.")

        if class_bias:
            issues.append("High accuracy but low balanced accuracy → model is biased toward majority class.")
            suggestions.append("Use class weights, SMOTE, or threshold tuning.")

        if low_perf:
            issues.append("Overall performance is low (F1_macro < 0.60).")
            suggestions.append("Try alternative models or improve preprocessing.")

        if imb >= 3.0:
            suggestions.append("Dataset is imbalanced: consider SMOTE or cost-sensitive learning.")

        # -----------------------------
        # STATUS + REPLAN DECISION
        # -----------------------------
        status = "needs_attention" if issues else "ok"

        replan_recommended = (
            low_perf or
            class_bias or
            (dummy_gap is not None and dummy_gap < 0.05)
        )

        return {
            "status": status,
            "best_model": best_model,
            "issues": issues,
            "suggestions": suggestions,
            "replan_recommended": replan_recommended,
            "diagnostics": diagnostics,
        }

    except Exception as e:
        # -----------------------------
        # FAIL-SAFE (prevents crash)
        # -----------------------------
        return {
            "status": "ok",
            "best_model": evaluation.get("model"),
            "issues": [],
            "suggestions": [],
            "replan_recommended": False,
            "diagnostics": {"error": str(e)},
        }


# ------------------------------------------------------------------
# REPLAN DECISION
# ------------------------------------------------------------------

def should_replan(reflection: Dict[str, Any]) -> bool:
    diagnostics = reflection.get("diagnostics", {})

    if diagnostics.get("low_performance"):
        return True

    if diagnostics.get("class_bias"):
        return True

    if diagnostics.get("dummy_gap") is not None and diagnostics["dummy_gap"] < 0.05:
        return True

    return False


# ------------------------------------------------------------------
# APPLY REPLAN STRATEGY (PIVOT LOGIC)
# ------------------------------------------------------------------

def apply_replan_strategy(
    plan: List[str],
    dataset_profile: Dict[str, Any],
    reflection: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:

    # Copy to avoid modifying originals
    new_plan = list(plan)
    new_profile = dict(dataset_profile)

    diagnostics = reflection.get("diagnostics", {})

    # Helper to safely add steps before training
    def add_step(step: str):
        if step not in new_plan:
            if "train_models" in new_plan:
                new_plan.insert(new_plan.index("train_models"), step)
            else:
                new_plan.append(step)

    # -----------------------------
    # CASE 1: LOW PERFORMANCE
    # -----------------------------
    if diagnostics.get("low_performance"):
        add_step("increase_model_complexity")

    # -----------------------------
    # CASE 2: CLASS BIAS
    # -----------------------------
    if diagnostics.get("class_bias"):
        add_step("handle_imbalance")
        add_step("use_class_weights")

    # -----------------------------
    # CASE 3: WEAK SIGNAL
    # -----------------------------
    if diagnostics.get("dummy_gap") is not None and diagnostics["dummy_gap"] < 0.05:
        add_step("feature_engineering")

    # -----------------------------
    # SAFETY: Prevent runaway plan growth
    # -----------------------------
    if len(new_plan) > 20:
        return plan, dataset_profile

    # -----------------------------
    # ADD NOTE TO PROFILE
    # -----------------------------
    notes = list(new_profile.get("notes", []))
    notes.append("Replan applied: adjusted strategy based on reflection.")
    new_profile["notes"] = notes

    return new_plan, new_profile