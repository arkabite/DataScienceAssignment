# """
# Planner Agent - Students must extend this significantly

# The planner analyzes dataset characteristics and generates an execution plan.
# Your task is to implement sophisticated planning logic that adapts to different
# dataset types, sizes, and characteristics.

# TODO: Extend this module with:
# 1. Sophisticated planning logic based on dataset profiles
# 2. Different plan templates for different scenarios
# 3. Memory-guided planning (use past successful strategies)
# 4. Dependency management (task ordering)
# 5. Conditional planning (if X then Y else Z)
# 6. Fallback strategies for edge cases
# """

# from typing import Any, Dict, List, Optional


# def create_plan(
#     dataset_profile: Dict[str, Any], 
#     memory_hint: Optional[Dict[str, Any]] = None
# ) -> List[str]:
#     """
#     Generate an execution plan based on dataset characteristics.
    
#     This is a basic implementation. Students should extend this significantly.
    
#     Args:
#         dataset_profile: Dictionary containing dataset metadata including:
#             - shape: {rows: int, cols: int}
#             - feature_types: {numeric: List[str], categorical: List[str]}
#             - imbalance_ratio: float (majority/minority class ratio)
#             - missing_pct: Dict[str, float] (missing % per column)
#             - is_classification: bool
#             - notes: List[str] (warnings/observations)
#         memory_hint: Optional dict with info from previous runs on similar datasets
    
#     Returns:
#         List of task names representing the execution plan
        
#     Example:
#         >>> profile = {"shape": {"rows": 5000}, "imbalance_ratio": 4.5}
#         >>> plan = create_plan(profile)
#         >>> print(plan)
#         ['profile_dataset', 'consider_imbalance_strategy', 'train_models', ...]
    
#     TODO for students:
#     - Implement conditional logic based on dataset size
#     - Add different strategies for imbalanced datasets
#     - Handle high-cardinality categorical features
#     - Use memory hints to prioritize successful models
#     - Create plan templates for common scenarios
#     - Add preprocessing steps based on data quality
#     """
    
#     # Basic plan structure (students should make this much more sophisticated)
#     plan: List[str] = [
#         "profile_dataset",
#         "build_preprocessor",
#         "select_models",
#         "train_models",
#         "evaluate",
#         "reflect",
#         "write_report",
#     ]
    
#     # TODO: Add sophisticated logic here
#     # Example: Check for imbalance
#     imb = dataset_profile.get("imbalance_ratio") or 1.0
#     if imb >= 3.0:
#         # TODO: Make this more sophisticated
#         # Consider: SMOTE, class weights, threshold tuning, etc.
#         plan.insert(plan.index("train_models"), "consider_imbalance_strategy")
    
#     # TODO: Add logic for small datasets
#     # if dataset_profile["shape"]["rows"] < 1000:
#     #     plan.append("apply_regularization")
    
#     # TODO: Add logic for high-cardinality categoricals
#     # high_card_cats = [c for c in categorical_cols if n_unique[c] > 50]
#     # if high_card_cats:
#     #     plan.insert(..., "apply_target_encoding")
    
#     # TODO: Use memory hints
#     # if memory_hint and memory_hint.get("best_model"):
#     #     plan.append(f"prioritize_model:{memory_hint['best_model']}")
    
#     # TODO: Add logic based on missing data
#     # max_missing = max(dataset_profile["missing_pct"].values())
#     # if max_missing > 20:
#     #     plan.insert(..., "handle_severe_missing_data")
    
#     return plan


# # TODO: Add helper functions for planning
# # def create_small_dataset_plan(...):
# # def create_imbalanced_dataset_plan(...):
# # def create_high_dimensional_plan(...):
# # def select_preprocessing_strategy(...):
# # def estimate_plan_cost(...):  # For cost-aware planning


from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------
# 1. PREPROCESSING STRATEGY (CORE INTELLIGENCE)
# ------------------------------------------------------------------

def get_preprocessing_tasks(profile: Dict[str, Any]) -> List[str]:
    tasks: List[str] = []

    # Missing data
    missing = profile.get("missing_pct", {})
    if missing:
        max_missing = max(missing.values())
        if max_missing > 30:
            tasks.append("handle_severe_missing_data")
        elif max_missing > 0:
            tasks.append("impute_missing_values")

    # Constant features
    if profile.get("constant_features"):
        tasks.append("drop_constant_features")

    if profile.get("near_constant_features"):
        tasks.append("drop_near_constant_features")

    # High-cardinality categoricals
    if profile.get("high_cardinality_categoricals"):
        tasks.append("apply_target_encoding")
    else:
        tasks.append("apply_one_hot_encoding")

    # Skewness
    if profile.get("highly_skewed_features"):
        tasks.append("apply_power_transform")

    # Outliers
    outliers = profile.get("outliers", {})
    if outliers.get("any_severe"):
        tasks.append("use_robust_scaler")
    else:
        tasks.append("use_standard_scaler")

    # Correlation
    if profile.get("high_correlation_pairs"):
        tasks.append("reduce_multicollinearity")

    return tasks


# ------------------------------------------------------------------
# 2. PLAN TEMPLATES (DATA PERSONAS)
# ------------------------------------------------------------------

def select_plan_template(profile: Dict[str, Any]) -> List[str]:
    rows = profile["shape"]["rows"]
    cols = profile["shape"]["cols"]

    # Small dataset → avoid overfitting
    if rows < 1000:
        return [
            "profile_dataset",
            "build_preprocessor",
            "cross_validation",
            "select_simple_models",
            "train_models",
            "evaluate",
            "reflect",
            "write_report",
        ]

    # High dimensional → feature reduction needed
    if cols > 100:
        return [
            "profile_dataset",
            "feature_selection",
            "dimensionality_reduction",
            "build_preprocessor",
            "select_models",
            "train_models",
            "evaluate",
            "reflect",
            "write_report",
        ]

    # Default
    return [
        "profile_dataset",
        "build_preprocessor",
        "select_models",
        "train_models",
        "evaluate",
        "reflect",
        "write_report",
    ]


# ------------------------------------------------------------------
# 3. MEMORY-GUIDED PLANNING
# ------------------------------------------------------------------

def apply_memory_guidance(
    plan: List[str], memory_hint: Optional[Dict[str, Any]]
) -> List[str]:

    if not memory_hint:
        return plan

    best_model = memory_hint.get("best_model")

    if best_model and "train_models" in plan:
        idx = plan.index("train_models")

        plan.insert(idx, f"validate_previous_model:{best_model}")
        plan.insert(idx + 1, "conditional_retrain")

    return plan


# ------------------------------------------------------------------
# 4. DEPENDENCY MANAGEMENT
# ------------------------------------------------------------------

def validate_task_order(plan: List[str]) -> List[str]:
    if "train_models" not in plan:
        return plan

    train_idx = plan.index("train_models")

    preprocessing_tasks = [
        "impute_missing_values",
        "apply_power_transform",
        "apply_one_hot_encoding",
        "apply_target_encoding",
        "use_standard_scaler",
        "use_robust_scaler",
    ]

    for task in preprocessing_tasks:
        if task in plan and plan.index(task) > train_idx:
            plan.remove(task)
            plan.insert(train_idx, task)

    return plan


# ------------------------------------------------------------------
# 5. MAIN FUNCTION (EXTENDED)
# ------------------------------------------------------------------

def create_plan(
    dataset_profile: Dict[str, Any],
    memory_hint: Optional[Dict[str, Any]] = None
) -> List[str]:

    # 1. Choose base template
    plan = select_plan_template(dataset_profile)

    # 2. Inject preprocessing tasks
    preprocessing_tasks = get_preprocessing_tasks(dataset_profile)

    if "build_preprocessor" in plan:
        insert_idx = plan.index("build_preprocessor") + 1
        for task in reversed(preprocessing_tasks):
            plan.insert(insert_idx, task)

    # 3. Handle imbalance (advanced logic)
    imb = dataset_profile.get("imbalance_ratio") or 1.0
    rows = dataset_profile["shape"]["rows"]

    if imb >= 3.0:
        if rows < 5000:
            plan.insert(plan.index("train_models"), "apply_smote")
        else:
            plan.insert(plan.index("train_models"), "use_class_weights")

    # 4. Model strategy decisions
    cols = dataset_profile["shape"]["cols"]

    if cols > 100 and "select_models" in plan:
        plan.insert(plan.index("select_models"), "prefer_linear_models")

    if rows < 1000 and "select_simple_models" in plan:
        plan.insert(plan.index("select_simple_models"), "prefer_simple_models")

    # 5. Memory guidance
    plan = apply_memory_guidance(plan, memory_hint)

    # 6. Validate ordering
    plan = validate_task_order(plan)

    return plan