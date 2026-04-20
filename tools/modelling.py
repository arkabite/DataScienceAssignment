# from typing import Any, Dict, List, Tuple

# import os
# import pandas as pd
# import numpy as np

# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, RobustScaler
# from sklearn.feature_selection import VarianceThreshold

# from sklearn.dummy import DummyClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC

# from sklearn.metrics import (
#     accuracy_score,
#     balanced_accuracy_score,
#     f1_score,
#     precision_score,
#     recall_score,
# )


# # def build_preprocessor(profile: Dict[str, Any]) -> ColumnTransformer:
# #     num_cols = profile["feature_types"]["numeric"]
# #     cat_cols = profile["feature_types"]["categorical"]

# #     numeric_transformer = Pipeline(steps=[
# #         ("imputer", SimpleImputer(strategy="median")),
# #         ("scaler", StandardScaler(with_mean=True)),
# #     ])

# #     # scikit-learn renamed `sparse` -> `sparse_output` (v1.2+). Support both.
# #     try:
# #         ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# #     except TypeError:
# #         ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

# #     categorical_transformer = Pipeline(steps=[
# #         ("imputer", SimpleImputer(strategy="most_frequent")),
# #         ("onehot", ohe),
# #     ])

# #     return ColumnTransformer(
# #         transformers=[
# #             ("num", numeric_transformer, num_cols),
# #             ("cat", categorical_transformer, cat_cols),
# #         ],
# #         remainder="drop",
# #     )

# def build_preprocessor(profile: Dict[str, Any], plan: Optional[List[str]] = None) -> ColumnTransformer:
#     plan = plan or []

#     num_cols = profile["feature_types"]["numeric"]
#     cat_cols = profile["feature_types"]["categorical"]

#     # -----------------------------
#     # SCALER SELECTION
#     # -----------------------------
#     if "use_robust_scaler" in plan:
#         scaler = RobustScaler()
#     else:
#         scaler = StandardScaler(with_mean=True)

#     numeric_steps = [
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", scaler),
#     ]

#     # -----------------------------
#     # CORRELATION / REDUCTION
#     # -----------------------------
#     if "reduce_multicollinearity" in plan:
#         numeric_steps.append(("variance_filter", VarianceThreshold(threshold=0.0)))

#     numeric_transformer = Pipeline(steps=numeric_steps)

#     # -----------------------------
#     # ENCODING STRATEGY
#     # -----------------------------
#     try:
#         ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
#     except TypeError:
#         ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

#     categorical_steps = [
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#     ]

#     if "apply_target_encoding" in plan:
#         # fallback (since target encoding not implemented)
#         categorical_steps.append(("onehot_fallback", ohe))
#     else:
#         categorical_steps.append(("onehot", ohe))

#     categorical_transformer = Pipeline(steps=categorical_steps)

#     return ColumnTransformer(
#         transformers=[
#             ("num", numeric_transformer, num_cols),
#             ("cat", categorical_transformer, cat_cols),
#         ],
#         remainder="drop",
#     )


# # def select_models(profile: Dict[str, Any], seed: int = 42) -> List[Tuple[str, Any]]:
# #     rows = profile["shape"]["rows"]
# #     cols = profile["shape"]["cols"]
# #     imb = float(profile.get("imbalance_ratio") or 1.0)
# #     class_weight = "balanced" if imb >= 3.0 else None

# #     candidates: List[Tuple[str, Any]] = [
# #         ("DummyMostFrequent", DummyClassifier(strategy="most_frequent")),
# #         ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight=class_weight)),
# #         ("RandomForest", RandomForestClassifier(
# #             n_estimators=300, random_state=seed, n_jobs=-1, class_weight=class_weight
# #         )),
# #     ]

# #     if rows <= 50000:
# #         candidates.append(("GradientBoosting", GradientBoostingClassifier(random_state=seed)))

# #     # SVC can be expensive after one-hot; keep for smaller problems
# #     if rows <= 20000 and cols <= 200:
# #         candidates.append(("SVC_RBF", SVC(kernel="rbf", probability=True, class_weight=class_weight)))

# #     return candidates

# def select_models(
#     profile: Dict[str, Any],
#     plan: Optional[List[str]] = None,
#     seed: int = 42
# ) -> List[Tuple[str, Any]]:

#     plan = plan or []

#     rows = profile["shape"]["rows"]
#     cols = profile["shape"]["cols"]
#     imb = float(profile.get("imbalance_ratio") or 1.0)
#     class_weight = "balanced" if imb >= 3.0 else None

#     # -----------------------------
#     # BASE MODELS
#     # -----------------------------
#     simple_models = [
#         ("DummyMostFrequent", DummyClassifier(strategy="most_frequent")),
#         ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight=class_weight)),
#     ]

#     complex_models = [
#         ("RandomForest", RandomForestClassifier(
#             n_estimators=300, random_state=seed, n_jobs=-1, class_weight=class_weight
#         )),
#         ("GradientBoosting", GradientBoostingClassifier(random_state=seed)),
#         ("SVC_RBF", SVC(kernel="rbf", probability=True, class_weight=class_weight)),
#     ]

#     # -----------------------------
#     # PLAN-BASED FILTERING
#     # -----------------------------

#     if "prefer_simple_models" in plan:
#         return simple_models

#     if "prefer_linear_models" in plan:
#         return [
#             ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight=class_weight))
#         ]

#     # -----------------------------
#     # DEFAULT BEHAVIOUR (original logic)
#     # -----------------------------
#     candidates: List[Tuple[str, Any]] = simple_models + [
#         complex_models[0]  # RandomForest
#     ]

#     if rows <= 50000:
#         candidates.append(complex_models[1])  # GradientBoosting

#     if rows <= 20000 and cols <= 200:
#         candidates.append(complex_models[2])  # SVC

#     return candidates


# def train_models(
#     df: pd.DataFrame,
#     target: str,
#     preprocessor: ColumnTransformer,
#     candidates: List[Tuple[str, Any]],
#     seed: int,
#     test_size: float,
#     output_dir: str,
#     verbose: bool = True,
# ) -> Dict[str, Any]:
#     if target not in df.columns:
#         raise ValueError(f"Target '{target}' not found.")

#     X = df.drop(columns=[target]).copy()
#     y = df[target].copy()

#     # Drop missing target rows
#     mask = ~y.isna()
#     X = X.loc[mask]
#     y = y.loc[mask]

#     # Stratify if possible
#     stratify = y if (y.nunique(dropna=True) > 1 and y.value_counts().min() >= 2) else None

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=seed, stratify=stratify
#     )

#     results: List[Dict[str, Any]] = []

#     for name, model in candidates:
#         if verbose:
#             print(f"[Modelling] Training: {name}")

#         pipe = Pipeline(steps=[
#             ("preprocess", preprocessor),
#             ("model", model),
#         ])
#         pipe.fit(X_train, y_train)

#         y_pred = pipe.predict(X_test)

#         metrics = {
#             "model": name,
#             "accuracy": float(accuracy_score(y_test, y_pred)),
#             "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
#             "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
#             "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
#             "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
#         }

#         results.append({
#             "name": name,
#             "pipeline": pipe,
#             "metrics": metrics,
#             "X_test": X_test,
#             "y_test": y_test,
#             "y_pred": y_pred,
#         })

#     # Sort by balanced accuracy then macro F1
#     results.sort(key=lambda r: (r["metrics"]["balanced_accuracy"], r["metrics"]["f1_macro"]), reverse=True)

#     return {
#         "results": results,
#         "best": results[0],
#         "all_metrics": [r["metrics"] for r in results],
#     }


from typing import Any, Dict, List, Tuple, Optional

import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# ------------------------------------------------------------------
# PREPROCESSOR (PLAN-AWARE)
# ------------------------------------------------------------------

def build_preprocessor(
    profile: Dict[str, Any],
    plan: Optional[List[str]] = None
) -> ColumnTransformer:

    plan = plan or []

    num_cols = profile["feature_types"]["numeric"]
    cat_cols = profile["feature_types"]["categorical"]

    # -----------------------------
    # SCALER SELECTION
    # -----------------------------
    if "use_robust_scaler" in plan:
        scaler = RobustScaler()
    else:
        scaler = StandardScaler(with_mean=True)

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ]

    # -----------------------------
    # BASIC FEATURE REDUCTION
    # NOTE: This is NOT true multicollinearity removal,
    # but a lightweight variance-based filter.
    # -----------------------------
    if "reduce_multicollinearity" in plan:
        numeric_steps.append(("variance_filter", VarianceThreshold(threshold=0.01)))

    numeric_transformer = Pipeline(steps=numeric_steps)

    # -----------------------------
    # ENCODING STRATEGY
    # -----------------------------
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ]

    if "apply_target_encoding" in plan:
        # Fallback since target encoding not implemented
        categorical_steps.append(("onehot_fallback", ohe))
    else:
        categorical_steps.append(("onehot", ohe))

    categorical_transformer = Pipeline(steps=categorical_steps)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )


# ------------------------------------------------------------------
# MODEL SELECTION (PLAN-AWARE)
# ------------------------------------------------------------------

def select_models(
    profile: Dict[str, Any],
    plan: Optional[List[str]] = None,
    seed: int = 42
) -> List[Tuple[str, Any]]:

    plan = plan or []

    rows = profile["shape"]["rows"]
    cols = profile["shape"]["cols"]
    imb = float(profile.get("imbalance_ratio") or 1.0)
    class_weight = "balanced" if imb >= 3.0 else None

    # -----------------------------
    # SIMPLE MODELS
    # -----------------------------
    simple_models = [
        ("DummyMostFrequent", DummyClassifier(strategy="most_frequent")),
        ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight=class_weight)),
    ]

    # -----------------------------
    # COMPLEX MODELS
    # -----------------------------
    complex_models = [
        ("RandomForest", RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1, class_weight=class_weight
        )),
        ("GradientBoosting", GradientBoostingClassifier(random_state=seed)),
        ("SVC_RBF", SVC(kernel="rbf", probability=True, class_weight=class_weight)),
    ]

    # -----------------------------
    # PLAN-BASED FILTERING
    # -----------------------------

    # Prefer simple models but KEEP a strong baseline
    if "prefer_simple_models" in plan:
        return simple_models + [complex_models[0]]  # keep RandomForest

    # Prefer linear only
    if "prefer_linear_models" in plan:
        return [
            ("LogisticRegression", LogisticRegression(max_iter=2000, class_weight=class_weight))
        ]

    # -----------------------------
    # DEFAULT LOGIC (ADAPTIVE)
    # -----------------------------
    candidates: List[Tuple[str, Any]] = simple_models + [complex_models[0]]

    if rows <= 50000:
        candidates.append(complex_models[1])  # GradientBoosting

    if rows <= 20000 and cols <= 200:
        candidates.append(complex_models[2])  # SVC

    return candidates


# ------------------------------------------------------------------
# TRAINING + EVALUATION
# ------------------------------------------------------------------

def train_models(
    df: pd.DataFrame,
    target: str,
    preprocessor: ColumnTransformer,
    candidates: List[Tuple[str, Any]],
    seed: int,
    test_size: float,
    output_dir: str,
    verbose: bool = True,
) -> Dict[str, Any]:

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")

    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # Drop missing target rows
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Stratify if possible
    stratify = y if (y.nunique(dropna=True) > 1 and y.value_counts().min() >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    results: List[Dict[str, Any]] = []

    for name, model in candidates:
        if verbose:
            print(f"[Modelling] Training: {name}")

        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        }

        results.append({
            "name": name,
            "pipeline": pipe,
            "metrics": metrics,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
        })

    # Sort by best performance
    results.sort(
        key=lambda r: (
            r["metrics"]["balanced_accuracy"],
            r["metrics"]["f1_macro"]
        ),
        reverse=True
    )

    return {
        "results": results,
        "best": results[0],
        "all_metrics": [r["metrics"] for r in results],
    }