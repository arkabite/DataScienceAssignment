# from typing import Any, Dict, Optional
# import pandas as pd


# def infer_target_column(df: pd.DataFrame) -> Optional[str]:
#     """
#     Heuristic target inference:
#       - prefer common target-like column names
#       - else last column if it has relatively low cardinality
#     """
#     candidates = ["target", "label", "class", "y", "outcome"]
#     lower_map = {c.lower(): c for c in df.columns}
#     for k in candidates:
#         if k in lower_map:
#             return lower_map[k]

#     last = df.columns[-1]
#     uniq = df[last].nunique(dropna=True)
#     n = len(df)
#     if n > 0 and (uniq <= 50 or (uniq / max(n, 1) < 0.05)):
#         return last
#     return None


# def is_classification_target(series: pd.Series) -> bool:
#     if series.dtype == "object" or str(series.dtype).startswith("category"):
#         return True
#     uniq = series.nunique(dropna=True)
#     return uniq <= 50


# def dataset_fingerprint(df: pd.DataFrame, target: str) -> str:
#     cols = ",".join(df.columns.astype(str).tolist())
#     shape = f"{df.shape[0]}x{df.shape[1]}"
#     base = f"{shape}|{target}|{cols}"
#     h = abs(hash(base)) % (10**12)
#     return f"fp_{h}"


# def profile_dataset(df: pd.DataFrame, target: str) -> Dict[str, Any]:
#     if target not in df.columns:
#         raise ValueError(f"Target column '{target}' not found in dataset columns.")

#     y = df[target]
#     profile: Dict[str, Any] = {}

#     profile["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
#     profile["columns"] = df.columns.astype(str).tolist()

#     missing = (df.isna().mean() * 100).round(2).to_dict()
#     profile["missing_pct"] = {str(k): float(v) for k, v in missing.items()}

#     profile["target"] = str(target)
#     profile["target_dtype"] = str(y.dtype)
#     profile["is_classification"] = bool(is_classification_target(y))

#     # Feature types
#     X = df.drop(columns=[target])
#     numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.astype(str).tolist()
#     cat_cols = [c for c in X.columns.astype(str).tolist() if c not in numeric_cols]

#     profile["feature_types"] = {"numeric": numeric_cols, "categorical": cat_cols}
#     profile["n_unique_by_col"] = {str(c): int(df[c].nunique(dropna=True)) for c in df.columns.astype(str)}

#     notes = []
#     if profile["shape"]["rows"] < 1000:
#         notes.append("Small dataset (<1000 rows): prefer simpler models / guard against overfitting.")
#     if profile["shape"]["cols"] > 100:
#         notes.append("High dimensionality (>100 columns): watch one-hot expansion and overfitting.")
#     profile["notes"] = notes

#     # Class balance if classification
#     if profile["is_classification"]:
#         vc = y.value_counts(dropna=False)
#         profile["class_counts"] = {str(k): int(v) for k, v in vc.items()}
#         if len(vc) >= 2:
#             ratio = float(vc.max() / max(vc.min(), 1))
#         else:
#             ratio = 1.0
#         profile["imbalance_ratio"] = round(ratio, 3)
#         if ratio >= 3.0:
#             profile["notes"].append("Imbalance detected (ratio >= 3.0): prioritise macro metrics / balanced accuracy.")
#     else:
#         profile["class_counts"] = None
#         profile["imbalance_ratio"] = None
#         profile["notes"].append("Non-classification target detected: this template focuses on classification.")

#     return profile

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers (new)
# ---------------------------------------------------------------------------

def _constant_and_near_constant_features(
    X: pd.DataFrame, threshold: float = 0.99
) -> Dict[str, List[str]]:
    """
    Detect columns whose most-frequent value covers >= `threshold` of rows.
    Constant  : coverage == 1.0  (a single unique value)
    Near-const: threshold <= coverage < 1.0
    """
    constant, near_constant = [], []
    n = len(X)
    if n == 0:
        return {"constant": constant, "near_constant": near_constant}

    for col in X.columns:
        top_freq = X[col].value_counts(dropna=False).iloc[0] / n
        if top_freq >= 1.0:
            constant.append(str(col))
        elif top_freq >= threshold:
            near_constant.append(str(col))

    return {"constant": constant, "near_constant": near_constant}


def _high_cardinality_categoricals(
    X: pd.DataFrame, cat_cols: List[str], max_unique: int = 50
) -> List[str]:
    """
    Flag categorical columns whose unique-value count exceeds `max_unique`.
    These will explode OneHotEncoder and should be dropped or target-encoded.
    """
    flagged = []
    for col in cat_cols:
        if col in X.columns and X[col].nunique(dropna=True) > max_unique:
            flagged.append(str(col))
    return flagged


def _skewness(X: pd.DataFrame, num_cols: List[str]) -> Dict[str, float]:
    """
    Return skewness for each numeric column (rounded to 3 dp).
    |skew| > 1  → highly skewed  (log / PowerTransformer recommended)
    0.5–1       → moderately skewed
    """
    result = {}
    for col in num_cols:
        if col in X.columns:
            try:
                result[str(col)] = round(float(X[col].skew()), 3)
            except Exception:
                result[str(col)] = float("nan")
    return result


def _outlier_summary(
    X: pd.DataFrame, num_cols: List[str], iqr_factor: float = 3.0
) -> Dict[str, Any]:
    """
    IQR-based outlier detection.  Uses a wider fence (3 × IQR) to flag only
    extreme outliers rather than mild ones.

    Returns for each numeric column:
      - outlier_count : int
      - outlier_pct   : float  (% of non-null rows)
    Also returns `any_severe` True if any column has >1 % extreme outliers,
    suggesting RobustScaler over StandardScaler.
    """
    per_column: Dict[str, Dict[str, Any]] = {}
    any_severe = False

    for col in num_cols:
        if col not in X.columns:
            continue
        s = X[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
        count = int(((s < lower) | (s > upper)).sum())
        pct = round(count / len(s) * 100, 2)
        per_column[str(col)] = {"outlier_count": count, "outlier_pct": pct}
        if pct > 1.0:
            any_severe = True

    return {"per_column": per_column, "any_severe": bool(any_severe)}


def _correlation_pairs(
    X: pd.DataFrame, num_cols: List[str], threshold: float = 0.90
) -> List[Dict[str, Any]]:
    """
    Return pairs of numeric features whose absolute Pearson correlation
    exceeds `threshold`.  Useful for PCA / feature-dropping decisions.
    """
    if len(num_cols) < 2:
        return []

    sub = X[num_cols].select_dtypes(include="number")
    if sub.shape[1] < 2:
        return []

    corr = sub.corr(method="pearson").abs()
    # Upper triangle only (no self-pairs)
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val) and val >= threshold:
                pairs.append({
                    "feature_a": str(cols[i]),
                    "feature_b": str(cols[j]),
                    "correlation": round(float(val), 4),
                })
    pairs.sort(key=lambda p: p["correlation"], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Original public API — signatures unchanged
# ---------------------------------------------------------------------------

def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristic target inference:
      - prefer common target-like column names
      - else last column if it has relatively low cardinality
    """
    candidates = ["target", "label", "class", "y", "outcome"]
    lower_map = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in lower_map:
            return lower_map[k]

    last = df.columns[-1]
    uniq = df[last].nunique(dropna=True)
    n = len(df)
    if n > 0 and (uniq <= 50 or (uniq / max(n, 1) < 0.05)):
        return last
    return None


def is_classification_target(series: pd.Series) -> bool:
    if series.dtype == "object" or str(series.dtype).startswith("category"):
        return True
    uniq = series.nunique(dropna=True)
    return uniq <= 50


def dataset_fingerprint(df: pd.DataFrame, target: str) -> str:
    cols = ",".join(df.columns.astype(str).tolist())
    shape = f"{df.shape[0]}x{df.shape[1]}"
    base = f"{shape}|{target}|{cols}"
    h = abs(hash(base)) % (10**12)
    return f"fp_{h}"


def profile_dataset(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset columns.")

    y = df[target]
    profile: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. Basic metadata  (unchanged)
    # ------------------------------------------------------------------
    profile["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    profile["columns"] = df.columns.astype(str).tolist()

    missing = (df.isna().mean() * 100).round(2).to_dict()
    profile["missing_pct"] = {str(k): float(v) for k, v in missing.items()}

    profile["target"] = str(target)
    profile["target_dtype"] = str(y.dtype)
    profile["is_classification"] = bool(is_classification_target(y))

    X = df.drop(columns=[target])
    numeric_cols: List[str] = X.select_dtypes(include=["number", "bool"]).columns.astype(str).tolist()
    cat_cols: List[str] = [c for c in X.columns.astype(str).tolist() if c not in numeric_cols]

    profile["feature_types"] = {"numeric": numeric_cols, "categorical": cat_cols}
    profile["n_unique_by_col"] = {str(c): int(df[c].nunique(dropna=True)) for c in df.columns.astype(str)}

    notes: List[str] = []
    if profile["shape"]["rows"] < 1000:
        notes.append("Small dataset (<1000 rows): prefer simpler models / guard against overfitting.")
    if profile["shape"]["cols"] > 100:
        notes.append("High dimensionality (>100 columns): watch one-hot expansion and overfitting.")

    # ------------------------------------------------------------------
    # 2. Class balance  (unchanged)
    # ------------------------------------------------------------------
    if profile["is_classification"]:
        vc = y.value_counts(dropna=False)
        profile["class_counts"] = {str(k): int(v) for k, v in vc.items()}
        ratio = float(vc.max() / max(vc.min(), 1)) if len(vc) >= 2 else 1.0
        profile["imbalance_ratio"] = round(ratio, 3)
        if ratio >= 3.0:
            notes.append("Imbalance detected (ratio >= 3.0): prioritise macro metrics / balanced accuracy.")
    else:
        profile["class_counts"] = None
        profile["imbalance_ratio"] = None
        notes.append("Non-classification target detected: this template focuses on classification.")

    # ------------------------------------------------------------------
    # 3. NEW — Constant / near-constant features
    # ------------------------------------------------------------------
    variance_check = _constant_and_near_constant_features(X, threshold=0.99)
    profile["constant_features"] = variance_check["constant"]
    profile["near_constant_features"] = variance_check["near_constant"]

    if variance_check["constant"]:
        notes.append(
            f"Constant features detected (drop before modelling): "
            f"{variance_check['constant']}"
        )
    if variance_check["near_constant"]:
        notes.append(
            f"Near-constant features (≥99 % same value, consider dropping): "
            f"{variance_check['near_constant']}"
        )

    # ------------------------------------------------------------------
    # 4. NEW — High-cardinality categorical columns
    # ------------------------------------------------------------------
    high_card_cats = _high_cardinality_categoricals(X, cat_cols, max_unique=50)
    profile["high_cardinality_categoricals"] = high_card_cats

    if high_card_cats:
        notes.append(
            f"High-cardinality categoricals (>50 unique values, will explode OHE): "
            f"{high_card_cats}. Consider target-encoding or dropping."
        )

    # ------------------------------------------------------------------
    # 5. NEW — Skewness of numeric features
    # ------------------------------------------------------------------
    skew_by_col = _skewness(X, numeric_cols)
    profile["skewness"] = skew_by_col

    highly_skewed = [c for c, s in skew_by_col.items() if abs(s) > 1.0]
    moderately_skewed = [c for c, s in skew_by_col.items() if 0.5 < abs(s) <= 1.0]
    profile["highly_skewed_features"] = highly_skewed
    profile["moderately_skewed_features"] = moderately_skewed

    if highly_skewed:
        notes.append(
            f"Highly skewed numeric features (|skew|>1, consider PowerTransformer/log): "
            f"{highly_skewed}"
        )
    if moderately_skewed:
        notes.append(
            f"Moderately skewed numeric features (0.5<|skew|≤1): "
            f"{moderately_skewed}"
        )

    # ------------------------------------------------------------------
    # 6. NEW — Outlier detection (IQR ×3 fence)
    # ------------------------------------------------------------------
    outlier_info = _outlier_summary(X, numeric_cols, iqr_factor=3.0)
    profile["outliers"] = outlier_info

    if outlier_info["any_severe"]:
        severe_cols = [
            c for c, v in outlier_info["per_column"].items() if v["outlier_pct"] > 1.0
        ]
        notes.append(
            f"Extreme outliers detected (>1 % of rows beyond 3×IQR fence) in: "
            f"{severe_cols}. Consider RobustScaler instead of StandardScaler."
        )

    # ------------------------------------------------------------------
    # 7. NEW — High inter-feature correlation (redundancy check)
    # ------------------------------------------------------------------
    corr_pairs = _correlation_pairs(X, numeric_cols, threshold=0.90)
    profile["high_correlation_pairs"] = corr_pairs

    if corr_pairs:
        notes.append(
            f"{len(corr_pairs)} highly correlated feature pair(s) found (|r|≥0.90). "
            f"Data may benefit from PCA or feature dropping to reduce redundancy."
        )

    profile["notes"] = notes
    return profile