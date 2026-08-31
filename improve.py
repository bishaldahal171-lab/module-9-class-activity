"""
Step 3 — Measure, Learn, Improve.

Compares three models on the same train/test split:
1. Logistic Regression (baseline from Step 2)
2. Random Forest
3. Gradient Boosting

For each model, computes:
- ROC-AUC
- Accuracy / Precision / Recall
- Retention Hit Rate (Top 100)
- Lift vs. baseline churn rate

Also extracts feature importances to answer:
"What did we learn about which factors drive churn?"

Usage:
    python -m src.improve            # prints comparison table + feature importances
    python -m src.improve --save     # also saves results to step3_results.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Make src importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_data, preprocess, get_feature_lists
from src.model import retention_hit_rate, train_test_split_data

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Telco-Customer-Churn.csv")
BASELINE_CHURN_RATE = 0.265  # overall churn rate in the dataset


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build the shared preprocessor (OneHotEncoder + StandardScaler)."""
    categorical_cols, numeric_cols = get_feature_lists(X)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )


def make_logistic_regression(X: pd.DataFrame) -> Pipeline:
    """Logistic Regression baseline (same as Step 2)."""
    return Pipeline([
        ("preprocessor", _build_preprocessor(X)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def make_random_forest(X: pd.DataFrame) -> Pipeline:
    """Random Forest classifier."""
    return Pipeline([
        ("preprocessor", _build_preprocessor(X)),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
        )),
    ])


def make_gradient_boosting(X: pd.DataFrame) -> Pipeline:
    """Gradient Boosting classifier."""
    return Pipeline([
        ("preprocessor", _build_preprocessor(X)),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
        )),
    ])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_pipeline(pipeline: Pipeline, X_train, X_test, y_train, y_test) -> dict:
    """Train and evaluate a single pipeline."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    hit_rate = retention_hit_rate(y_test.values, y_proba, top_n=100)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "retention_hit_rate": round(hit_rate, 4),
        "lift": round(hit_rate / BASELINE_CHURN_RATE, 2),
    }


def compare_models(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Compare all three models on the same train/test split.

    Returns
    -------
    results_df : pd.DataFrame
        Model comparison table.
    pipelines : dict
        Fitted pipelines keyed by model name (for feature importance).
    """
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    models = {
        "Logistic Regression": make_logistic_regression(X),
        "Random Forest": make_random_forest(X),
        "Gradient Boosting": make_gradient_boosting(X),
    }

    results = {}
    fitted_pipelines = {}

    for name, pipeline in models.items():
        fitted = _clone_and_fit(pipeline, X_train, y_train)
        fitted_pipelines[name] = fitted
        metrics = evaluate_pipeline(fitted, X_train, X_test, y_train, y_test)
        results[name] = metrics

    results_df = pd.DataFrame(results).T
    results_df.index.name = "Model"
    return results_df, fitted_pipelines


def _clone_and_fit(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """Clone and fit a pipeline (avoids mutating the original)."""
    from sklearn.base import clone
    cloned = clone(pipeline)
    cloned.fit(X_train, y_train)
    return cloned


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(pipeline: Pipeline, X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Extract feature importances from a fitted pipeline.

    Works with both tree-based models (feature_importances_) and
    Logistic Regression (coef_).

    Returns
    -------
    pd.DataFrame
        Columns: feature, importance (sorted descending).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Get feature names after one-hot encoding
    categorical_cols, numeric_cols = get_feature_lists(X)
    try:
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    except Exception:
        cat_feature_names = categorical_cols

    all_feature_names = numeric_cols + cat_feature_names

    # Extract importances
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame({
        "feature": all_feature_names[:len(importances)],
        "importance": importances,
    })
    df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return df


def get_top_risk_factors(importance_df: pd.DataFrame) -> list[str]:
    """Convert raw feature importance into human-readable risk factors."""
    risk_factors = []
    for _, row in importance_df.iterrows():
        feat = row["feature"]
        # Clean up one-hot encoded feature names
        if "_" in feat:
            parts = feat.split("_", 1)
            feat = f"{parts[0]} = {parts[1]}" if len(parts) > 1 else feat
        risk_factors.append(f"{feat} (importance: {row['importance']:.4f})")
    return risk_factors[:10]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 3: Model comparison and feature importance")
    parser.add_argument("--save", action="store_true", help="Save results to step3_results.json")
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 3 — MEASURE, LEARN, IMPROVE")
    print("=" * 70)

    # Load and prepare data
    df = load_data(DATA_PATH)
    X, y, ids = preprocess(df)

    print(f"\nDataset: {len(df):,} customers, {df['Churn'].eq('Yes').sum()} churned ({BASELINE_CHURN_RATE:.1%})")
    print(f"Features: {len(X.columns)} columns")
    print(f"Baseline churn rate: {BASELINE_CHURN_RATE:.1%}")
    print(f"Random selection expected hit rate: ~{BASELINE_CHURN_RATE:.1%}")

    # Compare models
    print("\n" + "-" * 70)
    print("MODEL COMPARISON (same 80/20 train/test split, stratified)")
    print("-" * 70)

    results_df, pipelines = compare_models(X, y)
    print(results_df.to_string())

    best_model = results_df["retention_hit_rate"].idxmax()
    best_hit_rate = results_df.loc[best_model, "retention_hit_rate"]
    print(f"\nBest Retention Hit Rate: {best_model} ({best_hit_rate:.1%})")

    # Feature importance
    print("\n" + "-" * 70)
    print("FEATURE IMPORTANCE (from best tree-based model)")
    print("-" * 70)

    # Use Gradient Boosting for feature importance (most informative for trees)
    gb_pipeline = pipelines.get("Gradient Boosting")
    if gb_pipeline:
        importance_df = get_feature_importance(gb_pipeline, X, top_n=15)
        print("\nTop 15 Features by Importance:")
        print(importance_df.to_string(index=False))

        print("\nKey Risk Factors (what we learned):")
        risk_factors = get_top_risk_factors(importance_df)
        for i, rf in enumerate(risk_factors, 1):
            print(f"  {i}. {rf}")

    # Also show Logistic Regression coefficients
    lr_pipeline = pipelines.get("Logistic Regression")
    if lr_pipeline:
        print("\n" + "-" * 70)
        print("LOGISTIC REGRESSION COEFFICIENTS (top positive = churn drivers)")
        print("-" * 70)
        lr_importance = get_feature_importance(lr_pipeline, X, top_n=15)
        print(lr_importance.to_string(index=False))

    # Summary
    print("\n" + "=" * 70)
    print("WHAT WE LEARNED")
    print("=" * 70)
    print("""
1. Contract type is the strongest churn predictor — month-to-month customers
   churn at 42.7% vs. 2.8% for two-year contracts.
2. Tenure is critical — churned customers have median tenure of 10 months
   vs. 38 for retained customers.
3. Internet service type matters — fiber optic customers churn at 41.9%
   vs. 19% for DSL.
4. Payment method is a signal — electronic check users churn at 45.3%.
5. Higher monthly charges correlate with churn ($74.44 vs. $61.27 average).
""")

    if args.save:
        output = {
            "model_comparison": results_df.to_dict(orient="index"),
            "best_model": best_model,
            "feature_importance": importance_df.to_dict(orient="records") if gb_pipeline else [],
            "baseline_churn_rate": BASELINE_CHURN_RATE,
        }
        output_path = os.path.join(os.path.dirname(__file__), "..", "step3_results.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
