"""
Churn prediction model for the Telco Customer Retention MVP.

Supports multiple classifiers selected via the ``model_type`` parameter:
- ``"gradient_boosting"`` (default, best Retention Hit Rate)
- ``"logistic_regression"`` (most interpretable)
- ``"random_forest"``

All models share the same preprocessing pipeline:
- OneHotEncoder for categorical features
- StandardScaler for numeric features
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import get_feature_lists

# Available model types
MODEL_TYPES = ("gradient_boosting", "logistic_regression", "random_forest")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build the shared preprocessor (OneHotEncoder + StandardScaler).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (used to detect column types).

    Returns
    -------
    ColumnTransformer
    """
    categorical_cols, numeric_cols = get_feature_lists(X)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )


def _make_classifier(model_type: str):
    """Create a classifier instance for the given model type."""
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
        )
    elif model_type == "logistic_regression":
        return LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from {MODEL_TYPES}."
        )


def build_pipeline(X: pd.DataFrame, model_type: str = "gradient_boosting") -> Pipeline:
    """Build a preprocessing + classification pipeline.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (used to detect column types).
    model_type : str
        Which classifier to use. One of ``MODEL_TYPES``.
        Default: ``"gradient_boosting"`` (best Retention Hit Rate).

    Returns
    -------
    Pipeline
        A scikit-learn Pipeline ready to fit.
    """
    preprocessor = build_preprocessor(X)
    classifier = _make_classifier(model_type)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_model(
    X: pd.DataFrame, y: pd.Series, model_type: str = "gradient_boosting"
) -> Pipeline:
    """Train the churn prediction model on the given data.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target (1 = churned).
    model_type : str
        Which classifier to use. Default: ``"gradient_boosting"``.

    Returns
    -------
    Pipeline
        Fitted pipeline.
    """
    pipeline = build_pipeline(X, model_type=model_type)
    pipeline.fit(X, y)
    return pipeline


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate the trained model on test data.

    Computes accuracy, precision, recall, ROC-AUC, and the Retention Hit Rate
    (how many actual churners are in the Top 100 predicted-risk customers).

    Parameters
    ----------
    model : Pipeline
        Fitted pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "retention_hit_rate": retention_hit_rate(y_test.values, y_proba, top_n=100),
    }
    return metrics


def retention_hit_rate(y_true: np.ndarray, y_scores: np.ndarray, top_n: int = 100) -> float:
    """Compute the Retention Hit Rate.

    Retention Hit Rate = (actual churners in Top N) / N

    This is equivalent to Precision@N — of the N customers with the highest
    predicted churn probability, what fraction actually churned?

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (1 = churned).
    y_scores : np.ndarray
        Predicted churn probabilities.
    top_n : int
        Number of top-ranked customers to consider (default 100).

    Returns
    -------
    float
        Hit rate between 0 and 1.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    n = min(top_n, len(y_scores))
    # Get indices of the top_n highest probability customers
    top_indices = np.argsort(y_scores)[::-1][:n]
    # Count how many of those actually churned
    hits = y_true[top_indices].sum()

    return hits / n


def train_test_split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target.
    test_size : float
        Fraction of data for testing (default 0.2).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
