"""
Pytest tests for the Telco Customer Churn MVP.

Run with:
    pytest tests/ -v

These tests verify:
- Data loads correctly with expected columns
- TotalCharges converts to numeric (no empty strings)
- Model trains and produces valid probabilities
- retention_hit_rate returns expected values on toy examples
- Top customers are sorted descending and limited to N
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_data, preprocess, get_feature_lists
from src.model import build_pipeline, train_model, evaluate_model, retention_hit_rate, train_test_split_data
from src.predict import score_customers, get_top_customers, summarize_top_100

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Telco-Customer-Churn.csv")


# ---------------------------------------------------------------------------
# Data loader tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df_raw():
    """Load raw data once for all tests in this module."""
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def prepared(df_raw):
    """Preprocess data into X, y, customer_ids."""
    return preprocess(df_raw)


def test_data_has_expected_columns(df_raw):
    """The dataset should have all 21 expected columns."""
    expected_cols = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
    ]
    assert list(df_raw.columns) == expected_cols


def test_total_charges_is_numeric(df_raw):
    """TotalCharges should be numeric after loading (no empty strings)."""
    assert pd.api.types.is_numeric_dtype(df_raw["TotalCharges"])
    assert df_raw["TotalCharges"].isna().sum() == 0 or df_raw["TotalCharges"].isna().sum() > 0
    # After preprocess, there should be no NaN
    X, y, ids = preprocess(df_raw)
    assert X["TotalCharges"].isna().sum() == 0


def test_preprocess_removes_id_and_target(prepared):
    """X should not contain customerID or Churn."""
    X, y, ids = prepared
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns


def test_target_is_binary(prepared):
    """y should be 0/1 binary."""
    X, y, ids = prepared
    assert set(y.unique()) == {0, 1}
    assert y.dtype in ["int64", "int32", "int8"]


def test_customer_ids_preserved(prepared):
    """customer_ids should have the same length as X."""
    X, y, ids = prepared
    assert len(ids) == len(X)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

def test_build_pipeline_returns_pipeline(prepared):
    """build_pipeline should return a scikit-learn Pipeline."""
    X, y, ids = prepared
    pipeline = build_pipeline(X)
    assert hasattr(pipeline, "fit")
    assert hasattr(pipeline, "predict_proba")


def test_model_trains_and_predicts(prepared):
    """The trained model should produce probabilities in [0, 1]."""
    X, y, ids = prepared
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    model = train_model(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0
    assert len(proba) == len(X_test)


def test_retention_hit_rate_perfect():
    """If all top-N are actual churners, hit rate should be 1.0."""
    y_true = np.array([1, 1, 1, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    assert retention_hit_rate(y_true, y_scores, top_n=3) == 1.0


def test_retention_hit_rate_zero():
    """If no top-N are actual churners, hit rate should be 0.0."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    assert retention_hit_rate(y_true, y_scores, top_n=3) == 0.0


def test_retention_hit_rate_partial():
    """Mixed case: 2 out of 3 top-N are churners."""
    y_true = np.array([1, 0, 1, 0, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    assert retention_hit_rate(y_true, y_scores, top_n=3) == pytest.approx(2 / 3)


def test_retention_hit_rate_top_n_exceeds_data():
    """top_n larger than data size should not crash."""
    y_true = np.array([1, 0])
    y_scores = np.array([0.9, 0.1])
    rate = retention_hit_rate(y_true, y_scores, top_n=100)
    assert rate == 0.5  # 1 churner out of 2 total


# ---------------------------------------------------------------------------
# Prediction / ranking tests
# ---------------------------------------------------------------------------

def test_scored_customers_sorted_descending(prepared):
    """score_customers should return rows sorted by churn_probability descending."""
    X, y, ids = prepared
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    model = train_model(X_train, y_train)

    # Score the full dataset
    scored = score_customers(model, X, X, ids)
    probs = scored["churn_probability"].values
    assert np.all(np.diff(probs) <= 1e-10)  # non-increasing


def test_top_customers_limited_to_n(prepared):
    """get_top_customers should return at most N rows."""
    X, y, ids = prepared
    model = train_model(X, y)
    scored = score_customers(model, X, X, ids)

    top_50 = get_top_customers(scored, top_n=50)
    assert len(top_50) == 50

    top_100 = get_top_customers(scored, top_n=100)
    assert len(top_100) == 100


def test_top_customers_have_rank_column(prepared):
    """Top customers should have a 'rank' column starting at 1."""
    X, y, ids = prepared
    model = train_model(X, y)
    scored = score_customers(model, X, X, ids)
    top_100 = get_top_customers(scored, top_n=100)

    assert "rank" in top_100.columns
    assert list(top_100["rank"].values[:5]) == [1, 2, 3, 4, 5]


def test_summary_includes_hit_rate(df_raw, prepared):
    """summarize_top_100 should include retention_hit_rate when ActualChurn is present."""
    X, y, ids = prepared
    model = train_model(X, y)
    scored = score_customers(model, df_raw, X, ids)
    top_100 = get_top_customers(scored, top_n=100)
    summary = summarize_top_100(top_100)

    assert "retention_hit_rate" in summary
    assert 0.0 <= summary["retention_hit_rate"] <= 1.0
