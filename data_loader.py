"""
Data loading and preprocessing for the Telco Customer Churn MVP.

Handles:
- Reading the raw CSV
- Converting TotalCharges to numeric (empty strings → NaN → median)
- Encoding the target (Churn: Yes → 1, No → 0)
- Separating features (X), target (y), and customer IDs
"""

import pandas as pd
import numpy as np


# Columns that are categorical (object or low-cardinality)
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

# Columns that are numeric
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Column that identifies customers (not a feature)
ID_COL = "customerID"

# Target column
TARGET_COL = "Churn"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with TotalCharges coerced to numeric.
    """
    df = pd.read_csv(path)
    # TotalCharges has empty-string values for new customers; coerce to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataframe into features (X), target (y), and customer IDs.

    - Fills missing TotalCharges with the median value.
    - Encodes Churn: "Yes" → 1, "No" → 0.
    - Drops customerID from features (kept separately).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe (after load_data).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (no customerID, no Churn).
    y : pd.Series
        Binary target (1 = churned, 0 = retained).
    customer_ids : pd.Series
        Customer ID column preserved for output.
    """
    df = df.copy()

    # Fill missing TotalCharges (NaN from empty strings) with median
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Preserve customer IDs
    customer_ids = df[ID_COL].copy()

    # Encode target
    y = (df[TARGET_COL] == "Yes").astype(int)

    # Drop ID and target from features
    X = df.drop(columns=[ID_COL, TARGET_COL])

    return X, y, customer_ids


def get_feature_lists(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Determine which feature columns are categorical vs numeric.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    categorical_cols : list[str]
    numeric_cols : list[str]
    """
    categorical_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return categorical_cols, numeric_cols
