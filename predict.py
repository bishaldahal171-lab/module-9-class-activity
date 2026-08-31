"""
Customer scoring and ranking for the Telco Customer Retention MVP.

Takes a trained model and produces a prioritized list of customers
for retention outreach, sorted by churn probability (highest first).
"""

import pandas as pd

from .data_loader import ID_COL, TARGET_COL


def score_customers(model, df: pd.DataFrame, X: pd.DataFrame, customer_ids: pd.Series) -> pd.DataFrame:
    """Score all customers with churn probability and return a ranked list.

    Parameters
    ----------
    model : Pipeline
        Fitted scikit-learn pipeline.
    df : pd.DataFrame
        Original (preprocessed) dataframe — used to attach context columns.
    X : pd.DataFrame
        Feature matrix used for prediction.
    customer_ids : pd.Series
        Customer IDs corresponding to X.

    Returns
    -------
    pd.DataFrame
        Sorted by churn_probability descending. Contains:
        - rank
        - customerID
        - churn_probability
        - actual Churn label (if present in df)
        - key context columns (Contract, tenure, MonthlyCharges, InternetService, PaymentMethod)
    """
    # Predict churn probabilities
    probabilities = model.predict_proba(X)[:, 1]

    # Build the scored dataframe
    scored = pd.DataFrame({
        "customerID": customer_ids.values,
        "churn_probability": probabilities,
    })

    # Attach context columns for the retention team
    context_cols = [
        "Contract", "tenure", "MonthlyCharges", "InternetService",
        "PaymentMethod", "PaperlessBilling", "SeniorCitizen",
        "Partner", "Dependents",
    ]
    available_cols = [c for c in context_cols if c in df.columns]
    for col in available_cols:
        scored[col] = df[col].values

    # Attach actual churn label if available (for measuring MVP performance)
    if TARGET_COL in df.columns:
        scored["ActualChurn"] = df[TARGET_COL].values

    # Sort by churn probability descending
    scored = scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)

    # Add rank column
    scored.insert(0, "rank", range(1, len(scored) + 1))

    return scored


def get_top_customers(scored_df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """Get the Top N highest-risk customers.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Output of score_customers().
    top_n : int
        Number of top customers to return (default 100).

    Returns
    -------
    pd.DataFrame
        Top N rows of the scored dataframe.
    """
    return scored_df.head(top_n).copy()


def summarize_top_100(top_df: pd.DataFrame) -> dict:
    """Produce a summary of the Top 100 customers for quick insights.

    Parameters
    ----------
    top_df : pd.DataFrame
        Top 100 customers from get_top_customers().

    Returns
    -------
    dict
        Summary statistics.
    """
    summary = {
        "total_customers": len(top_df),
        "avg_churn_probability": round(top_df["churn_probability"].mean(), 4),
        "min_churn_probability": round(top_df["churn_probability"].min(), 4),
        "max_churn_probability": round(top_df["churn_probability"].max(), 4),
    }

    if "ActualChurn" in top_df.columns:
        actual_churners = (top_df["ActualChurn"] == "Yes").sum()
        summary["actual_churners_in_top_100"] = int(actual_churners)
        summary["retention_hit_rate"] = round(actual_churners / len(top_df), 4)

    if "Contract" in top_df.columns:
        summary["contract_breakdown"] = top_df["Contract"].value_counts().to_dict()

    return summary
