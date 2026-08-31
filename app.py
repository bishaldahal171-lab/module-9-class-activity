"""
Streamlit web app for the Telco Customer Retention MVP.

Run with:
    streamlit run app.py

The app allows the retention team to:
1. Load the customer dataset
2. Train a churn prediction model (with train/test evaluation)
3. View evaluation metrics including Retention Hit Rate
4. See the Top 100 highest-risk customers
5. Download the prioritized list as CSV
"""

import io
import os
import sys

import pandas as pd
import streamlit as st

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.data_loader import load_data, preprocess
from src.model import train_model, evaluate_model, train_test_split_data
from src.predict import score_customers, get_top_customers, summarize_top_100

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Telco-Customer-Churn.csv")


st.set_page_config(
    page_title="Telco Customer Retention MVP",
    page_icon="📞",
    layout="wide",
)

st.title("📞 Telco Customer Retention MVP")
st.markdown("""
**Which 100 customers should we contact first?**

This tool scores every customer's churn probability and ranks the Top 100
highest-risk customers for retention outreach.
""")

st.divider()

# --- Load data ---
st.subheader("1. Load Customer Data")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    st.success(f"Loaded {len(df_raw):,} customers from uploaded file.")
else:
    # Use default dataset
    if os.path.exists(DATA_PATH):
        df_raw = load_data(DATA_PATH)
        st.info(f"Using default dataset: {len(df_raw):,} customers from `data/Telco-Customer-Churn.csv`.")
    else:
        st.warning("No data file found. Please upload a CSV or place the dataset in `data/`.")
        st.stop()

# Show data preview
with st.expander("Preview raw data"):
    st.dataframe(df_raw.head(10), use_container_width=True)

st.divider()

# --- Preprocess & train ---
st.subheader("2. Train Churn Prediction Model")

X, y, customer_ids = preprocess(df_raw)

model_type = st.selectbox(
    "Select model",
    options=["gradient_boosting", "logistic_regression", "random_forest"],
    format_func=lambda x: {
        "gradient_boosting": "Gradient Boosting (best hit rate)",
        "logistic_regression": "Logistic Regression (most interpretable)",
        "random_forest": "Random Forest",
    }[x],
    help="Gradient Boosting achieved 80% Retention Hit Rate vs. 73% for Logistic Regression.",
)

if st.button("Train Model", type="primary"):
    with st.spinner(f"Training {model_type} model..."):
        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split_data(X, y)

        # Train on training set
        model = train_model(X_train, y_train, model_type=model_type)

        # Evaluate on test set
        metrics = evaluate_model(model, X_test, y_test)

        # Now train on full data for the production ranking
        full_model = train_model(X, y, model_type=model_type)

        # Score all customers
        scored_df = score_customers(full_model, df_raw, X, customer_ids)
        top_100 = get_top_customers(scored_df, top_n=100)
        summary = summarize_top_100(top_100)

        # Store in session state
        st.session_state["model"] = full_model
        st.session_state["scored_df"] = scored_df
        st.session_state["top_100"] = top_100
        st.session_state["summary"] = summary
        st.session_state["metrics"] = metrics

    st.success("Model trained successfully!")

st.divider()

# --- Display results ---
if "metrics" in st.session_state:
    st.subheader("3. Model Performance (Test Set)")

    metrics = st.session_state["metrics"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    with col2:
        st.metric("Precision", f"{metrics['precision']:.1%}")
    with col3:
        st.metric("Recall", f"{metrics['recall']:.1%}")
    with col4:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.divider()

    # --- Retention Hit Rate ---
    st.subheader("4. Retention Hit Rate (Top 100)")

    summary = st.session_state["summary"]

    if "retention_hit_rate" in summary:
        hit_rate = summary["retention_hit_rate"]
        actual_churners = summary["actual_churners_in_top_100"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Retention Hit Rate", f"{hit_rate:.1%}")
        with col2:
            st.metric("Actual Churners in Top 100", f"{actual_churners}")
        with col3:
            baseline = 0.265  # Overall churn rate
            lift = hit_rate / baseline
            st.metric("Lift vs. Random", f"{lift:.2f}x")

        # Visual indicator
        progress = min(hit_rate, 1.0)
        st.progress(progress, text=f"Hit Rate: {hit_rate:.1%} (target: ≥ 60%)")

        if hit_rate >= 0.60:
            st.success(f"Target met! {actual_churners} out of 100 recommended customers actually churned.")
        else:
            st.warning(f"{actual_churners} out of 100 recommended customers churned (target: ≥ 60).")

    st.divider()

    # --- Top 100 table ---
    st.subheader("5. Top 100 Customers to Contact")

    top_100 = st.session_state["top_100"]

    # Format probability as percentage
    display_df = top_100.copy()
    display_df["churn_probability"] = display_df["churn_probability"].apply(lambda x: f"{x:.1%}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Download button
    csv_buffer = io.StringIO()
    top_100.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Top 100 as CSV",
        data=csv_buffer.getvalue(),
        file_name="top_100_retention_list.csv",
        mime="text/csv",
    )

    st.divider()

    # --- Summary insights ---
    st.subheader("6. Summary Insights")

    st.write(f"**Average churn probability (Top 100):** {summary['avg_churn_probability']:.1%}")
    st.write(f"**Probability range:** {summary['min_churn_probability']:.1%} – {summary['max_churn_probability']:.1%}")

    if "contract_breakdown" in summary:
        st.write("**Contract type breakdown (Top 100):**")
        contract_df = pd.DataFrame(
            list(summary["contract_breakdown"].items()),
            columns=["Contract Type", "Count"],
        )
        st.bar_chart(contract_df.set_index("Contract Type"))

    st.divider()
    st.markdown("""
    ---
    **MVP — Build, Measure, Learn**

    This is a minimum viable product. The model is trained on historical data
    to test the hypothesis that contract type, tenure, monthly charges,
    internet service type, and payment method can identify at-risk customers.

    **Next steps:** Collect feedback from the retention team, measure real-world
    hit rates from outreach campaigns, and iterate on features and model choice.
    """)
else:
    st.info("👆 Click **Train Model** to score customers and see the Top 100 priority list.")
