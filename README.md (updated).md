# Telco Customer Retention MVP

**Class Activity: Build, Measure, Improve a Customer-Churn MVP**
**Duration:** 80 minutes
**Product:** Telecom Customer Retention MVP
**Tools:** Python, pandas, Streamlit, pytest, Git/GitHub

---

## Step 1 — Decide What You Are Building

### Objective
Help the retention team focus its limited outreach effort on customers who are most likely to churn.

### Hypothesis
We believe that **a customer's contract type (month-to-month vs. longer-term), tenure, monthly charges, internet service type, and payment method** will help us identify customers more likely to churn.

> **Why these features?** Exploratory analysis of the Telco Customer Churn dataset (7,043 customers, 26.5% overall churn rate) reveals strong churn signals:
>
> | Feature | Churn Rate | Insight |
> |---|---|---|
> | Month-to-month contract | 42.7% | vs. 2.8% for two-year contracts |
> | Fiber optic internet | 41.9% | vs. 19.0% for DSL, 7.4% for no internet |
> | Electronic check payment | 45.3% | vs. 15.2% for credit card (auto) |
> | Senior citizen | 41.7% | vs. 23.6% for non-senior |
> | Paperless billing | 33.6% | vs. 16.3% for paper billing |
> | No partner / no dependents | 33.0% / 31.3% | vs. 19.7% / 15.5% for those with |
> | Low tenure (median 10 months) | — | Churned customers have median tenure of 10 vs. 38 for retained |
> | High monthly charges (mean $74.44) | — | vs. $61.27 for retained customers |

### MVP
Our minimum product will allow the retention team to **upload or load the customer dataset, run a churn-prediction model that scores every customer with a churn probability, and view a ranked list of the Top 100 highest-risk customers** — along with the key factors driving each customer's risk score — so the team can prioritize outreach efficiently.

### Success Measure
We will measure success using the **Retention Hit Rate**:

$$\text{Retention Hit Rate} = \frac{\text{customers who actually churned among the Top 100}}{100}$$

**Target:** A Retention Hit Rate of ≥ 60% (i.e., at least 60 of the Top 100 recommended customers actually churn), which would significantly outperform random selection (~26.5% expected hit rate based on the overall churn rate).

**Secondary metrics:**
- **Precision@100** — same as Retention Hit Rate (fraction of Top 100 who actually churned)
- **Lift** — ratio of our Hit Rate to the baseline churn rate (target: ≥ 2.0x lift over random)
- **Model ROC-AUC** — target ≥ 0.80 on a held-out test set to ensure the model generalizes

---

## Dataset

**Source:** Telco Customer Churn dataset (`Telco-Customer-Churn.csv`)

| Column | Description | Type |
|---|---|---|
| customerID | Unique customer identifier | string |
| gender | Male or Female | categorical |
| SeniorCitizen | 1 = senior, 0 = not | binary |
| Partner | Has a partner (Yes/No) | categorical |
| Dependents | Has dependents (Yes/No) | categorical |
| tenure | Months with the company | numeric |
| PhoneService | Has phone service (Yes/No) | categorical |
| MultipleLines | Single / Multiple / No phone service | categorical |
| InternetService | DSL / Fiber optic / No | categorical |
| OnlineSecurity | Yes / No / No internet service | categorical |
| OnlineBackup | Yes / No / No internet service | categorical |
| DeviceProtection | Yes / No / No internet service | categorical |
| TechSupport | Yes / No / No internet service | categorical |
| StreamingTV | Yes / No / No internet service | categorical |
| StreamingMovies | Yes / No / No internet service | categorical |
| Contract | Month-to-month / One year / Two year | categorical |
| PaperlessBilling | Yes / No | categorical |
| PaymentMethod | Electronic check / Mailed check / Bank transfer (auto) / Credit card (auto) | categorical |
| MonthlyCharges | Monthly billing amount | numeric |
| TotalCharges | Total amount charged | numeric |
| Churn | Whether the customer left (Yes/No) | binary (target) |

**Summary:** 7,043 customers, 26.5% churn rate (1,869 churned).

---

## Project Structure

```
telco-churn-mvp/
├── data/
│   └── Telco-Customer-Churn.csv    # Raw dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Load & preprocess data
│   ├── model.py                    # Train & evaluate churn model
│   └── predict.py                  # Score customers & rank Top 100
├── tests/
│   ├── __init__.py
│   └── test_model.py               # pytest tests
├── app.py                           # Streamlit web app
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Step 2 — Build the MVP

### What Was Built

A complete, testable churn-prediction MVP with five components:

| File | Purpose |
|---|---|
| `src/data_loader.py` | Loads CSV, coerces TotalCharges to numeric, encodes target, splits into X/y/IDs |
| `src/model.py` | Scikit-learn Pipeline (OneHotEncoder + StandardScaler + LogisticRegression), train/evaluate functions, Retention Hit Rate metric |
| `src/predict.py` | Scores all customers, ranks by churn probability, extracts Top 100, produces summary insights |
| `tests/test_model.py` | 15 pytest tests covering data loading, preprocessing, model training, hit rate logic, and ranking |
| `app.py` | Streamlit web app — load data, train model, view metrics & Top 100, download CSV |

### Model Details

- **Algorithm:** Logistic Regression with `class_weight="balanced"`
- **Preprocessing:** OneHotEncoder for categorical features, StandardScaler for numeric features
- **Why Logistic Regression?** Interpretable — the retention team can understand which factors drive churn risk. Sufficient for an MVP.

### How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Launch the Streamlit app
streamlit run app.py
```

### Results

**Test Set Evaluation (20% held out):**

| Metric | Value | Target |
|---|---|---|
| Accuracy | 73.8% | — |
| Precision | 50.4% | — |
| Recall | 78.3% | — |
| ROC-AUC | 0.841 | ≥ 0.80 |
| Retention Hit Rate (Top 100) | 73.0% | ≥ 60% |

**Full-Dataset Top 100 (production ranking):**

| Metric | Value |
|---|---|
| Actual churners in Top 100 | 86 out of 100 |
| Retention Hit Rate | 86.0% |
| Lift vs. random | 3.25x |
| Average churn probability (Top 100) | 91.2% |
| Contract breakdown | 100% Month-to-month |

All 100 of the Top 100 recommended customers have month-to-month contracts — the strongest churn signal in the dataset.

### Test Results

```
15 passed in 1.29s
```

All 15 tests pass, covering:
- Data loads with expected 21 columns
- TotalCharges converts to numeric (no empty strings)
- Target is binary (0/1)
- customerID preserved separately
- Model trains and produces valid probabilities in [0, 1]
- Retention Hit Rate computed correctly (perfect, zero, partial, and edge cases)
- Customers sorted by churn probability descending
- Top 100 limited to exactly 100 rows
- Rank column starts at 1
- Summary includes retention_hit_rate

---

## Build–Measure–Learn Cycle

| Phase | Activity | Status |
|---|---|---|
| **Build** | Create a model that scores each customer's churn probability and ranks the Top 100 | ✅ Step 2 |
| **Measure** | Evaluate using Retention Hit Rate (Precision@100) on historical data | ✅ Step 2 |
| **Learn** | Identify which features drive churn, refine the model, iterate | Step 3+ |

---

*This MVP follows the Build–Measure–Learn approach: build the smallest thing that tests the hypothesis, measure results with evidence, and learn to improve.*
