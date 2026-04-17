# Customer Segmentation & Churn Pattern Analytics in European Banking

> Machine Learning Internship Project | Unified Mentor Private Limited

---

## Business Problem

Customer churn is one of the largest hidden costs in retail banking. This project analyzes **10,000 customer records** across three European markets (France, Germany, Spain) to uncover churn patterns, identify high-risk segments, and build predictive models to support data-driven retention strategies.

## Dataset

| Field | Description |
|-------|-------------|
| CustomerId | Unique customer identifier |
| CreditScore | Customer creditworthiness (350-850) |
| Geography | France, Germany, Spain |
| Gender | Male / Female |
| Age | Customer age |
| Tenure | Years with the bank (0-10) |
| Balance | Account balance |
| NumOfProducts | Number of bank products held |
| HasCrCard | Credit card ownership (0/1) |
| IsActiveMember | Activity indicator (0/1) |
| EstimatedSalary | Estimated annual salary |
| Exited | **Target** - Churn indicator (0/1) |

**Source:** European Central Bank (10,000 rows, 14 columns, 0 missing values, 0 duplicates)

## Methodology

```
Data Ingestion --> Cleaning --> Segmentation --> EDA --> ML Modeling --> Dashboard
```

### Segmentation Dimensions
- **Geographic:** France, Germany, Spain
- **Age:** <30, 30-45, 46-60, 60+
- **Credit Score:** Low (<580), Medium (580-740), High (740+)
- **Tenure:** New (0-2yr), Mid (3-6yr), Long (7-10yr)
- **Balance:** Zero, Low, Medium, High

### Machine Learning Models

| Model | Accuracy | Precision | Recall | AUC-ROC |
|-------|----------|-----------|--------|---------|
| Logistic Regression | 80.50% | 58.59% | 14.25% | 0.771 |
| **Random Forest** | **86.75%** | **82.87%** | **43.98%** | **0.859** |
| Gradient Boosting | 85.90% | 73.95% | 47.42% | 0.856 |

**Best Model:** Random Forest (86.75% accuracy, 0.859 AUC-ROC)

### Key Performance Indicators

| KPI | Value |
|-----|-------|
| Overall Churn Rate | 20.37% |
| High-Value Churn Ratio | 23.12% |
| Inactive Member Churn | 26.85% |
| Germany Risk Index | 1.59x |
| France Risk Index | 0.79x |
| Spain Risk Index | 0.82x |

## Key Findings

1. **Germany** has the highest churn rate (1.59x above average) despite fewer customers
2. **Age 46-60** is the highest-risk demographic across all countries
3. **Inactive members** churn at 26.85% vs 20.37% baseline
4. Customers with **3+ products** show anomalously high churn
5. **High-balance customers** churn more than average (23.12%)

## Dashboard Features

Interactive Streamlit dashboard with 5 modules:

- **Overview** - KPI cards, churn distribution, gender/product/activity analysis
- **Geography Analysis** - Country-wise churn rates, Geographic Risk Index
- **Age & Tenure** - Age group analysis, tenure comparison, geography-age heatmap
- **High-Value Explorer** - Balance segments, credit bands, revenue risk metrics
- **ML Model Results** - Model comparison, feature importance, confusion matrix

### Filters
- Geography (multi-select)
- Gender (multi-select)
- Age Group (multi-select)
- Member Status (All / Active / Inactive)

## Project Structure

```
european-bank-churn-analytics/
|-- European_Bank.csv          # Dataset (10,000 records)
|-- analysis.py                # EDA + ML pipeline
|-- app.py                     # Streamlit dashboard
|-- requirements.txt           # Python dependencies
|-- EXECUTIVE_SUMMARY.md       # Business summary
|-- RESEARCH_PAPER.md          # Research paper
|-- README.md                  # This file
|-- charts/                    # 14 EDA visualizations
|   |-- 01_overall_churn.png
|   |-- 02_churn_geography.png
|   |-- ...
|   |-- 14_model_comparison.png
|-- models/
    |-- best_model.pkl         # Trained Random Forest
    |-- scaler.pkl             # StandardScaler
```

## Setup & Run

### Prerequisites
- Python 3.9+

### Installation
```bash
git clone https://github.com/rupac4530-creator/european-bank-churn-analytics.git
cd european-bank-churn-analytics
pip install -r requirements.txt
```

### Run Analysis Pipeline
```bash
python analysis.py
```
This generates all 14 charts and trains 3 ML models.

### Launch Dashboard
```bash
streamlit run app.py
```
Dashboard opens at `http://localhost:8501`

## Tech Stack

- **Python** - pandas, numpy, scikit-learn
- **Visualization** - matplotlib, seaborn, plotly
- **Dashboard** - Streamlit
- **ML Models** - Logistic Regression, Random Forest, Gradient Boosting

## Business Recommendations

1. Deploy targeted retention programs for German customers aged 46-60
2. Create re-engagement campaigns for inactive members
3. Audit customers with 3+ products for satisfaction
4. Implement dedicated relationship management for high-balance customers
5. Deploy monthly churn scoring using the Random Forest model

---

**Author:** Bedanta | **Organization:** Unified Mentor Private Limited | **Role:** Machine Learning Intern
