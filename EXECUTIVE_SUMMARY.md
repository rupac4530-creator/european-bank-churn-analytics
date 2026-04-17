# Executive Summary
## Customer Segmentation & Churn Pattern Analytics in European Banking
### Unified Mentor Private Limited — Machine Learning Internship Project

---

## 1. Overview

This project analyzes customer churn patterns across 10,000 banking customers in three European
markets (France, Germany, Spain). Using segmentation-driven analytics and machine learning,
we identify high-risk customer groups and provide actionable retention strategies.

---

## 2. Key Findings

### Overall Churn Rate: 20.37%
Out of 10,000 customers, **2,037 customers** (20.37%) have exited the bank.

### Geographic Insights
| Country | Risk Index | Interpretation |
|---------|-----------|----------------|
| **Germany** | **1.59** | **Highest risk** — 59% above average churn |
| Spain | 0.82 | Below average churn |
| France | 0.79 | Below average churn |

**Key Insight:** Germany represents the highest geographic churn risk despite having fewer
customers than France. Targeted retention programs should prioritize German operations.

### Demographic Patterns
- **Age Group 46–60** shows the highest churn rate across all geographies
- **Female customers** churn at a higher rate than male customers
- **Inactive members** churn at **26.85%** — significantly above the 20.37% baseline

### High-Value Customer Risk
- **High-value churn ratio: 23.12%** — premium customers with high balances churn
  more than average customers
- This represents significant **revenue risk** as these customers hold the largest deposits

### Product Engagement
- Customers with **3+ products** show anomalously high churn, suggesting over-selling
  or product complexity issues
- **Zero-balance customers** have different churn patterns than active-balance customers

---

## 3. Machine Learning Model Results

Three classification models were trained to predict customer churn:

| Model | Accuracy | Precision | Recall | AUC-ROC |
|-------|----------|-----------|--------|---------|
| Logistic Regression | 80.50% | 58.59% | 14.25% | 0.771 |
| **Random Forest** | **86.75%** | **82.87%** | **43.98%** | **0.859** |
| Gradient Boosting | 85.90% | 73.95% | 47.42% | 0.856 |

**Best Model:** Random Forest — highest accuracy (86.75%) and best AUC-ROC (0.859)

### Top Predictive Features (Feature Importance)
1. **Age** — strongest predictor of churn
2. **NumOfProducts** — number of bank products held
3. **Balance** — account balance level
4. **IsActiveMember** — engagement indicator
5. **Geography** — regional effects (especially Germany)

---

## 4. Business Recommendations

### Immediate Actions (0–3 months)
1. **Germany Retention Program** — Deploy targeted retention offers for German customers,
   especially those aged 46–60 with high balances
2. **Inactive Member Re-engagement** — Create personalized outreach campaigns for
   inactive members before they churn
3. **Product Audit** — Review customers with 3+ products for satisfaction and simplification

### Strategic Initiatives (3–12 months)
4. **High-Value Customer Protection** — Implement a dedicated relationship management
   program for customers with balances above the 75th percentile
5. **Early Warning System** — Deploy the Random Forest churn prediction model to flag
   at-risk customers monthly for proactive intervention
6. **Age-Segmented Marketing** — Design retention strategies tailored to the 46–60
   age group, which shows highest churn propensity

### Long-Term (12+ months)
7. **Cross-Country Benchmarking** — Investigate why Germany has significantly higher
   churn and apply lessons from France/Spain operations
8. **Predictive Analytics Pipeline** — Automate monthly churn scoring and integrate
   with CRM systems for real-time alerts

---

## 5. KPI Dashboard Summary

| KPI | Value | Status |
|-----|-------|--------|
| Overall Churn Rate | 20.37% | Needs Improvement |
| High-Value Churn Ratio | 23.12% | Critical |
| Inactive Member Churn | 26.85% | Critical |
| Germany Risk Index | 1.59 | High Risk |
| France Risk Index | 0.79 | Stable |
| Spain Risk Index | 0.82 | Stable |
| Best Model Accuracy | 86.75% | Good |

---

## 6. Deliverables

| Deliverable | File | Description |
|-------------|------|-------------|
| Analysis Script | `analysis.py` | Complete EDA + ML pipeline |
| Streamlit Dashboard | `app.py` | Interactive analytics dashboard |
| EDA Charts | `charts/` | 14 visualization charts |
| Trained Model | `models/` | Random Forest model + scaler |
| Requirements | `requirements.txt` | Python dependencies |
| Executive Summary | `EXECUTIVE_SUMMARY.md` | This document |

---

## 7. Technical Setup

### Running the Dashboard
```bash
pip install -r requirements.txt
python analysis.py          # Generate charts & train models
streamlit run app.py        # Launch interactive dashboard
```

### Running the Analysis
```bash
python analysis.py
```
This generates all 14 charts in `charts/` and trains 3 ML models, saving the best one.

---

*Prepared by: Bedanta*
*Organization: Unified Mentor Private Limited*
*Project: Machine Learning Internship*
*Date: April 2026*
