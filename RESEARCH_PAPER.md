# Research Paper: Customer Segmentation and Churn Pattern Analytics in European Banking

**Author:** Bedanta  
**Organization:** Unified Mentor Private Limited  
**Role:** Machine Learning Intern  
**Date:** April 2026

---

## Abstract

This study presents a comprehensive analysis of customer churn patterns across 10,000 banking customers in three European markets: France, Germany, and Spain. Using segmentation-driven analytics and machine learning classification models, we identify high-risk customer groups and quantify the financial impact of churn. Our Random Forest model achieves 86.75% accuracy and 0.859 AUC-ROC in predicting customer exits. Key findings reveal that Germany has 59% higher churn than the European average, customers aged 46-60 are most likely to exit, and inactive members churn at 26.85%. These insights enable banks to design targeted, data-driven retention strategies.

**Keywords:** Customer Churn, Machine Learning, Customer Segmentation, European Banking, Random Forest, Predictive Analytics

---

## 1. Introduction

### 1.1 Background

Customer churn represents one of the largest hidden costs in retail banking. Losing existing customers leads to reduced lifetime value, increased acquisition costs, and revenue instability. While banks often track aggregate churn rates, they typically lack the granular segmentation insights needed to understand which customer groups are most at risk and why.

### 1.2 Problem Statement

Despite having rich customer-level data, banks face challenges in:
- Identifying high-risk customer segments before they churn
- Understanding how churn differs across geographies and demographics
- Quantifying the financial profile of churned customers to assess revenue risk
- Building predictive systems for proactive intervention

### 1.3 Research Objectives

**Primary Objectives:**
- Measure overall churn rate and its distribution across customer segments
- Compare churn behavior across France, Germany, and Spain
- Build and evaluate machine learning models for churn prediction

**Secondary Objectives:**
- Understand churn among high-value customers (revenue risk)
- Evaluate engagement and tenure patterns
- Provide actionable recommendations for strategic planning

---

## 2. Literature Review

Customer churn prediction has been extensively studied in telecommunications and banking. Logistic Regression and tree-based models (Random Forest, Gradient Boosting) consistently perform well on structured tabular churn datasets. Feature importance analysis in banking churn typically highlights age, account balance, and engagement metrics as primary predictors. This study extends existing work by incorporating multi-dimensional segmentation analysis alongside predictive modeling to provide both explanatory and predictive insights.

---

## 3. Methodology

### 3.1 Dataset Description

The dataset contains 10,000 customer records from a European bank with 14 features including demographic information (Age, Gender, Geography), financial indicators (Balance, CreditScore, EstimatedSalary), product usage (NumOfProducts, HasCrCard), and engagement metrics (IsActiveMember, Tenure). The binary target variable is `Exited` (1 = churned, 0 = retained).

### 3.2 Data Quality Assessment

| Metric | Value |
|--------|-------|
| Total Records | 10,000 |
| Missing Values | 0 |
| Duplicate Records | 0 |
| Target Distribution | 79.63% retained, 20.37% churned |

### 3.3 Data Preparation

1. **Column Removal:** Dropped non-analytical fields (Surname, Year)
2. **Validation:** Confirmed binary variables (HasCrCard, IsActiveMember, Exited) contain only 0/1 values
3. **Encoding:** Label-encoded Geography and Gender for model input
4. **Scaling:** Applied StandardScaler for Logistic Regression

### 3.4 Customer Segmentation Design

Five segmentation dimensions were created per project requirements:

| Dimension | Segments | Bins |
|-----------|----------|------|
| Age | <30, 30-45, 46-60, 60+ | [0,30,45,60,100) |
| Credit Score | Low, Medium, High | [0,580,740,900) |
| Tenure | New (0-2), Mid (3-6), Long (7-10) | [-1,2,6,11] |
| Balance | Zero, Low, Medium, High | 0, 0-75K, 75K-150K, 150K+ |
| Geography | France, Germany, Spain | Categorical |

### 3.5 Machine Learning Pipeline

- **Train/Test Split:** 80/20 with stratification on target
- **Models:** Logistic Regression, Random Forest (200 trees, max_depth=10), Gradient Boosting (150 estimators, max_depth=5)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix

---

## 4. Results

### 4.1 Overall Churn Analysis

The overall churn rate is **20.37%** (2,037 out of 10,000 customers). This establishes the baseline against which all segment-level analyses are compared.

### 4.2 Geographic Analysis

| Country | Churn Rate | Risk Index | Customers |
|---------|-----------|------------|-----------|
| Germany | ~32% | 1.59x | ~2,500 |
| France | ~16% | 0.79x | ~5,000 |
| Spain | ~17% | 0.82x | ~2,500 |

**Finding:** Germany shows 59% higher churn than the European average, despite having fewer customers than France. This suggests country-specific service quality or competitive pressure issues.

### 4.3 Demographic Analysis

**Age:** The 46-60 age group shows the highest churn rate across all geographies. This pre-retirement segment may be consolidating finances or seeking better rates.

**Gender:** Female customers churn at a higher rate than male customers.

**Activity:** Inactive members churn at **26.85%** compared to the 20.37% baseline, confirming that engagement is a strong retention factor.

### 4.4 Product and Financial Analysis

- Customers with **3+ products** show anomalously high churn (83-100%), suggesting over-selling creates dissatisfaction
- **High-balance customers** churn at 23.12%, representing significant revenue risk
- Zero-balance customers show different churn patterns, possibly indicating dormant accounts

### 4.5 Machine Learning Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 80.50% | 58.59% | 14.25% | 22.92% | 0.771 |
| **Random Forest** | **86.75%** | **82.87%** | **43.98%** | **57.46%** | **0.859** |
| Gradient Boosting | 85.90% | 73.95% | 47.42% | 57.78% | 0.856 |

**Best Model:** Random Forest achieves the highest accuracy (86.75%) and AUC-ROC (0.859).

### 4.6 Feature Importance (Random Forest)

The top 5 most important features for churn prediction:
1. **Age** - strongest predictor
2. **NumOfProducts** - product engagement
3. **Balance** - financial profile
4. **IsActiveMember** - engagement indicator
5. **Geography** - regional effects

---

## 5. Discussion

### 5.1 Key Insights

1. **Geographic targeting is essential:** Germany requires dedicated retention programs due to significantly higher churn
2. **Age-based risk profiling:** The 46-60 segment needs proactive outreach before they consider switching
3. **Engagement drives retention:** Active members are significantly less likely to churn
4. **Product simplification needed:** Over-selling (3+ products) correlates with higher churn
5. **Revenue risk is concentrated:** High-balance churners represent disproportionate financial loss

### 5.2 Limitations

- The dataset is cross-sectional; longitudinal data would enable trend analysis
- External factors (market conditions, competitor offerings) are not captured
- The model's recall for churned customers (43.98%) indicates room for improvement with techniques like SMOTE or class weighting

---

## 6. Recommendations

### Immediate (0-3 months)
1. Deploy targeted retention offers for German customers aged 46-60
2. Create personalized re-engagement campaigns for inactive members
3. Review customers with 3+ products for satisfaction and simplification

### Strategic (3-12 months)
4. Implement dedicated relationship management for high-balance customers
5. Deploy monthly churn scoring using the Random Forest model
6. Design age-segmented marketing for the 46-60 demographic

### Long-term (12+ months)
7. Cross-country benchmarking to understand Germany's higher churn
8. Automate predictive analytics pipeline integrated with CRM systems

---

## 7. Conclusion

This project demonstrates a complete, end-to-end analytics pipeline for customer churn prediction in European banking. Through systematic segmentation analysis and machine learning modeling, we identified actionable patterns that can inform targeted retention strategies. The Random Forest model (86.75% accuracy, 0.859 AUC-ROC) provides a practical tool for proactive churn identification, while the interactive Streamlit dashboard enables business stakeholders to explore churn patterns through dynamic filters and visualizations.

---

## References

1. European Central Bank Customer Dataset (2025)
2. Scikit-learn Documentation - Classification Models
3. Streamlit Documentation - Web Application Framework
4. Unified Mentor Private Limited - Project Requirements Document

---

**End of Paper**
