"""
Customer Segmentation & Churn Pattern Analytics in European Banking
===================================================================
Complete EDA + ML Pipeline
Unified Mentor Private Limited — Machine Learning Internship
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────
CHART_DIR = os.path.join(os.path.dirname(__file__), 'charts')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

COLORS = {
    'primary': '#1a73e8',
    'danger': '#e53935',
    'success': '#43a047',
    'warning': '#fb8c00',
    'purple': '#8e24aa',
    'teal': '#00897b',
    'palette': ['#1a73e8', '#e53935', '#43a047', '#fb8c00', '#8e24aa', '#00897b'],
    'churn': ['#43a047', '#e53935'],
}

plt.rcParams.update({
    'figure.facecolor': '#fafafa',
    'axes.facecolor': '#fafafa',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
})


# ══════════════════════════════════════════════════════════════
# PHASE 1 — DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════
def load_and_clean(csv_path):
    """Load CSV and perform data cleaning."""
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDuplicates: {df.duplicated().sum()}")
    print(f"\nData types:\n{df.dtypes}")

    # Drop non-analytical columns
    drop_cols = [c for c in ['Surname', 'Year'] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Validate binary columns
    for col in ['HasCrCard', 'IsActiveMember', 'Exited']:
        if col in df.columns:
            assert df[col].isin([0, 1]).all(), f"{col} has invalid values"

    print(f"\n✅ Cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ══════════════════════════════════════════════════════════════
# PHASE 2 — FEATURE ENGINEERING & SEGMENTATION
# ══════════════════════════════════════════════════════════════
def add_segments(df):
    """Create segmentation columns per project requirements."""
    # Age segments
    df['AgeGroup'] = pd.cut(df['Age'],
                            bins=[0, 30, 45, 60, 100],
                            labels=['<30', '30-45', '46-60', '60+'],
                            right=False)

    # Credit score bands
    df['CreditBand'] = pd.cut(df['CreditScore'],
                              bins=[0, 580, 740, 900],
                              labels=['Low', 'Medium', 'High'],
                              right=False)

    # Tenure groups
    df['TenureGroup'] = pd.cut(df['Tenure'],
                               bins=[-1, 2, 6, 11],
                               labels=['New (0-2)', 'Mid (3-6)', 'Long (7-10)'])

    # Balance segments
    conditions = [
        df['Balance'] == 0,
        df['Balance'].between(0.01, 75000),
        df['Balance'].between(75000.01, 150000),
        df['Balance'] > 150000,
    ]
    labels = ['Zero', 'Low', 'Medium', 'High']
    df['BalanceSegment'] = np.select(conditions, labels, default='Unknown')

    print("✅ Segmentation columns added: AgeGroup, CreditBand, TenureGroup, BalanceSegment")
    return df


# ══════════════════════════════════════════════════════════════
# PHASE 3 — EDA CHARTS
# ══════════════════════════════════════════════════════════════
def _save(fig, name):
    path = os.path.join(CHART_DIR, f'{name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   📊 Saved {name}.png")


def plot_overall_churn(df):
    """Pie chart: overall churn rate."""
    counts = df['Exited'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts, labels=['Retained', 'Churned'], autopct='%1.1f%%',
           colors=COLORS['churn'], startangle=90,
           textprops={'fontsize': 13, 'fontweight': 'bold'},
           explode=(0, 0.06), shadow=True)
    ax.set_title('Overall Customer Churn Rate')
    _save(fig, '01_overall_churn')


def plot_churn_by_geography(df):
    """Bar chart: churn rate by country."""
    geo = df.groupby('Geography')['Exited'].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(geo.index, geo.values, color=COLORS['palette'][:3], edgecolor='white', width=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                f'{b.get_height():.1f}%', ha='center', fontweight='bold')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_title('Churn Rate by Geography')
    ax.set_ylim(0, max(geo.values) * 1.2)
    _save(fig, '02_churn_geography')


def plot_churn_by_age(df):
    """Bar chart: churn rate by age group."""
    age = df.groupby('AgeGroup')['Exited'].mean().sort_index() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(age.index.astype(str), age.values, color=COLORS['palette'][:4], edgecolor='white', width=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                f'{b.get_height():.1f}%', ha='center', fontweight='bold')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_title('Churn Rate by Age Group')
    _save(fig, '03_churn_age')


def plot_churn_by_gender(df):
    """Bar chart: churn rate by gender."""
    gen = df.groupby('Gender')['Exited'].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(gen.index, gen.values, color=[COLORS['primary'], COLORS['purple']], edgecolor='white', width=0.4)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                f'{b.get_height():.1f}%', ha='center', fontweight='bold')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_title('Churn Rate by Gender')
    _save(fig, '04_churn_gender')


def plot_churn_by_credit(df):
    """Bar chart: churn by credit score band."""
    cr = df.groupby('CreditBand')['Exited'].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(cr.index.astype(str), cr.values, color=[COLORS['danger'], COLORS['warning'], COLORS['success']], edgecolor='white', width=0.4)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                f'{b.get_height():.1f}%', ha='center', fontweight='bold')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_title('Churn Rate by Credit Score Band')
    _save(fig, '05_churn_credit')


def plot_churn_by_balance(df):
    """Bar chart: churn by balance segment."""
    order = ['Zero', 'Low', 'Medium', 'High']
    bal = df.groupby('BalanceSegment')['Exited'].mean().reindex(order) * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(bal.index, bal.values, color=COLORS['palette'][:4], edgecolor='white', width=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                f'{b.get_height():.1f}%', ha='center', fontweight='bold')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_title('Churn Rate by Balance Segment')
    _save(fig, '06_churn_balance')


def plot_churn_by_tenure(df):
    """Bar chart: churn by tenure group."""
    ten = df.groupby('TenureGroup')['Exited'].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(ten.index.astype(str), ten.values, color=COLORS['palette'][:3], edgecolor='white', width=0.4)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                f'{b.get_height():.1f}%', ha='center', fontweight='bold')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_title('Churn Rate by Tenure Group')
    _save(fig, '07_churn_tenure')


def plot_churn_by_products(df):
    """Bar chart: churn by number of products."""
    prod = df.groupby('NumOfProducts')['Exited'].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(prod.index.astype(str), prod.values, color=COLORS['palette'][:4], edgecolor='white', width=0.4)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                f'{b.get_height():.1f}%', ha='center', fontweight='bold')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_title('Churn Rate by Number of Products')
    _save(fig, '08_churn_products')


def plot_churn_active_vs_inactive(df):
    """Bar chart: active vs inactive churn."""
    act = df.groupby('IsActiveMember')['Exited'].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(['Inactive', 'Active'], act.values, color=[COLORS['danger'], COLORS['success']], edgecolor='white', width=0.4)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                f'{b.get_height():.1f}%', ha='center', fontweight='bold')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_title('Churn: Active vs Inactive Members')
    _save(fig, '09_churn_activity')


def plot_geo_age_heatmap(df):
    """Heatmap: geography × age group churn rates."""
    pivot = df.pivot_table(values='Exited', index='Geography',
                           columns='AgeGroup', aggfunc='mean') * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                linewidths=1, cbar_kws={'label': 'Churn Rate %'})
    ax.set_title('Churn Rate: Geography × Age Group')
    _save(fig, '10_heatmap_geo_age')


def plot_correlation(df):
    """Correlation heatmap of numerical features."""
    num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
                linewidths=0.5, square=True)
    ax.set_title('Feature Correlation Matrix')
    _save(fig, '11_correlation')


def run_all_eda(df):
    """Run all EDA charts."""
    print("\n📊 Generating EDA Charts...")
    plot_overall_churn(df)
    plot_churn_by_geography(df)
    plot_churn_by_age(df)
    plot_churn_by_gender(df)
    plot_churn_by_credit(df)
    plot_churn_by_balance(df)
    plot_churn_by_tenure(df)
    plot_churn_by_products(df)
    plot_churn_active_vs_inactive(df)
    plot_geo_age_heatmap(df)
    plot_correlation(df)
    print("✅ All 11 charts saved to charts/ folder")


# ══════════════════════════════════════════════════════════════
# PHASE 4 — KPI COMPUTATION
# ══════════════════════════════════════════════════════════════
def compute_kpis(df):
    """Compute all required KPIs."""
    total = len(df)
    churned = df['Exited'].sum()
    overall_rate = churned / total * 100

    # Geographic risk index
    overall_cr = df['Exited'].mean()
    geo_risk = (df.groupby('Geography')['Exited'].mean() / overall_cr).to_dict()

    # High-value churn
    high_val = df[df['BalanceSegment'] == 'High']
    hv_churn = high_val['Exited'].mean() * 100 if len(high_val) > 0 else 0

    # Engagement drop
    inactive = df[df['IsActiveMember'] == 0]
    engagement_drop = inactive['Exited'].mean() * 100

    kpis = {
        'total_customers': total,
        'churned_customers': int(churned),
        'retained_customers': int(total - churned),
        'overall_churn_rate': round(overall_rate, 2),
        'geographic_risk_index': {k: round(v, 2) for k, v in geo_risk.items()},
        'high_value_churn_rate': round(hv_churn, 2),
        'engagement_drop_indicator': round(engagement_drop, 2),
    }

    print("\n📈 KEY PERFORMANCE INDICATORS")
    print(f"   Total Customers:       {kpis['total_customers']:,}")
    print(f"   Churned:               {kpis['churned_customers']:,}")
    print(f"   Overall Churn Rate:    {kpis['overall_churn_rate']}%")
    print(f"   High-Value Churn:      {kpis['high_value_churn_rate']}%")
    print(f"   Inactive Churn:        {kpis['engagement_drop_indicator']}%")
    print(f"   Geographic Risk Index: {kpis['geographic_risk_index']}")
    return kpis


# ══════════════════════════════════════════════════════════════
# PHASE 5 — ML MODEL TRAINING
# ══════════════════════════════════════════════════════════════
def train_models(df):
    """Train Logistic Regression + Random Forest + Gradient Boosting."""
    # Prepare features
    feature_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                    'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    X = df[feature_cols].copy()

    # Encode Geography and Gender
    le_geo = LabelEncoder()
    le_gen = LabelEncoder()
    X['Geography'] = le_geo.fit_transform(df['Geography'])
    X['Gender'] = le_gen.fit_transform(df['Gender'])

    y = df['Exited']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10,
                                                 random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                                         random_state=42),
    }

    results = {}
    print("\n🤖 MODEL TRAINING RESULTS")
    print("=" * 65)

    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            y_prob = model.predict_proba(X_test_sc)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'auc_roc': round(auc, 4),
            'confusion_matrix': cm,
            'model': model,
        }

        print(f"\n📌 {name}")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall:    {rec:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   AUC-ROC:   {auc:.4f}")

    # Save best model (Random Forest)
    best_name = 'Random Forest'
    best = results[best_name]['model']
    joblib.dump(best, os.path.join(MODEL_DIR, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    print(f"\n✅ Best model ({best_name}) saved to models/")

    # Feature importance chart
    if hasattr(best, 'feature_importances_'):
        imp = pd.Series(best.feature_importances_, index=X.columns).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        imp.plot(kind='barh', color=COLORS['primary'], ax=ax, edgecolor='white')
        ax.set_title(f'Feature Importance — {best_name}')
        ax.set_xlabel('Importance')
        _save(fig, '12_feature_importance')

    # Confusion matrix chart for best
    cm = results[best_name]['confusion_matrix']
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Retained', 'Churned'],
                yticklabels=['Retained', 'Churned'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {best_name}')
    _save(fig, '13_confusion_matrix')

    # Model comparison chart
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_df = pd.DataFrame({
        name: {k: v for k, v in r.items() if k not in ['confusion_matrix', 'model']}
        for name, r in results.items()
    }).T
    metrics_df.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='white', width=0.7)
    ax.set_title('Model Comparison')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right')
    plt.xticks(rotation=0)
    _save(fig, '14_model_comparison')

    return results


# ══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    CSV_PATH = os.path.join(os.path.dirname(__file__), 'European_Bank.csv')

    # Phase 1
    df = load_and_clean(CSV_PATH)

    # Phase 2
    df = add_segments(df)

    # Phase 3
    run_all_eda(df)

    # Phase 4
    kpis = compute_kpis(df)

    # Phase 5
    results = train_models(df)

    print("\n" + "=" * 65)
    print("🎉 ANALYSIS COMPLETE — All charts, models & KPIs generated!")
    print("=" * 65)
