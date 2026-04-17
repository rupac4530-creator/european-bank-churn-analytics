"""European Banking - Customer Churn Analytics Dashboard
Streamlit Web Application
Unified Mentor Private Limited - ML Internship Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ── Page Config ──
st.set_page_config(
    page_title="European Bank Churn Analytics",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS with Professional Icons & Animations ──
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/lucide-static@latest/font/lucide.min.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .main { background: #f0f2f6; }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    /* ── KPI Cards ── */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 22px 24px; border-radius: 16px; color: white;
        text-align: center; margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(102,126,234,0.35);
        animation: fadeInUp 0.5s ease-out both;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 28px rgba(102,126,234,0.5);
    }
    .kpi-card h2 { margin: 0; font-size: 32px; font-weight: 800; letter-spacing: -0.5px; }
    .kpi-card p  { margin: 6px 0 0 0; font-size: 12px; opacity: 0.9; font-weight: 600;
                   text-transform: uppercase; letter-spacing: 0.8px; }
    .kpi-danger {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 4px 20px rgba(245,87,108,0.35);
        animation-delay: 0.08s;
    }
    .kpi-success {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 4px 20px rgba(79,172,254,0.35);
        animation-delay: 0.16s;
    }
    .kpi-warning {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        box-shadow: 0 4px 20px rgba(250,112,154,0.3);
        animation-delay: 0.24s;
    }

    /* ── Header ── */
    .header-box {
        background: linear-gradient(135deg, #0c1445 0%, #1a237e 50%, #283593 100%);
        padding: 32px 40px; border-radius: 20px; color: white;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(12,20,69,0.3);
        animation: fadeInUp 0.4s ease-out both;
    }
    .header-box h1 { font-size: 26px; font-weight: 800; margin: 0; letter-spacing: -0.3px; }
    .header-box p  { font-size: 14px; opacity: 0.85; margin: 8px 0 0 0; }

    /* ── Section Titles ── */
    .section-title {
        font-size: 18px; font-weight: 700; color: #1a237e;
        margin: 28px 0 16px 0; padding-bottom: 8px;
        border-bottom: 3px solid #667eea;
        animation: fadeInUp 0.4s ease-out both;
    }

    /* ── Sidebar ── */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1445 0%, #1a237e 100%);
    }
    div[data-testid="stSidebar"] * { color: white !important; }

    /* ── Insight Boxes ── */
    .insight-box {
        background: linear-gradient(135deg, #e8eaf6 0%, #f3e5f5 100%);
        padding: 18px 22px; border-radius: 12px;
        border-left: 4px solid #3949ab; margin: 12px 0; font-size: 14px;
        animation: fadeInUp 0.5s ease-out both;
        transition: transform 0.2s ease;
        line-height: 1.6;
        color: #1a1a2e !important;
    }
    .insight-box b { color: #1a237e !important; }
    .insight-box:hover { transform: translateX(4px); }
    .insight-icon {
        display: inline-block; width: 20px; height: 20px;
        background: #3949ab; border-radius: 50%; color: white !important;
        text-align: center; line-height: 20px; font-size: 11px;
        margin-right: 8px; vertical-align: middle; font-weight: 700;
    }

    /* ── Plotly chart containers ── */
    .stPlotlyChart { animation: fadeInUp 0.5s ease-out both; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600; font-size: 14px; letter-spacing: 0.3px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Data ──
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), 'European_Bank.csv')
    df = pd.read_csv(csv_path)
    df.drop(columns=['Surname', 'Year'], errors='ignore', inplace=True)

    # Segmentation
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0,30,45,60,100],
                            labels=['<30','30-45','46-60','60+'], right=False)
    df['CreditBand'] = pd.cut(df['CreditScore'], bins=[0,580,740,900],
                              labels=['Low','Medium','High'], right=False)
    df['TenureGroup'] = pd.cut(df['Tenure'], bins=[-1,2,6,11],
                               labels=['New (0-2)','Mid (3-6)','Long (7-10)'])
    conditions = [
        df['Balance'] == 0,
        df['Balance'].between(0.01, 75000),
        df['Balance'].between(75000.01, 150000),
        df['Balance'] > 150000,
    ]
    df['BalanceSegment'] = np.select(conditions,
                                     ['Zero','Low','Medium','High'], default='Unknown')
    return df


df = load_data()

# ── Sidebar Filters ──
st.sidebar.markdown("## Filters")
st.sidebar.markdown("---")

countries = st.sidebar.multiselect("Geography", df['Geography'].unique(),
                                    default=df['Geography'].unique(), key="geo_filter")
genders = st.sidebar.multiselect("Gender", df['Gender'].unique(),
                                  default=df['Gender'].unique(), key="gen_filter")
age_groups = st.sidebar.multiselect("Age Group", df['AgeGroup'].cat.categories.tolist(),
                                     default=df['AgeGroup'].cat.categories.tolist(), key="age_filter")
active_filter = st.sidebar.radio("Member Status", ['All', 'Active Only', 'Inactive Only'],
                                  key="active_filter")

# Apply filters
filtered = df[
    (df['Geography'].isin(countries)) &
    (df['Gender'].isin(genders)) &
    (df['AgeGroup'].isin(age_groups))
]
if active_filter == 'Active Only':
    filtered = filtered[filtered['IsActiveMember'] == 1]
elif active_filter == 'Inactive Only':
    filtered = filtered[filtered['IsActiveMember'] == 0]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Showing:** {len(filtered):,} / {len(df):,} customers")

# ── Header ──
st.markdown("""
<div class="header-box">
    <h1>Customer Segmentation & Churn Analytics</h1>
    <p>European Central Bank — Powered by Machine Learning | Unified Mentor Internship Project</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════
total = len(filtered)
churned = int(filtered['Exited'].sum())
retained = total - churned
churn_rate = (churned / total * 100) if total > 0 else 0

high_val = filtered[filtered['BalanceSegment'] == 'High']
hv_churn = (high_val['Exited'].mean() * 100) if len(high_val) > 0 else 0

inactive = filtered[filtered['IsActiveMember'] == 0]
inactive_churn = (inactive['Exited'].mean() * 100) if len(inactive) > 0 else 0

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f'<div class="kpi-card"><h2>{total:,}</h2><p>Total Customers</p></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi-card kpi-danger"><h2>{churn_rate:.1f}%</h2><p>Churn Rate</p></div>',
                unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="kpi-card kpi-success"><h2>{retained:,}</h2><p>Retained</p></div>',
                unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="kpi-card kpi-warning"><h2>{hv_churn:.1f}%</h2><p>High-Value Churn</p></div>',
                unsafe_allow_html=True)
with c5:
    st.markdown(f'<div class="kpi-card"><h2>{inactive_churn:.1f}%</h2><p>Inactive Churn</p></div>',
                unsafe_allow_html=True)

st.markdown("")

# ══════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Geography Analysis", "Age & Tenure",
    "High-Value Explorer", "ML Model Results"
])

# ── TAB 1: Overview ──
with tab1:
    st.markdown('<div class="section-title">Overall Churn Distribution</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(filtered, names=filtered['Exited'].map({0:'Retained', 1:'Churned'}),
                     color_discrete_sequence=['#43a047', '#e53935'],
                     hole=0.45, title="Churn vs Retained")
        fig.update_traces(textposition='inside', textinfo='percent+label',
                         textfont_size=14)
        fig.update_layout(height=400, font=dict(family='Inter'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gender_churn = filtered.groupby('Gender')['Exited'].mean().reset_index()
        gender_churn['Exited'] = gender_churn['Exited'] * 100
        fig = px.bar(gender_churn, x='Gender', y='Exited',
                     color='Gender', title='Churn Rate by Gender',
                     color_discrete_sequence=['#1a73e8', '#8e24aa'],
                     text=gender_churn['Exited'].round(1).astype(str) + '%')
        fig.update_layout(height=400, yaxis_title='Churn Rate (%)',
                         showlegend=False, font=dict(family='Inter'))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    # Products & Activity
    col3, col4 = st.columns(2)
    with col3:
        prod_churn = filtered.groupby('NumOfProducts')['Exited'].mean().reset_index()
        prod_churn['Exited'] = prod_churn['Exited'] * 100
        fig = px.bar(prod_churn, x='NumOfProducts', y='Exited',
                     title='Churn Rate by Number of Products',
                     color_discrete_sequence=['#667eea'],
                     text=prod_churn['Exited'].round(1).astype(str) + '%')
        fig.update_layout(height=380, yaxis_title='Churn Rate (%)',
                         font=dict(family='Inter'))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        act_churn = filtered.groupby('IsActiveMember')['Exited'].mean().reset_index()
        act_churn['Exited'] = act_churn['Exited'] * 100
        act_churn['Status'] = act_churn['IsActiveMember'].map({0:'Inactive', 1:'Active'})
        fig = px.bar(act_churn, x='Status', y='Exited',
                     title='Churn: Active vs Inactive Members',
                     color='Status',
                     color_discrete_map={'Inactive':'#e53935', 'Active':'#43a047'},
                     text=act_churn['Exited'].round(1).astype(str) + '%')
        fig.update_layout(height=380, yaxis_title='Churn Rate (%)',
                         showlegend=False, font=dict(family='Inter'))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    # Insight
    st.markdown("""
    <div class="insight-box">
        <span class="insight-icon">i</span><b>Key Insight:</b> Customers with 3-4 products show significantly higher churn,
        suggesting over-selling may backfire. Inactive members churn at nearly double the rate
        of active members — engagement programs are critical.
    </div>
    """, unsafe_allow_html=True)


# ── TAB 2: Geography ──
with tab2:
    st.markdown('<div class="section-title">Geographic Churn Analysis</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        geo_churn = filtered.groupby('Geography')['Exited'].agg(['mean','sum','count']).reset_index()
        geo_churn.columns = ['Geography','ChurnRate','Churned','Total']
        geo_churn['ChurnRate'] = geo_churn['ChurnRate'] * 100

        fig = px.bar(geo_churn, x='Geography', y='ChurnRate',
                     color='Geography', title='Churn Rate by Country',
                     color_discrete_sequence=['#1a73e8','#e53935','#43a047'],
                     text=geo_churn['ChurnRate'].round(1).astype(str) + '%')
        fig.update_layout(height=420, yaxis_title='Churn Rate (%)',
                         showlegend=False, font=dict(family='Inter'))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.pie(geo_churn, names='Geography', values='Churned',
                     title='Churn Distribution by Country',
                     color_discrete_sequence=['#1a73e8','#e53935','#43a047'],
                     hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=420, font=dict(family='Inter'))
        st.plotly_chart(fig, use_container_width=True)

    # Geographic Risk Index
    overall_cr = filtered['Exited'].mean()
    if overall_cr > 0:
        geo_risk = filtered.groupby('Geography')['Exited'].mean() / overall_cr
        st.markdown('<div class="section-title">Geographic Risk Index</div>',
                    unsafe_allow_html=True)
        risk_cols = st.columns(len(geo_risk))
        for i, (country, risk) in enumerate(geo_risk.items()):
            cls = 'kpi-danger' if risk > 1.2 else ('kpi-success' if risk < 0.9 else 'kpi-card')
            with risk_cols[i]:
                st.markdown(f'<div class="kpi-card {cls}"><h2>{risk:.2f}x</h2>'
                           f'<p>{country}</p></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        <span class="insight-icon">i</span><b>Key Insight:</b> Germany consistently shows the highest churn rate despite having
        fewer customers than France. This suggests country-specific service quality or
        competitive pressure issues that require targeted retention strategies.
    </div>
    """, unsafe_allow_html=True)


# ── TAB 3: Age & Tenure ──
with tab3:
    st.markdown('<div class="section-title">Age & Tenure Churn Comparison</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age_churn = filtered.groupby('AgeGroup', observed=True)['Exited'].mean().reset_index()
        age_churn['Exited'] = age_churn['Exited'] * 100
        fig = px.bar(age_churn, x='AgeGroup', y='Exited',
                     title='Churn Rate by Age Group',
                     color='AgeGroup',
                     color_discrete_sequence=['#4facfe','#667eea','#f093fb','#f5576c'],
                     text=age_churn['Exited'].round(1).astype(str) + '%')
        fig.update_layout(height=420, yaxis_title='Churn Rate (%)',
                         showlegend=False, font=dict(family='Inter'))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        tenure_churn = filtered.groupby('TenureGroup', observed=True)['Exited'].mean().reset_index()
        tenure_churn['Exited'] = tenure_churn['Exited'] * 100
        fig = px.bar(tenure_churn, x='TenureGroup', y='Exited',
                     title='Churn Rate by Tenure Group',
                     color='TenureGroup',
                     color_discrete_sequence=['#fa709a','#fee140','#43a047'],
                     text=tenure_churn['Exited'].round(1).astype(str) + '%')
        fig.update_layout(height=420, yaxis_title='Churn Rate (%)',
                         showlegend=False, font=dict(family='Inter'))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap: Geography × Age
    st.markdown('<div class="section-title">Geography × Age Interaction</div>',
                unsafe_allow_html=True)
    pivot = filtered.pivot_table(values='Exited', index='Geography',
                                  columns='AgeGroup', aggfunc='mean', observed=True) * 100
    fig = px.imshow(pivot.values, x=pivot.columns.astype(str), y=pivot.index,
                    color_continuous_scale='YlOrRd', aspect='auto',
                    title='Churn Rate Heatmap: Geography × Age',
                    text_auto='.1f')
    fig.update_layout(height=350, font=dict(family='Inter'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <span class="insight-icon">i</span><b>Key Insight:</b> Customers aged 46-60 show the highest churn rate across all
        geographies. This is the "pre-retirement" segment that may be consolidating finances
        or seeking better rates. New customers (0-2 years) also show elevated churn — the
        onboarding experience is critical.
    </div>
    """, unsafe_allow_html=True)


# ── TAB 4: High-Value Explorer ──
with tab4:
    st.markdown('<div class="section-title">High-Value Customer Churn Explorer</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        bal_churn = filtered.groupby('BalanceSegment')['Exited'].mean().reset_index()
        bal_churn['Exited'] = bal_churn['Exited'] * 100
        order = ['Zero', 'Low', 'Medium', 'High']
        bal_churn['BalanceSegment'] = pd.Categorical(bal_churn['BalanceSegment'],
                                                      categories=order, ordered=True)
        bal_churn = bal_churn.sort_values('BalanceSegment')
        fig = px.bar(bal_churn, x='BalanceSegment', y='Exited',
                     title='Churn Rate by Balance Segment',
                     color='BalanceSegment',
                     color_discrete_sequence=['#90a4ae','#4facfe','#667eea','#f5576c'],
                     text=bal_churn['Exited'].round(1).astype(str) + '%')
        fig.update_layout(height=420, yaxis_title='Churn Rate (%)',
                         showlegend=False, font=dict(family='Inter'))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        credit_churn = filtered.groupby('CreditBand', observed=True)['Exited'].mean().reset_index()
        credit_churn['Exited'] = credit_churn['Exited'] * 100
        fig = px.bar(credit_churn, x='CreditBand', y='Exited',
                     title='Churn Rate by Credit Score Band',
                     color='CreditBand',
                     color_discrete_sequence=['#f5576c','#fee140','#43a047'],
                     text=credit_churn['Exited'].round(1).astype(str) + '%')
        fig.update_layout(height=420, yaxis_title='Churn Rate (%)',
                         showlegend=False, font=dict(family='Inter'))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    # Revenue Risk
    st.markdown('<div class="section-title">Revenue Risk from Churn</div>',
                unsafe_allow_html=True)
    churned_df = filtered[filtered['Exited'] == 1]
    retained_df = filtered[filtered['Exited'] == 0]

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        total_bal_lost = churned_df['Balance'].sum()
        st.markdown(f'<div class="kpi-card kpi-danger"><h2>€{total_bal_lost/1e6:.1f}M</h2>'
                   f'<p>Total Balance Lost to Churn</p></div>', unsafe_allow_html=True)
    with rc2:
        avg_bal_churned = churned_df['Balance'].mean()
        st.markdown(f'<div class="kpi-card kpi-warning"><h2>€{avg_bal_churned:,.0f}</h2>'
                   f'<p>Avg Balance (Churned)</p></div>', unsafe_allow_html=True)
    with rc3:
        avg_bal_retained = retained_df['Balance'].mean()
        st.markdown(f'<div class="kpi-card kpi-success"><h2>€{avg_bal_retained:,.0f}</h2>'
                   f'<p>Avg Balance (Retained)</p></div>', unsafe_allow_html=True)

    # Balance vs Salary scatter
    sample = filtered.sample(min(2000, len(filtered)), random_state=42)
    fig = px.scatter(sample, x='Balance', y='EstimatedSalary',
                     color=sample['Exited'].map({0:'Retained', 1:'Churned'}),
                     color_discrete_map={'Retained':'#43a047', 'Churned':'#e53935'},
                     opacity=0.5, title='Balance vs Salary — Churn Pattern',
                     labels={'color': 'Status'})
    fig.update_layout(height=450, font=dict(family='Inter'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <span class="insight-icon">i</span><b>Key Insight:</b> High-balance customers who churn represent a significant
        revenue risk. The bank is losing its most valuable segment. Targeted premium
        retention programs (dedicated relationship managers, preferential rates) are
        essential for this group.
    </div>
    """, unsafe_allow_html=True)


# ── TAB 5: ML Model Results ──
with tab5:
    st.markdown('<div class="section-title">Machine Learning Model Results</div>',
                unsafe_allow_html=True)

    st.info("Run `python analysis.py` first to train models and generate charts. "
            "Below shows pre-computed model performance benchmarks.")

    # Show model metrics
    st.markdown("### Model Performance Comparison")

    metrics_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
        'Accuracy': ['~0.81', '~0.86', '~0.86'],
        'Precision': ['~0.56', '~0.75', '~0.74'],
        'Recall': ['~0.20', '~0.47', '~0.48'],
        'F1-Score': ['~0.29', '~0.58', '~0.58'],
        'AUC-ROC': ['~0.77', '~0.86', '~0.87'],
    }
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)

    # Show saved charts if they exist
    chart_dir = os.path.join(os.path.dirname(__file__), 'charts')
    for chart_name in ['12_feature_importance.png', '13_confusion_matrix.png', '14_model_comparison.png']:
        chart_path = os.path.join(chart_dir, chart_name)
        if os.path.exists(chart_path):
            st.image(chart_path, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <b>Key Insight:</b> Random Forest and Gradient Boosting significantly outperform
        Logistic Regression. Age and Balance are the most important features for predicting
        churn. The model can help identify at-risk customers for proactive retention outreach.
    </div>
    """, unsafe_allow_html=True)


# ── Footer ──
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; font-size:13px; padding:10px;">
    Customer Segmentation & Churn Pattern Analytics in European Banking<br>
    Unified Mentor Private Limited — Machine Learning Internship Project<br>
    Built with Streamlit + Python + Scikit-learn
</div>
""", unsafe_allow_html=True)
