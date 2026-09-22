"""
app.py - Data Analyzer AI Platform (Matplotlib / Seaborn / Plotly Dashboard Engine)
Enterprise Streamlit application for automated data analysis, multi-dashboard visual analytics,
machine learning driver modeling, and high-resolution PDF executive reporting.
"""

import io
import streamlit as st
import pandas as pd
import numpy as np

# Safe Matplotlib & Seaborn Imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cloud execution
import matplotlib.pyplot as plt
import seaborn as sns

# ReportLab Imports for Direct High-Res PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.multiclass import type_of_target

# Custom module safety wrappers
try:
    from web_research import detect_domain
except ImportError:
    def detect_domain(df):
        return {"domain": "General / Operations", "confidence": "Low", "matched_terms": []}

try:
    from insight_engine import analyze_data_quality, compute_relationships
except ImportError:
    def analyze_data_quality(df):
        return {
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "is_perfect_quality": df.isnull().sum().sum() == 0 and df.duplicated().sum() == 0,
            "total_missing": int(df.isnull().sum().sum()),
            "cols_with_missing": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "empty_cols": [col for col in df.columns if df[col].isnull().all()],
            "outliers": {}
        }
    def compute_relationships(df):
        return []

try:
    from question_engine import generate_analytical_questions
except ImportError:
    def generate_analytical_questions(df, domain_info):
        return [
            {"category": "Performance", "question": "Which categories drive the highest metric totals?", "purpose": "Identify core operational drivers."},
            {"category": "Risk", "question": "Are there extreme statistical anomalies in numerical metrics?", "purpose": "Mitigate operational risks."},
            {"category": "Trend", "question": "What is the continuous metric trajectory over time?", "purpose": "Evaluate performance stability."}
        ]

try:
    from recommendation_engine import generate_recommendations
except ImportError:
    def generate_recommendations(df, quality_info, relationships, domain_info):
        return []


# Streamlit Page Setup
st.set_page_config(
    page_title="Data Analyzer AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #F8FAFC;
    }
    .rec-box {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-left: 5px solid #2563EB;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .rec-title {
        font-size: 18px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 10px;
    }
    .field-label {
        font-weight: 700;
        color: #334155;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)


# Column Cleaning Helper: Ignore Non-Analytical ID Columns
def get_analytical_columns(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    clean_num_cols = [c for c in num_cols if not (c.lower().endswith('id') or c.lower().startswith('id') or df[c].nunique() == len(df))]
    if not clean_num_cols and num_cols:
        clean_num_cols = num_cols

    clean_cat_cols = [c for c in cat_cols if not (c.lower().endswith('id') or c.lower().startswith('id')) and df[c].nunique() <= 50]
    if not clean_cat_cols and cat_cols:
        clean_cat_cols = cat_cols[:5]

    return clean_num_cols, clean_cat_cols


# Data Loader with Memory Optimization
@st.cache_data
def load_data(uploaded_file, row_limit=5000):
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            try:
                df = pd.read_excel(uploaded_file, sheet_name=0, engine='calamine')
            except Exception:
                df = pd.read_excel(uploaded_file, sheet_name=0, engine='openpyxl')
            
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')

        # Auto-parse datetime columns if applicable
        for col in df.select_dtypes(include=['object']).columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

        if len(df) > row_limit:
            st.warning(f"⚠️ Dataset contains {len(df):,} records. Sampled down to {row_limit:,} rows for performance optimization.")
            df = df.sample(n=row_limit, random_state=42).reset_index(drop=True)
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# Random Forest Driver Model
def train_risk_model(df, target_col):
    data = df.copy().dropna()
    if data.empty:
        raise ValueError("Dataset has no complete rows after dropping null values.")
        
    encoders = {}
    
    date_cols = data.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
    for col in date_cols:
        if col != target_col:
            data[f"{col}_Year"] = data[col].dt.year
            data[f"{col}_Month"] = data[col].dt.month
            data = data.drop(columns=[col])

    for col in list(data.columns):
        if col != target_col and (col.lower().endswith('id') or data[col].nunique() > 100):
            data = data.drop(columns=[col])

    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        if col != target_col:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
        
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    if X.empty or X.shape[1] == 0:
        raise ValueError("Insufficient analytical features remaining to train model.")

    target_type = type_of_target(y)
    
    if target_type == 'continuous':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
    else:
        if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
            target_le = LabelEncoder()
            y = target_le.fit_transform(y.astype(str))
            encoders[target_col] = target_le
            
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
    
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return model, encoders, importances, X.columns.tolist()


# High-Resolution Native PDF Generator
def generate_native_pdf_report(df, domain_info, quality_info, recommendations):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor('#1E3A8A'))
    h2_style = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontSize=11, leading=14, textColor=colors.HexColor('#1E40AF'), spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontSize=8.5, leading=11, textColor=colors.HexColor('#334155'))
    bold_style = ParagraphStyle('ReportBold', parent=body_style, fontName='Helvetica-Bold')

    story.append(Paragraph("Data Analyzer AI — Executive Analytics Report", title_style))
    story.append(Spacer(1, 4))
    domain_text = f"<b>Detected Domain:</b> {domain_info.get('domain', 'General')} | <b>Confidence:</b> {domain_info.get('confidence', 'Low')}"
    story.append(Paragraph(domain_text, body_style))
    story.append(Spacer(1, 6))

    # Overview Table
    story.append(Paragraph("1. Dataset Overview Metrics", h2_style))
    overview_data = [
        ["Total Records", "Total Columns", "Duplicate Rows", "Missing Values"],
        [
            f"{quality_info.get('total_rows', len(df)):,}",
            f"{quality_info.get('total_cols', len(df.columns)):,}",
            f"{quality_info.get('duplicate_rows', 0)}",
            f"{quality_info.get('total_missing', 0)}"
        ]
    ]
    t_overview = Table(overview_data, colWidths=[130, 130, 130, 130])
    t_overview.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_overview)
    story.append(Spacer(1, 8))

    # Visual Analytics High-Res Render Section
    story.append(Paragraph("2. High-Resolution Visual Dashboards", h2_style))
    num_cols, cat_cols = get_analytical_columns(df)

    if num_cols:
        try:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.5), dpi=300)
            fig.subplots_adjust(hspace=0.45, wspace=0.35)

            if cat_cols:
                top_cats = df.groupby(cat_cols[0])[num_cols[0]].sum().nlargest(6)
                top_cats.plot(kind='bar', ax=axs[0,0], color='#2563EB', edgecolor='none')
                axs[0,0].set_title(f"1. Total {num_cols[0]} by {cat_cols[0]}", fontsize=8.5, fontweight='bold', pad=6)
                axs[0,0].tick_params(axis='x', rotation=25, labelsize=7)
            else:
                axs[0,0].hist(df[num_cols[0]].dropna(), bins=15, color='#2563EB', edgecolor='white')
                axs[0,0].set_title(f"1. Distribution of {num_cols[0]}", fontsize=8.5, fontweight='bold', pad=6)

            if len(cat_cols) > 1:
                pie_data = df[cat_cols[1]].value_counts().head(5)
                axs[0,1].pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140,
                            colors=['#2563EB', '#3B82F6', '#60A5FA', '#93C5FD', '#BFDBFE'], textprops={'fontsize': 7})
                axs[0,1].set_title(f"2. {cat_cols[1]} Market Share", fontsize=8.5, fontweight='bold', pad=6)
            else:
                axs[0,1].boxplot(df[num_cols[0]].dropna(), vert=True, patch_artist=True,
                                boxprops=dict(facecolor="#3B82F6", color="#1E40AF"))
                axs[0,1].set_title(f"2. Outliers: {num_cols[0]}", fontsize=8.5, fontweight='bold', pad=6)

            if cat_cols:
                avg_data = df.groupby(cat_cols[0])[num_cols[0]].mean().nlargest(6).sort_values()
                avg_data.plot(kind='barh', ax=axs[1,0], color='#1D4ED8')
                axs[1,0].set_title(f"3. Avg {num_cols[0]} per {cat_cols[0]}", fontsize=8.5, fontweight='bold', pad=6)
                axs[1,0].tick_params(axis='y', labelsize=7)

            if cat_cols:
                vol_data = df[cat_cols[0]].value_counts().head(6)
                vol_data.plot(kind='bar', ax=axs[1,1], color='#60A5FA', edgecolor='none')
                axs[1,1].set_title(f"4. Record Volume by {cat_cols[0]}", fontsize=8.5, fontweight='bold', pad=6)
                axs[1,1].tick_params(axis='x', rotation=25, labelsize=7)

            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            img_buf.seek(0)
            
            story.append(Image(img_buf, width=500, height=330))
        except Exception as err:
            story.append(Paragraph(f"<i>Visual Rendering Exception: {err}</i>", body_style))

    story.append(Spacer(1, 8))
    story.append(Paragraph("3. Executive Analyst Recommendations", h2_style))
    if recommendations:
        for idx, rec in enumerate(recommendations[:3], 1):
            story.append(Paragraph(f"<b>Rec {idx}: {rec.get('business_area', 'Strategy')}</b>", bold_style))
            story.append(Paragraph(f"• <b>Action:</b> {rec.get('action_development', 'N/A')}", body_style))
            story.append(Spacer(1, 3))
    else:
        story.append(Paragraph("<i>No automated recommendations generated for this dataset.</i>", body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# Sidebar Navigation
st.sidebar.title("🧠 Data Analyzer AI")
st.sidebar.write("Upload a CSV or Excel dataset to begin visual analytics.")

uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel File", type=["csv", "xlsx"])
row_limit = st.sidebar.slider("Sampling Row Limit:", min_value=1000, max_value=10000, value=5000, step=1000)

if uploaded_file is not None:
    df = load_data(uploaded_file, row_limit=row_limit)

    if df is not None and not df.empty:
        num_cols, cat_cols = get_analytical_columns(df)
        
        target_field = st.sidebar.selectbox("Select Target / Driver Field:", df.columns)
        cmap_choice = st.sidebar.selectbox("Matplotlib Color Map:", ["Blues", "viridis", "plasma", "cividis", "coolwarm", "YlGnBu"], index=0)

        try:
            domain_info = detect_domain(df)
        except Exception:
            domain_info = {"domain": "General / Operations", "confidence": "Low", "matched_terms": []}

        try:
            quality_info = analyze_data_quality(df)
        except Exception:
            quality_info = {
                "total_rows": len(df), "total_cols": len(df.columns),
                "is_perfect_quality": False, "cols_with_missing": {},
                "duplicate_rows": 0, "empty_cols": [], "outliers": {}
            }

        try:
            relationships = compute_relationships(df)
        except Exception:
            relationships = []

        try:
            questions = generate_analytical_questions(df, domain_info)
        except Exception:
            questions = []

        try:
            recommendations = generate_recommendations(df, quality_info, relationships, domain_info)
        except Exception:
            recommendations = []

        # Sidebar PDF Export Button
        st.sidebar.markdown("---")
        pdf_bytes = generate_native_pdf_report(df, domain_info, quality_info, recommendations)
        st.sidebar.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name="Executive_Analytics_Report.pdf",
            mime="application/pdf",
            type="primary"
        )

        # App Header
        st.title("Data Analyzer AI")
        st.caption("Intelligent Visual Analytics & Reporting Platform (Matplotlib / Seaborn)")
        st.info(f"Detected Industry / Domain: **{domain_info.get('domain', 'General')}** (Confidence: **{domain_info.get('confidence', 'Low')}**)")

        # Main Navigation Tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📋 Overview",
            "📊 Dashboard Analytics",
            "🔍 Data Quality",
            "📈 Statistics",
            "❓ Analytical Questions",
            "🤖 Risk & Feature Importance",
            "💡 Recommendations"
        ])

        # TAB 1: OVERVIEW
        with tab1:
            st.subheader("Dataset Structure & Preview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Loaded Records", f"{quality_info.get('total_rows', len(df)):,}")
            c2.metric("Total Columns", f"{quality_info.get('total_cols', len(df.columns)):,}")
            c3.metric("Numeric Metrics", len(num_cols))
            c4.metric("Categorical Dimensions", len(cat_cols))

            st.markdown("---")
            st.subheader("Dataset First 10 Rows")
            st.dataframe(df.head(10), use_container_width=True)

        # TAB 2: DASHBOARD ANALYTICS
        with tab2:
            st.subheader("Matplotlib Dashboard Analytics (4 Figures Per Dashboard)")
            
            date_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()

            db_tab1, db_tab2, db_tab3, db_tab4 = st.tabs([
                "Dashboard 1: Categorical Breakdown",
                "Dashboard 2: Metric Relationships",
                "Dashboard 3: Distribution & Trends",
                "Dashboard 4: Interactive Query Engine"
            ])

            # DASHBOARD 1
            with db_tab1:
                st.markdown("#### Dashboard 1: Categorical & Dimension Breakdown")
                if cat_cols and num_cols:
                    col_a, col_b = st.columns(2)
                    c_dim = col_a.selectbox("Categorical Dimension:", cat_cols, index=0, key="db1_cat")
                    m_metric = col_b.selectbox("Numerical Metric:", num_cols, index=0, key="db1_num")

                    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
                    fig.subplots_adjust(hspace=0.4, wspace=0.3)

                    # 1. Total Bar
                    agg_sum = df.groupby(c_dim)[m_metric].sum().nlargest(8)
                    sns.barplot(x=agg_sum.index, y=agg_sum.values, ax=axs[0,0], palette=cmap_choice)
                    axs[0,0].set_title(f"1. Total {m_metric} by {c_dim}", fontweight='bold')
                    axs[0,0].tick_params(axis='x', rotation=30)

                    # 2. Pie Chart
                    pie_data = df[c_dim].value_counts().head(5)
                    axs[0,1].pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette(cmap_choice, len(pie_data)))
                    axs[0,1].set_title(f"2. {c_dim} Frequency Share", fontweight='bold')

                    # 3. Horizontal Average Bar
                    agg_avg = df.groupby(c_dim)[m_metric].mean().nlargest(8).sort_values()
                    sns.barplot(x=agg_avg.values, y=agg_avg.index, ax=axs[1,0], palette=cmap_choice)
                    axs[1,0].set_title(f"3. Avg {m_metric} by {c_dim}", fontweight='bold')

                    # 4. Count Bar
                    agg_cnt = df[c_dim].value_counts().head(8)
                    sns.barplot(x=agg_cnt.index, y=agg_cnt.values, ax=axs[1,1], palette="Blues_r")
                    axs[1,1].set_title(f"4. Record Count per {c_dim}", fontweight='bold')
                    axs[1,1].tick_params(axis='x', rotation=30)

                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("Requires categorical and numerical fields to display Dashboard 1.")

            # DASHBOARD 2
            with db_tab2:
                st.markdown("#### Dashboard 2: Cross-Metric Relationship Analysis")
                if len(num_cols) >= 2:
                    col_a, col_b = st.columns(2)
                    x_met = col_a.selectbox("X-Axis Metric:", num_cols, index=0, key="db2_x")
                    y_met = col_b.selectbox("Y-Axis Metric:", num_cols, index=1 if len(num_cols) > 1 else 0, key="db2_y")

                    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
                    fig.subplots_adjust(hspace=0.4, wspace=0.3)

                    # 1. Scatter Regression
                    sns.regplot(data=df, x=x_met, y=y_met, ax=axs[0,0], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
                    axs[0,0].set_title(f"1. Scatter ({x_met} vs {y_met})", fontweight='bold')

                    # 2. KDE Density Heatmap
                    sns.kdeplot(data=df, x=x_met, y=y_met, ax=axs[0,1], cmap=cmap_choice, fill=True)
                    axs[0,1].set_title(f"2. Density Kernel ({x_met} vs {y_met})", fontweight='bold')

                    # 3. Profile Line Chart
                    df_sorted = df.sort_values(by=x_met).head(150)
                    axs[1,0].plot(df_sorted[x_met].values, df_sorted[y_met].values, color='#2563EB', linewidth=1.5)
                    axs[1,0].set_title(f"3. Profile Trajectory ({x_met} vs {y_met})", fontweight='bold')
                    axs[1,0].set_xlabel(x_met)
                    axs[1,0].set_ylabel(y_met)

                    # 4. Correlation Matrix Heatmap
                    corr_matrix = df[num_cols[:6]].corr()
                    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", ax=axs[1,1], cbar=False)
                    axs[1,1].set_title("4. Multi-Metric Correlation", fontweight='bold')

                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("Dashboard 2 requires at least two numerical metrics.")

            # DASHBOARD 3
            with db_tab3:
                st.markdown("#### Dashboard 3: Statistical Distributions & Time Trends")
                if num_cols:
                    col_a, col_b = st.columns(2)
                    dist_col = col_a.selectbox("Metric for Distribution:", num_cols, index=0, key="db3_dist")
                    group_cat = col_b.selectbox("Category Dimension Split:", [None] + cat_cols, index=0, key="db3_split")

                    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
                    fig.subplots_adjust(hspace=0.4, wspace=0.3)

                    # 1. Boxplot
                    sns.boxplot(data=df, x=group_cat, y=dist_col, ax=axs[0,0], palette=cmap_choice)
                    axs[0,0].set_title(f"1. Boxplot Spread for '{dist_col}'", fontweight='bold')
                    if group_cat:
                        axs[0,0].tick_params(axis='x', rotation=30)

                    # 2. Violin Plot
                    sns.violinplot(data=df, x=group_cat, y=dist_col, ax=axs[0,1], palette=cmap_choice)
                    axs[0,1].set_title(f"2. Violin Density for '{dist_col}'", fontweight='bold')
                    if group_cat:
                        axs[0,1].tick_params(axis='x', rotation=30)

                    # 3. Histplot
                    sns.histplot(data=df, x=dist_col, hue=group_cat, kde=True, ax=axs[1,0], palette="Set2")
                    axs[1,0].set_title(f"3. Histogram & KDE of '{dist_col}'", fontweight='bold')

                    # 4. Timeline or ECDF
                    if date_cols:
                        dt_col = date_cols[0]
                        trend_df = df.groupby(dt_col)[dist_col].mean().reset_index()
                        axs[1,1].plot(trend_df[dt_col], trend_df[dist_col], color='#1D4ED8', marker='o', markersize=3)
                        axs[1,1].set_title(f"4. Average '{dist_col}' Timeline", fontweight='bold')
                        axs[1,1].tick_params(axis='x', rotation=30)
                    else:
                        sns.ecdfplot(data=df, x=dist_col, hue=group_cat, ax=axs[1,1])
                        axs[1,1].set_title(f"4. ECDF Cumulative Spread for '{dist_col}'", fontweight='bold')

                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("Dashboard 3 requires numerical fields.")

            # DASHBOARD 4
            with db_tab4:
                st.markdown("#### Dashboard 4: Interactive Query Engine")
                if questions:
                    suggested_q = [q.get('question', '') for q in questions if q.get('question')]
                    selected_q = st.selectbox("Select a generated question:", ["-- Select or type below --"] + suggested_q)
                else:
                    selected_q = "-- Select or type below --"

                custom_q = st.text_input("Or type your question:", value="" if selected_q == "-- Select or type below --" else selected_q)

                if st.button("🔎 Render Dashboard 4", type="primary"):
                    if custom_q.strip():
                        matched_num = [c for c in num_cols if c.lower() in custom_q.lower()]
                        matched_cat = [c for c in cat_cols if c.lower() in custom_q.lower()]

                        m_col = matched_num[0] if matched_num else num_cols[0]
                        c_col = matched_cat[0] if matched_cat else (cat_cols[0] if cat_cols else None)

                        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
                        fig.subplots_adjust(hspace=0.4, wspace=0.3)

                        if c_col:
                            res1 = df.groupby(c_col)[m_col].sum().nlargest(8)
                            sns.barplot(x=res1.index, y=res1.values, ax=axs[0,0], palette="Blues_r")
                            axs[0,0].set_title(f"1. Total {m_col} by {c_col}", fontweight='bold')
                            axs[0,0].tick_params(axis='x', rotation=30)

                            res2 = df.groupby(c_col)[m_col].mean().nlargest(8)
                            sns.barplot(x=res2.index, y=res2.values, ax=axs[0,1], palette="Blues")
                            axs[0,1].set_title(f"2. Avg {m_col} by {c_col}", fontweight='bold')
                            axs[0,1].tick_params(axis='x', rotation=30)

                            res3 = df[c_col].value_counts().head(8)
                            axs[1,0].pie(res3, labels=res3.index, autopct='%1.1f%%')
                            axs[1,0].set_title(f"3. {c_col} Proportion", fontweight='bold')
                        else:
                            sns.histplot(df[m_col], ax=axs[0,0], kde=True)
                            axs[0,0].set_title(f"1. Histogram of {m_col}", fontweight='bold')

                            sns.boxplot(y=df[m_col], ax=axs[0,1])
                            axs[0,1].set_title(f"2. Boxplot of {m_col}", fontweight='bold')

                            axs[1,0].plot(df.index, df[m_col], alpha=0.6)
                            axs[1,0].set_title(f"3. Sequential Trend of {m_col}", fontweight='bold')

                        if len(num_cols) >= 2:
                            second_m = num_cols[1] if num_cols[1] != m_col else num_cols[0]
                            sns.scatterplot(data=df, x=m_col, y=second_m, ax=axs[1,1])
                            axs[1,1].set_title(f"4. {m_col} vs {second_m}", fontweight='bold')
                        else:
                            sns.ecdfplot(data=df, x=m_col, ax=axs[1,1])
                            axs[1,1].set_title(f"4. ECDF of {m_col}", fontweight='bold')

                        st.pyplot(fig)
                        plt.close(fig)

        # TAB 3: DATA QUALITY
        with tab3:
            st.subheader("Data Quality & Integrity Audit")
            if quality_info.get("is_perfect_quality", False):
                st.success("✅ Dataset is completely clean with zero missing values or duplicate rows.")
            else:
                st.warning("⚠️ Data Quality Notice: Missing values or duplicate rows detected.")

            col_q1, col_q2 = st.columns(2)
            with col_q1:
                st.write("**Missing Values Audit**")
                missing = quality_info.get("cols_with_missing", {})
                if missing:
                    st.dataframe(pd.DataFrame(list(missing.items()), columns=["Column", "Missing Count"]), use_container_width=True)
                else:
                    st.info("Zero missing entries across all columns.")

            with col_q2:
                st.write("**Data Integrity Summary**")
                st.write(f"- Duplicate Records: **{quality_info.get('duplicate_rows', 0)}**")
                st.write(f"- Empty Columns: **{len(quality_info.get('empty_cols', []))}**")
                st.write(f"- Columns with Outliers: **{len(quality_info.get('outliers', {}))}**")

        # TAB 4: STATISTICS
        with tab4:
            st.subheader("Statistical Summary")
            num_df = df[num_cols] if num_cols else pd.DataFrame()
            if not num_df.empty:
                st.write("**Numerical Summary**")
                st.dataframe(num_df.describe().T, use_container_width=True)

            cat_df = df[cat_cols] if cat_cols else pd.DataFrame()
            if not cat_df.empty:
                st.write("**Categorical Summary**")
                st.dataframe(cat_df.describe().T, use_container_width=True)

        # TAB 5: ANALYTICAL QUESTIONS
        with tab5:
            st.subheader("Domain Analytical Questions & Hypotheses")
            if questions:
                for q in questions:
                    with st.expander(f"📌 {q.get('category', 'Analysis')}: {q.get('question', '')}"):
                        st.write(f"**Purpose:** {q.get('purpose', '')}")
            else:
                st.info("No analytical questions generated for this dataset.")

        # TAB 6: RISK & FEATURE IMPORTANCE
        with tab6:
            st.subheader(f"Feature Importance Driver Model: '{target_field}'")
            st.write("Train a Random Forest model on demand to uncover primary metric impact drivers.")
            
            if st.button("🚀 Train & Calculate Feature Importances", type="primary"):
                with st.spinner("Training Random Forest model..."):
                    try:
                        model, encoders, importances, feature_names = train_risk_model(df, target_field)
                        st.success("Model trained successfully!")
                        
                        m_col1, m_col2 = st.columns([2, 1])
                        with m_col1:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            sns.barplot(data=importances, x='Importance', y='Feature', palette="viridis", ax=ax)
                            ax.set_title(f"Feature Importances Driving '{target_field}'", fontweight='bold')
                            st.pyplot(fig)
                            plt.close(fig)
                            
                        with m_col2:
                            st.write("**Feature Importance Data**")
                            st.dataframe(importances, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error training model: {e}")

        # TAB 7: RECOMMENDATIONS
        with tab7:
            st.subheader("Executive Analyst Recommendations")
            if recommendations:
                for rec in recommendations:
                    st.markdown(f"""
                    <div class="rec-box">
                        <div class="rec-title">📍 Area: {rec.get('business_area', 'General Strategy')}</div>
                        <p><span class="field-label">Action:</span> {rec.get('action_development', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No automated recommendations generated for this dataset.")

else:
    st.info("👆 Upload a CSV or Excel dataset from the sidebar to launch the analytics suite.")
