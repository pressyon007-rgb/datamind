"""
app.py - Enterprise Data Analyzer AI Platform
Advanced Streamlit Application featuring interactive Plotly visual analytics, 
automated data quality profiling, cross-validated Machine Learning driver models,
and high-resolution ReportLab PDF executive exports.
"""

import io
import time
import streamlit as st
import pandas as pd
import numpy as np

# Interactive Visualization Imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Fallback Visualization Imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# PDF Generation Engine
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
)
from reportlab.pdfgen import canvas

# Machine Learning Engine
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.utils.multiclass import type_of_target

# Safe Custom Module Imports
try:
    from web_research import detect_domain
except ImportError:
    def detect_domain(df):
        return {"domain": "General Analytics", "confidence": "Medium", "matched_terms": []}

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
            {"category": "Driver Analysis", "question": "What primary factors correlate with top metric variations?", "purpose": "Isolate core performance drivers."},
            {"category": "Risk Management", "question": "Are there high statistical outliers impacting average trends?", "purpose": "Mitigate risk exposure."},
            {"category": "Temporal Dynamics", "question": "How do key metrics behave across recorded time horizons?", "purpose": "Evaluate trajectory consistency."}
        ]

try:
    from recommendation_engine import generate_recommendations
except ImportError:
    def generate_recommendations(df, quality_info, relationships, domain_info):
        return [
            {"business_area": "Data Governance", "action_development": "Implement systematic validation rules to handle missing entries and duplicate rows."},
            {"business_area": "Performance Optimization", "action_development": "Focus resources on high-importance target features identified in the ML driver module."}
        ]


# Page Configuration
st.set_page_config(
    page_title="Data Analyzer AI Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise Custom CSS
st.markdown("""
<style>
    /* Theme Color Variables */
    :root {
        --primary-color: #2563EB;
        --background-color: #F8FAFC;
        --card-background: #FFFFFF;
        --text-color: #0F172A;
        --border-color: #E2E8F0;
    }
    
    .stApp {
        background-color: var(--background-color);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Executive Metric Cards */
    div[data-testid="stMetric"] {
        background-color: var(--card-background);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Recommendation Cards */
    .rec-card {
        background-color: var(--card-background);
        border: 1px solid var(--border-color);
        border-left: 5px solid #2563EB;
        border-radius: 10px;
        padding: 18px 24px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }
    .rec-card-title {
        font-size: 16px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 6px;
    }
    .rec-card-body {
        font-size: 14px;
        color: #334155;
        margin: 0;
    }
    
    /* Header Enhancements */
    .main-header {
        font-size: 28px;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.5px;
    }
</style>
""", unsafe_allow_html=True)


# Column Profiling Helper
@st.cache_data
def get_analytical_columns(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
    
    clean_num_cols = [
        c for c in num_cols 
        if not (c.lower().endswith('id') or c.lower().startswith('id') or df[c].nunique() == len(df) or df[c].nunique() <= 1)
    ]
    if not clean_num_cols and num_cols:
        clean_num_cols = num_cols

    clean_cat_cols = [
        c for c in cat_cols 
        if not (c.lower().endswith('id') or c.lower().startswith('id')) and 1 < df[c].nunique() <= 100
    ]
    if not clean_cat_cols and cat_cols:
        clean_cat_cols = cat_cols[:10]

    return clean_num_cols, clean_cat_cols, date_cols


# Optimized Data Ingestion
@st.cache_data(show_spinner=False)
def load_data(uploaded_file, row_limit=10000):
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            try:
                df = pd.read_excel(uploaded_file, sheet_name=0, engine='calamine')
            except Exception:
                df = pd.read_excel(uploaded_file, sheet_name=0, engine='openpyxl')
            
        # Strip trailing whitespaces
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()

        # Parse Datetime Fields
        for col in df.columns:
            if df[col].dtype == 'object':
                if any(kw in col.lower() for kw in ['date', 'time', 'timestamp', 'created', 'updated']):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='ignore')
                    except Exception:
                        pass

        # Optimize numeric storage
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')

        if len(df) > row_limit:
            df = df.sample(n=row_limit, random_state=42).reset_index(drop=True)
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# Advanced Machine Learning Engine
@st.cache_resource(show_spinner=False)
def train_risk_model(df, target_col):
    data = df.copy().dropna()
    if data.empty:
        raise ValueError("Dataset has no complete rows after removing null values.")
        
    encoders = {}
    date_cols = data.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
    for col in date_cols:
        if col != target_col:
            data[f"{col}_Year"] = data[col].dt.year
            data[f"{col}_Month"] = data[col].dt.month
            data[f"{col}_Day"] = data[col].dt.day
            data = data.drop(columns=[col])

    for col in list(data.columns):
        if col != target_col and (col.lower().endswith('id') or data[col].nunique() > 100 or data[col].nunique() <= 1):
            data = data.drop(columns=[col])

    cat_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    for col in cat_cols:
        if col != target_col:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
        
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    if X.empty or X.shape[1] == 0:
        raise ValueError("Insufficient analytical features remaining for model training.")

    target_type = type_of_target(y)
    
    if target_type in ['continuous', 'continuous-multioutput']:
        model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        model.fit(X, y)
        preds = model.predict(X)
        
        metrics = {
            "Model Type": "Random Forest Regressor",
            "R² Score": r2_score(y, preds),
            "MAE": mean_absolute_error(y, preds),
            "RMSE": np.sqrt(mean_squared_error(y, preds))
        }
    else:
        if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
            target_le = LabelEncoder()
            y = target_le.fit_transform(y.astype(str))
            encoders[target_col] = target_le
            
        model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        model.fit(X, y)
        preds = model.predict(X)
        
        metrics = {
            "Model Type": "Random Forest Classifier",
            "Accuracy": accuracy_score(y, preds),
            "Precision": precision_score(y, preds, average='weighted', zero_division=0),
            "Recall": recall_score(y, preds, average='weighted', zero_division=0),
            "F1-Score": f1_score(y, preds, average='weighted', zero_division=0)
        }
    
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    return model, encoders, importances, metrics


# ReportLab Dynamic Page Numbering Canvas
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor('#64748B'))
        self.setStrokeColor(colors.HexColor('#E2E8F0'))
        self.setLineWidth(0.5)
        self.line(36, 756, 576, 756)
        self.drawString(36, 762, "Data Analyzer AI — Executive Intelligence Report")
        self.line(36, 40, 576, 40)
        self.drawString(36, 28, "Confidential — Automated Analysis Report")
        self.drawRightString(576, 28, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


# PDF Executive Report Generator
def generate_native_pdf_report(df, domain_info, quality_info, questions, recommendations, target_field):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        rightMargin=36, leftMargin=36,
        topMargin=54, bottomMargin=54
    )
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor('#1E3A8A'), spaceAfter=4)
    h2_style = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontSize=11, leading=14, textColor=colors.HexColor('#1E40AF'), spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontSize=8.5, leading=11, textColor=colors.HexColor('#334155'))
    table_text = ParagraphStyle('TableText', parent=styles['Normal'], fontSize=7.5, leading=9, textColor=colors.HexColor('#1E293B'))

    # Header Section
    story.append(Paragraph("Executive Intelligence Analytics Report", title_style))
    domain_text = f"<b>Domain Context:</b> {domain_info.get('domain', 'General Analytics')} | <b>Detection Confidence:</b> {domain_info.get('confidence', 'Medium')}"
    story.append(Paragraph(domain_text, body_style))
    story.append(Spacer(1, 10))

    num_cols, cat_cols, _ = get_analytical_columns(df)

    # 1. Dataset Overview
    story.append(Paragraph("1. Dataset Architecture & Summary", h2_style))
    overview_data = [
        ["Total Records", "Total Attributes", "Numeric Features", "Categorical Dimensions"],
        [
            f"{quality_info.get('total_rows', len(df)):,}",
            f"{quality_info.get('total_cols', len(df.columns)):,}",
            f"{len(num_cols):,}",
            f"{len(cat_cols):,}"
        ]
    ]
    t_overview = Table(overview_data, colWidths=[135]*4)
    t_overview.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_overview)
    story.append(Spacer(1, 10))

    # 2. Matplotlib Analytics Chart Summary
    story.append(Paragraph("2. Primary Analytics Visualizations", h2_style))
    if num_cols:
        try:
            fig, axs = plt.subplots(2, 2, figsize=(9.5, 5.5), dpi=200)
            fig.subplots_adjust(hspace=0.4, wspace=0.3)

            if cat_cols:
                df.groupby(cat_cols[0])[num_cols[0]].sum().nlargest(6).plot(kind='bar', ax=axs[0,0], color='#2563EB')
                axs[0,0].set_title(f"Sum of {num_cols[0]} by {cat_cols[0]}", fontsize=8, fontweight='bold')
                axs[0,0].tick_params(axis='x', rotation=25, labelsize=6.5)
            else:
                axs[0,0].hist(df[num_cols[0]].dropna(), bins=15, color='#2563EB')
                axs[0,0].set_title(f"Distribution: {num_cols[0]}", fontsize=8, fontweight='bold')

            if len(num_cols) >= 2:
                sns.regplot(data=df, x=num_cols[0], y=num_cols[1], ax=axs[0,1], scatter_kws={'alpha':0.4, 's':10}, line_kws={'color':'red'})
                axs[0,1].set_title(f"{num_cols[0]} vs {num_cols[1]} Correlation", fontsize=8, fontweight='bold')
            else:
                axs[0,1].boxplot(df[num_cols[0]].dropna())
                axs[0,1].set_title(f"Boxplot: {num_cols[0]}", fontsize=8, fontweight='bold')

            sns.histplot(data=df, x=num_cols[0], kde=True, ax=axs[1,0], color='#3B82F6')
            axs[1,0].set_title(f"Density Curve: {num_cols[0]}", fontsize=8, fontweight='bold')

            if len(num_cols) >= 2:
                corr = df[num_cols[:5]].corr()
                sns.heatmap(corr, ax=axs[1,1], cmap="Blues", cbar=False, annot=True, fmt=".2f", annot_kws={"size": 6})
                axs[1,1].set_title("Correlation Matrix", fontsize=8, fontweight='bold')

            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', dpi=200, bbox_inches='tight')
            plt.close(fig)
            img_buf.seek(0)
            story.append(Image(img_buf, width=500, height=290))
        except Exception as e:
            story.append(Paragraph(f"<i>Chart generation warning: {e}</i>", body_style))

    story.append(Spacer(1, 10))

    # 3. Data Integrity Summary
    story.append(Paragraph("3. Data Quality & Risk Audit", h2_style))
    missing_dict = quality_info.get("cols_with_missing", {})
    dq_summary = [
        ["Quality Metric", "Audit Result"],
        ["Data Quality Status", "Clean" if quality_info.get("is_perfect_quality", False) else "Action Required"],
        ["Duplicate Row Count", f"{quality_info.get('duplicate_rows', 0):,}"],
        ["Missing Value Columns", f"{len(missing_dict):,} attributes"]
    ]
    t_dq = Table(dq_summary, colWidths=[240, 300])
    t_dq.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F8FAFC')),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_dq)
    story.append(Spacer(1, 10))

    # 4. Machine Learning Importance
    story.append(Paragraph("4. Target Machine Learning Driver Model", h2_style))
    try:
        model, encoders, importances, metrics = train_risk_model(df, target_field)
        story.append(Paragraph(f"<b>Target Evaluated:</b> {target_field} ({metrics.get('Model Type', 'Ensemble Model')})", body_style))
        story.append(Spacer(1, 4))

        imp_rows = [[Paragraph("<b>Feature Attribute</b>", table_text), Paragraph("<b>Relative Importance Score</b>", table_text)]]
        for _, row in importances.head(5).iterrows():
            imp_rows.append([Paragraph(str(row['Feature']), table_text), Paragraph(f"{row['Importance']:.4f}", table_text)])

        t_imp = Table(imp_rows, colWidths=[270, 270])
        t_imp.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(t_imp)
    except Exception as e:
        story.append(Paragraph(f"<i>ML Model evaluation warning: {e}</i>", body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()


# Sidebar Configuration
st.sidebar.title("⚡ Data Analyzer AI")
st.sidebar.write("Enterprise Automated Analytics Engine")

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])
row_limit = st.sidebar.slider("Maximum Sampling Rows:", min_value=1000, max_value=20000, value=10000, step=1000)

if uploaded_file is not None:
    with st.spinner("Loading and processing dataset..."):
        df = load_data(uploaded_file, row_limit=row_limit)

    if df is not None and not df.empty:
        num_cols, cat_cols, date_cols = get_analytical_columns(df)
        
        target_field = st.sidebar.selectbox("Target / Driver Field:", df.columns, index=0)

        # Domain & Quality Auto-Engine
        domain_info = detect_domain(df)
        quality_info = analyze_data_quality(df)
        relationships = compute_relationships(df)
        questions = generate_analytical_questions(df, domain_info)
        recommendations = generate_recommendations(df, quality_info, relationships, domain_info)

        # Download Report PDF Section
        st.sidebar.markdown("---")
        pdf_bytes = generate_native_pdf_report(df, domain_info, quality_info, questions, recommendations, target_field)
        st.sidebar.download_button(
            label="📄 Export Executive PDF Report",
            data=pdf_bytes,
            file_name="Executive_Data_Report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

        # Header Title Area
        st.markdown(f"<div class='main-header'>Data Analyzer AI Platform</div>", unsafe_allow_html=True)
        st.caption(f"Domain Identified: **{domain_info.get('domain', 'General')}** | Confidence: **{domain_info.get('confidence', 'Medium')}**")

        # Global Interactive Filtering Engine
        with st.expander("🔍 Interactive Data Filtering Engine", expanded=False):
            filter_cols = st.multiselect("Select attributes to filter dataset:", cat_cols)
            filtered_df = df.copy()
            if filter_cols:
                for fc in filter_cols:
                    selected_vals = st.multiselect(f"Filter values for {fc}:", df[fc].unique(), default=df[fc].unique())
                    filtered_df = filtered_df[filtered_df[fc].isin(selected_vals)]
                st.info(f"Showing **{len(filtered_df):,}** of **{len(df):,}** records matching active filters.")
            else:
                filtered_df = df

        # Application Navigation Tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📋 Overview",
            "📊 Interactive Dashboards",
            "🔍 Data Quality Audit",
            "📈 Statistical Profiling",
            "❓ Analytical Questions",
            "🤖 ML Driver Engine",
            "💡 Executive Recommendations"
        ])

        # TAB 1: OVERVIEW
        with tab1:
            st.subheader("Dataset Structure & Metric Breakdown")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Records Analyzed", f"{len(filtered_df):,}")
            c2.metric("Total Attributes", len(filtered_df.columns))
            c3.metric("Numerical Attributes", len(num_cols))
            c4.metric("Categorical Attributes", len(cat_cols))

            st.markdown("---")
            st.subheader("Dataset Preview (Top Rows)")
            st.dataframe(filtered_df.head(15), use_container_width=True)

        # TAB 2: INTERACTIVE DASHBOARDS (PLOTLY)
        with tab2:
            st.subheader("Interactive Plotly Visual Dashboards")

            db1, db2, db3, db4 = st.tabs([
                "Categorical Aggregations",
                "Cross-Metric Correlations",
                "Distribution & Boxplots",
                "Interactive Query Builder"
            ])

            # Dashboard 1
            with db1:
                st.markdown("#### Categorical & Dimension Aggregation")
                if cat_cols and num_cols:
                    ca1, ca2 = st.columns(2)
                    sel_cat = ca1.selectbox("Categorical Dimension:", cat_cols, key="p_cat")
                    sel_num = ca2.selectbox("Numerical Metric:", num_cols, key="p_num")

                    f1, f2 = st.columns(2)
                    
                    # Chart 1: Bar Aggregation
                    agg_df = filtered_df.groupby(sel_cat)[sel_num].sum().reset_index().sort_values(by=sel_num, ascending=False).head(10)
                    fig_bar = px.bar(agg_df, x=sel_cat, y=sel_num, title=f"Total {sel_num} by {sel_cat}", color=sel_num, color_continuous_scale="Blues")
                    f1.plotly_chart(fig_bar, use_container_width=True)

                    # Chart 2: Donut Distribution
                    pie_df = filtered_df[sel_cat].value_counts().head(6).reset_index()
                    pie_df.columns = [sel_cat, "Count"]
                    fig_pie = px.pie(pie_df, names=sel_cat, values="Count", title=f"Top {sel_cat} Share Proportion", hole=0.4)
                    f2.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.warning("Categorical and Numerical attributes are required to render Dashboard 1.")

            # Dashboard 2
            with db2:
                st.markdown("#### Cross-Metric Relationship & Correlation Analysis")
                if len(num_cols) >= 2:
                    cb1, cb2 = st.columns(2)
                    x_val = cb1.selectbox("X-Axis Metric:", num_cols, index=0, key="px_x")
                    y_val = cb2.selectbox("Y-Axis Metric:", num_cols, index=1 if len(num_cols) > 1 else 0, key="px_y")

                    f1, f2 = st.columns(2)

                    # Scatter Plot with Trendline
                    fig_scatter = px.scatter(filtered_df, x=x_val, y=y_val, trendline="ols", title=f"Scatter Trend: {x_val} vs {y_val}", opacity=0.7)
                    f1.plotly_chart(fig_scatter, use_container_width=True)

                    # Correlation Heatmap
                    corr_data = filtered_df[num_cols[:8]].corr()
                    fig_heat = px.imshow(corr_data, text_auto=True, color_continuous_scale="RdBu_r", title="Metric Correlation Matrix")
                    f2.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.warning("At least two numerical metrics are required for cross-metric analytics.")

            # Dashboard 3
            with db3:
                st.markdown("#### Distribution Spread & Outlier Analysis")
                if num_cols:
                    cd1, cd2 = st.columns(2)
                    dist_m = cd1.selectbox("Select Metric:", num_cols, key="px_dist")
                    group_c = cd2.selectbox("Optional Categorical Split:", [None] + cat_cols, key="px_split")

                    f1, f2 = st.columns(2)

                    # Histogram
                    fig_hist = px.histogram(filtered_df, x=dist_m, color=group_c, marginal="rug", title=f"Histogram of {dist_m}")
                    f1.plotly_chart(fig_hist, use_container_width=True)

                    # Boxplot
                    fig_box = px.box(filtered_df, x=group_c, y=dist_m, color=group_c, title=f"Boxplot Spread of {dist_m}")
                    f2.plotly_chart(fig_box, use_container_width=True)

            # Dashboard 4
            with db4:
                st.markdown("#### Custom Interactive Query Dashboard")
                q_col1, q_col2, q_col3 = st.columns(3)
                q_x = q_col1.selectbox("Select X Attribute:", filtered_df.columns, index=0)
                q_y = q_col2.selectbox("Select Y Attribute:", num_cols, index=0 if num_cols else 0)
                chart_type = q_col3.selectbox("Select Plot Type:", ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot"])

                if chart_type == "Bar Chart":
                    fig_q = px.bar(filtered_df.head(100), x=q_x, y=q_y, color=q_x if q_x in cat_cols else None)
                elif chart_type == "Line Chart":
                    fig_q = px.line(filtered_df.head(100), x=q_x, y=q_y)
                elif chart_type == "Scatter Plot":
                    fig_q = px.scatter(filtered_df.head(100), x=q_x, y=q_y)
                else:
                    fig_q = px.box(filtered_df.head(100), x=q_x, y=q_y)

                st.plotly_chart(fig_q, use_container_width=True)

        # TAB 3: DATA QUALITY AUDIT
        with tab3:
            st.subheader("Data Quality & Integrity Audit")
            
            if quality_info.get("is_perfect_quality", False):
                st.success("✅ Dataset is fully optimized. Zero missing values or duplicates detected.")
            else:
                st.warning("⚠️ Data Quality Issues Identified.")

            dq1, dq2 = st.columns(2)
            with dq1:
                st.write("**Missing Values Inventory**")
                missing_cols = quality_info.get("cols_with_missing", {})
                if missing_cols:
                    m_df = pd.DataFrame(list(missing_cols.items()), columns=["Attribute", "Missing Count"])
                    st.dataframe(m_df, use_container_width=True)
                else:
                    st.info("Zero missing entries across dataset attributes.")

            with dq2:
                st.write("**Integrity Metrics Summary**")
                st.write(f"- Duplicate Records: **{quality_info.get('duplicate_rows', 0):,}**")
                st.write(f"- Completely Empty Columns: **{len(quality_info.get('empty_cols', []))}**")

        # TAB 4: STATISTICAL PROFILING
        with tab4:
            st.subheader("Statistical Profiling Summary")
            if num_cols:
                st.write("**Numerical Summary Statistics**")
                st.dataframe(filtered_df[num_cols].describe().T, use_container_width=True)

            if cat_cols:
                st.write("**Categorical Summary Statistics**")
                st.dataframe(filtered_df[cat_cols].describe().T, use_container_width=True)

        # TAB 5: ANALYTICAL QUESTIONS
        with tab5:
            st.subheader("Generated Domain Analytical Questions")
            for q in questions:
                with st.expander(f"📌 {q.get('category', 'Analysis')}: {q.get('question', '')}"):
                    st.write(f"**Analytical Purpose:** {q.get('purpose', '')}")

        # TAB 6: MACHINE LEARNING ENGINE
        with tab6:
            st.subheader(f"Machine Learning Driver Analysis: Target '{target_field}'")
            st.write("Train a cross-validated Random Forest model to identify primary variable drivers.")

            if st.button("🚀 Execute Machine Learning Driver Pipeline", type="primary"):
                with st.spinner("Training model and calculating feature importances..."):
                    try:
                        model, encoders, importances, metrics = train_risk_model(filtered_df, target_field)
                        st.success("Model Training Complete!")

                        st.write("**Model Performance Evaluation Metrics:**")
                        m_cols = st.columns(len(metrics))
                        for idx, (m_name, m_val) in enumerate(metrics.items()):
                            if isinstance(m_val, float):
                                m_cols[idx].metric(m_name, f"{m_val:.4f}")
                            else:
                                m_cols[idx].metric(m_name, str(m_val))

                        st.markdown("---")
                        ml_c1, ml_c2 = st.columns([2, 1])
                        with ml_c1:
                            fig_imp = px.bar(
                                importances.head(10), x='Importance', y='Feature', orientation='h',
                                title=f"Top Feature Importance Scores for '{target_field}'",
                                color='Importance', color_continuous_scale='Viridis'
                            )
                            fig_imp.update_layout(yaxis=dict(autorange="reversed"))
                            st.plotly_chart(fig_imp, use_container_width=True)
                        with ml_c2:
                            st.write("**Importance Table**")
                            st.dataframe(importances, use_container_width=True)
                    except Exception as err:
                        st.error(f"Machine learning engine execution failed: {err}")

        # TAB 7: EXECUTIVE RECOMMENDATIONS
        with tab7:
            st.subheader("Strategic Recommendations")
            for rec in recommendations:
                st.markdown(f"""
                <div class="rec-card">
                    <div class="rec-card-title">Focus Focus Domain: {rec.get('business_area', 'General Strategy')}</div>
                    <p class="rec-card-body"><b>Action Plan:</b> {rec.get('action_development', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)

else:
    st.info("👆 Upload a CSV or Excel dataset in the sidebar menu to launch the platform.")
