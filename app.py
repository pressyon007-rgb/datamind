""""
app.py - Data Analyzer AI Platform
Interactive Streamlit application for data analysis, visual exploration, machine learning, and decision support.
Optimized for high-performance cloud deployment with native PDF chart export and interactive Q&A.
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ReportLab Imports for Direct PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

# Safe Matplotlib Import for Cloud PDF Execution
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for cloud execution
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.multiclass import type_of_target

# Import custom modules with safety guards
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
            {"category": "Performance", "question": "Which categories or dimensions drive the highest metric yield?", "purpose": "Identify high-value revenue drivers."},
            {"category": "Risk", "question": "Are there extreme statistical anomalies present in continuous columns?", "purpose": "Mitigate risk exposure in operational metrics."},
            {"category": "Trend", "question": "What is the historical pattern across recorded temporal intervals?", "purpose": "Evaluate seasonal stability."}
        ]

try:
    from recommendation_engine import generate_recommendations
except ImportError:
    def generate_recommendations(df, quality_info, relationships, domain_info):
        return []


# Page Configuration
st.set_page_config(
    page_title="Data Analyzer AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom UI Styling
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
    </style>
""", unsafe_allow_html=True)


# Helper Function: Memory-Optimized Data Loader
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
            
        # Downcast numbers for performance
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')

        # Smart Sampling for large datasets
        if len(df) > row_limit:
            st.warning(f"⚠️ Dataset has {len(df):,} rows. Sampled to {row_limit:,} rows for cloud memory optimization.")
            df = df.sample(n=row_limit, random_state=42).reset_index(drop=True)
            
        return df
    except Exception as e:
        st.error(f"Error loading file '{uploaded_file.name}': {e}")
        return None


# Machine Learning Driver Model
def train_risk_model(df, target_col):
    data = df.copy().dropna()
    if data.empty:
        raise ValueError("Dataset has no complete rows after dropping missing values.")
        
    encoders = {}
    
    # Convert dates to numerical features
    date_cols = data.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
    for col in date_cols:
        if col != target_col:
            data[f"{col}_Year"] = data[col].dt.year
            data[f"{col}_Month"] = data[col].dt.month
            data = data.drop(columns=[col])

    # Drop high-cardinality ID columns
    for col in list(data.columns):
        if col != target_col and (col.lower().endswith('id') or data[col].nunique() > 100):
            data = data.drop(columns=[col])

    # Encode categorical features
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        if col != target_col:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
        
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    if X.empty or X.shape[1] == 0:
        raise ValueError("Not enough features available to train a prediction model.")

    target_type = type_of_target(y)
    
    if target_type == 'continuous':
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)
    else:
        if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
            target_le = LabelEncoder()
            y = target_le.fit_transform(y.astype(str))
            encoders[target_col] = target_le
            
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)
    
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return model, encoders, importances, X.columns.tolist()


# Helper Function: Native PDF Generator Rendering All Dashboards
def generate_native_pdf_report(df, domain_info, quality_info, recommendations):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()

    # Paragraph Styles
    title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=20, leading=24, textColor=colors.HexColor('#1E3A8A'))
    h2_style = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontSize=13, leading=16, textColor=colors.HexColor('#1E40AF'), spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontSize=9, leading=13, textColor=colors.HexColor('#334155'))
    bold_style = ParagraphStyle('ReportBold', parent=body_style, fontName='Helvetica-Bold')

    # Header Title
    story.append(Paragraph("Data Analyzer AI — Complete Executive Report", title_style))
    story.append(Spacer(1, 6))
    domain_text = f"<b>Detected Domain:</b> {domain_info.get('domain', 'General')} (Confidence: {domain_info.get('confidence', 'Low')})"
    story.append(Paragraph(domain_text, body_style))
    story.append(Spacer(1, 10))

    # Overview Table
    story.append(Paragraph("1. Executive Overview & Key Metrics", h2_style))
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
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_overview)
    story.append(Spacer(1, 10))

    # Render All 4 Dashboards into PDF
    story.append(Paragraph("2. Multi-Dashboard Analytics Charts Export", h2_style))

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    valid_cat_cols = [c for c in cat_cols if not c.lower().endswith('id') and df[c].nunique() <= 50]

    if HAS_MATPLOTLIB:
        try:
            # Dashboard 1 Chart: Categorical Ranking
            if valid_cat_cols and num_cols:
                fig, ax = plt.subplots(figsize=(7, 2.8))
                agg_df = df.groupby(valid_cat_cols[0])[num_cols[0]].sum().sort_values(ascending=False).head(8)
                agg_df.plot(kind='bar', ax=ax, color='#2563EB', edgecolor='#1E3A8A')
                ax.set_title(f"DB 1: Top {valid_cat_cols[0]} by {num_cols[0]}", fontsize=10, fontweight='bold', color='#1E3A8A')
                plt.xticks(rotation=25, ha='right', fontsize=8)
                plt.tight_layout()

                img_buf1 = io.BytesIO()
                plt.savefig(img_buf1, format='png', dpi=150)
                plt.close(fig)
                img_buf1.seek(0)
                story.append(Image(img_buf1, width=460, height=180))
                story.append(Spacer(1, 6))

            # Dashboard 2 Chart: Metric Correlation
            if len(num_cols) >= 2:
                fig, ax = plt.subplots(figsize=(7, 2.8))
                ax.scatter(df[num_cols[0]], df[num_cols[1]], alpha=0.6, color='#0284C7', edgecolors='none', s=20)
                ax.set_title(f"DB 2: {num_cols[0]} vs {num_cols[1]} Correlation", fontsize=10, fontweight='bold', color='#1E3A8A')
                ax.set_xlabel(num_cols[0], fontsize=8)
                ax.set_ylabel(num_cols[1], fontsize=8)
                plt.tight_layout()

                img_buf2 = io.BytesIO()
                plt.savefig(img_buf2, format='png', dpi=150)
                plt.close(fig)
                img_buf2.seek(0)
                story.append(Image(img_buf2, width=460, height=180))
                story.append(Spacer(1, 6))

            # Dashboard 3 Chart: Box Plot Outliers
            if num_cols:
                fig, ax = plt.subplots(figsize=(7, 2.8))
                ax.boxplot(df[num_cols[0]].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor='#93C5FD', color='#1E3A8A'))
                ax.set_title(f"DB 3: Outlier Spread for '{num_cols[0]}'", fontsize=10, fontweight='bold', color='#1E3A8A')
                plt.tight_layout()

                img_buf3 = io.BytesIO()
                plt.savefig(img_buf3, format='png', dpi=150)
                plt.close(fig)
                img_buf3.seek(0)
                story.append(Image(img_buf3, width=460, height=180))

        except Exception as err:
            story.append(Paragraph(f"<i>Dashboard Chart Export Notice: {err}</i>", body_style))
    else:
        story.append(Paragraph("<i>Matplotlib missing. Please install matplotlib to render chart images in PDF exports.</i>", body_style))

    story.append(Spacer(1, 10))

    # Recommendations
    story.append(Paragraph("3. Executive Analyst Recommendations", h2_style))
    if recommendations:
        for idx, rec in enumerate(recommendations, 1):
            rec_title = f"<b>Recommendation {idx}: {rec.get('business_area', 'Strategy')}</b>"
            story.append(Paragraph(rec_title, bold_style))
            story.append(Paragraph(f"• <b>Chart Outcome:</b> {rec.get('chart_outcome', 'N/A')}", body_style))
            story.append(Paragraph(f"• <b>Analyst Recommendation:</b> {rec.get('analyst_recommendation', 'N/A')}", body_style))
            story.append(Paragraph(f"• <b>Action Plan:</b> {rec.get('action_development', 'N/A')}", body_style))
            story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# Sidebar Configuration
st.sidebar.title("🧠 Data Analyzer AI")
st.sidebar.write("Upload a CSV or Excel file to analyze structure, quality, and recommendations.")

uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel File", type=["csv", "xlsx"])
row_limit = st.sidebar.slider("Dataset Sampling Limit (Rows):", min_value=1000, max_value=10000, value=5000, step=1000)

if uploaded_file is not None:
    df = load_data(uploaded_file, row_limit=row_limit)

    if df is not None and not df.empty:
        target_field = st.sidebar.selectbox("Select Target / Risk Field:", df.columns)
        palette = st.sidebar.selectbox("Select Dashboard Color Theme:", ["Blues", "Viridis", "Cividis", "Plasma", "Turbo", "Magma"], index=0)

        # Domain Detection & Web Research Integration
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

        # Main Header
        st.title("Data Analyzer AI")
        st.caption("Intelligent Data Analysis & Decision Support Platform")
        st.info(f"Detected Industry / Domain: **{domain_info.get('domain', 'General')}** (Confidence: **{domain_info.get('confidence', 'Low')}**)")

        # Navigation Tabs
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
            st.subheader("Dataset Summary & Structure")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Loaded Records", f"{quality_info.get('total_rows', len(df)):,}")
            c2.metric("Total Columns", f"{quality_info.get('total_cols', len(df.columns)):,}")
            c3.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
            c4.metric("Categorical Columns", len(df.select_dtypes(include=['object', 'category']).columns))

            st.markdown("---")
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

        # TAB 2: DASHBOARD ANALYTICS (4 COMPLETE DASHBOARDS)
        with tab2:
            st.subheader("Interactive Visual Analytics (4 Comprehensive Dashboards)")
            
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()

            if not date_cols:
                for c in cat_cols:
                    if 'date' in c.lower() or 'time' in c.lower():
                        try:
                            df[c] = pd.to_datetime(df[c])
                            date_cols.append(c)
                        except Exception:
                            pass

            valid_cat_cols = [c for c in cat_cols if not c.lower().endswith('id') and df[c].nunique() <= 50]

            db_tab1, db_tab2, db_tab3, db_tab4 = st.tabs([
                "Dashboard 1: Categorical Breakdown",
                "Dashboard 2: Metric Relationships",
                "Dashboard 3: Distribution & Trends",
                "Dashboard 4: Interactive Q&A Engine"
            ])

            # DASHBOARD 1
            with db_tab1:
                st.markdown("#### Dashboard 1: Categorical & Dimension Breakdown")
                if valid_cat_cols and num_cols:
                    col_a, col_b = st.columns(2)
                    c_dim = col_a.selectbox("Select Categorical Dimension:", valid_cat_cols, index=0, key="db1_cat")
                    m_metric = col_b.selectbox("Select Numerical Metric:", num_cols, index=0, key="db1_num")

                    agg_df = df.groupby(c_dim)[m_metric].sum().reset_index().sort_values(by=m_metric, ascending=False).head(10)
                    
                    fig1 = px.bar(agg_df, x=c_dim, y=m_metric, color=m_metric,
                                  color_continuous_scale=palette.lower(), text_auto='.2s',
                                  title=f"Top 10 {c_dim} by {m_metric}")
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.warning("Insufficient categorical or numerical columns to construct Dashboard 1.")

            # DASHBOARD 2
            with db_tab2:
                st.markdown("#### Dashboard 2: Cross-Metric Relationship Analysis")
                if len(num_cols) >= 2:
                    col_a, col_b = st.columns(2)
                    x_met = col_a.selectbox("X-Axis Metric:", num_cols, index=0, key="db2_x")
                    y_met = col_b.selectbox("Y-Axis Metric:", num_cols, index=1 if len(num_cols) > 1 else 0, key="db2_y")

                    fig2 = px.scatter(df, x=x_met, y=y_met, color=valid_cat_cols[0] if valid_cat_cols else None,
                                      title=f"Relationship Scatter: {x_met} vs {y_met}")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("Dashboard 2 requires at least two numerical metrics.")

            # DASHBOARD 3
            with db_tab3:
                st.markdown("#### Dashboard 3: Statistical Distributions & Time Trends")
                col_a, col_b = st.columns(2)
                with col_a:
                    if num_cols:
                        box_col = st.selectbox("Select Metric for Outliers:", num_cols, index=0, key="db3_box")
                        fig3a = px.box(df, y=box_col, title=f"Outlier Spread in '{box_col}'")
                        st.plotly_chart(fig3a, use_container_width=True)
                with col_b:
                    if date_cols and num_cols:
                        dt_col = st.selectbox("Select Date Column:", date_cols, index=0, key="db3_dt")
                        tm_met = st.selectbox("Select Metric for Trend:", num_cols, index=0, key="db3_tm")
                        trend_df = df.groupby(dt_col)[tm_met].sum().reset_index()
                        fig3b = px.line(trend_df, x=dt_col, y=tm_met, title=f"{tm_met} Trend over Time")
                        st.plotly_chart(fig3b, use_container_width=True)
                    elif len(num_cols) >= 2:
                        fig3b = px.imshow(df[num_cols].corr(), text_auto=True, color_continuous_scale="Blues", title="Correlation Heatmap")
                        st.plotly_chart(fig3b, use_container_width=True)

            # DASHBOARD 4: INTERACTIVE Q&A ENGINE
            with db_tab4:
                st.markdown("#### Dashboard 4: Interactive Q&A Engine & Query Execution")
                st.write("Query your dataset directly using analytical questions or custom prompts.")

                if questions:
                    st.markdown("##### 💡 Suggested Analytical Questions")
                    suggested_q = [q.get('question', '') for q in questions if q.get('question')]
                    selected_q = st.selectbox("Select a generated domain question:", ["-- Select or type below --"] + suggested_q)
                else:
                    selected_q = "-- Select or type below --"

                custom_q = st.text_input("Or enter your custom question about this dataset:", value="" if selected_q == "-- Select or type below --" else selected_q)

                if st.button("🔎 Execute Query & Get Answer", type="primary"):
                    if custom_q.strip():
                        st.markdown(f"**Query:** *{custom_q}*")
                        with st.spinner("Analyzing dataset structure to construct query result..."):
                            matched_num = [c for c in num_cols if c.lower() in custom_q.lower()]
                            matched_cat = [c for c in valid_cat_cols if c.lower() in custom_q.lower()]

                            if matched_cat and matched_num:
                                group_res = df.groupby(matched_cat[0])[matched_num[0]].agg(['sum', 'mean', 'count']).reset_index().head(10)
                                st.write(f"**Automated Breakdown:** Grouping `{matched_num[0]}` by `{matched_cat[0]}`")
                                st.dataframe(group_res, use_container_width=True)
                                fig_q = px.bar(group_res, x=matched_cat[0], y='sum', title=f"Sum of {matched_num[0]} by {matched_cat[0]}")
                                st.plotly_chart(fig_q, use_container_width=True)
                            elif matched_num:
                                desc_res = df[matched_num].describe().T[['mean', 'std', 'min', '50%', 'max']]
                                st.write(f"**Statistical Summary for Identified Metrics:**")
                                st.dataframe(desc_res, use_container_width=True)
                            else:
                                st.info("Could not map specific column names automatically. Here is the general correlation and distribution summary:")
                                if num_cols:
                                    st.dataframe(df[num_cols].describe().T.head(8), use_container_width=True)
                    else:
                        st.warning("Please enter or select a valid question to execute.")

        # TAB 3: DATA QUALITY
        with tab3:
            st.subheader("Data Quality & Integrity")
            if quality_info.get("is_perfect_quality", False):
                st.success("✅ Positive Finding: No missing values or duplicates detected.")
            else:
                st.warning("⚠️ Data Quality Notice: Missing values, duplicates, or outliers detected.")

            col_q1, col_q2 = st.columns(2)
            with col_q1:
                st.write("**Missing Values**")
                missing = quality_info.get("cols_with_missing", {})
                if missing:
                    missing_df = pd.DataFrame(list(missing.items()), columns=["Column", "Missing Count"])
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.info("Zero missing values across all columns.")

            with col_q2:
                st.write("**Data Integrity Metrics**")
                st.write(f"- Duplicate Rows: **{quality_info.get('duplicate_rows', 0)}**")
                st.write(f"- Empty Columns: **{len(quality_info.get('empty_cols', []))}**")
                st.write(f"- Columns with Outliers: **{len(quality_info.get('outliers', {}))}**")

        # TAB 4: STATISTICS
        with tab4:
            st.subheader("Statistical Analysis")
            num_df = df.select_dtypes(include=[np.number])
            if not num_df.empty:
                st.write("**Numerical Summary**")
                st.dataframe(num_df.describe().T, use_container_width=True)

            cat_df = df.select_dtypes(include=['object', 'category'])
            if not cat_df.empty:
                st.write("**Categorical Summary**")
                st.dataframe(cat_df.describe().T, use_container_width=True)

        # TAB 5: ANALYTICAL QUESTIONS
        with tab5:
            st.subheader("Analytical Questions & Domain Hypotheses")
            if questions:
                for q in questions:
                    with st.expander(f"📌 {q.get('category', 'Analysis')}: {q.get('question', '')}"):
                        st.write(f"**Purpose:** {q.get('purpose', '')}")
            else:
                st.info("No domain hypothesis questions generated for this dataset.")

        # TAB 6: RISK & FEATURE IMPORTANCE
        with tab6:
            st.subheader(f"Feature Importance Driver Analysis: '{target_field}'")
            st.write("Train a Random Forest model to calculate feature importances on demand.")
            
            if st.button("🚀 Train & Calculate Drivers", type="primary"):
                with st.spinner("Training Random Forest model..."):
                    try:
                        model, encoders, importances, feature_names = train_risk_model(df, target_field)
                        fig_imp = px.bar(importances.head(10), x='Importance', y='Feature', orientation='h',
                                         title=f"Top Drivers Influencing '{target_field}'")
                        fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                        st.plotly_chart(fig_imp, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not build feature importance model: {e}")

        # TAB 7: RECOMMENDATIONS & DIRECT PDF DOWNLOAD
        with tab7:
            st.subheader("Evidence-Based Data Analyst Recommendations & Executive Export")
            
            pdf_bytes = generate_native_pdf_report(df, domain_info, quality_info, recommendations)
            
            st.download_button(
                label="📄 Download Complete Executive Report with Charts (PDF)",
                data=pdf_bytes,
                file_name="Data_Analyzer_AI_Executive_Report.pdf",
                mime="application/pdf",
                type="primary"
            )
            
            st.markdown("---")

            if recommendations:
                for idx, rec in enumerate(recommendations, 1):
                    st.markdown(f"""
                    <div class="rec-box">
                        <div class="rec-title">Recommendation {idx}: {rec.get('business_area', 'Strategy')}</div>
                        <p><span class="field-label">Chart / Data Outcome:</span> {rec.get('chart_outcome', 'N/A')}</p>
                        <p><span class="field-label">What This Means:</span> {rec.get('what_this_means', 'N/A')}</p>
                        <p><span class="field-label">Limitation:</span> {rec.get('limitation', 'N/A')}</p>
                        <p><span class="field-label">Data Analyst Recommendation:</span> {rec.get('analyst_recommendation', 'N/A')}</p>
                        <p><span class="field-label">Action / Development:</span> {rec.get('action_development', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No explicit recommendations generated for the current dataset structure.")

else:
    st.info("👈 Upload a CSV or Excel dataset in the sidebar to begin.")
