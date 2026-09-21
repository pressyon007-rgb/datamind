""""
app.py - Data Analyzer AI Platform
Enterprise Streamlit application for automated data analysis, multi-dashboard visual exploration,
machine learning driver modeling, and automated PDF executive reporting.
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
            {"category": "Performance", "question": "Which categories or operational dimensions yield the highest metrics?", "purpose": "Identify primary growth and revenue drivers."},
            {"category": "Risk Analysis", "question": "Are extreme statistical anomalies present across continuous numerical fields?", "purpose": "Mitigate operational exposure and data variance."},
            {"category": "Trends", "question": "What is the performance metric trajectory across temporal intervals?", "purpose": "Evaluate seasonal stability."}
        ]

try:
    from recommendation_engine import generate_recommendations
except ImportError:
    def generate_recommendations(df, quality_info, relationships, domain_info):
        return []


# Page Setup
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


# Data Loader & Downcaster
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
            
        # Memory optimization
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')

        if len(df) > row_limit:
            st.warning(f"⚠️ Dataset has {len(df):,} records. Sampled down to {row_limit:,} rows for cloud responsiveness.")
            df = df.sample(n=row_limit, random_state=42).reset_index(drop=True)
            
        return df
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        return None


# Random Forest Machine Learning Engine
def train_risk_model(df, target_col):
    data = df.copy().dropna()
    if data.empty:
        raise ValueError("Dataset contains no complete rows after dropping missing values.")
        
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
        raise ValueError("Insufficient feature columns available for modeling.")

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


# PDF Generator with Native Matplotlib Export
def generate_native_pdf_report(df, domain_info, quality_info, recommendations):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor('#1E3A8A'))
    h2_style = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontSize=11, leading=14, textColor=colors.HexColor('#1E40AF'), spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontSize=8.5, leading=11, textColor=colors.HexColor('#334155'))
    bold_style = ParagraphStyle('ReportBold', parent=body_style, fontName='Helvetica-Bold')

    story.append(Paragraph("Data Analyzer AI — Executive Report", title_style))
    story.append(Spacer(1, 4))
    domain_text = f"<b>Detected Domain:</b> {domain_info.get('domain', 'General')} | <b>Confidence:</b> {domain_info.get('confidence', 'Low')}"
    story.append(Paragraph(domain_text, body_style))
    story.append(Spacer(1, 8))

    # Overview Metrics Table
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

    # Render Visual Chart Matrix into PDF
    story.append(Paragraph("2. Visual Analytics Dashboard Export", h2_style))

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    valid_cat_cols = [c for c in cat_cols if not c.lower().endswith('id') and df[c].nunique() <= 50]

    if HAS_MATPLOTLIB and num_cols:
        try:
            fig, axs = plt.subplots(2, 2, figsize=(7.5, 4.5))
            
            if valid_cat_cols:
                df.groupby(valid_cat_cols[0])[num_cols[0]].sum().head(6).plot(kind='bar', ax=axs[0,0], color='#2563EB')
                axs[0,0].set_title(f"1. {num_cols[0]} by {valid_cat_cols[0]}", fontsize=8, fontweight='bold')
            else:
                df[num_cols[0]].plot(kind='hist', ax=axs[0,0], color='#2563EB')
                axs[0,0].set_title(f"1. {num_cols[0]} Distribution", fontsize=8, fontweight='bold')

            if len(valid_cat_cols) > 1:
                df[valid_cat_cols[1]].value_counts().head(5).plot(kind='pie', ax=axs[0,1], autopct='%1.0f%%')
                axs[0,1].set_title(f"2. {valid_cat_cols[1]} Share", fontsize=8, fontweight='bold')
                axs[0,1].set_ylabel('')
            else:
                df[num_cols[0]].plot(kind='box', ax=axs[0,1], color='#1D4ED8')
                axs[0,1].set_title(f"2. {num_cols[0]} Boxplot", fontsize=8, fontweight='bold')

            if valid_cat_cols:
                df.groupby(valid_cat_cols[0])[num_cols[0]].mean().head(6).plot(kind='barh', ax=axs[1,0], color='#3B82F6')
                axs[1,0].set_title(f"3. Avg {num_cols[0]}", fontsize=8, fontweight='bold')

            if valid_cat_cols:
                df[valid_cat_cols[0]].value_counts().head(6).plot(kind='bar', ax=axs[1,1], color='#60A5FA')
                axs[1,1].set_title(f"4. Record Volume", fontsize=8, fontweight='bold')

            plt.tight_layout()
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', dpi=130)
            plt.close(fig)
            img_buf.seek(0)
            story.append(Image(img_buf, width=480, height=250))

        except Exception as err:
            story.append(Paragraph(f"<i>Chart Export Exception: {err}</i>", body_style))
    else:
        story.append(Paragraph("<i>Matplotlib missing. Install matplotlib to include rendered visual charts in exported reports.</i>", body_style))

    story.append(Spacer(1, 8))

    # Recommendations
    story.append(Paragraph("3. Executive Analyst Recommendations", h2_style))
    if recommendations:
        for idx, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"<b>Rec {idx}: {rec.get('business_area', 'Strategy')}</b>", bold_style))
            story.append(Paragraph(f"• <b>Action:</b> {rec.get('action_development', 'N/A')}", body_style))
            story.append(Spacer(1, 4))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# Sidebar Navigation
st.sidebar.title("🧠 Data Analyzer AI")
st.sidebar.write("Upload a CSV or Excel dataset to perform end-to-end automated visual analytics.")

uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel File", type=["csv", "xlsx"])
row_limit = st.sidebar.slider("Sampling Row Limit:", min_value=1000, max_value=10000, value=5000, step=1000)

if uploaded_file is not None:
    df = load_data(uploaded_file, row_limit=row_limit)

    if df is not None and not df.empty:
        target_field = st.sidebar.selectbox("Select Target / Driver Field:", df.columns)
        palette = st.sidebar.selectbox("Color Palette:", ["Blues", "Viridis", "Cividis", "Plasma", "Turbo", "Magma"], index=0)

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

        # App Header
        st.title("Data Analyzer AI")
        st.caption("Intelligent Visual Analytics & Automated Reporting Platform")
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
            c3.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
            c4.metric("Categorical Columns", len(df.select_dtypes(include=['object', 'category']).columns))

            st.markdown("---")
            st.subheader("Dataset First 10 Rows")
            st.dataframe(df.head(10), use_container_width=True)

        # TAB 2: DASHBOARD ANALYTICS (EXACTLY 4 CHARTS PER DASHBOARD)
        with tab2:
            st.subheader("Interactive Visual Dashboards (4 Charts Per Dashboard)")
            
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

            # DASHBOARD 1 (4 CHARTS)
            with db_tab1:
                st.markdown("#### Dashboard 1: Categorical & Dimension Breakdown")
                if valid_cat_cols and num_cols:
                    col_a, col_b = st.columns(2)
                    c_dim = col_a.selectbox("Select Categorical Dimension:", valid_cat_cols, index=0, key="db1_cat")
                    m_metric = col_b.selectbox("Select Numerical Metric:", num_cols, index=0, key="db1_num")

                    r1_c1, r1_c2 = st.columns(2)
                    r2_c1, r2_c2 = st.columns(2)

                    with r1_c1:
                        agg_sum = df.groupby(c_dim)[m_metric].sum().reset_index().sort_values(by=m_metric, ascending=False).head(10)
                        fig1_1 = px.bar(agg_sum, x=c_dim, y=m_metric, color=m_metric,
                                        color_continuous_scale=palette.lower(), text_auto='.2s',
                                        title=f"Chart 1: Total {m_metric} by {c_dim}")
                        st.plotly_chart(fig1_1, use_container_width=True)

                    with r1_c2:
                        top_cats = agg_sum[c_dim].tolist()
                        df_filtered = df[df[c_dim].isin(top_cats)]
                        fig1_2 = px.pie(df_filtered, names=c_dim, values=m_metric, hole=0.4,
                                        title=f"Chart 2: % Share of {m_metric} across {c_dim}",
                                        color_discrete_sequence=px.colors.qualitative.Set2)
                        st.plotly_chart(fig1_2, use_container_width=True)

                    with r2_c1:
                        agg_avg = df.groupby(c_dim)[m_metric].mean().reset_index().sort_values(by=m_metric, ascending=False).head(10)
                        fig1_3 = px.bar(agg_avg, y=c_dim, x=m_metric, orientation='h', color=m_metric,
                                        color_continuous_scale=palette.lower(), text_auto='.2f',
                                        title=f"Chart 3: Average {m_metric} by {c_dim}")
                        fig1_3.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig1_3, use_container_width=True)

                    with r2_c2:
                        agg_cnt = df[c_dim].value_counts().reset_index().head(10)
                        agg_cnt.columns = [c_dim, 'Record Count']
                        fig1_4 = px.bar(agg_cnt, x=c_dim, y='Record Count', color='Record Count',
                                        color_continuous_scale="Viridis", text_auto=True,
                                        title=f"Chart 4: Volume Frequency per {c_dim}")
                        st.plotly_chart(fig1_4, use_container_width=True)
                else:
                    st.warning("Requires categorical and numerical fields to display Dashboard 1.")

            # DASHBOARD 2 (4 CHARTS WITH SAFE TRENDLINE GUARD)
            with db_tab2:
                st.markdown("#### Dashboard 2: Cross-Metric Relationship Analysis")
                if len(num_cols) >= 2:
                    col_a, col_b = st.columns(2)
                    x_met = col_a.selectbox("X-Axis Metric:", num_cols, index=0, key="db2_x")
                    y_met = col_b.selectbox("Y-Axis Metric:", num_cols, index=1 if len(num_cols) > 1 else 0, key="db2_y")

                    r1_c1, r1_c2 = st.columns(2)
                    r2_c1, r2_c2 = st.columns(2)

                    # Chart 1: Safe Scatter with Exception Guard for statsmodels
                    with r1_c1:
                        try:
                            fig2_1 = px.scatter(df, x=x_met, y=y_met, color=valid_cat_cols[0] if valid_cat_cols else None,
                                                title=f"Chart 1: Scatter ({x_met} vs {y_met})", trendline="ols")
                        except Exception:
                            fig2_1 = px.scatter(df, x=x_met, y=y_met, color=valid_cat_cols[0] if valid_cat_cols else None,
                                                title=f"Chart 1: Scatter ({x_met} vs {y_met})")
                        st.plotly_chart(fig2_1, use_container_width=True)

                    with r1_c2:
                        fig2_2 = px.density_heatmap(df, x=x_met, y=y_met, text_auto=True, color_continuous_scale=palette.lower(),
                                                    title=f"Chart 2: Density Heatmap ({x_met} vs {y_met})")
                        st.plotly_chart(fig2_2, use_container_width=True)

                    with r2_c1:
                        df_sorted = df.sort_values(by=x_met).head(200)
                        fig2_3 = px.line(df_sorted, x=x_met, y=y_met, title=f"Chart 3: Metric Profile Line ({x_met} vs {y_met})")
                        st.plotly_chart(fig2_3, use_container_width=True)

                    with r2_c2:
                        corr_matrix = df[num_cols[:6]].corr()
                        fig2_4 = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="Blues",
                                           title="Chart 4: Multi-Metric Correlation Matrix")
                        st.plotly_chart(fig2_4, use_container_width=True)
                else:
                    st.warning("Dashboard 2 requires at least two numerical metrics.")

            # DASHBOARD 3 (4 CHARTS)
            with db_tab3:
                st.markdown("#### Dashboard 3: Statistical Distributions & Time Trends")
                if num_cols:
                    col_a, col_b = st.columns(2)
                    dist_col = col_a.selectbox("Metric for Distribution:", num_cols, index=0, key="db3_dist")
                    group_cat = col_b.selectbox("Category Dimension Split:", valid_cat_cols if valid_cat_cols else [None], index=0, key="db3_split")

                    r1_c1, r1_c2 = st.columns(2)
                    r2_c1, r2_c2 = st.columns(2)

                    with r1_c1:
                        fig3_1 = px.box(df, y=dist_col, x=group_cat, color=group_cat, points="outliers",
                                        title=f"Chart 1: Boxplot Spread for '{dist_col}'")
                        st.plotly_chart(fig3_1, use_container_width=True)

                    with r2_c1:
                        fig3_2 = px.histogram(df, x=dist_col, color=group_cat, marginal="rug",
                                              title=f"Chart 2: Histogram & Density of '{dist_col}'")
                        st.plotly_chart(fig3_2, use_container_width=True)

                    with r1_c2:
                        fig3_3 = px.violin(df, y=dist_col, x=group_cat, box=True, points="all",
                                           title=f"Chart 3: Violin Kernel Density for '{dist_col}'")
                        st.plotly_chart(fig3_3, use_container_width=True)

                    with r2_c2:
                        if date_cols:
                            dt_col = date_cols[0]
                            trend_df = df.groupby(dt_col)[dist_col].mean().reset_index()
                            fig3_4 = px.line(trend_df, x=dt_col, y=dist_col, title=f"Chart 4: Average '{dist_col}' Timeline")
                        else:
                            fig3_4 = px.ecdf(df, x=dist_col, color=group_cat, title=f"Chart 4: ECDF Cumulative Curve for '{dist_col}'")
                        st.plotly_chart(fig3_4, use_container_width=True)
                else:
                    st.warning("Dashboard 3 requires numerical fields.")

            # DASHBOARD 4 (4 CHARTS & Q&A)
            with db_tab4:
                st.markdown("#### Dashboard 4: Interactive Q&A Engine & Automated Visual Querying")
                st.write("Ask questions about your data or choose from generated hypotheses.")

                if questions:
                    suggested_q = [q.get('question', '') for q in questions if q.get('question')]
                    selected_q = st.selectbox("Select a generated analytical question:", ["-- Select or type below --"] + suggested_q)
                else:
                    selected_q = "-- Select or type below --"

                custom_q = st.text_input("Or type your question:", value="" if selected_q == "-- Select or type below --" else selected_q)

                if st.button("🔎 Execute Query & Render Dashboard 4", type="primary"):
                    if custom_q.strip():
                        st.markdown(f"**Query Executed:** *{custom_q}*")
                        matched_num = [c for c in num_cols if c.lower() in custom_q.lower()]
                        matched_cat = [c for c in valid_cat_cols if c.lower() in custom_q.lower()]

                        m_col = matched_num[0] if matched_num else num_cols[0]
                        c_col = matched_cat[0] if matched_cat else (valid_cat_cols[0] if valid_cat_cols else None)

                        r1_c1, r1_c2 = st.columns(2)
                        r2_c1, r2_c2 = st.columns(2)

                        with r1_c1:
                            if c_col:
                                res1 = df.groupby(c_col)[m_col].sum().reset_index().head(8)
                                fig4_1 = px.bar(res1, x=c_col, y=m_col, color=m_col, title=f"Chart 1: Total {m_col} by {c_col}")
                            else:
                                fig4_1 = px.histogram(df, x=m_col, title=f"Chart 1: Distribution of {m_col}")
                            st.plotly_chart(fig4_1, use_container_width=True)

                        with r1_c2:
                            if c_col:
                                res2 = df.groupby(c_col)[m_col].mean().reset_index().head(8)
                                fig4_2 = px.bar(res2, x=c_col, y=m_col, color=m_col, title=f"Chart 2: Avg {m_col} by {c_col}")
                            else:
                                fig4_2 = px.box(df, y=m_col, title=f"Chart 2: Boxplot Spread of {m_col}")
                            st.plotly_chart(fig4_2, use_container_width=True)

                        with r2_c1:
                            if c_col:
                                res3 = df[c_col].value_counts().reset_index().head(8)
                                res3.columns = [c_col, 'Count']
                                fig4_3 = px.pie(res3, names=c_col, values='Count', title=f"Chart 3: {c_col} Proportion")
                            else:
                                fig4_3 = px.scatter(df, x=df.index, y=m_col, title=f"Chart 3: Sequential Scatter of {m_col}")
                            st.plotly_chart(fig4_3, use_container_width=True)

                        with r2_c2:
                            if date_cols:
                                res4 = df.groupby(date_cols[0])[m_col].sum().reset_index()
                                fig4_4 = px.line(res4, x=date_cols[0], y=m_col, title=f"Chart 4: {m_col} Timeline")
                            elif len(num_cols) >= 2:
                                second_m = num_cols[1] if num_cols[1] != m_col else num_cols[0]
                                fig4_4 = px.scatter(df, x=m_col, y=second_m, title=f"Chart 4: {m_col} vs {second_m}")
                            else:
                                fig4_4 = px.ecdf(df, x=m_col, title=f"Chart 4: ECDF Curve for {m_col}")
                            st.plotly_chart(fig4_4, use_container_width=True)
                    else:
                        st.warning("Please enter or select a query to generate Dashboard 4.")

        # TAB 3: DATA QUALITY
        with tab3:
            st.subheader("Data Quality & Integrity")
            if quality_info.get("is_perfect_quality", False):
                st.success("✅ Dataset is completely clean with zero missing values or duplicate rows.")
            else:
                st.warning("⚠️ Quality Notice: Missing entries, duplicate records, or extreme anomalies detected.")

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
                        fig_imp = px.bar(importances.head(10), x='Importance', y='Feature', orientation='h',
                                         title=f"Top Drivers Influencing '{target_field}'")
                        fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                        st.plotly_chart(fig_imp, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not build feature importance model: {e}")

        # TAB 7: RECOMMENDATIONS & DIRECT PDF DOWNLOAD
        with tab7:
            st.subheader("Executive Analyst Recommendations & Complete PDF Export")
            
            pdf_bytes = generate_native_pdf_report(df, domain_info, quality_info, recommendations)
            
            st.download_button(
                label="📄 Download Executive Report with Visual Charts (PDF)",
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
    st.info("👈 Upload a CSV or Excel file in the sidebar to get started.")
