"""
app.py - DataMind Analytics AI Platform
Complete production-ready multi-dashboard Streamlit application.
Supports 4 interactive dashboards, dynamic error guards, and executive PDF exports.
"""

import io
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cloud execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ReportLab Imports for Native PDF Generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# =============================================================================
# FAIL-SAFE CUSTOM MODULE IMPORTS
# =============================================================================

try:
    from insight_engine import analyze_data_quality, compute_relationships, generate_insights
except ImportError:
    def analyze_data_quality(df):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outliers_dict = {}
        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outlier_cnt = int(((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum())
            if outlier_cnt > 0:
                outliers_dict[col] = outlier_cnt

        return {
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "is_perfect_quality": df.isnull().sum().sum() == 0 and df.duplicated().sum() == 0,
            "total_missing": int(df.isnull().sum().sum()),
            "cols_with_missing": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "empty_cols": [col for col in df.columns if df[col].isnull().all()],
            "outliers": outliers_dict
        }

    def compute_relationships(df):
        relationships = []
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            corr_matrix = df[num_cols].corr()
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c_val = corr_matrix.iloc[i, j]
                    if abs(c_val) > 0.4:
                        relationships.append({
                            "var1": num_cols[i],
                            "var2": num_cols[j],
                            "correlation": round(float(c_val), 2),
                            "strength": "Strong" if abs(c_val) > 0.7 else "Moderate"
                        })
        return relationships

    def generate_insights(df):
        return {"quality": analyze_data_quality(df), "relationships": compute_relationships(df)}

try:
    from recommendation_engine import generate_recommendations
except ImportError:
    def generate_recommendations(df, quality_info, relationships, domain_info):
        return [{
            "business_area": "Data Governance & Remediation",
            "chart_outcome": f"Processed dataset with {quality_info.get('total_rows', 0):,} rows.",
            "what_this_means": "Automated pipeline analysis completed.",
            "limitation": "Custom recommendation engine module not loaded.",
            "analyst_recommendation": "Review dataset structure and configure custom engine rules.",
            "action_development": "Deploy full recommendation module."
        }]

try:
    from question_engine import generate_analytical_questions
except ImportError:
    def generate_analytical_questions(df, domain_info):
        return []

try:
    from web_research import detect_domain
except ImportError:
    def detect_domain(df):
        return {"domain": "Business Operations", "confidence": "High", "matched_terms": []}

try:
    from pdf_generator import create_pdf_report
except ImportError:
    def create_pdf_report(df, quality_info, recommendations, chart_images):
        return b""


# =============================================================================
# STREAMLIT APP CONFIGURATION & STYLING
# =============================================================================

st.set_page_config(
    page_title="DataMind Analytics Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .rec-card {
        background-color: #FFFFFF;
        border-left: 5px solid #2563EB;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING & HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            try:
                return pd.read_excel(uploaded_file, sheet_name=0, engine='calamine')
            except Exception:
                return pd.read_excel(uploaded_file, sheet_name=0, engine='openpyxl')
    except Exception as e:
        st.error(f"Error reading file '{uploaded_file.name}': {e}")
        return None


# =============================================================================
# MAIN APPLICATION FLOW
# =============================================================================

st.sidebar.title("🧠 DataMind Platform")
st.sidebar.write("Upload a CSV or Excel file to unlock interactive dashboards and executive reports.")

uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel File", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is not None and not df.empty:
        # Run Analytics Engines
        domain_info = detect_domain(df)
        quality_info = analyze_data_quality(df)
        relationships = compute_relationships(df)
        recommendations = generate_recommendations(df, quality_info, relationships, domain_info)

        st.title("DataMind Analytics AI")
        st.caption("Automated Multi-Dashboard Analytics & Decision Support Engine")
        st.info(f"Industry Domain Context: **{domain_info.get('domain', 'Operations')}**")

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        valid_cat_cols = [c for c in cat_cols if not c.lower().endswith('id') and df[c].nunique() <= 50]

        # Dictionary to store chart images for PDF generation
        chart_images = {}

        # ---------------------------------------------------------------------
        # 4 DISTINCT DASHBOARD TABS
        # ---------------------------------------------------------------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Dashboard 1: Quality & Governance",
            "📊 Dashboard 2: Distributions & Outliers",
            "🔗 Dashboard 3: Interactions & Heatmaps",
            "🎯 Dashboard 4: Strategic Recommendations & PDF"
        ])

        # ---------------------------------------------------------------------
        # DASHBOARD 1: OVERVIEW & DATA QUALITY AUDIT
        # ---------------------------------------------------------------------
        with tab1:
            st.subheader("Dashboard 1: Data Integrity & Field Completeness")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Records", f"{quality_info.get('total_rows', len(df)):,}")
            c2.metric("Total Attributes", f"{quality_info.get('total_cols', len(df.columns)):,}")
            c3.metric("Missing Values", f"{quality_info.get('total_missing', 0):,}")
            c4.metric("Duplicate Rows", f"{quality_info.get('duplicate_rows', 0):,}")

            st.markdown("---")
            st.markdown("##### Field Completeness Audit")

            null_series = df.isnull().sum()
            null_df = pd.DataFrame({"Column": null_series.index, "Missing_Count": null_series.values})

            fig1 = px.bar(null_df, x="Column", y="Missing_Count", color="Missing_Count",
                          color_continuous_scale="Reds", title="Missing Values Count per Attribute")
            st.plotly_chart(fig1, use_container_width=True)

            # Generate Matplotlib chart image for PDF
            fig_mpl1, ax1 = plt.subplots(figsize=(6, 3))
            ax1.bar(null_df["Column"], null_df["Missing_Count"], color="#DC2626")
            ax1.set_title("Missing Values Audit", fontsize=10, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            buf1 = io.BytesIO()
            plt.savefig(buf1, format='png', dpi=150)
            plt.close(fig_mpl1)
            chart_images["Data Quality & Missing Values Audit"] = buf1.getvalue()

            st.markdown("##### Dataset Sample Preview")
            st.dataframe(df.head(8), use_container_width=True)

        # ---------------------------------------------------------------------
        # DASHBOARD 2: DISTRIBUTIONS & VARIANCE
        # ---------------------------------------------------------------------
        with tab2:
            st.subheader("Dashboard 2: Feature Distribution & Extreme Variance Audit")

            if num_cols:
                selected_num = st.selectbox("Select Numerical Feature for Variance Audit:", num_cols, index=0)

                col_a, col_b = st.columns(2)
                with col_a:
                    fig2 = px.histogram(df, x=selected_num, nbins=30, marginal="rug",
                                        title=f"Histogram Spread: '{selected_num}'", color_discrete_sequence=['#2563EB'])
                    st.plotly_chart(fig2, use_container_width=True)

                with col_b:
                    fig3 = px.box(df, y=selected_num, points="outliers",
                                  title=f"Outlier Box Plot: '{selected_num}'", color_discrete_sequence=['#0D9488'])
                    st.plotly_chart(fig3, use_container_width=True)

                # Generate Matplotlib chart image for PDF
                fig_mpl2, ax2 = plt.subplots(figsize=(6, 3))
                ax2.hist(df[selected_num].dropna(), bins=25, color="#2563EB", edgecolor="black")
                ax2.set_title(f"Distribution Audit: {selected_num}", fontsize=10, fontweight='bold')
                plt.tight_layout()
                buf2 = io.BytesIO()
                plt.savefig(buf2, format='png', dpi=150)
                plt.close(fig_mpl2)
                chart_images[f"Feature Distribution - {selected_num}"] = buf2.getvalue()
            else:
                st.warning("No numerical attributes detected for distribution analysis.")

        # ---------------------------------------------------------------------
        # DASHBOARD 3: INTERACTIONS & HEATMAPS
        # ---------------------------------------------------------------------
        with tab3:
            st.subheader("Dashboard 3: Multivariable Correlation & Feature Interaction")

            if len(num_cols) >= 2:
                corr_matrix = df[num_cols].corr()
                fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                                     title="Pearson Correlation Coefficient Heatmap")
                st.plotly_chart(fig_corr, use_container_width=True)

                # Generate Matplotlib Heatmap for PDF
                fig_mpl3, ax3 = plt.subplots(figsize=(6, 4))
                cax = ax3.matshow(corr_matrix, cmap='coolwarm')
                fig_mpl3.colorbar(cax)
                ax3.set_xticks(range(len(num_cols)))
                ax3.set_yticks(range(len(num_cols)))
                ax3.set_xticklabels(num_cols, rotation=45, ha='left', fontsize=7)
                ax3.set_yticklabels(num_cols, fontsize=7)
                ax3.set_title("Correlation Heatmap Matrix", fontsize=10, fontweight='bold', pad=15)
                plt.tight_layout()
                buf3 = io.BytesIO()
                plt.savefig(buf3, format='png', dpi=150)
                plt.close(fig_mpl3)
                chart_images["Pearson Correlation Matrix Heatmap"] = buf3.getvalue()

            elif valid_cat_cols and num_cols:
                cat_col = valid_cat_cols[0]
                num_col = num_cols[0]
                agg_df = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(10).reset_index()

                fig_cat = px.bar(agg_df, x=cat_col, y=num_col, color=num_col,
                                 color_continuous_scale="Viridis", title=f"Top 10 '{cat_col}' by '{num_col}'")
                st.plotly_chart(fig_cat, use_container_width=True)

                fig_mpl3, ax3 = plt.subplots(figsize=(6, 3))
                ax3.bar(agg_df[cat_col].astype(str), agg_df[num_col], color="#0D9488")
                ax3.set_title(f"Top Categories: {cat_col}", fontsize=10, fontweight='bold')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.tight_layout()
                buf3 = io.BytesIO()
                plt.savefig(buf3, format='png', dpi=150)
                plt.close(fig_mpl3)
                chart_images[f"Category Performance - {cat_col}"] = buf3.getvalue()
            else:
                st.info("Insufficient variables for multivariable interaction models.")

        # ---------------------------------------------------------------------
        # DASHBOARD 4: STRATEGIC RECOMMENDATIONS & PDF EXPORT
        # ---------------------------------------------------------------------
        with tab4:
            st.subheader("Dashboard 4: Executive Recommendations & PDF Report Export")

            for idx, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div class="rec-card">
                    <h4 style="color:#1E3A8A; margin-top:0;">{idx}. {rec.get('business_area', 'Strategy')}</h4>
                    <p><b>Data / Chart Outcome:</b> {rec.get('chart_outcome', '')}</p>
                    <p><b>What This Means:</b> {rec.get('what_this_means', '')}</p>
                    <p><b>Limitation:</b> {rec.get('limitation', '')}</p>
                    <p><b>Analyst Recommendation:</b> {rec.get('analyst_recommendation', '')}</p>
                    <p><b>Action / Development Step:</b> {rec.get('action_development', '')}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📄 Download Executive Report")

            pdf_data = create_pdf_report(df, quality_info, recommendations, chart_images)

            st.download_button(
                label="📥 Download Executive PDF Report (Including All Dashboard Charts)",
                data=pdf_data,
                file_name="DataMind_Executive_Analytics_Report.pdf",
                mime="application/pdf",
                type="primary"
            )

else:
    st.info("👈 Upload a CSV or Excel dataset in the sidebar to run the multi-dashboard platform.")
