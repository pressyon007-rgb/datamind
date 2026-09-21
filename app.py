"""
app.py - Multi-Dashboard Streamlit Application with PDF Report Generation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as io
import io
import matplotlib.pyplot as plt

from insight_engine import generate_insights
from question_engine import answer_question
from recommendation_engine import generate_recommendations
from web_research import perform_web_research

st.set_page_config(page_title="DataMind Analytics", layout="wide")

st.title("DataMind Analytics Platform")
st.write("Upload your dataset to generate interactive dashboards, automated insights, and exported PDF reports.")

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

if uploaded_file is not None:
    # 1. Load Data
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Pre-calculate Data Quality Metrics
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    quality_info = {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "cols_with_missing": df.columns[df.isnull().any()].tolist(),
        "is_perfect_quality": (df.isnull().sum().sum() == 0 and df.duplicated().sum() == 0),
        "outliers": {}
    }
    
    # Calculate basic outliers via IQR
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outlier_cnt = int(((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum())
        if outlier_cnt > 0:
            quality_info["outliers"][col] = outlier_cnt

    # Correlation relationships
    relationships = []
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

    domain_info = {"domain": "Business Operations"}

    # Dictionary to hold rendered chart images for the PDF report
    chart_images = {}

    # Create 4 Separate Dashboard Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 1. Overview & Data Quality",
        "📈 2. Distributions & Variance",
        "🔗 3. Relationships & Correlations",
        "🎯 4. Strategic Recommendations"
    ])

    # -------------------------------------------------------------------------
    # DASHBOARD 1: OVERVIEW & DATA QUALITY
    # -------------------------------------------------------------------------
    with tab1:
        st.header("Dashboard 1: Data Overview & Quality Audit")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{quality_info['total_rows']:,}")
        col2.metric("Total Attributes", f"{quality_info['total_cols']:,}")
        col3.metric("Missing Values", f"{quality_info['total_missing']:,}")
        col4.metric("Duplicate Rows", f"{quality_info['duplicate_rows']:,}")

        st.subheader("Data Completeness by Field")
        null_counts = df.isnull().sum().reset_index()
        null_counts.columns = ["Column", "Missing_Count"]
        
        fig1 = px.bar(null_counts, x="Column", y="Missing_Count", title="Missing Values per Column")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Save static image for PDF using matplotlib
        fig_mpl, ax = plt.subplots(figsize=(6, 3))
        ax.bar(null_counts["Column"], null_counts["Missing_Count"], color="#3182bd")
        ax.set_title("Missing Values per Column")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        img_buf1 = io.BytesIO()
        plt.savefig(img_buf1, format='png', dpi=150)
        plt.close()
        chart_images["Data Quality Chart"] = img_buf1.getvalue()

    # -------------------------------------------------------------------------
    # DASHBOARD 2: DISTRIBUTIONS & VARIANCE
    # -------------------------------------------------------------------------
    with tab2:
        st.header("Dashboard 2: Numerical Distributions & Outliers")
        if num_cols:
            selected_num = st.selectbox("Select Numerical Feature to Analyze", num_cols)
            
            col_a, col_b = st.columns(2)
            with col_a:
                fig2 = px.histogram(df, x=selected_num, title=f"Distribution of {selected_num}")
                st.plotly_chart(fig2, use_container_width=True)
            with col_b:
                fig3 = px.box(df, y=selected_num, title=f"Outlier Box Plot of {selected_num}")
                st.plotly_chart(fig3, use_container_width=True)

            # Save static image for PDF
            fig_mpl, ax = plt.subplots(figsize=(6, 3))
            ax.hist(df[selected_num].dropna(), bins=20, color="#2ca02c", edgecolor="black")
            ax.set_title(f"Distribution of {selected_num}")
            plt.tight_layout()
            img_buf2 = io.BytesIO()
            plt.savefig(img_buf2, format='png', dpi=150)
            plt.close()
            chart_images[f"Distribution - {selected_num}"] = img_buf2.getvalue()
        else:
            st.info("No numerical columns found in the dataset.")

    # -------------------------------------------------------------------------
    # DASHBOARD 3: RELATIONSHIPS & CORRELATIONS
    # -------------------------------------------------------------------------
    with tab3:
        st.header("Dashboard 3: Feature Interactions & Correlation Matrix")
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)

            # Matplotlib version for PDF
            fig_mpl, ax = plt.subplots(figsize=(5, 4))
            cax = ax.matshow(corr, cmap='coolwarm')
            fig_mpl.colorbar(cax)
            ax.set_xticks(range(len(num_cols)))
            ax.set_yticks(range(len(num_cols)))
            ax.set_xticklabels(num_cols, rotation=90)
            ax.set_yticklabels(num_cols)
            ax.set_title("Correlation Heatmap", pad=20)
            plt.tight_layout()
            img_buf3 = io.BytesIO()
            plt.savefig(img_buf3, format='png', dpi=150)
            plt.close()
            chart_images["Correlation Heatmap"] = img_buf3.getvalue()
        else:
            st.info("At least two numerical features are required to compute correlations.")

    # -------------------------------------------------------------------------
    # DASHBOARD 4: STRATEGIC RECOMMENDATIONS & PDF EXPORT
    # -------------------------------------------------------------------------
    with tab4:
        st.header("Dashboard 4: Decision Support & Executive Export")
        
        recs = generate_recommendations(df, quality_info, relationships, domain_info)
        
        for r in recs:
            with st.expander(f"📌 {r['business_area']}", expanded=True):
                st.write(f"**Outcome:** {r['chart_outcome']}")
                st.write(f"**Meaning:** {r['what_this_means']}")
                st.write(f"**Recommendation:** {r['analyst_recommendation']}")
                st.write(f"**Action:** {r['action_development']}")

        st.markdown("---")
        st.subheader("📄 Export Executive PDF Report")
        
        # PDF Generator Import
        from pdf_generator import create_pdf_report
        
        if st.button("Generate & Download PDF Report"):
            pdf_data = create_pdf_report(df, quality_info, recs, chart_images)
            st.download_button(
                label="📥 Download Complete Report (with All Charts)",
                data=pdf_data,
                file_name="DataMind_Executive_Report.pdf",
                mime="application/pdf"
            )
