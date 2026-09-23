"""
app.py - Data Analyzer AI Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.multiclass import type_of_target

# Import custom modules
from web_research import detect_domain
from insight_engine import analyze_data_quality, compute_relationships
from question_engine import generate_analytical_questions
from recommendation_engine import generate_recommendations
from ai_analyst import ask_data_analyst
from data_context import create_data_context

# Page Configuration
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
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_uploaded_file(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def train_risk_model(df, target_col):
    data = df.copy().dropna()
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
    
    target_type = type_of_target(y)
    
    if target_type == 'continuous':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
    else:
        if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
            target_le = LabelEncoder()
            y = target_le.fit_transform(y.astype(str))
            encoders[target_col] = target_le
            
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
    
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return model, encoders, importances, X.columns.tolist()


# Sidebar Configuration
st.sidebar.title("🧠 Data Analyzer AI")
uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel File", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = load_uploaded_file(uploaded_file)

    if df is not None:
        target_field = st.sidebar.selectbox("Select Target / Risk Field:", df.columns)
        palette = st.sidebar.selectbox("Select Dashboard Color Theme:", ["Blues", "Viridis", "Cividis", "Plasma", "Turbo", "Magma"], index=0)

        # Run Domain Discovery with Type Guard
        domain_info = detect_domain(df)
        if not isinstance(domain_info, dict):
            domain_info = {"domain": "General / Operations", "confidence": "Low", "matched_terms": []}

        quality_info = analyze_data_quality(df)
        relationships = compute_relationships(df)
        questions = generate_analytical_questions(df, domain_info)
        recommendations = generate_recommendations(df, quality_info, relationships, domain_info)

        # Main Display
        st.title("Data Analyzer AI")
        st.caption("Intelligent Data Analysis & Decision Support Platform")
        st.info(f"Detected Industry / Domain: **{domain_info.get('domain', 'General')}** (Confidence: **{domain_info.get('confidence', 'N/A')}**)")

        # Main Navigation Tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📋 Overview",
            "📊 Dashboard Analytics",
            "🔍 Data Quality",
            "📈 Statistics",
            "❓ Analytical Questions",
            "🤖 Risk & Feature Importance",
            "💡 Recommendations",
            "💬 AI Analyst Chat"
        ])

        with tab1:
            st.subheader("Dataset Summary & Structure")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Records", f"{quality_info.get('total_rows', 0):,}" if isinstance(quality_info, dict) else len(df))
            c2.metric("Total Columns", f"{quality_info.get('total_cols', 0):,}" if isinstance(quality_info, dict) else len(df.columns))
            c3.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
            c4.metric("Categorical Columns", len(df.select_dtypes(include=['object', 'category']).columns))

            st.markdown("---")
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

        with tab2:
            st.subheader("Interactive Visual Dashboard & Custom Chart Builder")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            with st.expander("🎨 Custom Chart Builder", expanded=True):
                b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                chart_type = b_col1.selectbox("Select Chart Type:", ["Bar Chart", "Pie / Donut Chart", "Line Chart", "Scatter Plot", "Box Plot", "Histogram", "Heatmap"])
                x_axis = b_col2.selectbox("Select X-Axis / Grouping Column:", df.columns, index=0)
                y_default_idx = 1 if len(df.columns) > 1 else 0
                y_axis = b_col3.selectbox("Select Y-Axis Column:", df.columns, index=y_default_idx)
                agg_func = b_col4.selectbox("Aggregation Method:", ["Sum", "Mean", "Count", "Median"])

                try:
                    fig = None
                    if chart_type in ["Bar Chart", "Pie / Donut Chart", "Line Chart"]:
                        if agg_func == "Sum":
                            grouped_df = df.groupby(x_axis, as_index=False)[y_axis].sum().head(15)
                        elif agg_func == "Mean":
                            grouped_df = df.groupby(x_axis, as_index=False)[y_axis].mean().head(15)
                        elif agg_func == "Median":
                            grouped_df = df.groupby(x_axis, as_index=False)[y_axis].median().head(15)
                        else:
                            grouped_df = df[x_axis].value_counts().reset_index().head(15)
                            grouped_df.columns = [x_axis, y_axis]

                        if chart_type == "Bar Chart":
                            fig = px.bar(grouped_df, x=x_axis, y=y_axis, text_auto='.2s', title=f"{agg_func} of {y_axis} by {x_axis}")
                        elif chart_type == "Pie / Donut Chart":
                            fig = px.pie(grouped_df, names=x_axis, values=y_axis, hole=0.4, title=f"{agg_func} Share of {y_axis} by {x_axis}")
                        elif chart_type == "Line Chart":
                            fig = px.line(grouped_df, x=x_axis, y=y_axis, markers=True, title=f"{agg_func} Trend: {y_axis} over {x_axis}")

                    elif chart_type == "Scatter Plot":
                        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot: {x_axis} vs {y_axis}", opacity=0.8)

                    elif chart_type == "Box Plot":
                        fig = px.box(df, x=x_axis, y=y_axis, points="outliers", title=f"Distribution of {y_axis} across {x_axis}")

                    elif chart_type == "Histogram":
                        fig = px.histogram(df, x=x_axis, nbins=30, marginal="box", title=f"Histogram of {x_axis}")

                    elif chart_type == "Heatmap":
                        if len(num_cols) >= 2:
                            corr_data = df[num_cols].corr()
                            fig = px.imshow(corr_data, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Matrix Heatmap")
                        else:
                            st.warning("Heatmap requires at least two numerical columns.")

                    if fig is not None:
                        fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as err:
                    st.error(f"Could not build chart with selected parameters: {err}")

        with tab3:
            st.subheader("Data Quality & Integrity")
            if isinstance(quality_info, dict):
                if quality_info.get("is_perfect_quality", False):
                    st.success("✅ Positive Finding: No missing values or duplicates detected.")
                else:
                    st.warning("⚠️ Data Quality Notice: Missing values or duplicates were found.")

                col_q1, col_q2 = st.columns(2)
                with col_q1:
                    st.write("**Missing Values**")
                    missing_cols = quality_info.get("cols_with_missing", {})
                    if missing_cols:
                        st.dataframe(pd.DataFrame(list(missing_cols.items()), columns=["Column", "Missing Count"]), use_container_width=True)
                    else:
                        st.info("Zero missing values across all columns.")

                with col_q2:
                    st.write("**Data Integrity Metrics**")
                    st.write(f"- Duplicate Rows: **{quality_info.get('duplicate_rows', 0)}**")
                    st.write(f"- Empty Columns: **{len(quality_info.get('empty_cols', []))}**")
                    st.write(f"- Columns with Outliers: **{len(quality_info.get('outliers', {}))}**")

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

        with tab5:
            st.subheader("Analytical Questions & Hypotheses")
            if isinstance(questions, list):
                for q in questions:
                    if isinstance(q, dict):
                        with st.expander(f"📌 {q.get('category', 'Analysis')}: {q.get('question', '')}"):
                            st.write(f"**Purpose:** {q.get('purpose', 'N/A')}")

        with tab6:
            st.subheader(f"Feature Importance for Target: '{target_field}'")
            try:
                model, encoders, importances, feature_names = train_risk_model(df, target_field)
                fig_imp = px.bar(importances.head(10), x='Importance', y='Feature', orientation='h', title="Top Drivers / Features Influencing Target Variable")
                fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e:
                st.error(f"Could not build feature importance model: {e}")

        with tab7:
            st.subheader("Evidence-Based Data Analyst Recommendations")
            if isinstance(recommendations, list):
                for idx, rec in enumerate(recommendations, 1):
                    if isinstance(rec, dict):
                        st.markdown(f"""
                        <div class="rec-box">
                            <div class="rec-title">Recommendation {idx}: {rec.get('business_area', 'General')}</div>
                            <p><span class="field-label">Chart / Data Outcome:</span> {rec.get('chart_outcome', '')}</p>
                            <p><span class="field-label">What This Means:</span> {rec.get('what_this_means', '')}</p>
                            <p><span class="field-label">Limitation:</span> {rec.get('limitation', '')}</p>
                            <p><span class="field-label">Data Analyst Recommendation:</span> {rec.get('analyst_recommendation', '')}</p>
                            <p><span class="field-label">Action / Development:</span> {rec.get('action_development', '')}</p>
                        </div>
                        """, unsafe_allow_html=True)

        with tab8:
            st.subheader("💬 Ask the Gemini AI Data Analyst")
            user_question = st.text_input("Ask a custom business question about this dataset:")
            if st.button("Analyze with Gemini AI"):
                if user_question:
                    with st.spinner("Analyzing dataset with Gemini..."):
                        context_str = create_data_context(df)
                        ai_response = ask_data_analyst(
                            question=user_question,
                            data_context=context_str,
                            insights=quality_info,
                            recommendations=recommendations
                        )
                        st.markdown("### AI Analyst Insights")
                        st.write(ai_response)
                else:
                    st.warning("Please enter a question to analyze.")
else:
    st.info("👈 Upload a CSV or Excel dataset in the sidebar to begin.")
