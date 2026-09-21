"""
app.py - Data Analyzer AI Platform (Diagnostic Version)
"""

import sys
import traceback
import io
import streamlit as st
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(page_title="Data Analyzer AI", page_icon="⚡", layout="wide")

try:
    # ---------------------------------------------------------
    # Safe Library Imports
    # ---------------------------------------------------------
    import plotly.express as px
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import (
        r2_score, mean_squared_error, mean_absolute_error,
        accuracy_score, precision_score, recall_score, f1_score
    )
    from sklearn.utils.multiclass import type_of_target

    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.pdfgen import canvas

    # Custom Modules
    try:
        from web_research import detect_domain
    except ImportError:
        def detect_domain(df):
            return {"domain": "General Analytics", "confidence": "Medium"}

    try:
        from insight_engine import analyze_data_quality, compute_relationships
    except ImportError:
        def analyze_data_quality(df):
            return {
                "total_rows": len(df),
                "total_cols": len(df.columns),
                "is_perfect_quality": df.isnull().sum().sum() == 0 and df.duplicated().sum() == 0,
                "cols_with_missing": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
                "duplicate_rows": int(df.duplicated().sum()),
                "empty_cols": [c for c in df.columns if df[c].isnull().all()]
            }
        def compute_relationships(df):
            return []

    try:
        from question_engine import generate_analytical_questions
    except ImportError:
        def generate_analytical_questions(df, domain_info):
            return [{"category": "Analysis", "question": "What are key drivers?", "purpose": "Isolate core factors."}]

    try:
        from recommendation_engine import generate_recommendations
    except ImportError:
        def generate_recommendations(df, quality_info, relationships, domain_info):
            return [{"business_area": "Governance", "action_development": "Clean missing entries."}]


    # ---------------------------------------------------------
    # Helper Functions
    # ---------------------------------------------------------
    def get_analytical_columns(df):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
        return num_cols, cat_cols, date_cols

    def load_data(uploaded_file, row_limit=10000):
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, sheet_name=0)
            
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()

        if len(df) > row_limit:
            df = df.sample(n=row_limit, random_state=42).reset_index(drop=True)
            
        return df

    def train_risk_model(df, target_col):
        data = df.copy().dropna()
        if data.empty:
            raise ValueError("Dataset has no complete rows after removing missing values.")
            
        encoders = {}
        date_cols = data.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
        for col in date_cols:
            if col != target_col:
                data[f"{col}_Year"] = data[col].dt.year
                data[f"{col}_Month"] = data[col].dt.month
                data = data.drop(columns=[col])

        for col in list(data.columns):
            if col != target_col and (data[col].nunique() > 100 or data[col].nunique() <= 1):
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
            raise ValueError("No valid features remaining to train the model.")

        target_type = type_of_target(y)
        
        if target_type in ['continuous', 'continuous-multioutput']:
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X, y)
            preds = model.predict(X)
            metrics = {
                "Model Type": "Random Forest Regressor",
                "R² Score": r2_score(y, preds),
                "MAE": mean_absolute_error(y, preds)
            }
        else:
            if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):
                target_le = LabelEncoder()
                y = target_le.fit_transform(y.astype(str))
                encoders[target_col] = target_le
                
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X, y)
            preds = model.predict(X)
            metrics = {
                "Model Type": "Random Forest Classifier",
                "Accuracy": accuracy_score(y, preds),
                "F1-Score": f1_score(y, preds, average='weighted', zero_division=0)
            }
        
        importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        return model, encoders, importances, metrics


    # ---------------------------------------------------------
    # UI App Pipeline
    # ---------------------------------------------------------
    st.title("⚡ Data Analyzer AI Platform")

    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        num_cols, cat_cols, date_cols = get_analytical_columns(df)
        
        target_field = st.sidebar.selectbox("Target / Driver Field:", df.columns, index=0)

        domain_info = detect_domain(df)
        quality_info = analyze_data_quality(df)
        relationships = compute_relationships(df)
        questions = generate_analytical_questions(df, domain_info)
        recommendations = generate_recommendations(df, quality_info, relationships, domain_info)

        tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📊 Visualizations", "🤖 ML Engine"])

        with tab1:
            st.subheader("Dataset Summary")
            st.write(f"Total Rows: **{len(df):,}** | Total Columns: **{len(df.columns)}**")
            st.dataframe(df.head(10), use_container_width=True)

        with tab2:
            st.subheader("Quick Charts")
            if cat_cols and num_cols:
                fig = px.bar(df.head(50), x=cat_cols[0], y=num_cols[0], color=cat_cols[0])
                st.plotly_chart(fig, use_container_width=True)
            elif num_cols:
                fig = px.histogram(df, x=num_cols[0])
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader(f"Model Driver Evaluation for '{target_field}'")
            if st.button("Run ML Pipeline"):
                model, encoders, importances, metrics = train_risk_model(df, target_field)
                st.write("**Model Metrics:**", metrics)
                fig_imp = px.bar(importances.head(10), x='Importance', y='Feature', orientation='h')
                st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Upload a CSV or XLSX file to begin.")

except Exception as err:
    st.error("❌ An Error Occurred While Running the Application:")
    st.code(traceback.format_exc(), language="python")
