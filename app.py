"""
app.py - Data Analyzer AI Platform
Interactive Streamlit application for data analysis, visual exploration, machine learning, and decision support.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
        return []

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


# Helper Function: Robust Data Loading
@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, sheet_name=0, engine='openpyxl')
            
        # Safeguard: Downsample if dataset exceeds 3,000 rows to prevent Streamlit RAM crashes
        if len(df) > 3000:
            st.warning("⚠️ Large dataset detected. Downsampling to 3,000 rows for high-performance cloud processing.")
            df = df.sample(n=3000, random_state=42).reset_index(drop=True)
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


# Machine Learning Trainer Function (Handles Classifier + Regressor)
def train_risk_model(df, target_col):
    data = df.copy().dropna()
    if data.empty:
        raise ValueError("Dataset has no complete rows after dropping missing values.")
        
    encoders = {}
    
    # Convert datetime columns into numerical year/month features
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

    # Determine target variable type
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
st.sidebar.write("Upload a CSV or Excel file to analyze structure, quality, and recommendations.")

uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel File", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is not None and not df.empty:
        # Target Selection in Sidebar
        target_field = st.sidebar.selectbox("Select Target / Risk Field:", df.columns)
        
        # Dashboard Color Theme
        palette = st.sidebar.selectbox("Select Dashboard Color Theme:", ["Blues", "Viridis", "Cividis", "Plasma", "Turbo", "Magma"], index=0)

        # Run Analytical Pipeline Safely
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
            st.subheader("Dataset Summary & Structure")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Records", f"{quality_info.get('total_rows', len(df)):,}")
            c2.metric("Total Columns", f"{quality_info.get('total_cols', len(df.columns)):,}")
            c3.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
            c4.metric("Categorical Columns", len(df.select_dtypes(include=['object', 'category']).columns))

            st.markdown("---")
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

        # TAB 2: EXPANDED DASHBOARD ANALYTICS
        with tab2:
            st.subheader("Interactive Visual Dashboard & Custom Chart Builder")
            
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()

            # Attempt auto-parsing date columns if stored as strings
            if not date_cols:
                for c in cat_cols:
                    if 'date' in c.lower() or 'time' in c.lower():
                        try:
                            df[c] = pd.to_datetime(df[c])
                            date_cols.append(c)
                        except Exception:
                            pass

            valid_cat_cols = [c for c in cat_cols if not c.lower().endswith('id') and df[c].nunique() <= 50]

            # Custom Chart Builder
            with st.expander("🎨 Custom Chart Builder (Interactive Options)", expanded=True):
                b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                
                chart_type = b_col1.selectbox(
                    "Select Chart Type:",
                    ["Bar Chart", "Pie / Donut Chart", "Line Chart", "Scatter Plot", "Box Plot", "Histogram", "Heatmap"]
                )
                
                x_axis = b_col2.selectbox("Select X-Axis / Grouping Column:", df.columns, index=0)
                y_default_idx = 1 if len(df.columns) > 1 else 0
                y_axis = b_col3.selectbox("Select Y-Axis Column:", df.columns, index=y_default_idx)
                agg_func = b_col4.selectbox("Aggregation Method:", ["Sum", "Mean", "Count", "Median"])

                # Render Custom Chart
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
                            fig = px.bar(grouped_df, x=x_axis, y=y_axis, color=y_axis,
                                         color_continuous_scale=palette.lower(), text_auto='.2s',
                                         title=f"{agg_func} of {y_axis} by {x_axis}")
                        elif chart_type == "Pie / Donut Chart":
                            fig = px.pie(grouped_df, names=x_axis, values=y_axis, hole=0.4,
                                         color_discrete_sequence=px.colors.qualitative.Set2,
                                         title=f"{agg_func} Share of {y_axis} by {x_axis}")
                        elif chart_type == "Line Chart":
                            fig = px.line(grouped_df, x=x_axis, y=y_axis, markers=True,
                                          title=f"{agg_func} Trend: {y_axis} over {x_axis}")

                    elif chart_type == "Scatter Plot":
                        color_by = valid_cat_cols[0] if valid_cat_cols else None
                        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by,
                                         title=f"Scatter Plot: {x_axis} vs {y_axis}", opacity=0.8)

                    elif chart_type == "Box Plot":
                        fig = px.box(df, x=x_axis, y=y_axis, points="outliers",
                                     title=f"Distribution & Outliers of {y_axis} across {x_axis}")

                    elif chart_type == "Histogram":
                        fig = px.histogram(df, x=x_axis, nbins=30, marginal="box",
                                           title=f"Distribution Histogram of {x_axis}")

                    elif chart_type == "Heatmap":
                        if len(num_cols) >= 2:
                            corr_data = df[num_cols].corr()
                            fig = px.imshow(corr_data, text_auto=True, color_continuous_scale="RdBu_r",
                                            title="Correlation Matrix Heatmap")
                        else:
                            st.warning("Heatmap requires at least two numerical columns.")

                    if fig is not None:
                        fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as err:
                    st.error(f"Could not build chart with selected parameters: {err}")

            st.markdown("---")

            # Automated Multi-View Grid
            st.markdown("### 📊 Automated Multi-Metric Visual Insights")
            grid_col1, grid_col2 = st.columns(2)

            with grid_col1:
                if valid_cat_cols and num_cols:
                    cat_field = grid_col1.selectbox("Select Categorical Breakdown:", valid_cat_cols, index=0, key="cat_break")
                    num_field = grid_col1.selectbox("Select Numerical Metric:", num_cols, index=0, key="num_break")
                    
                    top_agg = df.groupby(cat_field)[num_field].sum().sort_values(ascending=False).head(10).reset_index()
                    fig_top = px.bar(top_agg, x=cat_field, y=num_field, color=num_field,
                                     color_continuous_scale=palette.lower(), text_auto='.2s',
                                     title=f"Top 10 {cat_field} by Total {num_field}")
                    fig_top.update_layout(height=380)
                    st.plotly_chart(fig_top, use_container_width=True)

            with grid_col2:
                if len(num_cols) >= 2:
                    scat_x = grid_col2.selectbox("Select X Metric:", num_cols, index=0, key="scat_x")
                    scat_y_idx = 1 if len(num_cols) > 1 else 0
                    scat_y = grid_col2.selectbox("Select Y Metric:", num_cols, index=scat_y_idx, key="scat_y")
                    
                    fig_scat = px.scatter(df, x=scat_x, y=scat_y,
                                          color=valid_cat_cols[0] if valid_cat_cols else None,
                                          trendline=None,
                                          title=f"Relationship: {scat_x} vs {scat_y}")
                    fig_scat.update_layout(height=380)
                    st.plotly_chart(fig_scat, use_container_width=True)

            grid_col3, grid_col4 = st.columns(2)

            with grid_col3:
                if num_cols:
                    dist_col = grid_col3.selectbox("Select Feature for Distribution Check:", num_cols, index=0, key="dist_col")
                    fig_box = px.box(df, y=dist_col, color_discrete_sequence=['#2563EB'],
                                     title=f"Spread & Outliers in '{dist_col}'")
                    fig_box.update_layout(height=350)
                    st.plotly_chart(fig_box, use_container_width=True)

            with grid_col4:
                if date_cols and num_cols:
                    t_date = grid_col4.selectbox("Select Date Column:", date_cols, index=0, key="t_date")
                    t_metric = grid_col4.selectbox("Select Metric over Time:", num_cols, index=0, key="t_metric")
                    
                    time_df = df.groupby(t_date)[t_metric].sum().reset_index()
                    fig_time = px.line(time_df, x=t_date, y=t_metric, title=f"Trend of {t_metric} over Time")
                    fig_time.update_layout(height=350)
                    st.plotly_chart(fig_time, use_container_width=True)
                elif len(num_cols) >= 2:
                    st.markdown("##### Feature Correlation Heatmap")
                    corr_sub = df[num_cols].corr()
                    fig_map = px.imshow(corr_sub, text_auto=True, color_continuous_scale="Blues")
                    fig_map.update_layout(height=350)
                    st.plotly_chart(fig_map, use_container_width=True)

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
            st.subheader("Analytical Questions & Hypotheses")
            if questions:
                for q in questions:
                    with st.expander(f"📌 {q.get('category', 'Analysis')}: {q.get('question', '')}"):
                        st.write(f"**Purpose:** {q.get('purpose', '')}")
            else:
                st.info("No domain hypothesis questions generated for this dataset.")

        # TAB 6: RISK & FEATURE IMPORTANCE
        with tab6:
            st.subheader(f"Feature Importance for Target: '{target_field}'")
            try:
                model, encoders, importances, feature_names = train_risk_model(df, target_field)
                fig_imp = px.bar(importances.head(10), x='Importance', y='Feature', orientation='h',
                                 title="Top Drivers / Features Influencing Target Variable")
                fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
                st.plotly_chart(fig_imp, use_container_width=True)
            except Exception as e:
                st.error(f"Could not build feature importance model: {e}")

        # TAB 7: RECOMMENDATIONS
        with tab7:
            st.subheader("Evidence-Based Data Analyst Recommendations")
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
