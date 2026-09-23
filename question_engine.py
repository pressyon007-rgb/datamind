"""
question_engine.py - Analytical Exploration & Deterministic Q&A Engine
"""

import numpy as np
import pandas as pd


def _find_column(df, keywords):
    for keyword in keywords:
        for column in df.columns:
            if keyword.lower() in str(column).lower():
                return column
    return None


def generate_analytical_questions(df, domain_info=None):
    """
    Produces dynamic analytical questions based on dataset structure and domain.
    """
    questions = []
    domain = domain_info.get("domain", "General") if isinstance(domain_info, dict) else "General"
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if cat_cols and num_cols:
        questions.append({
            "category": "Group Comparisons",
            "question": f"How does the distribution of '{num_cols[0]}' vary across different '{cat_cols[0]}' categories?",
            "purpose": "Identifies operational variances across organizational or categorical divisions."
        })

    if len(num_cols) >= 2:
        questions.append({
            "category": "Linear Correlation",
            "question": f"Is there a statistical correlation between '{num_cols[0]}' and '{num_cols[1]}'?",
            "purpose": "Evaluates potential co-dependence between primary numerical metrics."
        })

    if domain == "Healthcare / Clinical":
        questions.append({
            "category": "Clinical Diagnostic Variance",
            "question": "Which diagnostic measurements exhibit the highest statistical dispersion among target patient groups?",
            "purpose": "Highlights critical physical dimensions useful for classification modeling."
        })
    elif domain == "Human Resources (HR)":
        questions.append({
            "category": "Workforce Retention",
            "question": "Are specific job levels or departments experiencing disproportionate compensation or tenure variance?",
            "purpose": "Evaluates equity and potential turnover risk factors across departments."
        })
    else:
        questions.append({
            "category": "Distribution Dynamics",
            "question": "Are there skewed distributions or multi-modal clusters in the core numeric attributes?",
            "purpose": "Determines whether data normalization or segmentation is required."
        })

    return questions


def answer_question(df, question, insights=None, recommendations=None):
    if df is None or df.empty:
        return "There is no uploaded data available to answer this question."

    if not question:
        return "Please select or enter a question."

    q_lower = question.lower()
    sales_column = _find_column(df, ["sales", "revenue", "amount", "income", "profit"])
    product_column = _find_column(df, ["product", "item", "product name"])

    if sales_column and pd.api.types.is_numeric_dtype(df[sales_column]):
        if ("highest" in q_lower or "top" in q_lower) and product_column:
            try:
                res = df.groupby(product_column)[sales_column].sum().sort_values(ascending=False)
                if not res.empty:
                    return f"The product with the highest {sales_column} is **{res.index[0]}** ({res.iloc[0]:,.2f})."
            except Exception:
                pass

        if "total" in q_lower:
            total_val = df[sales_column].sum()
            return f"The total **{sales_column}** is **{total_val:,.2f}**."

    return (
        f"Query evaluated against columns: {', '.join(map(str, df.columns[:8]))}. "
        "Use the dynamic dashboard tab for deeper metric visualisations."
    )
