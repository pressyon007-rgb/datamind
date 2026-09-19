"""
question_engine.py - Analytical Exploration Generator
"""

import numpy as np


def generate_analytical_questions(df, domain_info):
    """
    Produces dynamic analytical questions based on dataset structure and domain.
    """
    questions = []
    domain = domain_info["domain"]
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