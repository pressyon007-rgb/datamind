"""
insight_engine.py
Core analysis module for data quality audits, correlation computation, and statistical insight generation.
"""

import pandas as pd
import numpy as np


def analyze_data_quality(df):
    """
    Computes quality indicators including null counts, duplicates, empty fields, and outliers.
    """
    if df is None or df.empty:
        return {
            "total_rows": 0, "total_cols": 0, "is_perfect_quality": False,
            "total_missing": 0, "cols_with_missing": {}, "duplicate_rows": 0,
            "empty_cols": [], "outliers": {}
        }

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
    """
    Calculates pairwise Pearson correlation coefficients across numerical attributes.
    """
    relationships = []
    if df is None or df.empty:
        return relationships

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
    """
    Master helper returning combined quality and relationship payloads.
    """
    quality = analyze_data_quality(df)
    relationships = compute_relationships(df)
    return {"quality": quality, "relationships": relationships}
