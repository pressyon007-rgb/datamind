"""
insight_engine.py - Statistical Profiling & Quality Audit
"""

import pandas as pd
import numpy as np


def analyze_data_quality(df):
    """
    Audits data quality. Identifies positive traits (zero nulls/duplicates)
    alongside genuine anomalies or warnings.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    cols_with_missing = missing_counts[missing_counts > 0].to_dict()
    
    duplicate_rows = int(df.duplicated().sum())
    empty_cols = [col for col in df.columns if df[col].isnull().all()]

    # Numeric distribution & potential outlier checks (IQR method)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_info = {}
    
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))]
            if len(outliers) > 0:
                outlier_info[col] = len(outliers)

    is_perfect_quality = (total_missing == 0) and (duplicate_rows == 0) and (len(empty_cols) == 0)

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "is_perfect_quality": is_perfect_quality,
        "total_missing": int(total_missing),
        "cols_with_missing": cols_with_missing,
        "duplicate_rows": duplicate_rows,
        "empty_cols": empty_cols,
        "outliers": outlier_info
    }


def compute_relationships(df):
    """
    Calculates pairwise numerical correlations and returns key relationships.
    Avoids modifying read-only NumPy array diagonals.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return []

    # Calculate absolute correlation matrix and make an explicit copy
    corr_matrix = num_df.corr().abs().copy()

    relationships = []
    visited = set()

    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 == col2:
                continue  # Skip comparing the same column with itself
                
            pair_key = tuple(sorted([col1, col2]))
            if pair_key not in visited:
                visited.add(pair_key)
                val = corr_matrix.loc[col1, col2]
                if not np.isnan(val) and val >= 0.4:
                    strength = "Strong" if val >= 0.7 else "Moderate"
                    relationships.append({
                        "var1": col1,
                        "var2": col2,
                        "correlation": round(float(val), 3),
                        "strength": strength
                    })

    return sorted(relationships, key=lambda x: x["correlation"], reverse=True)