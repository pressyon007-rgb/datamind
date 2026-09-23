"""
insight_engine.py - Statistical Profiling & Quality Audit
"""

import pandas as pd
import numpy as np


def analyze_data_quality(df):
    """
    Audits data quality. Identifies positive traits (zero nulls/duplicates)
    alongside genuine anomalies or warnings.
    """"""
insight_engine.py - Data Quality & Structural Relationship Calculations
"""

import pandas as pd
import numpy as np


def analyze_data_quality(df):
    """
    Computes summary metrics for data quality and integrity.
    Always returns a clean dictionary.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return {
            "total_rows": 0,
            "total_cols": 0,
            "is_perfect_quality": False,
            "total_missing": 0,
            "cols_with_missing": {},
            "duplicate_rows": 0,
            "empty_cols": [],
            "outliers": {}
        }

    total_rows = len(df)
    total_cols = len(df.columns)

    missing_series = df.isnull().sum()
    cols_with_missing = missing_series[missing_series > 0].to_dict()
    cols_with_missing = {str(k): int(v) for k, v in cols_with_missing.items()}
    total_missing = int(missing_series.sum())

    duplicate_rows = int(df.duplicated().sum())
    empty_cols = [str(col) for col in df.columns if df[col].isnull().all()]

    # Detect outliers using Interquartile Range (IQR) for numeric fields
    outliers = {}
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                count = int(((col_data < lower) | (col_data > upper)).sum())
                if count > 0:
                    outliers[str(col)] = count

    is_perfect = (total_missing == 0) and (duplicate_rows == 0)

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "is_perfect_quality": is_perfect,
        "total_missing": total_missing,
        "cols_with_missing": cols_with_missing,
        "duplicate_rows": duplicate_rows,
        "empty_cols": empty_cols,
        "outliers": outliers
    }


def compute_relationships(df):
    """
    Computes pairwise correlations among numeric features.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return []

    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return []

    try:
        corr_matrix = num_df.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)
        
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                v1 = str(corr_matrix.columns[i])
                v2 = str(corr_matrix.columns[j])
                val = float(corr_matrix.iloc[i, j])
                
                if not np.isnan(val) and val > 0.3:
                    strength = "Strong" if val >= 0.7 else ("Moderate" if val >= 0.4 else "Weak")
                    pairs.append({
                        "var1": v1,
                        "var2": v2,
                        "correlation": round(val, 3),
                        "strength": strength
                    })

        return sorted(pairs, key=lambda x: x["correlation"], reverse=True)
    except Exception:
        return []
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
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            outliers = series[(series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))]
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
    num_df = df.select_dtypes(include=[np.number]).dropna(how="all", axis=1)
    if num_df.shape[1] < 2:
        return []

    corr_matrix = num_df.corr().abs().copy()

    relationships = []
    visited = set()

    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 == col2:
                continue
                
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
