"""
data_context.py - Prepares DataFrame summaries for LLM prompt context.
"""

import pandas as pd
import numpy as np


def create_data_context(df, max_sample_rows=5):
    """
    Generates a structured text overview of the uploaded DataFrame.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "No valid dataset uploaded."

    total_rows, total_cols = df.shape
    columns_list = list(df.columns)
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()

    context_lines = [
        f"DATASET STRUCTURE OVERVIEW:",
        f"- Total Rows: {total_rows:,}",
        f"- Total Columns: {total_cols}",
        f"- Columns: {', '.join([str(c) for c in columns_list])}",
        f"- Numerical Fields ({len(num_cols)}): {', '.join([str(c) for c in num_cols]) if num_cols else 'None'}",
        f"- Categorical Fields ({len(cat_cols)}): {', '.join([str(c) for c in cat_cols]) if cat_cols else 'None'}",
        f"- Datetime Fields ({len(date_cols)}): {', '.join([str(c) for c in date_cols]) if date_cols else 'None'}",
        "\nFIRST 5 SAMPLE ROWS:"
    ]

    # Sample rows formatted as text
    sample_df = df.head(max_sample_rows)
    context_lines.append(sample_df.to_string(index=False))

    # Numerical Summary Statistics
    if num_cols:
        context_lines.append("\nNUMERICAL SUMMARY STATISTICS:")
        stats = df[num_cols].describe().T[['mean', 'std', 'min', '50%', 'max']].to_string()
        context_lines.append(stats)

    return "\n".join(context_lines)
