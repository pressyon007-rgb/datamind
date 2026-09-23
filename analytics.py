import pandas as pd
import numpy as np


def calculate_kpis(df):
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    kpis = {
        "Total Records": len(df),
        "Total Columns": len(df.columns),
        "Missing Values": int(df.isna().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum())
    }

    for column in numeric_columns[:6]:
        series = df[column].dropna()
        if len(series) == 0:
            continue

        kpis[f"{column} Total"] = round(float(series.sum()), 2)
        kpis[f"{column} Average"] = round(float(series.mean()), 2)

    return kpis


def calculate_correlations(df):
    numeric_df = df.select_dtypes(include=np.number).dropna(how="all", axis=1)

    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    return numeric_df.corr().round(2)


def find_top_categories(df, category_column, value_column, n=10):
    if (
        category_column not in df.columns
        or value_column not in df.columns
        or not pd.api.types.is_numeric_dtype(df[value_column])
    ):
        return pd.DataFrame()

    return (
        df.groupby(category_column)[value_column]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )


def get_numeric_summary(df):
    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        return pd.DataFrame()

    valid_cols = [c for c in numeric.columns if not numeric[c].dropna().empty]
    if not valid_cols:
        return pd.DataFrame()

    sub_df = numeric[valid_cols]

    return pd.DataFrame({
        "Metric": sub_df.columns,
        "Total": sub_df.sum().round(2).values,
        "Average": sub_df.mean().round(2).values,
        "Median": sub_df.median().round(2).values,
        "Minimum": sub_df.min().round(2).values,
        "Maximum": sub_df.max().round(2).values
    })
