import pandas as pd
import numpy as np


def calculate_kpis(df):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    kpis = {

        "Total Records":
            len(df),

        "Total Columns":
            len(df.columns),

        "Missing Values":
            int(df.isna().sum().sum()),

        "Duplicate Rows":
            int(df.duplicated().sum())
    }

    for column in numeric_columns[:6]:

        series = df[column].dropna()

        if len(series) == 0:
            continue

        kpis[f"{column} Total"] = round(
            series.sum(), 2
        )

        kpis[f"{column} Average"] = round(
            series.mean(), 2
        )

    return kpis


def calculate_correlations(df):

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    return numeric_df.corr().round(2)


def find_top_categories(
    df,
    category_column,
    value_column,
    n=10
):

    if (
        category_column not in df.columns
        or value_column not in df.columns
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

    numeric = df.select_dtypes(
        include=np.number
    )

    if numeric.empty:
        return pd.DataFrame()

    return pd.DataFrame({

        "Metric":
            numeric.columns,

        "Total":
            numeric.sum().round(2).values,

        "Average":
            numeric.mean().round(2).values,

        "Median":
            numeric.median().round(2).values,

        "Minimum":
            numeric.min().round(2).values,

        "Maximum":
            numeric.max().round(2).values
    })