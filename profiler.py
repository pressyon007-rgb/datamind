import pandas as pd
import numpy as np


def get_data_profile(df):

    profile = pd.DataFrame({

        "Column": df.columns,

        "Data Type":
            df.dtypes.astype(str).values,

        "Non-Null":
            df.notna().sum().values,

        "Missing":
            df.isna().sum().values,

        "Unique":
            df.nunique().values
    })

    profile["Missing %"] = (
        profile["Missing"] /
        max(len(df), 1) *
        100
    ).round(2)

    return profile


def get_missing_values(df):

    result = pd.DataFrame({

        "Column": df.columns,

        "Missing Count":
            df.isna().sum().values
    })

    result["Missing %"] = (
        result["Missing Count"] /
        max(len(df), 1) *
        100
    ).round(2)

    return result.sort_values(
        "Missing Count",
        ascending=False
    )


def get_data_quality_score(df):

    total_cells = df.shape[0] * df.shape[1]

    if total_cells == 0:
        return 0

    missing_cells = df.isna().sum().sum()

    duplicate_rows = df.duplicated().sum()

    missing_penalty = (
        missing_cells /
        total_cells *
        100
    )

    duplicate_penalty = (
        duplicate_rows /
        max(len(df), 1) *
        100
    )

    score = (
        100 -
        missing_penalty -
        duplicate_penalty
    )

    return round(max(score, 0), 2)


def get_summary_statistics(df):

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.empty:
        return pd.DataFrame()

    result = numeric_df.describe().T

    result["median"] = numeric_df.median()

    result["missing"] = numeric_df.isna().sum()

    result["missing_%"] = (
        numeric_df.isna().mean() * 100
    ).round(2)

    result["skewness"] = numeric_df.skew().round(2)

    result["variance"] = numeric_df.var().round(2)

    return result


def detect_outliers(df):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    results = []

    for column in numeric_columns:

        series = df[column].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = (
            (series < lower) |
            (series > upper)
        ).sum()

        results.append({
            "Column": column,
            "Outliers": int(count),
            "Outlier %": round(
                count / len(series) * 100,
                2
            )
        })

    return pd.DataFrame(results)