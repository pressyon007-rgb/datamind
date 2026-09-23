import pandas as pd
import numpy as np


def _find_column(df, keywords):
    for keyword in keywords:
        for column in df.columns:
            if keyword.lower() in str(column).lower():
                return column
    return None


def generate_insights(df):
    insights = []

    employee_column = _find_column(df, ["employee", "staff", "worker", "person"])
    attrition_column = _find_column(df, ["attrition", "left", "churn", "exit", "status"])
    department_column = _find_column(df, ["department", "team", "division"])
    job_column = _find_column(df, ["jobrole", "job role", "role", "position"])
    travel_column = _find_column(df, ["travel", "business travel"])
    satisfaction_column = _find_column(df, ["satisfaction", "job satisfaction", "environment satisfaction"])

    if attrition_column:
        values = df[attrition_column].dropna().astype(str).str.lower()
        leaving_words = ["yes", "left", "leave", "leaving", "exit", "churn"]
        leaving_count = values.isin(leaving_words).sum()

        if leaving_count > 0:
            insights.append({
                "Severity": "HIGH",
                "Area": "Employee Retention",
                "Problem": "Some employees are leaving the company.",
                "Business Impact": "The company may lose experienced employees and spend resources replacing them.",
                "Evidence": f"The {attrition_column} column indicates employee departures."
            })

    if department_column:
        counts = df[department_column].dropna().astype(str).value_counts()
        if not counts.empty:
            top_dept = counts.index[0]
            insights.append({
                "Severity": "MEDIUM",
                "Area": "Department Management",
                "Problem": f"{top_dept} is the primary department in the dataset.",
                "Business Impact": "Larger teams require focused managerial support and resources.",
                "Evidence": f"{top_dept} has the highest representation in {department_column}."
            })

    sales_column = _find_column(df, ["sales", "revenue", "income", "amount"])
    product_column = _find_column(df, ["product", "item", "product name"])

    if sales_column and product_column:
        try:
            if pd.api.types.is_numeric_dtype(df[sales_column]):
                grouped = df.groupby(product_column)[sales_column].sum().sort_values(ascending=False)
                if not grouped.empty:
                    top_prod = grouped.index[0]
                    insights.append({
                        "Severity": "LOW",
                        "Area": "Product Performance",
                        "Problem": f"{top_prod} is the top revenue-generating product.",
                        "Business Impact": "Replicate success factors from top products across lower performers.",
                        "Evidence": f"{top_prod} generated the highest total in {sales_column}."
                    })
        except Exception:
            pass

    missing_total = int(df.isna().sum().sum())
    if missing_total > 0:
        insights.append({
            "Severity": "MEDIUM",
            "Area": "Information Quality",
            "Problem": "Missing values were detected in the dataset.",
            "Business Impact": "Incomplete records can distort strategic analysis.",
            "Evidence": f"{missing_total:,} missing values found."
        })

    if not insights:
        insights.append({
            "Severity": "LOW",
            "Area": "Business Data",
            "Problem": "Dataset ingested successfully with standard attributes.",
            "Business Impact": "Use dashboard visualizations for key metric breakdowns.",
            "Evidence": f"{len(df):,} rows and {len(df.columns)} columns."
        })

    final_rows = []
    used_areas = set()
    for item in insights:
        area = item.get("Area", "Business Area")
        if area in used_areas:
            continue
        used_areas.add(area)
        final_rows.append(item)

    return pd.DataFrame(final_rows)
