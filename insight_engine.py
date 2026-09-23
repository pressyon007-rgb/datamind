# ==========================================================
# BUSINESS INSIGHT ENGINE
# ==========================================================

import pandas as pd
import numpy as np


def _find_column(df, keywords):
    """
    Find a column based on business-related keywords.
    """

    for keyword in keywords:

        for column in df.columns:

            if keyword.lower() in str(column).lower():

                return column

    return None


def _clean_value(value):
    """
    Convert a dataframe value into simple readable text.
    """

    if pd.isna(value):
        return "unknown"

    return str(value)


def generate_insights(df):
    """
    Generate simple business insights from the uploaded dataset.

    The output is designed for non-technical users.
    """

    insights = []

    # ======================================================
    # COLUMN GROUPS
    # ======================================================

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    # ======================================================
    # 1. EMPLOYEE / PEOPLE DATA
    # ======================================================

    employee_column = _find_column(
        df,
        [
            "employee",
            "staff",
            "worker",
            "person"
        ]
    )

    attrition_column = _find_column(
        df,
        [
            "attrition",
            "left",
            "churn",
            "exit",
            "status"
        ]
    )

    department_column = _find_column(
        df,
        [
            "department",
            "team",
            "division"
        ]
    )

    job_column = _find_column(
        df,
        [
            "jobrole",
            "job role",
            "role",
            "position"
        ]
    )

    travel_column = _find_column(
        df,
        [
            "travel",
            "business travel"
        ]
    )

    satisfaction_column = _find_column(
        df,
        [
            "satisfaction",
            "job satisfaction",
            "environment satisfaction"
        ]
    )

    if attrition_column:

        values = (
            df[attrition_column]
            .dropna()
            .astype(str)
            .str.lower()
        )

        leaving_words = [
            "yes",
            "left",
            "leave",
            "leaving",
            "exit",
            "churn"
        ]

        leaving_count = values.isin(
            leaving_words
        ).sum()

        if leaving_count > 0:

            insights.append({
                "Severity": "HIGH",
                "Area": "Employee Retention",

                "Problem":
                    "Some employees are leaving the company.",

                "Business Impact":
                    "The company may lose experienced employees "
                    "and may need to spend more time finding and "
                    "training replacements.",

                "Evidence":
                    f"The {attrition_column} information shows "
                    f"employees who have left."
            })

        else:

            insights.append({
                "Severity": "LOW",
                "Area": "Employee Stability",

                "Problem":
                    "Most employees appear to be continuing "
                    "with the company.",

                "Business Impact":
                    "Keeping employees for longer can help the "
                    "company maintain experience and reduce "
                    "the need for frequent replacements.",

                "Evidence":
                    f"The {attrition_column} information shows "
                    f"employee status."
            })

    # ======================================================
    # 2. DEPARTMENT
    # ======================================================

    if department_column:

        counts = (
            df[department_column]
            .dropna()
            .astype(str)
            .value_counts()
        )

        if not counts.empty:

            top_department = (
                counts.index[0]
            )

            insights.append({
                "Severity": "MEDIUM",
                "Area": "Department Management",

                "Problem":
                    f"{top_department} is one of the main "
                    f"departments represented in the data.",

                "Business Impact":
                    "A department with many employees may need "
                    "enough managers, resources and development "
                    "opportunities.",

                "Evidence":
                    f"{top_department} has the largest employee "
                    f"representation in the dataset."
            })

    # ======================================================
    # 3. JOB ROLE
    # ======================================================

    if job_column:

        counts = (
            df[job_column]
            .dropna()
            .astype(str)
            .value_counts()
        )

        if not counts.empty:

            top_role = counts.index[0]

            insights.append({
                "Severity": "MEDIUM",
                "Area": "Job Role Development",

                "Problem":
                    f"{top_role} is one of the main job roles "
                    "in the dataset.",

                "Business Impact":
                    "Important job roles should have suitable "
                    "training, career growth and support.",

                "Evidence":
                    f"{top_role} appears frequently in the "
                    f"{job_column} information."
            })

    # ======================================================
    # 4. WORK TRAVEL
    # ======================================================

    if travel_column:

        counts = (
            df[travel_column]
            .dropna()
            .astype(str)
            .value_counts()
        )

        if not counts.empty:

            common_travel = counts.index[0]

            if "rare" in common_travel.lower():

                insights.append({
                    "Severity": "MEDIUM",
                    "Area": "Work Travel",

                    "Problem":
                        "Most employees rarely travel for work.",

                    "Business Impact":
                        "The company can check whether employees "
                        "have enough opportunities for client "
                        "interaction, training and career development.",

                    "Evidence":
                        f"{common_travel} is the most common "
                        f"value in {travel_column}."
                })

            else:

                insights.append({
                    "Severity": "LOW",
                    "Area": "Work Travel",

                    "Problem":
                        f"{common_travel} is the most common "
                        "work-travel pattern.",

                    "Business Impact":
                        "The company should make sure travel "
                        "requirements are suitable for employees "
                        "and business needs.",

                    "Evidence":
                        f"{common_travel} is the most common "
                        f"value in {travel_column}."
                })

    # ======================================================
    # 5. SATISFACTION
    # ======================================================

    if satisfaction_column:

        series = pd.to_numeric(
            df[satisfaction_column],
            errors="coerce"
        ).dropna()

        if not series.empty:

            average = series.mean()

            if average <= 2:

                severity = "HIGH"

                problem = (
                    "Employee satisfaction appears to need attention."
                )

                impact = (
                    "Low satisfaction may affect employee "
                    "experience and retention."
                )

            elif average <= 3:

                severity = "MEDIUM"

                problem = (
                    "Employee satisfaction could be improved."
                )

                impact = (
                    "Improving employee experience may help "
                    "the company keep employees and create "
                    "a better workplace."
                )

            else:

                severity = "LOW"

                problem = (
                    "Employee satisfaction appears generally positive."
                )

                impact = (
                    "The company can continue the practices "
                    "that are supporting employee satisfaction."
                )

            insights.append({
                "Severity": severity,
                "Area": "Employee Satisfaction",
                "Problem": problem,
                "Business Impact": impact,
                "Evidence":
                    f"{satisfaction_column} contains employee "
                    "satisfaction information."
            })

    # ======================================================
    # 6. CUSTOMER DATA
    # ======================================================

    customer_column = _find_column(
        df,
        [
            "customer",
            "client",
            "buyer"
        ]
    )

    segment_column = _find_column(
        df,
        [
            "segment",
            "customer type",
            "customer segment"
        ]
    )

    if customer_column:

        unique_customers = (
            df[customer_column]
            .nunique()
        )

        insights.append({
            "Severity": "MEDIUM",
            "Area": "Customer Management",

            "Problem":
                "The dataset contains information about "
                "different customers.",

            "Business Impact":
                "The business needs to understand customer "
                "needs and provide suitable services to "
                "important customer groups.",

            "Evidence":
                f"The data contains approximately "
                f"{unique_customers:,} different customer values."
        })

    if segment_column:

        counts = (
            df[segment_column]
            .dropna()
            .astype(str)
            .value_counts()
        )

        if not counts.empty:

            common_segment = counts.index[0]

            insights.append({
                "Severity": "MEDIUM",
                "Area": "Customer Segments",

                "Problem":
                    f"{common_segment} is the most common "
                    "customer group.",

                "Business Impact":
                    "The business can understand this group "
                    "better and create services or offers "
                    "that match its needs.",

                "Evidence":
                    f"{common_segment} appears most often "
                    f"in {segment_column}."
            })

    # ======================================================
    # 7. SALES / REVENUE
    # ======================================================

    sales_column = _find_column(
        df,
        [
            "sales",
            "revenue",
            "income",
            "amount"
        ]
    )

    product_column = _find_column(
        df,
        [
            "product",
            "item",
            "product name"
        ]
    )

    region_column = _find_column(
        df,
        [
            "region",
            "state",
            "city",
            "location",
            "market"
        ]
    )

    if sales_column:

        if product_column:

            try:

                grouped = (
                    df.groupby(
                        product_column
                    )[sales_column]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                if not grouped.empty:

                    top_product = (
                        grouped.index[0]
                    )

                    insights.append({
                        "Severity": "LOW",
                        "Area": "Product Performance",

                        "Problem":
                            f"{top_product} is one of the "
                            "main products in the data.",

                        "Business Impact":
                            "The business can understand why "
                            "customers are buying this product "
                            "and use that learning for other products.",

                        "Evidence":
                            f"{top_product} has the highest "
                            f"total {sales_column} in the available data."
                    })

            except Exception:
                pass

        if region_column:

            try:

                grouped = (
                    df.groupby(
                        region_column
                    )[sales_column]
                    .sum()
                    .sort_values(
                        ascending=False
                    )
                )

                if not grouped.empty:

                    top_region = (
                        grouped.index[0]
                    )

                    insights.append({
                        "Severity": "MEDIUM",
                        "Area": "Market Performance",

                        "Problem":
                            f"{top_region} is one of the "
                            "strongest markets in the data.",

                        "Business Impact":
                            "The business can study what works "
                            "well in this market and consider "
                            "whether similar ideas can be used "
                            "in other markets.",

                        "Evidence":
                            f"{top_region} has the highest "
                            f"total {sales_column}."
                    })

            except Exception:
                pass

    # ======================================================
    # 8. MARKETING DATA
    # ======================================================

    campaign_column = _find_column(
        df,
        [
            "campaign",
            "promotion",
            "offer",
            "marketing"
        ]
    )

    response_column = _find_column(
        df,
        [
            "response",
            "converted",
            "conversion",
            "accepted"
        ]
    )

    channel_column = _find_column(
        df,
        [
            "channel",
            "source",
            "medium"
        ]
    )

    if campaign_column:

        counts = (
            df[campaign_column]
            .dropna()
            .astype(str)
            .value_counts()
        )

        if not counts.empty:

            common_campaign = (
                counts.index[0]
            )

            insights.append({
                "Severity": "MEDIUM",
                "Area": "Marketing Campaigns",

                "Problem":
                    f"{common_campaign} is one of the main "
                    "campaigns represented in the data.",

                "Business Impact":
                    "The business can review how customers "
                    "respond to different campaigns and improve "
                    "future marketing activities.",

                "Evidence":
                    f"{common_campaign} appears frequently "
                    f"in {campaign_column}."
            })

    if response_column:

        insights.append({
            "Severity": "MEDIUM",
            "Area": "Customer Response",

            "Problem":
                "The dataset contains information about "
                "how customers respond to business activities.",

            "Business Impact":
                "Understanding customer responses can help "
                "the business improve future offers and campaigns.",

            "Evidence":
                f"{response_column} contains customer response information."
        })

    if channel_column:

        counts = (
            df[channel_column]
            .dropna()
            .astype(str)
            .value_counts()
        )

        if not counts.empty:

            common_channel = counts.index[0]

            insights.append({
                "Severity": "LOW",
                "Area": "Business Channels",

                "Problem":
                    f"{common_channel} is a commonly used "
                    "business channel.",

                "Business Impact":
                    "The business can understand which channels "
                    "customers use and improve communication "
                    "through those channels.",

                "Evidence":
                    f"{common_channel} appears most often "
                    f"in {channel_column}."
            })

    # ======================================================
    # 9. GENERAL DATA QUALITY
    # ======================================================

    missing_total = int(
        df.isna().sum().sum()
    )

    if missing_total > 0:

        insights.append({
            "Severity": "MEDIUM",
            "Area": "Information Quality",

            "Problem":
                "Some information is missing from the dataset.",

            "Business Impact":
                "Missing information can make it harder "
                "for the business to understand customers, "
                "employees, products or other important areas.",

            "Evidence":
                f"The dataset contains {missing_total:,} "
                "missing values."
        })

    # ======================================================
    # 10. DUPLICATE DATA
    # ======================================================

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count > 0:

        insights.append({
            "Severity": "MEDIUM",
            "Area": "Data Records",

            "Problem":
                "Some records appear more than once.",

            "Business Impact":
                "Repeated records may make the business "
                "information less reliable.",

            "Evidence":
                f"{duplicate_count:,} duplicate records "
                "were found."
        })

    # ======================================================
    # 11. FALLBACK
    # ======================================================

    if not insights:

        insights.append({
            "Severity": "LOW",
            "Area": "Business Data",

            "Problem":
                "The uploaded data contains several business "
                "information areas that can be explored.",

            "Business Impact":
                "The business can use the dashboard to "
                "understand its customers, operations or "
                "performance.",

            "Evidence":
                f"The dataset contains {len(df):,} records "
                f"and {len(df.columns)} columns."
        })

    # ======================================================
    # REMOVE DUPLICATE AREAS
    # ======================================================

    final_rows = []

    used_areas = set()

    for item in insights:

        area = item.get(
            "Area",
            "Business Area"
        )

        if area in used_areas:
            continue

        used_areas.add(area)

        final_rows.append({
            "Severity":
                item.get(
                    "Severity",
                    "MEDIUM"
                ),

            "Area":
                area,

            "Problem":
                item.get(
                    "Problem",
                    "A business situation needs attention."
                ),

            "Business Impact":
                item.get(
                    "Business Impact",
                    "This situation may affect business performance."
                ),

            "Evidence":
                item.get(
                    "Evidence",
                    "Found in the uploaded data."
                )
        })

    # ======================================================
    # RETURN DATAFRAME
    # ======================================================

    return pd.DataFrame(
        final_rows,
        columns=[
            "Severity",
            "Area",
            "Problem",
            "Business Impact",
            "Evidence"
        ]
    )